// CUDA kernels for the VoxCPM2 engine. Storage is bf16, arithmetic fp32.
// Where PyTorch rounds an intermediate to bf16, the kernels round at the same point (rb), so the
// engine tracks the reference implementation to within accumulation-order noise.
//
// Every kernel reads positions and step counters from device memory, never from launch arguments,
// so one captured CUDA graph can be replayed for every decoding step.
#include <cuda_bf16.h>

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 b(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float rb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// Sum over the block; blockDim.x must be a multiple of 32 and at most 1024.
__device__ float block_sum(float v) {
    __shared__ float part[32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) part[w] = v;
    __syncthreads();
    int nw = blockDim.x >> 5;
    v = l < nw ? part[l] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__device__ float block_max(float v) {
    __shared__ float part[32];
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) part[w] = v;
    __syncthreads();
    int nw = blockDim.x >> 5;
    v = l < nw ? part[l] : -INFINITY;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// out = bf16(bf16(x * rsqrt(mean(x^2) + eps)) * w), one block per row.
extern "C" __global__ void rmsnorm(const bf16* x, const bf16* w, bf16* out, int dim, float eps) {
    const bf16* xr = x + (long)blockIdx.x * dim;
    bf16* yr = out + (long)blockIdx.x * dim;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) { float v = f(xr[i]); ss += v * v; }
    float r = rsqrtf(block_sum(ss) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) yr[i] = b(rb(f(xr[i]) * r) * f(w[i]));
}

// res += d (bf16), then out = rmsnorm(res). One block per row.
extern "C" __global__ void add_rmsnorm(bf16* res, const bf16* d, const bf16* w, bf16* out, int dim, float eps) {
    bf16* xr = res + (long)blockIdx.x * dim;
    const bf16* dr = d + (long)blockIdx.x * dim;
    bf16* yr = out + (long)blockIdx.x * dim;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = rb(f(xr[i]) + f(dr[i]));
        xr[i] = b(v);
        ss += v * v;
    }
    float r = rsqrtf(block_sum(ss) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) yr[i] = b(rb(f(xr[i]) * r) * f(w[i]));
}

// res += d elementwise (bf16).
extern "C" __global__ void add_inplace(bf16* res, const bf16* d, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) res[i] = b(f(res[i]) + f(d[i]));
}

// y[r, c] = bias[c] for an output whose rows are ld apart; the GEMM then accumulates with beta = 1,
// which rounds acc + bias once, like torch's addmm.
extern "C" __global__ void fill_bias(bf16* y, const bf16* bias, int rows, int n, int ld) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (long)rows * n) { int r = i / n, c = i % n; y[(long)r * ld + c] = bias[c]; }
}

extern "C" __global__ void silu_inplace(bf16* x, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float g = f(x[i]); x[i] = b(g / (1.f + __expf(-g))); }
}

// dst[r, 0..n) = src[r, 0..n) with independent row strides.
extern "C" __global__ void copy2d(const bf16* src, int lds, bf16* dst, int ldd, int rows, int n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (long)rows * n) { int r = i / n, c = i % n; dst[(long)r * ldd + c] = src[(long)r * lds + c]; }
}

// gu = [gate | up] per row (2F wide) -> out = bf16(bf16(silu(gate)) * up)
extern "C" __global__ void silu_mul(const bf16* gu, bf16* out, int rows, int F) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)rows * F) return;
    int r = i / F, c = i % F;
    float g = f(gu[(long)r * 2 * F + c]), u = f(gu[(long)r * 2 * F + F + c]);
    out[i] = b(rb(g / (1.f + __expf(-g))) * u);
}

// In-place rotary embedding on the q and k heads of a fused qkv buffer (row = [q 16x128 | k 2x128 | v 2x128]).
// Position of row r: r % seq when seq > 0 (independent sequences of length seq), else *pos (decode).
extern "C" __global__ void rope(bf16* qkv, const float* cos_t, const float* sin_t, const int* pos, int rows,
                                int seq, int nq, int nk, int hd) {
    int r = blockIdx.x, h = blockIdx.y;
    if (h >= nq + nk || r >= rows) return;
    int p = seq > 0 ? r % seq : *pos;
    int half = hd / 2;
    bf16* v = qkv + (long)r * (nq + 2 * nk) * hd + h * hd;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float c = cos_t[(long)p * half + i], s = sin_t[(long)p * half + i];
        float x1 = f(v[i]), x2 = f(v[i + half]);
        v[i] = b(x1 * c - x2 * s);
        v[i + half] = b(x2 * c + x1 * s);
    }
}

// Copy k and v of `rows` consecutive positions starting at *pos (or at 0 when pos is null) of every slot
// into the per-layer cache [slots, nk, maxlen, hd]. qkv rows are [slot * rows + i].
extern "C" __global__ void kv_write(const bf16* qkv, bf16* kc, bf16* vc, const int* pos, int rows, int nq,
                                    int nk, int hd, int maxlen) {
    int i = blockIdx.x, s = blockIdx.y, h = blockIdx.z;
    int p = (pos ? *pos : 0) + i;
    const bf16* src = qkv + ((long)s * rows + i) * (nq + 2 * nk) * hd;
    long dst = (((long)s * nk + h) * maxlen + p) * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        kc[dst + d] = src[(nq + h) * hd + d];
        vc[dst + d] = src[(nq + nk + h) * hd + d];
    }
}

// Decode attention, split over the sequence (flash-decoding). grid (slots, nq, splits), block 128.
// Each block handles positions [split * CH, split * CH + CH) up to *pos and writes a partial (max, sum, out);
// attn_combine merges the partials. The grid is fixed so the launch can live in a CUDA graph.
#define ATT_CH 64
extern "C" __global__ void attn_decode(const bf16* qkv, const bf16* kc, const bf16* vc, float* part,
                                       const int* pos, int nq, int nk, int hd, int maxlen, float scale) {
    __shared__ float q[128];
    __shared__ float sc[ATT_CH];
    int s = blockIdx.x, h = blockIdx.y, sp = blockIdx.z, nsp = gridDim.z, kh = h / (nq / nk);
    int L = *pos + 1, j0 = sp * ATT_CH;
    float* pr = part + (((long)s * nq + h) * nsp + sp) * (hd + 2);
    if (j0 >= L) { if (threadIdx.x == 0) { pr[0] = -INFINITY; pr[1] = 0.f; } return; }
    int n = min(ATT_CH, L - j0);
    const bf16* qr = qkv + (long)s * (nq + 2 * nk) * hd + h * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) q[d] = f(qr[d]);
    __syncthreads();
    const bf16* K = kc + (((long)s * nk + kh) * maxlen + j0) * hd;
    const bf16* V = vc + (((long)s * nk + kh) * maxlen + j0) * hd;
    int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int j = w; j < n; j += nw) {
        const __nv_bfloat162* k2 = (const __nv_bfloat162*)(K + (long)j * hd);
        float a = 0.f;
        for (int d = l; d < hd / 2; d += 32) { float2 kv = __bfloat1622float2(k2[d]); a += q[2 * d] * kv.x + q[2 * d + 1] * kv.y; }
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
        if (l == 0) sc[j] = a * scale;
    }
    __syncthreads();
    float m = -INFINITY;
    for (int j = 0; j < n; j++) m = fmaxf(m, sc[j]);
    float sum = 0.f;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float a = 0.f;
        sum = 0.f;
        for (int j = 0; j < n; j++) { float e = __expf(sc[j] - m); sum += e; a += e * f(V[(long)j * hd + d]); }
        pr[2 + d] = a;
    }
    if (threadIdx.x == 0) { pr[0] = m; pr[1] = sum; }
}

// Merge the split partials of attn_decode. grid (slots, nq), block hd.
extern "C" __global__ void attn_combine(const float* part, bf16* out, int nq, int hd, int nsp) {
    int s = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const float* pr = part + ((long)s * nq + h) * nsp * (hd + 2);
    float m = -INFINITY;
    for (int i = 0; i < nsp; i++) m = fmaxf(m, pr[i * (hd + 2)]);
    float sum = 0.f, a = 0.f;
    for (int i = 0; i < nsp; i++) {
        float pm = pr[i * (hd + 2)];
        if (pm == -INFINITY) continue;
        float w = __expf(pm - m);
        sum += w * pr[i * (hd + 2) + 1];
        a += w * pr[i * (hd + 2) + 2 + d];
    }
    out[(long)s * nq * hd + h * hd + d] = b(a / sum);
}

// Skinny GEMM for decode-sized M: y[m, n] = x[m, :] . W[n, :] (+ y when beta), M <= MR.
// Each warp owns COLS output columns and streams their weight rows once with 16-byte loads;
// x is staged in shared memory in K chunks and reused by every column of the block.
template <int MR, int COLS, int KC>
__device__ __forceinline__ void gemv_body(const bf16* __restrict__ x, int ldx, const bf16* __restrict__ W, bf16* y,
                                          int ldy, int M, int N, int K, int beta) {
    extern __shared__ __align__(16) unsigned char smraw[];
    bf16* xs = (bf16*)smraw;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n0 = (blockIdx.x * (blockDim.x >> 5) + warp) * COLS;
    float acc[COLS][MR];
#pragma unroll
    for (int c = 0; c < COLS; c++)
#pragma unroll
        for (int m = 0; m < MR; m++) acc[c][m] = 0.f;
    for (int k0 = 0; k0 < K; k0 += KC) {
        int kc = min(KC, K - k0);
        __syncthreads();
        for (int i = threadIdx.x; i < M * (kc / 8); i += blockDim.x) {
            int m = i / (kc / 8), j = i % (kc / 8);
            ((uint4*)(xs + m * KC))[j] = *(const uint4*)(x + (long)m * ldx + k0 + j * 8);
        }
        __syncthreads();
        for (int kk = lane * 8; kk < kc; kk += 256) {
            float w[COLS][8];
#pragma unroll
            for (int c = 0; c < COLS; c++) {
                if (n0 + c < N) {
                    uint4 u = __ldg((const uint4*)(W + (long)(n0 + c) * K + k0 + kk));
                    const __nv_bfloat162* p = (const __nv_bfloat162*)&u;
#pragma unroll
                    for (int j = 0; j < 4; j++) { float2 v = __bfloat1622float2(p[j]); w[c][2 * j] = v.x; w[c][2 * j + 1] = v.y; }
                } else {
#pragma unroll
                    for (int j = 0; j < 8; j++) w[c][j] = 0.f;
                }
            }
#pragma unroll
            for (int m = 0; m < MR; m++) {
                if (m < M) {
                    uint4 u = *(const uint4*)(xs + m * KC + kk);
                    const __nv_bfloat162* p = (const __nv_bfloat162*)&u;
                    float xf[8];
#pragma unroll
                    for (int j = 0; j < 4; j++) { float2 v = __bfloat1622float2(p[j]); xf[2 * j] = v.x; xf[2 * j + 1] = v.y; }
#pragma unroll
                    for (int c = 0; c < COLS; c++)
#pragma unroll
                        for (int j = 0; j < 8; j++) acc[c][m] += w[c][j] * xf[j];
                }
            }
        }
    }
#pragma unroll
    for (int c = 0; c < COLS; c++) {
#pragma unroll
        for (int m = 0; m < MR; m++) {
            float v = acc[c][m];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
            if (lane == (m & 31) && m < M && n0 + c < N) {
                bf16* yp = y + (long)m * ldy + n0 + c;
                *yp = b(beta ? v + f(*yp) : v);
            }
        }
    }
}

#define GEMV(MR, COLS, KC) \
    extern "C" __global__ void gemv_##MR(const bf16* x, int ldx, const bf16* W, bf16* y, int ldy, int M, int N, int K, int beta) { \
        gemv_body<MR, COLS, KC>(x, ldx, W, y, ldy, M, N, K, beta); }
GEMV(1, 4, 2048)
GEMV(2, 4, 2048)
GEMV(4, 4, 1024)
GEMV(8, 4, 1024)

// Attention within independent sequences of length seq held in a fused qkv buffer.
// grid (nseq * seq query rows, nq), block 128; causal masks future keys.
extern "C" __global__ void attn_full(const bf16* qkv, bf16* out, int seq, int causal, int nq, int nk, int hd,
                                     float scale) {
    extern __shared__ float sm[];
    float* q = sm;
    float* sc = sm + hd;
    int r = blockIdx.x, h = blockIdx.y, kh = h / (nq / nk);
    int s0 = r - r % seq, i = r % seq;
    int L = causal ? i + 1 : seq;
    int stride = (nq + 2 * nk) * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) q[d] = f(qkv[(long)r * stride + h * hd + d]);
    __syncthreads();
    int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int j = w; j < L; j += nw) {
        const bf16* k = qkv + (long)(s0 + j) * stride + (nq + kh) * hd;
        float a = 0.f;
        for (int d = l; d < hd; d += 32) a += q[d] * f(k[d]);
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
        if (l == 0) sc[j] = a * scale;
    }
    __syncthreads();
    float m = -INFINITY;
    for (int j = threadIdx.x; j < L; j += blockDim.x) m = fmaxf(m, sc[j]);
    m = block_max(m);
    float sum = 0.f;
    for (int j = threadIdx.x; j < L; j += blockDim.x) { float e = __expf(sc[j] - m); sc[j] = e; sum += e; }
    sum = block_sum(sum);
    __syncthreads();
    float inv = 1.f / sum;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float a = 0.f;
        for (int j = 0; j < L; j++) a += sc[j] * f(qkv[(long)(s0 + j) * stride + (nq + nk + kh) * hd + d]);
        out[(long)r * nq * hd + h * hd + d] = b(a * inv);
    }
}

// DiT input sequence per CFG row: [mu0, mu1, t, cond x4, x x4] (11 tokens x H).
// Rows 0..B-1 are conditional (mu = dit_hidden), rows B..2B-1 unconditional (mu = 0).
// condp and xp are [B, 4, H] (projected once per slot, shared by both halves).
extern "C" __global__ void dit_build(bf16* seq, const bf16* mu, const bf16* temb, const bf16* condp,
                                     const bf16* xp, int B, int H) {
    int r = blockIdx.x, tok = blockIdx.y;
    int s = r % B;
    bf16* dst = seq + ((long)r * 11 + tok) * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
        bf16 v;
        if (tok < 2) v = r < B ? mu[(long)s * 2 * H + tok * H + c] : b(0.f);
        else if (tok == 2) v = temb[c];
        else if (tok < 7) v = condp[((long)s * 4 + tok - 3) * H + c];
        else v = xp[((long)s * 4 + tok - 7) * H + c];
        dst[c] = v;
    }
}

// Gather rows [first, first + n) of every length-`seq` sequence: out[r * n + i] = in[r * seq + first + i].
extern "C" __global__ void gather_rows(const bf16* in, bf16* out, int seq, int first, int n, int H) {
    int r = blockIdx.x, i = blockIdx.y;
    const bf16* src = in + ((long)r * seq + first + i) * H;
    bf16* dst = out + ((long)r * n + i) * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x) dst[c] = src[c];
}

// Local-encoder input: [special, x0..x3] per patch. xp is [n, 4, H].
extern "C" __global__ void enc_build(bf16* seq, const bf16* special, const bf16* xp, int H) {
    int r = blockIdx.x, tok = blockIdx.y;
    bf16* dst = seq + ((long)r * 5 + tok) * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x)
        dst[c] = tok == 0 ? special[c] : xp[((long)r * 4 + tok - 1) * H + c];
}

// Classifier-free guidance with CFG-Zero* and one Euler step, mirroring UnifiedCFM.solve_euler in bf16.
// out is [2B, 4*64] (cond rows then uncond rows), x is [B, 4*64]. One block (256 threads) per slot.
extern "C" __global__ void cfg_euler(bf16* x, const bf16* out, int B, float dt, float cfg) {
    int s = blockIdx.x, i = threadIdx.x;
    float pos = f(out[(long)s * 256 + i]), neg = f(out[(long)(B + s) * 256 + i]);
    float dot = rb(block_sum(rb(pos * neg)));
    float nn = rb(rb(block_sum(rb(neg * neg))) + 1e-8f);
    float st = rb(dot / nn);
    float a = rb(neg * st);
    float d = rb(a + rb(cfg * rb(pos - a)));
    long xi = (long)s * 256 + i;
    x[xi] = b(f(x[xi]) - rb(dt * d));
}

// Diffusion start noise for every slot, [B, 4, 64] time-major.
// table != null: replay a [N, 64, 4] table (verification); else counter-based Gaussian noise keyed by (seed, slot, step).
__device__ unsigned int mix32(unsigned int h) {
    h ^= h >> 16; h *= 0x7feb352d; h ^= h >> 15; h *= 0x846ca68b; h ^= h >> 16;
    return h;
}
extern "C" __global__ void noise(bf16* x, const float* table, const int* step, const unsigned int* seeds, int B) {
    int s = blockIdx.x, i = threadIdx.x;  // i = t * 64 + c
    int t = i / 64, c = i % 64;
    float v;
    if (table) {
        v = table[((long)*step * 64 + c) * 4 + t];
    } else {
        unsigned int k = mix32(seeds[s] * 0x9E3779B9u ^ mix32((unsigned int)*step * 0x85EBCA6Bu + i));
        unsigned int k2 = mix32(k ^ 0x68E31DA4u);
        float u1 = ((k >> 8) + 1) * (1.f / 16777217.f), u2 = (k2 >> 8) * (1.f / 16777216.f);
        v = sqrtf(-2.f * __logf(u1)) * __cosf(6.2831853f * u2);
    }
    x[(long)s * 256 + i] = b(v);
}

// h = round(tanh(h) * 9) / 9 with torch's bf16 roundings (ScalarQuantizationLayer, inference).
extern "C" __global__ void fsq_act(bf16* h, long n, float scale) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float t = rb(tanhf(f(h[i])));
    h[i] = b(rintf(rb(t * scale)) / scale);
}

// Prefill input: combined = text_mask * embed[token] + audio_mask * feat_embed. feat_embed [T, H].
extern "C" __global__ void prefill_combine(bf16* out, const int* tokens, const bf16* emb, const bf16* feat_embed,
                                           const int* tmask, const int* amask, int H) {
    int r = blockIdx.x;
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
        float te = tmask[r] ? f(emb[(long)tokens[r] * H + c]) : 0.f;
        float fe = amask[r] ? f(feat_embed[(long)r * H + c]) : 0.f;
        out[(long)r * H + c] = b(te + fe);
    }
}

// After base-LM prefill: mix = fsq * audio_mask + enc * text_mask, and the residual LM input
// resin = [mix | audio_mask * feat_embed] (T x 2H).
extern "C" __global__ void prefill_mix(const bf16* enc, const bf16* fsq, const bf16* feat_embed, const int* tmask,
                                       const int* amask, bf16* mix, bf16* resin, int H) {
    int r = blockIdx.x;
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
        long i = (long)r * H + c;
        float m = rb((amask[r] ? f(fsq[i]) : 0.f) + (tmask[r] ? f(enc[i]) : 0.f));
        mix[i] = b(m);
        resin[(long)r * 2 * H + c] = b(m);
        resin[(long)r * 2 * H + H + c] = amask[r] ? feat_embed[i] : b(0.f);
    }
}

// Copy the last row of a [T, H] buffer into every one of B slots of a [B, ld] buffer (prefill -> decode state).
extern "C" __global__ void broadcast_row(const bf16* src, bf16* dst, int T, int H, int ld) {
    int s = blockIdx.x;
    for (int c = threadIdx.x; c < H; c += blockDim.x) dst[(long)s * ld + c] = src[(long)(T - 1) * H + c];
}

// Replicate slot 0 of a per-layer KV cache [slots, nk, maxlen, hd] into slots 1..B-1 for positions < T.
extern "C" __global__ void kv_replicate(bf16* c, int B, int nk, int maxlen, int hd, int T) {
    int p = blockIdx.x, h = blockIdx.y, s = blockIdx.z + 1;
    if (p >= T || s >= B) return;
    for (int d = threadIdx.x; d < hd; d += blockDim.x)
        c[(((long)s * nk + h) * maxlen + p) * hd + d] = c[((long)h * maxlen + p) * hd + d];
}

// End of a decoding step: argmax of the stop logits per slot, record the patch, advance the counters.
// ctr[0] = KV position of the next token, ctr[1] = generated patches so far.
// stop[s] holds the first step at which slot s asked to stop (or -1).
extern "C" __global__ void step_end(const bf16* stop_logits, const bf16* patch, bf16* patches, int* stop,
                                    int* ctr, int B, int min_len, int max_steps) {
    int n = ctr[1];
    if (n < max_steps)
        for (int i = threadIdx.x; i < B * 256; i += blockDim.x) {
            int s = i / 256;
            patches[((long)s * max_steps + n) * 256 + i % 256] = patch[i];
        }
    if (threadIdx.x < B) {
        int s = threadIdx.x;
        bool want = f(stop_logits[s * 2 + 1]) > f(stop_logits[s * 2]);
        if (want && n > min_len && stop[s] < 0) stop[s] = n;
    }
    __syncthreads();
    if (threadIdx.x == 0) { ctr[0] += 1; ctr[1] = n + 1; }
}

#include "vae.cu"
