// AudioVAE kernels, all fp32 like the reference (which runs the VAE in float32).
// Activations are [channels][time] with time contiguous, as in PyTorch's [B, C, T] with B = 1.
// Snake: x + 1 / (alpha + 1e-9) * sin(alpha * x)^2, evaluated in the same order as the reference.

__device__ __forceinline__ float snake(float x, float a) {
    float s = sinf(a * x);
    return x + (1.f / (a + 1e-9f)) * (s * s);
}

#define DW_TILE 512

// Front half of a causal residual unit, per channel c:
//   out[c][t] = snake2(bias[c] + sum_k w[c][k] * snake1(x[c][t - (6 - k) * dil]))   (zero before t = 0)
// alpha1 / alpha2 may be null (no snake). Row C of out (when ones != 0) is set to 1, the bias row of the
// following 1x1 convolution's GEMM. grid (ceil(T / DW_TILE), C + ones), block DW_TILE.
extern "C" __global__ void vae_dw(const float* x, float* out, const float* w, const float* bias, const float* alpha1,
                                  const float* alpha2, int C, int T, int dil, int ones) {
    __shared__ float s[DW_TILE + 6 * 9];
    int c = blockIdx.y, t0 = blockIdx.x * DW_TILE, halo = 6 * dil;
    if (c == C) {
        int t = t0 + threadIdx.x;
        if (ones && t < T) out[(long)C * T + t] = 1.f;
        return;
    }
    const float* xr = x + (long)c * T;
    float a1 = alpha1 ? alpha1[c] : 0.f;
    for (int i = threadIdx.x; i < DW_TILE + halo; i += blockDim.x) {
        int t = t0 - halo + i;
        float v = (t >= 0 && t < T) ? xr[t] : 0.f;
        s[i] = alpha1 ? snake(v, a1) : v;
    }
    __syncthreads();
    int t = t0 + threadIdx.x;
    if (t >= T) return;
    const float* wr = w + c * 7;
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 7; k++) acc += wr[k] * s[threadIdx.x + k * dil];
    acc += bias[c];
    out[(long)c * T + t] = alpha2 ? snake(acc, alpha2[c]) : acc;
}

// Input of a causal transposed convolution as one GEMM: rows [0, C) hold v[c][i] and rows [C, 2C) hold v[c][i - 1]
// (zero at i = 0), v = snake(x * scale + shift) with the sample-rate condition. grid over C * T.
extern "C" __global__ void vae_tconv_in(const float* x, float* out, const float* scale, const float* shift,
                                        const float* alpha, int C, int T) {
    long n = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= (long)C * T) return;
    int c = n / T, t = n % T;
    float v = x[n];
    if (scale) v = v * scale[c] + shift[c];
    v = snake(v, alpha[c]);
    out[n] = v;
    long o = (long)(C + c) * T;
    if (t + 1 < T) out[o + t + 1] = v;
    if (t == 0) out[o] = 0.f;
}

// Transposed-convolution weight [cin][cout][2s] to the GEMM layout [(r, o)][(h, c)] = w[c][o][r + h s].
extern "C" __global__ void vae_tconv_w(const float* w, float* tw, int cin, int cout, int s) {
    long n = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= (long)s * cout * 2 * cin) return;
    int c = n % cin;
    long q = n / cin;
    int h = q % 2;
    q /= 2;
    int o = q % cout, r = q / cout;
    tw[n] = w[((long)c * cout + o) * 2 * s + r + h * s];
}

// y[o][s * i + r] = p[(r * Cout + o)][i] + bias[o]: the phase-major GEMM output of a transposed convolution
// interleaved into time order. grid over Cout * T * s.
extern "C" __global__ void vae_unshuffle(const float* p, float* y, const float* bias, int Cout, int T, int s) {
    long n = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long L = (long)T * s;
    if (n >= (long)Cout * L) return;
    int o = n / L, u = n % L, i = u / s, r = u % s;
    y[n] = p[((long)r * Cout + o) * T + i] + bias[o];
}

// out[c * K + k][i] = f(x[c][s * i + k - pad]) (zero outside [0, L)), f = snake(alpha[c]) or identity when alpha is
// null; row C * K is set to 1 when ones != 0 (the GEMM bias row). Lout output positions. grid over (C * K + ones) * Lout.
extern "C" __global__ void vae_im2col(const float* x, float* out, const float* alpha, int C, int L, int K, int s,
                                      int pad, int Lout, int ones) {
    long n = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long rows = (long)C * K;
    if (n >= (rows + ones) * Lout) return;
    long row = n / Lout;
    int i = n % Lout;
    if (row == rows) {
        out[n] = 1.f;
        return;
    }
    int c = row / K, k = row % K;
    long t = (long)s * i + k - pad;
    float v = (t >= 0 && t < L) ? x[(long)c * L + t] : 0.f;
    out[n] = alpha ? snake(v, alpha[c]) : v;
}

#define OUT_TILE 256
#define OUT_C 32

// Last decoder layer: wav[t] = tanh(bias + sum_c sum_k w[c][k] * snake(x[c][t - 6 + k])), C <= OUT_C channels.
// grid ceil(T / OUT_TILE), block OUT_TILE.
extern "C" __global__ void vae_out(const float* x, float* wav, const float* w, const float* bias, const float* alpha,
                                   int C, int T) {
    __shared__ float s[OUT_C][OUT_TILE + 6];
    int t0 = blockIdx.x * OUT_TILE;
    for (int c = 0; c < C; c++) {
        float a = alpha[c];
        for (int i = threadIdx.x; i < OUT_TILE + 6; i += blockDim.x) {
            int t = t0 - 6 + i;
            s[c][i] = (t >= 0 && t < T) ? snake(x[(long)c * T + t], a) : 0.f;
        }
    }
    __syncthreads();
    int t = t0 + threadIdx.x;
    if (t >= T) return;
    float acc = 0.f;
    for (int c = 0; c < C; c++)
#pragma unroll
        for (int k = 0; k < 7; k++) acc += w[c * 7 + k] * s[c][threadIdx.x + k];
    wav[t] = tanhf(acc + bias[0]);
}
