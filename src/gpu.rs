//! Thin layer over the CUDA driver and cuBLAS: raw device buffers, kernel launches, bf16 GEMM.
//! Buffers live for the whole process, so plain device pointers are used instead of owned slices.
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use cudarc::cublas::{CudaBlas, result as blas, sys as bsys};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, result as drv, sys};
use cudarc::nvrtc::Ptx;

pub type R<T> = Result<T, Box<dyn std::error::Error>>;
pub type Ptr = u64;

/// Alignment of tensors placed in an Arena.
const ALIGN: usize = 256;

const KERNEL_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/voxcpm.fatbin"));
const KERNELS: &[&str] = &[
    "rmsnorm",
    "add_rmsnorm",
    "add_inplace",
    "fill_bias",
    "silu_inplace",
    "copy2d",
    "silu_mul",
    "rope",
    "kv_write",
    "attn_decode",
    "attn_combine",
    "attn_full",
    "gemv_1",
    "gemv_2",
    "gemv_4",
    "gemv_8",
    "dit_build",
    "gather_rows",
    "enc_build",
    "cfg_euler",
    "noise",
    "fsq_act",
    "prefill_combine",
    "prefill_mix",
    "broadcast_row",
    "kv_replicate",
    "step_end",
    "vae_dw",
    "vae_tconv_in",
    "vae_tconv_w",
    "vae_unshuffle",
    "vae_im2col",
    "vae_out",
];

#[derive(Clone, Copy)]
pub enum A {
    P(Ptr),
    I(i32),
    L(i64),
    F(f32),
}

pub struct Gpu {
    _ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    blas: CudaBlas,
    funcs: HashMap<&'static str, CudaFunction>,
    pub allocated: usize,
    /// use cuBLAS for every GEMM (comparison); otherwise M <= 8 uses the gemv kernels
    pub cublas_only: bool,
}

pub fn f2b(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7fc0;
    }
    let u = x.to_bits();
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

pub fn b2f(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Round an f32 through bf16, like a PyTorch bf16 op.
pub fn rb(x: f32) -> f32 {
    b2f(f2b(x))
}

impl Gpu {
    pub fn new() -> R<Self> {
        let ctx = CudaContext::new(0)?;
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream()?; // graph capture is not allowed on the legacy default stream
        let blas = CudaBlas::new(stream.clone())?;
        let module = ctx.load_module(Ptx::from_binary(KERNEL_IMAGE.to_vec()))?;
        let mut funcs = HashMap::new();
        for &k in KERNELS {
            funcs.insert(k, module.load_function(k)?);
        }
        let mut g = Gpu { _ctx: ctx, stream, blas, funcs, allocated: 0, cublas_only: false };
        // fixed cuBLAS workspace so GEMMs never allocate while a graph is being captured
        let ws_bytes = 64 << 20;
        let ws = g.alloc(ws_bytes)?;
        unsafe { bsys::cublasSetWorkspace_v2(*g.blas.handle(), ws as *mut c_void, ws_bytes).result()? };
        Ok(g)
    }

    pub fn alloc(&mut self, bytes: usize) -> R<Ptr> {
        let p = unsafe { drv::malloc_sync(bytes.max(16))? };
        unsafe { drv::memset_d8_async(p, 0, bytes.max(16), self.cu())? };
        self.allocated += bytes;
        Ok(p)
    }

    pub fn free(&mut self, p: Ptr, bytes: usize) -> R<()> {
        self.sync()?;
        unsafe { drv::free_sync(p)? };
        self.allocated -= bytes;
        Ok(())
    }

    pub fn alloc_bf16(&mut self, n: usize) -> R<Ptr> {
        self.alloc(n * 2)
    }

    pub fn cu(&self) -> sys::CUstream {
        self.stream.cu_stream()
    }

    pub fn upload<T>(&self, dst: Ptr, src: &[T]) -> R<()> {
        unsafe { drv::memcpy_htod_async(dst, src, self.cu())? };
        self.sync()
    }

    pub fn upload_new<T>(&mut self, src: &[T]) -> R<Ptr> {
        let p = self.alloc(std::mem::size_of_val(src))?;
        self.upload(p, src)?;
        Ok(p)
    }

    pub fn download<T: Default + Clone>(&self, src: Ptr, n: usize) -> R<Vec<T>> {
        let mut v = vec![T::default(); n];
        unsafe { drv::memcpy_dtoh_async(&mut v[..], src, self.cu())? };
        self.sync()?;
        Ok(v)
    }

    pub fn download_bf16(&self, src: Ptr, n: usize) -> R<Vec<f32>> {
        Ok(self.download::<u16>(src, n)?.into_iter().map(b2f).collect())
    }

    pub fn copy(&self, dst: Ptr, src: Ptr, bytes: usize) -> R<()> {
        unsafe { drv::memcpy_dtod_async(dst, src, bytes, self.cu())? };
        Ok(())
    }

    pub fn sync(&self) -> R<()> {
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn k(&self, name: &str, grid: (u32, u32, u32), block: u32, smem: u32, args: &[A]) -> R<()> {
        let mut vals = args.to_vec();
        let mut ptrs: Vec<*mut c_void> = vals
            .iter_mut()
            .map(|a| match a {
                A::P(v) => v as *mut u64 as *mut c_void,
                A::I(v) => v as *mut i32 as *mut c_void,
                A::L(v) => v as *mut i64 as *mut c_void,
                A::F(v) => v as *mut f32 as *mut c_void,
            })
            .collect();
        let f = self.funcs.get(name).ok_or_else(|| format!("no kernel {name}"))?;
        unsafe { drv::launch_kernel(f.cu_function(), grid, (block, 1, 1), smem, self.cu(), &mut ptrs)? };
        Ok(())
    }

    /// Elementwise launch over n items.
    pub fn k1(&self, name: &str, n: usize, args: &[A]) -> R<()> {
        self.k(name, (n.div_ceil(256) as u32, 1, 1), 256, 0, args)
    }

    /// y[m, n] (row stride ldy) = x[m, k] (row stride ldx) · W[n, k]^T, all bf16, fp32 accumulation.
    /// beta = 1 adds onto what y already holds (used for the bias).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(&self, x: Ptr, ldx: usize, w: Ptr, y: Ptr, ldy: usize, m: usize, n: usize, k: usize, beta: f32) -> R<()> {
        // measured with --gemm-bench: the gemv kernels beat cuBLAS up to M = 8, cuBLAS wins above
        if m <= 8 && !self.cublas_only {
            let (name, mr, cols, kc) = match m {
                1 => ("gemv_1", 1, 4, 2048),
                2 => ("gemv_2", 2, 4, 2048),
                3..=4 => ("gemv_4", 4, 4, 1024),
                _ => ("gemv_8", 8, 4, 1024),
            };
            let per_block = 8 * cols;
            return self.k(
                name,
                (n.div_ceil(per_block) as u32, 1, 1),
                256,
                (mr * kc * 2) as u32,
                &[A::P(x), A::I(ldx as i32), A::P(w), A::P(y), A::I(ldy as i32), A::I(m as i32), A::I(n as i32), A::I(k as i32), A::I((beta != 0.0) as i32)],
            );
        }
        let alpha = 1.0f32;
        unsafe {
            blas::gemm_ex(
                *self.blas.handle(),
                bsys::cublasOperation_t::CUBLAS_OP_T,
                bsys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const c_void,
                w as *const c_void,
                bsys::cudaDataType::CUDA_R_16BF,
                k as i32,
                x as *const c_void,
                bsys::cudaDataType::CUDA_R_16BF,
                ldx as i32,
                &beta as *const f32 as *const c_void,
                y as *mut c_void,
                bsys::cudaDataType::CUDA_R_16BF,
                ldy as i32,
                bsys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                bsys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )?
        };
        Ok(())
    }

    /// fp32 GEMM in cuBLAS's column-major terms: c (m x n, ld ldc) = a (m x k, ld lda) · b (k x n, ld ldb) + beta · c.
    /// With [channels][time] activations, m is time: a is the input [k][m], b a weight stored [n][k].
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm(&self, m: usize, n: usize, k: usize, a: Ptr, lda: usize, b: Ptr, ldb: usize, c: Ptr, ldc: usize, beta: f32) -> R<()> {
        let alpha = 1.0f32;
        unsafe {
            blas::gemm_ex(
                *self.blas.handle(),
                bsys::cublasOperation_t::CUBLAS_OP_N,
                bsys::cublasOperation_t::CUBLAS_OP_N,
                m as i32,
                n as i32,
                k as i32,
                &alpha as *const f32 as *const c_void,
                a as *const c_void,
                bsys::cudaDataType::CUDA_R_32F,
                lda as i32,
                b as *const c_void,
                bsys::cudaDataType::CUDA_R_32F,
                ldb as i32,
                &beta as *const f32 as *const c_void,
                c as *mut c_void,
                bsys::cudaDataType::CUDA_R_32F,
                ldc as i32,
                bsys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                bsys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )?
        };
        Ok(())
    }
}

/// Device memory for many read-only tensors in one allocation, each placed at a 256-byte boundary. Bytes go
/// through two pinned staging buffers, so copying into one overlaps the transfer of the other.
pub struct Arena {
    base: Ptr,
    used: usize,
    cap: usize,
    stage: [*mut u8; 2],
    ev: [sys::CUevent; 2],
    cur: usize,
    /// arena offset where the current staging buffer starts, and the bytes in it
    win: usize,
    fill: usize,
}

const STAGE: usize = 32 << 20;

impl Arena {
    pub fn new(g: &mut Gpu, cap: usize) -> R<Self> {
        let base = unsafe { drv::malloc_sync(cap.max(16))? };
        g.allocated += cap;
        let mut stage = [std::ptr::null_mut(); 2];
        for s in stage.iter_mut() {
            *s = unsafe { drv::malloc_host(STAGE, 0)? } as *mut u8;
        }
        let ev = [drv::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)?, drv::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)?];
        Ok(Arena { base, used: 0, cap, stage, ev, cur: 0, win: 0, fill: 0 })
    }

    /// Place the parts back to back (one tensor) and return its device address.
    pub fn put(&mut self, g: &Gpu, parts: &[&[u8]]) -> R<Ptr> {
        let p = self.base + self.used as u64;
        let n: usize = parts.iter().map(|b| b.len()).sum();
        if self.used + n > self.cap {
            return Err("weight arena too small".into());
        }
        for b in parts {
            self.write(g, b)?;
        }
        let pad = self.used.next_multiple_of(ALIGN) - self.used;
        self.write(g, &[0u8; ALIGN][..pad])?;
        Ok(p)
    }

    fn write(&mut self, g: &Gpu, mut bytes: &[u8]) -> R<()> {
        while !bytes.is_empty() {
            if self.fill == STAGE {
                self.flush(g)?;
            }
            let n = (STAGE - self.fill).min(bytes.len());
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.stage[self.cur].add(self.fill), n) };
            self.fill += n;
            self.used += n;
            bytes = &bytes[n..];
        }
        Ok(())
    }

    /// Send the current staging buffer and switch to the other one once its previous transfer is done.
    fn flush(&mut self, g: &Gpu) -> R<()> {
        if self.fill > 0 {
            let src = unsafe { std::slice::from_raw_parts(self.stage[self.cur], self.fill) };
            unsafe {
                drv::memcpy_htod_async(self.base + self.win as u64, src, g.cu())?;
                drv::event::record(self.ev[self.cur], g.cu())?;
            }
        }
        self.cur ^= 1;
        unsafe { sys::cuEventSynchronize(self.ev[self.cur]).result()? };
        self.win = self.used;
        self.fill = 0;
        Ok(())
    }

    /// Wait for the last transfers and release the staging buffers.
    pub fn finish(mut self, g: &Gpu) -> R<()> {
        self.flush(g)?;
        g.sync()?;
        for i in 0..2 {
            unsafe {
                drv::free_host(self.stage[i] as *mut c_void)?;
                sys::cuEventDestroy_v2(self.ev[i]).result()?;
            }
        }
        Ok(())
    }
}
