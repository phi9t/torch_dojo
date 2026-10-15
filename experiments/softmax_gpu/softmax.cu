// Softmax forward/backward, one 256-thread block per row of a row-major
// float32 [rows, cols] matrix. Compiled at runtime by NVRTC (workout.py,
// softmax_cuda.py); also valid for nvcc. Launch: grid=(rows), block=(THREADS).
//   softmax_fwd(x, y, cols):      online (max, sum) per thread, tree merge in
//                                 shared memory, then y = exp(x - m) / l.
//   softmax_bwd(y, g, dx, cols):  d = <g, y> by tree reduction, dx = y * (g - d).
#define THREADS 256
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)   // NVRTC has no <math.h>
#endif

__device__ void merge(float& m, float& l, float m2, float l2) {
    if (m2 == -INFINITY) return;                 // empty partial: nothing to fold in
    if (m == -INFINITY) { m = m2; l = l2; return; }
    float mn = fmaxf(m, m2);
    l = l * expf(m - mn) + l2 * expf(m2 - mn);
    m = mn;
}

extern "C" __global__ void softmax_fwd(const float* x, float* y, int cols) {
    __shared__ float sm[THREADS], sl[THREADS];
    const float* xr = x + (size_t)blockIdx.x * cols;
    float* yr = y + (size_t)blockIdx.x * cols;
    float m = -INFINITY, l = 0.f;
    for (int i = threadIdx.x; i < cols; i += THREADS) merge(m, l, xr[i], 1.f);
    sm[threadIdx.x] = m; sl[threadIdx.x] = l;
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float m2 = sm[threadIdx.x + s], l2 = sl[threadIdx.x + s];
            merge(m, l, m2, l2);
            sm[threadIdx.x] = m; sl[threadIdx.x] = l;
        }
        __syncthreads();
    }
    m = sm[0]; l = sl[0];
    for (int i = threadIdx.x; i < cols; i += THREADS) yr[i] = expf(xr[i] - m) / l;
}

extern "C" __global__ void softmax_bwd(const float* y, const float* g, float* dx, int cols) {
    __shared__ float sd[THREADS];
    const float* yr = y + (size_t)blockIdx.x * cols;
    const float* gr = g + (size_t)blockIdx.x * cols;
    float* dr = dx + (size_t)blockIdx.x * cols;
    float d = 0.f;
    for (int i = threadIdx.x; i < cols; i += THREADS) d += yr[i] * gr[i];
    sd[threadIdx.x] = d;
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sd[threadIdx.x] += sd[threadIdx.x + s];
        __syncthreads();
    }
    d = sd[0];
    for (int i = threadIdx.x; i < cols; i += THREADS) dr[i] = yr[i] * (gr[i] - d);
}
