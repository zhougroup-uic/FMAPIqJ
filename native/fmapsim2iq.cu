// fmapsim2iq.cu - I(q) of an atomic structure on the GPU, see fmapsim2iq.h. Port of the sim2iq C core
// (grid_mapping.c, fft.c, binning.c, saxs_functions.c; reference/sim2iq_cuda/sim2iq_cuda.cu): same algorithm, single precision grids,
// double precision sums. Grid cell (x, y, z) is at x + n*(y + n*z).
#include "fmapsim2iq.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

static char g_err[512];
static int fail(const char *what, const char *msg) { snprintf(g_err, sizeof g_err, "%s: %s", what, msg); return -1; }
#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) return fail(#call, cudaGetErrorString(e_)); } while (0)
#define CKF(call) do { cufftResult r_ = (call); if (r_ != CUFFT_SUCCESS) { snprintf(g_err, sizeof g_err, "%s: cufft error %d", #call, (int)r_); return -1; } } while (0)

enum { NORM_ELECTRON_COUNT = 1, NORM_VOLUME = 2, NORM_NONE = 3 };

// Cromer-Mann 4-Gauss parameters by element type (cromer_mann_4gauss.h): a, b, van der Waals radius, electrons
static const double CM_A[8][4] = {
    {0, 0, 0, 0},
    {5.9565474326684456, -1.0902225877160387, 4.455816562289012, -9.253357957505079},
    {0.48964434865164597, 0.25795506418162684, 0.19862808650933614, 0.05398838736648019},
    {3.049502465329157, 0.9469490189353504, 0.9205181585530644, 1.0895088896753868},
    {5.551459140691302, 2.480555683008789, 2.773120764746902, -3.8187500178972695},
    {3.2844165713667874, 2.1863494590744375, 1.734158144526276, 0.7918335350536696},
    {7.138789042249852, 4.142606814307699, 2.128293691849062, 1.5929377824602673},
    {7.1088513543231375, 5.215028849933757, 2.091806378705691, 1.583611951864076}};
static const double CM_B[8][4] = {
    {0, 0, 0, 0},
    {18.395696887124288, 1.4191113040744165, 1.4200667487158216, 69.96135092707375},
    {20.659384838283998, 7.740482800259867, 49.551905131184874, 2.2010683273966882},
    {16.122830991045564, 0.879328843206143, 0.2466316869059288, 51.88436420676904},
    {0.3145270000398675, 7.917446957141198, 24.43261890668131, 0.3145232450476127},
    {13.427747243117944, 5.212997411416355, 0.24282907358098274, 32.87575302824983},
    {1.7548326959642166, 26.16163488546785, 0.12230615790259426, 68.26204711582082},
    {1.4377552638137123, 22.18043778213814, 0.10418335440325559, 56.17384058069765}};
static const double CM_VDW[8] = {0.0, 0.0, 1.2, 1.7, 1.55, 1.52, 1.9, 1.8};
static const int CM_EL[8] = {0, 0, 1, 6, 7, 8, 15, 16};

__constant__ double c_a[8][4], c_b[8][4], c_vdw[8];
__constant__ int c_el[8];

__device__ __forceinline__ double contribution(int mode, double r2, int et, double vol, double B, double rho0) {
    if (mode == NORM_NONE) return 1.0;
    if (mode == NORM_VOLUME) {
        double Bt = 4.0 * M_PI * pow(vol, 2.0 / 3.0);
        double pre = 8.0 * rho0 * pow(M_PI, 1.5) * vol / pow(Bt, 1.5);
        return pre * exp(-4.0 * M_PI * M_PI * r2 / Bt);
    }
    double ff = 0.0;
    for (int k = 0; k < 4; k++) {
        double Bt = c_b[et][k] + B;
        ff += pow(4.0 * M_PI / Bt, 1.5) * c_a[et][k] * exp(-4.0 * M_PI * M_PI * r2 / Bt);
    }
    return ff;
}

// One block (128 threads) per atom: reduce the total contribution, then add the normalised values.
__global__ void map_atoms_kernel(float *grid, int N, double spacing, const double *ax, const double *ay, const double *az,
                                 const double *avol, const int *aet, int mode, double cutoff_offset, double B, double rho0) {
    int i = blockIdx.x;
    int et = aet[i];
    if (et == 0) return;
    double cutoff = mode == NORM_ELECTRON_COUNT ? 3.0 : (mode == NORM_VOLUME ? 2.0 * c_vdw[et] : cutoff_offset + c_vdw[et]);
    double x = ax[i], y = ay[i], z = az[i], vol = avol[i];
    int small = 2 * (int)ceil(cutoff / spacing) + 1;
    int ox = (int)floor(x / spacing) - small / 2, oy = (int)floor(y / spacing) - small / 2, oz = (int)floor(z / spacing) - small / 2;
    double rx = x / spacing - ox, ry = y / spacing - oy, rz = z / spacing - oz;
    double cut2 = cutoff * cutoff;
    long long npts = (long long)small * small * small;
    int tid = threadIdx.x;
    __shared__ double red[32];
    __shared__ double tot;
    double loc = 0.0;
    for (long long p = tid; p < npts; p += 128) {
        int xi = (int)(p / ((long long)small * small)), yi = (int)((p / small) % small), zi = (int)(p % small);
        double dx = (xi - rx) * spacing, dy = (yi - ry) * spacing, dz = (zi - rz) * spacing;
        double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 <= cut2) loc += contribution(mode, r2, et, vol, B, rho0);
    }
    for (int off = 16; off > 0; off >>= 1) loc += __shfl_down_sync(0xffffffff, loc, off);
    if ((tid & 31) == 0) red[tid >> 5] = loc;
    __syncthreads();
    if (tid == 0) { double t = 0.0; for (int w = 0; w < 4; w++) t += red[w]; tot = t; }
    __syncthreads();
    double total = tot;
    if (!(total > 0.0)) return;
    double nf;
    if (mode == NORM_ELECTRON_COUNT) nf = c_el[et] > 0 ? c_el[et] / total : 1.0 / total;
    else if (mode == NORM_VOLUME)    nf = vol * rho0 / total;
    else                             nf = 1.0;
    for (long long p = tid; p < npts; p += 128) {
        int xi = (int)(p / ((long long)small * small)), yi = (int)((p / small) % small), zi = (int)(p % small);
        double dx = (xi - rx) * spacing, dy = (yi - ry) * spacing, dz = (zi - rz) * spacing;
        double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 <= cut2) {
            double c = contribution(mode, r2, et, vol, B, rho0) * nf;
            if (c > 0.0) {
                int fx = (ox + xi) % N; if (fx < 0) fx += N;
                int fy = (oy + yi) % N; if (fy < 0) fy += N;
                int fz = (oz + zi) % N; if (fz < 0) fz += N;
                atomicAdd(&grid[(size_t)fx + (size_t)fy * N + (size_t)fz * N * N], (float)c);
            }
        }
    }
}

// |F_vac * exp(B q^2 / 16 pi^2) + F_other|^2 binned in spherical shells (shared-memory bins per block).
__global__ void bin_kernel(double *g_sums, double *g_qs, double *g_cnt, const cufftComplex *vac, const cufftComplex *other, int have_other,
                           int N, double factor, double qmax, double bw, int nb, double b_factor) {
    extern __shared__ double s[];                                   // [sums | q sums | counts]
    for (int j = threadIdx.x; j < 3 * nb; j += blockDim.x) s[j] = 0.0;
    __syncthreads();
    size_t total = (size_t)N * N * N;
    int half = N / 2;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        if (idx == 0) continue;
        int i = (int)(idx % N), jj = (int)((idx / N) % N), k = (int)(idx / ((size_t)N * N));
        double qx = (i <= half ? i : i - N) * factor, qy = (jj <= half ? jj : jj - N) * factor, qz = (k <= half ? k : k - N) * factor;
        double q = sqrt(qx * qx + qy * qy + qz * qz);
        if (q > 0.0 && q < qmax) {
            int b = (int)(q / bw) + 1;
            if (b <= nb) {
                double bs = b_factor > 0.0 ? exp(b_factor * (q / (4.0 * M_PI)) * (q / (4.0 * M_PI))) : 1.0;
                cufftComplex v = vac[idx];
                double re = (double)v.x * bs, im = (double)v.y * bs;
                if (have_other) { cufftComplex o = other[idx]; re += (double)o.x; im += (double)o.y; }
                atomicAdd(&s[b - 1], re * re + im * im);
                atomicAdd(&s[nb + b - 1], q);
                atomicAdd(&s[2 * nb + b - 1], 1.0);
            }
        }
    }
    __syncthreads();
    for (int j = threadIdx.x; j < nb; j += blockDim.x)
        if (s[2 * nb + j] > 0.0) {
            atomicAdd(&g_sums[j], s[j]); atomicAdd(&g_qs[j], s[nb + j]); atomicAdd(&g_cnt[j], s[2 * nb + j]);
        }
}

__global__ void to_complex_kernel(cufftComplex *c, const float *r, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { c[i].x = r[i]; c[i].y = 0.0f; }
}
// other += work * w   (double arithmetic, stored as float, as in add2grid!)
__global__ void add_scaled_kernel(float *other, const float *work, double w, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        other[i] = (float)((double)other[i] + (double)work[i] * w);
}
// other += add where work > 0.5; counts those cells (addU2grid!)
__global__ void add_where_kernel(float *other, const float *work, double add, size_t n, unsigned long long *count) {
    unsigned long long c = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (work[i] > 0.5f) { other[i] = (float)((double)other[i] + add); c++; }
    for (int o = 16; o > 0; o >>= 1) c += __shfl_down_sync(0xffffffff, c, o);
    if ((threadIdx.x & 31) == 0 && c) atomicAdd(count, c);
}

struct fs2i_ctx {
    int N;
    size_t n3;
    cufftHandle plan;
    float *r1, *r2;                    // real grids: vacuum density, solvent / shell work (r1 is reused for the "other" sum)
    cufftComplex *fv, *fo;             // transforms
    double *ax, *ay, *az, *avol; int *aet; size_t acap;
    double *gs, *gq, *gc;              // shell sums
    unsigned long long *dcnt;
};

extern "C" {

const char *fs2i_error(void) { return g_err; }
int fs2i_device_count(void) { int n = 0; return cudaGetDeviceCount(&n) == cudaSuccess ? n : 0; }
double fs2i_u2B(double u) { return 8.0 * M_PI * M_PI * u * u; }
int fs2i_element_type(const char *e) {
    while (e && *e && isspace((unsigned char)*e)) e++;
    if (!e || !*e) return FS2I_UNKNOWN;
    switch (toupper((unsigned char)*e)) {
        case 'H': return FS2I_H; case 'C': return FS2I_C; case 'N': return FS2I_N;
        case 'O': return FS2I_O; case 'P': return FS2I_P; case 'S': return FS2I_S;
    }
    return FS2I_UNKNOWN;
}

fs2i_ctx *fs2i_open(int N, int device) {
    if (N < 4) { fail("fs2i_open", "grid size must be >= 4"); return NULL; }
    if (cudaSetDevice(device) != cudaSuccess) { fail("cudaSetDevice", cudaGetErrorString(cudaGetLastError())); return NULL; }
    fs2i_ctx *c = (fs2i_ctx *)calloc(1, sizeof *c);
    c->N = N; c->n3 = (size_t)N * N * N;
    cudaError_t e = cudaMemcpyToSymbol(c_a, CM_A, sizeof CM_A);
    if (e == cudaSuccess) e = cudaMemcpyToSymbol(c_b, CM_B, sizeof CM_B);
    if (e == cudaSuccess) e = cudaMemcpyToSymbol(c_vdw, CM_VDW, sizeof CM_VDW);
    if (e == cudaSuccess) e = cudaMemcpyToSymbol(c_el, CM_EL, sizeof CM_EL);
    if (e != cudaSuccess) { fail("cudaMemcpyToSymbol", cudaGetErrorString(e)); free(c); return NULL; }
    int nb = N / 2 + 2;
    if (cudaMalloc(&c->r1, c->n3 * 4) || cudaMalloc(&c->r2, c->n3 * 4) || cudaMalloc(&c->fv, c->n3 * 8) || cudaMalloc(&c->fo, c->n3 * 8) ||
        cudaMalloc(&c->gs, nb * 8) || cudaMalloc(&c->gq, nb * 8) || cudaMalloc(&c->gc, nb * 8) || cudaMalloc(&c->dcnt, 8)) {
        fail("fs2i_open", "out of device memory (24 n^3 bytes needed)"); return NULL;
    }
    if (cufftPlan3d(&c->plan, N, N, N, CUFFT_C2C) != CUFFT_SUCCESS) { fail("fs2i_open", "cufftPlan3d failed"); return NULL; }
    return c;
}

void fs2i_close(fs2i_ctx *c) {
    if (!c) return;
    cudaFree(c->r1); cudaFree(c->r2); cudaFree(c->fv); cudaFree(c->fo); cudaFree(c->ax); cudaFree(c->ay); cudaFree(c->az);
    cudaFree(c->avol); cudaFree(c->aet); cudaFree(c->gs); cudaFree(c->gq); cudaFree(c->gc); cudaFree(c->dcnt);
    cufftDestroy(c->plan);
    free(c);
}

static int map_step(fs2i_ctx *c, float *grid, int n, int mode, double spacing, double cutoff_offset, double B) {
    CK(cudaMemset(grid, 0, c->n3 * 4));
    if (n == 0) return 0;
    map_atoms_kernel<<<n, 128>>>(grid, c->N, spacing, c->ax, c->ay, c->az, c->avol, c->aet, mode, cutoff_offset, B, 0.334);
    CK(cudaGetLastError());
    return 0;
}
static int nblocks(size_t n) { size_t b = (n + 255) / 256; return (int)(b > 65535 ? 65535 : b); }

int fs2i_iq(fs2i_ctx *c, int natoms, const double *x, const double *y, const double *z, const double *vol, const int *et,
            double spacing, int steps, double cutoff_offset, double se_w, double ou_w, double in_w, double B,
            double *q, double *iq, int maxpts, double *shell_vol) {
    if (steps < 1 || steps > 4) return fail("fs2i_iq", "steps must be 1..4");
    int N = c->N; size_t n3 = c->n3;
    if ((size_t)natoms > c->acap) {
        cudaFree(c->ax); cudaFree(c->ay); cudaFree(c->az); cudaFree(c->avol); cudaFree(c->aet);
        size_t cap = (size_t)natoms;
        if (cudaMalloc(&c->ax, cap * 8) || cudaMalloc(&c->ay, cap * 8) || cudaMalloc(&c->az, cap * 8) || cudaMalloc(&c->avol, cap * 8) ||
            cudaMalloc(&c->aet, cap * 4)) return fail("fs2i_iq", "out of device memory (atoms)");
        c->acap = cap;
    }
    CK(cudaMemcpy(c->ax, x, natoms * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(c->ay, y, natoms * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c->az, z, natoms * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(c->avol, vol, natoms * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c->aet, et, natoms * 4, cudaMemcpyHostToDevice));
    double dV = spacing * spacing * spacing;
    float *vac = c->r1, *work = c->r2;

    // vacuum density -> fv (transformed at once, which frees r1 for the "other" sum)
    if (map_step(c, vac, natoms, NORM_ELECTRON_COUNT, spacing, 0.0, B)) return -1;
    to_complex_kernel<<<nblocks(n3), 256>>>(c->fv, vac, n3);
    CK(cudaGetLastError());
    CKF(cufftExecC2C(c->plan, c->fv, c->fv, CUFFT_FORWARD));
    if (steps > 1) {
        float *other = c->r1;
        CK(cudaMemset(other, 0, n3 * 4));
        if (map_step(c, work, natoms, NORM_VOLUME, spacing, 0.0, B)) return -1;
        add_scaled_kernel<<<nblocks(n3), 256>>>(other, work, se_w, n3);
        for (int s = 3; s <= steps; s++) {
            if (map_step(c, work, natoms, NORM_NONE, spacing, s == 3 ? cutoff_offset : 0.0, B)) return -1;
            CK(cudaMemset(c->dcnt, 0, 8));
            add_where_kernel<<<nblocks(n3), 256>>>(other, work, (s == 3 ? ou_w : in_w) * dV, n3, c->dcnt);
            CK(cudaGetLastError());
            unsigned long long cnt;
            CK(cudaMemcpy(&cnt, c->dcnt, 8, cudaMemcpyDeviceToHost));
            if (shell_vol) shell_vol[s - 3] = (double)cnt * dV;
        }
        to_complex_kernel<<<nblocks(n3), 256>>>(c->fo, other, n3);
        CK(cudaGetLastError());
        CKF(cufftExecC2C(c->plan, c->fo, c->fo, CUFFT_FORWARD));
    }
    CK(cudaDeviceSynchronize());

    double factor = 2.0 * M_PI / (N * spacing);
    double qmax = (N / 2 - 1) * factor - 1e-8, bw = factor - 1e-8;
    int nb = (int)floor(qmax / bw) + 1;
    if (nb + 1 > maxpts) return fail("fs2i_iq", "maxpts too small");
    CK(cudaMemset(c->gs, 0, nb * 8)); CK(cudaMemset(c->gq, 0, nb * 8)); CK(cudaMemset(c->gc, 0, nb * 8));
    bin_kernel<<<1024, 256, 3 * nb * sizeof(double)>>>(c->gs, c->gq, c->gc, c->fv, steps > 1 ? c->fo : c->fv, steps > 1, N, factor, qmax, bw, nb, B);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    cufftComplex v0, o0 = {0.0f, 0.0f};
    CK(cudaMemcpy(&v0, c->fv, sizeof v0, cudaMemcpyDeviceToHost));
    if (steps > 1) CK(cudaMemcpy(&o0, c->fo, sizeof o0, cudaMemcpyDeviceToHost));
    double re0 = (double)(v0.x + o0.x), im0 = (double)(v0.y + o0.y);
    double *sums = (double *)malloc(nb * 8), *qs = (double *)malloc(nb * 8), *cnt = (double *)malloc(nb * 8);
    CK(cudaMemcpy(sums, c->gs, nb * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(qs, c->gq, nb * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(cnt, c->gc, nb * 8, cudaMemcpyDeviceToHost));
    int np = 0;
    q[np] = 0.0; iq[np] = re0 * re0 + im0 * im0; np++;
    for (int i = 0; i < nb; i++) {
        if (!(cnt[i] > 0.0)) continue;
        q[np] = i == 0 ? 0.0 : qs[i] / cnt[i]; iq[np] = sums[i] / cnt[i]; np++;
    }
    free(sums); free(qs); free(cnt);
    return np;
}

}  // extern "C"
