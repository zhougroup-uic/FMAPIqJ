// fmapgrid.cu - GPU energy-map generation as a plain C library (CUDA runtime + cuFFT), called from Julia by ccall.
// Same algorithm as the CPU code in src/grid/cpu (softgrd.jl, mapping.jl) and the former CUDA.jl port:
// crowder grids are built once and kept as their Fourier transform on the device, every orientation maps
// the probe atoms and cross-correlates (corr = irfft(FT(crowd) * conj(FT(probe)))).
// Atom mapping uses atomic adds, so sums are not bit-reproducible; quantised cells agree with the CPU code
// except where a value sits on a rounding boundary.
//
// Array layout everywhere: cell (x=i, y=j, z=k) is at k + l*(j + l*i), the Julia (l,l,l) array grd[k+1,j+1,i+1].
// Atoms are passed as 7 consecutive arrays of n doubles: x y z (grid units) q r Asq Bsq.
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

#define FLT_MAX_EXP_D 128.0
#define NSTAT 4                       // per summary line: sum, occupancy, ave, net

enum { K_VOL = 0, K_ELE = 1, K_VDW = 2, K_V12 = 3 };

static char g_err[512];
static int fail(const char *what, const char *msg) { snprintf(g_err, sizeof g_err, "%s: %s", what, msg); return -1; }
#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) return fail(#call, cudaGetErrorString(e_)); } while (0)
#define CKF(call) do { cufftResult r_ = (call); if (r_ != CUFFT_SUCCESS) { snprintf(g_err, sizeof g_err, "%s: cufft error %d", #call, (int)r_); return -1; } } while (0)

struct fg_ctx {
    int l, tern;
    size_t n3, nc;                    // cells, complex cells (l/2+1)*l*l
    cufftHandle plan_f, plan_b;
    cufftDoubleComplex *crowd[4];     // FT of the crowder grids
    double *working, *lcEle, *lcVdw, *combined;
    uint8_t *vol;
    cufftDoubleComplex *ft;
    int8_t *out;                      // quantised grid of the last orientation
    double *atoms;                    // device atoms of the current call (7*cap)
    size_t atoms_cap;
    double *acoef, *arup, *ascl;      // per-atom mapping parameters (cap each)
    double *dsum; unsigned long long *dcount;
    unsigned int *dsel;               // pose selection counter
    int have_crowd;
};

// ---- radial functions (ext/rdf.h) ----
__device__ __forceinline__ double rdf(int kind, double r2, double sc) {
    if (kind == K_VOL) return r2 > sc ? 0.0 : 1.0;
    if (kind == K_ELE) { double r = sqrt(r2); return exp(-r * sc) / r; }
    double r6 = r2 * r2 * r2;
    return sc < 4.0 ? 1.0 / r6 : 1.0 / (r6 * r6);
}
__device__ __forceinline__ int wrapi(int a, int l) { int m = a % l; return m < 0 ? m + l : m; }

// ---- spherical mapping: block (bx, atom), threads stride over the atom's bounding box ----
__global__ void mapR_kernel(double *grd, const double *xs, const double *ys, const double *zs, const double *coeffs,
                            const double *rups, const double *scls, double rlow, double dx, int l, int kind) {
    int a = blockIdx.y;
    double x0 = xs[a], y0 = ys[a], z0 = zs[a];
    double rup = rups[a], rup2 = rup * rup, s = rup / dx;
    int imin = (int)floor(x0 - s), imax = (int)ceil(x0 + s);
    int jmin = (int)floor(y0 - s), jmax = (int)ceil(y0 + s);
    int kmin = (int)floor(z0 - s), kmax = (int)ceil(z0 + s);
    int nx = imax - imin + 1, ny = jmax - jmin + 1, nz = kmax - kmin + 1;
    long long ncell = (long long)nx * ny * nz;
    double coeff = coeffs[a], sc = scls[a], rlow2 = rlow * rlow;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < ncell; idx += stride) {
        int k = (int)(idx % nz), j = (int)((idx / nz) % ny), i = (int)(idx / ((long long)ny * nz));
        int gi = imin + i, gj = jmin + j, gk = kmin + k;
        double x = (gi - x0) * dx, y = (gj - y0) * dx, z = (gk - z0) * dx;
        double r2 = x * x + y * y + z * z;
        if (r2 <= rup2) {
            double var = coeff * (r2 > rlow2 ? rdf(kind, r2, sc) : rdf(kind, rlow2, sc));
            atomicAdd(&grd[(size_t)wrapi(gk, l) + (size_t)l * (wrapi(gj, l) + (size_t)l * wrapi(gi, l))], var);
        }
    }
}

// ---- trilinear mapping: one thread per atom ----
__global__ void mapL_kernel(double *grd, const double *xs, const double *ys, const double *zs, const double *coeffs, int n, int l) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n) return;
    double coeff = coeffs[a];
    double fa = xs[a] - trunc(xs[a]), fb = ys[a] - trunc(ys[a]), fc = zs[a] - trunc(zs[a]);
    int px = wrapi((int)trunc(xs[a]), l), py = wrapi((int)trunc(ys[a]), l), pz = wrapi((int)trunc(zs[a]), l);
    int pxn = wrapi((fa < 0.0 ? -1 : 1) + px, l), pyn = wrapi((fb < 0.0 ? -1 : 1) + py, l), pzn = wrapi((fc < 0.0 ? -1 : 1) + pz, l);
    double av = fabs(fa), bv = fabs(fb), cv = fabs(fc);
#define G(z, y, x) grd[(size_t)(z) + (size_t)l * ((y) + (size_t)l * (x))]
    atomicAdd(&G(pz,  py,  px),  coeff * (1.0 - av) * (1.0 - bv) * (1.0 - cv));
    atomicAdd(&G(pzn, py,  px),  coeff * (1.0 - av) * (1.0 - bv) * cv);
    atomicAdd(&G(pz,  pyn, px),  coeff * (1.0 - av) * bv * (1.0 - cv));
    atomicAdd(&G(pzn, pyn, px),  coeff * (1.0 - av) * bv * cv);
    atomicAdd(&G(pz,  py,  pxn), coeff * av * (1.0 - bv) * (1.0 - cv));
    atomicAdd(&G(pzn, py,  pxn), coeff * av * (1.0 - bv) * cv);
    atomicAdd(&G(pz,  pyn, pxn), coeff * av * bv * (1.0 - cv));
    atomicAdd(&G(pzn, pyn, pxn), coeff * av * bv * cv);
}

// ---- closed-form 10-point second-order spreading (map_L2nd! in mapping.jl) ----
__global__ void mapL2nd_kernel(double *grd, const double *xs, const double *ys, const double *zs, const double *coeffs, int n, int l) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n) return;
    double coeff = coeffs[a];
    double lx = floor(xs[a]), ly = floor(ys[a]), lz = floor(zs[a]);
    double fx = xs[a] - lx, fy = ys[a] - ly, fz = zs[a] - lz;
    bool nx = fx < 0.5, ny = fy < 0.5, nz = fz < 0.5;
    double u = nx ? fx : 1.0 - fx, v = ny ? fy : 1.0 - fy, w = nz ? fz : 1.0 - fz;
    int x0 = wrapi((int)lx + (nx ? 0 : 1), l), y0 = wrapi((int)ly + (ny ? 0 : 1), l), z0 = wrapi((int)lz + (nz ? 0 : 1), l);
    int xu = x0 + 1 == l ? 0 : x0 + 1, xd = x0 == 0 ? l - 1 : x0 - 1;
    int yu = y0 + 1 == l ? 0 : y0 + 1, yd = y0 == 0 ? l - 1 : y0 - 1;
    int zu = z0 + 1 == l ? 0 : z0 + 1, zd = z0 == 0 ? l - 1 : z0 - 1;
    int xn = nx ? xu : xd, xf = nx ? xd : xu;          // n: toward the charge, f: away from it
    int yn = ny ? yu : yd, yf = ny ? yd : yu;
    int zn = nz ? zu : zd, zf = nz ? zd : zu;
    double uv = u * v, uw = u * w, vw = v * w;
    atomicAdd(&G(z0, y0, x0), coeff * (1.0 - u * u - v * v - w * w + uv + uw + vw));
    atomicAdd(&G(z0, y0, xn), coeff * (u * (u + 1.0) / 2 - uv - uw));
    atomicAdd(&G(z0, y0, xf), coeff * (u * (u - 1.0) / 2));
    atomicAdd(&G(z0, yn, x0), coeff * (v * (v + 1.0) / 2 - uv - vw));
    atomicAdd(&G(z0, yf, x0), coeff * (v * (v - 1.0) / 2));
    atomicAdd(&G(zn, y0, x0), coeff * (w * (w + 1.0) / 2 - uw - vw));
    atomicAdd(&G(zf, y0, x0), coeff * (w * (w - 1.0) / 2));
    atomicAdd(&G(zn, yn, x0), coeff * vw);
    atomicAdd(&G(zn, y0, xn), coeff * uw);
    atomicAdd(&G(z0, yn, xn), coeff * uv);
}
#undef G

// ---- elementwise kernels ----
__global__ void mulconj_kernel(cufftDoubleComplex *ft, const cufftDoubleComplex *fc, size_t n, double inv) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    cufftDoubleComplex a = fc[i], b = ft[i];           // a * conj(b), scaled by 1/l^3 (cuFFT's inverse is unnormalised)
    ft[i].x = (a.x * b.x + a.y * b.y) * inv;
    ft[i].y = (a.y * b.x - a.x * b.y) * inv;
}
__global__ void addfilt_kernel(double *g1, double scl1, const double *g2, double scl2, const uint8_t *vol, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    g1[i] = vol[i] ? g1[i] * scl1 + FLT_MAX_EXP_D : g1[i] * scl1 + g2[i] * scl2;
}
// vRep: Boltzmann sum outside the mask and the mask count; for `vol` the mask is (re)defined first
__global__ void vrep_kernel(const double *grd, uint8_t *vol, int makevol, double cutoff, double scl, double mkbt, size_t n,
                            double *sum, unsigned long long *count) {
    double s = 0.0; unsigned long long c = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        if (makevol) vol[i] = grd[i] > cutoff;
        if (vol[i]) c++;
        else { double x = grd[i] * scl / mkbt; s += x > FLT_MAX_EXP_D ? 0.0 : exp(x); }
    }
    for (int o = 16; o > 0; o >>= 1) { s += __shfl_down_sync(0xffffffff, s, o); c += __shfl_down_sync(0xffffffff, c, o); }
    if ((threadIdx.x & 31) == 0) { atomicAdd(sum, s); atomicAdd(count, c); }
}
__device__ __forceinline__ int8_t qround(double x) { return (int8_t)(int)round(x * 8.0); }
// circshift by (h,h,h) and quantisation; tern 2: [elec, vdw] per cell, tern 1: the combined grid
__global__ void quant_kernel(int8_t *out, const double *a, const double *b, const uint8_t *vol, int l, int tern) {
    size_t n3 = (size_t)l * l * l;
    int h = l / 2;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n3; idx += (size_t)gridDim.x * blockDim.x) {
        int k = (int)(idx % l), j = (int)((idx / l) % l), i = (int)(idx / ((size_t)l * l));
        size_t o = (size_t)((k + h) % l) + (size_t)l * (((j + h) % l) + (size_t)l * ((i + h) % l));
        if (tern == 2) {
            out[2 * o]     = vol[idx] ? 127 : qround(a[idx]);
            out[2 * o + 1] = vol[idx] ? 127 : qround(b[idx]);
        } else out[o] = vol[idx] ? 127 : qround(a[idx]);
    }
}
__global__ void select_kernel(const double *comb, const uint8_t *vol, double ecut, size_t n, unsigned int *cnt,
                              int32_t *cell, double *val, int cap) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (!vol[i] && comb[i] < ecut) {
            unsigned int p = atomicAdd(cnt, 1u);
            if ((int)p < cap) { cell[p] = (int32_t)i; val[p] = comb[i]; }
        }
}

// ---- host side ----
static int ensure_atoms(fg_ctx *c, size_t n) {
    if (n <= c->atoms_cap) return 0;
    cudaFree(c->atoms); cudaFree(c->acoef); cudaFree(c->arup); cudaFree(c->ascl);
    CK(cudaMalloc(&c->atoms, 7 * n * sizeof(double)));
    CK(cudaMalloc(&c->acoef, n * sizeof(double)));
    CK(cudaMalloc(&c->arup, n * sizeof(double)));
    CK(cudaMalloc(&c->ascl, n * sizeof(double)));
    c->atoms_cap = n;
    return 0;
}
// no cap: addfilt/mulconj are one-thread-per-cell kernels (gridDim.x may be up to 2^31-1)
static int blocks_for(size_t n, int t) { return (int)((n + t - 1) / t); }

// spherical mapping of the atoms in c->atoms (device); coefficient/range/scale per atom are built on the host
static int map_R(fg_ctx *c, int n, int kind, const double *coeff, const double *rup, const double *scl, double rlow, double dx) {
    if (n == 0) return 0;
    CK(cudaMemcpy(c->acoef, coeff, n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c->arup, rup, n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c->ascl, scl, n * sizeof(double), cudaMemcpyHostToDevice));
    double maxrup = 0.0;
    for (int a = 0; a < n; a++) if (rup[a] > maxrup) maxrup = rup[a];
    long long side = (long long)ceil(2 * maxrup / dx) + 3;
    long long ncell = side * side * side;
    int threads = 256;
    int bx = (int)((ncell + threads - 1) / threads); if (bx > 4096) bx = 4096;
    const double *X = c->atoms, *Y = X + n, *Z = Y + n;
    for (int a0 = 0; a0 < n; a0 += 65535) {                   // gridDim.y is limited to 65535 atoms per launch
        int na = n - a0 < 65535 ? n - a0 : 65535;
        mapR_kernel<<<dim3(bx, na), threads>>>(c->working, X + a0, Y + a0, Z + a0, c->acoef + a0, c->arup + a0, c->ascl + a0,
                                               rlow, dx, c->l, kind);
    }
    CK(cudaGetLastError());
    return 0;
}

static int crowd_grid(fg_ctx *c, int kind, int n, const double *at, double dx, double cut, double kappa) {
    const double *q = at + 3 * (size_t)n, *r = at + 4 * (size_t)n, *A = at + 5 * (size_t)n, *B = at + 6 * (size_t)n;
    std::vector<double> co(n), ru(n), sc(n);
    for (int a = 0; a < n; a++) {
        if (kind == K_VOL)      { co[a] = 1.0;     ru[a] = r[a]; sc[a] = r[a] * r[a]; }
        else if (kind == K_ELE) { co[a] = q[a];    ru[a] = cut;  sc[a] = kappa; }
        else if (kind == K_VDW) { co[a] = -B[a];   ru[a] = cut;  sc[a] = 3.0; }
        else                    { co[a] = A[a];    ru[a] = cut;  sc[a] = 6.0; }
    }
    CK(cudaMemset(c->working, 0, c->n3 * sizeof(double)));
    return map_R(c, n, kind, co.data(), ru.data(), sc.data(), kind == K_VOL ? 0.0 : 1.0, dx);
}

static int probe_grid(fg_ctx *c, int kind, int n, const double *at, double dx) {
    const double *q = at + 3 * (size_t)n, *r = at + 4 * (size_t)n, *A = at + 5 * (size_t)n, *B = at + 6 * (size_t)n;
    const double *X = c->atoms, *Y = X + n, *Z = Y + n;
    CK(cudaMemset(c->working, 0, c->n3 * sizeof(double)));
    if (n == 0) return 0;
    if (kind == K_VOL) {
        std::vector<double> co(n, 1.0), ru(r, r + n), sc(n);
        for (int a = 0; a < n; a++) sc[a] = r[a] * r[a];
        return map_R(c, n, K_VOL, co.data(), ru.data(), sc.data(), 0.0, dx);
    }
    const double *coef = kind == K_ELE ? q : kind == K_VDW ? B : A;
    CK(cudaMemcpy(c->acoef, coef, n * sizeof(double), cudaMemcpyHostToDevice));
    if (kind == K_ELE) mapL2nd_kernel<<<(n + 127) / 128, 128>>>(c->working, X, Y, Z, c->acoef, n, c->l);
    else               mapL_kernel<<<(n + 127) / 128, 128>>>(c->working, X, Y, Z, c->acoef, n, c->l);
    CK(cudaGetLastError());
    return 0;
}

static int load_atoms(fg_ctx *c, int n, const double *at) {
    if (ensure_atoms(c, (size_t)(n > 0 ? n : 1))) return -1;
    CK(cudaMemcpy(c->atoms, at, 7 * (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    return 0;
}

extern "C" {

const char *fg_error(void) { return g_err; }

int fg_device_count(void) { int n = 0; return cudaGetDeviceCount(&n) == cudaSuccess ? n : 0; }

fg_ctx *fg_open(int l, int tern, int device) {
    if (l < 2 || l % 2 || (tern != 1 && tern != 2)) { fail("fg_open", "bad grid size or cell format"); return NULL; }
    if (cudaSetDevice(device) != cudaSuccess) { fail("cudaSetDevice", cudaGetErrorString(cudaGetLastError())); return NULL; }
    fg_ctx *c = (fg_ctx *)calloc(1, sizeof *c);
    c->l = l; c->tern = tern; c->n3 = (size_t)l * l * l; c->nc = (size_t)(l / 2 + 1) * l * l;
    int bad = 0;
    for (int k = 0; k < 4 && !bad; k++) bad = cudaMalloc(&c->crowd[k], c->nc * sizeof(cufftDoubleComplex)) != cudaSuccess;
    bad = bad || cudaMalloc(&c->working, c->n3 * 8) || cudaMalloc(&c->lcEle, c->n3 * 8) || cudaMalloc(&c->lcVdw, c->n3 * 8)
              || cudaMalloc(&c->combined, c->n3 * 8) || cudaMalloc(&c->vol, c->n3) || cudaMalloc(&c->ft, c->nc * sizeof(cufftDoubleComplex))
              || cudaMalloc(&c->out, c->n3 * tern) || cudaMalloc(&c->dsum, 8) || cudaMalloc(&c->dcount, 8) || cudaMalloc(&c->dsel, 4);
    if (bad) { fail("fg_open", "out of device memory (6 GB needed at blen=430)"); return NULL; }
    // cuFFT dims: slowest first; the fastest (z, the real axis) is halved in the complex output
    if (cufftPlan3d(&c->plan_f, l, l, l, CUFFT_D2Z) != CUFFT_SUCCESS || cufftPlan3d(&c->plan_b, l, l, l, CUFFT_Z2D) != CUFFT_SUCCESS) {
        fail("fg_open", "cufftPlan3d failed"); return NULL;
    }
    return c;
}

void fg_close(fg_ctx *c) {
    if (!c) return;
    for (int k = 0; k < 4; k++) cudaFree(c->crowd[k]);
    cudaFree(c->working); cudaFree(c->lcEle); cudaFree(c->lcVdw); cudaFree(c->combined); cudaFree(c->vol);
    cudaFree(c->ft); cudaFree(c->out); cudaFree(c->atoms); cudaFree(c->acoef); cudaFree(c->arup); cudaFree(c->ascl);
    cudaFree(c->dsum); cudaFree(c->dcount); cudaFree(c->dsel);
    cufftDestroy(c->plan_f); cufftDestroy(c->plan_b);
    free(c);
}

// Crowder atoms (never change during a run): build the four mapped grids and keep their transforms.
int fg_set_crowd(fg_ctx *c, int n, const double *at, double dx, double cut, double kappa) {
    if (load_atoms(c, n, at)) return -1;
    for (int k = 0; k < 4; k++) {
        if (crowd_grid(c, k, n, at, dx, cut, kappa)) return -1;
        CKF(cufftExecD2Z(c->plan_f, c->working, c->crowd[k]));
    }
    CK(cudaDeviceSynchronize());
    c->have_crowd = 1;
    return 0;
}

static int correlate(fg_ctx *c, int kind, int n, const double *pros, double dx) {
    if (probe_grid(c, kind, n, pros, dx)) return -1;
    CKF(cufftExecD2Z(c->plan_f, c->working, c->ft));
    mulconj_kernel<<<blocks_for(c->nc, 256), 256>>>(c->ft, c->crowd[kind], c->nc, 1.0 / (double)c->n3);
    CK(cudaGetLastError());
    CKF(cufftExecZ2D(c->plan_b, c->ft, c->working));
    return 0;
}

static int vrep(fg_ctx *c, const double *grd, int makevol, double scl, double kBT, double *stat) {
    CK(cudaMemset(c->dsum, 0, 8)); CK(cudaMemset(c->dcount, 0, 8));
    vrep_kernel<<<blocks_for(c->n3, 256), 256>>>(grd, c->vol, makevol, 0.001, scl, -kBT, c->n3, c->dsum, c->dcount);
    CK(cudaGetLastError());
    double s; unsigned long long cnt;
    CK(cudaMemcpy(&s, c->dsum, 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&cnt, c->dcount, 8, cudaMemcpyDeviceToHost));
    double dl3 = (double)c->n3, count = (double)cnt, u, un;
    if (makevol) { u = -kBT * log(1.0 - count / dl3); un = 0.0; }
    else         { u = -kBT * log(s / dl3); un = -kBT * log(s / (dl3 - count)); }
    stat[0] = s; stat[1] = count / dl3; stat[2] = u; stat[3] = un;
    return 0;
}

// One orientation: probe atoms (7 arrays, grid units, already rotated) -> quantised grid in `out` (tern*l^3 int8)
// and the five summary lines vol, ele, vdw, v12, v+e in `stat` (5 x NSTAT). es = escl*332/sdie, vscl from the parameters.
int fg_softgrd(fg_ctx *c, int n, const double *pros, double dx, double es, double vscl, double kBT, int8_t *out, double *stat) {
    if (!c->have_crowd) return fail("fg_softgrd", "fg_set_crowd not called");
    if (load_atoms(c, n, pros)) return -1;
    size_t n3 = c->n3; int tb = blocks_for(n3, 256);
    CK(cudaMemset(c->lcEle, 0, n3 * 8)); CK(cudaMemset(c->lcVdw, 0, n3 * 8)); CK(cudaMemset(c->combined, 0, n3 * 8));

    if (correlate(c, K_VOL, n, pros, dx)) return -1;
    if (vrep(c, c->working, 1, 1.0, kBT, stat + 0 * NSTAT)) return -1;

    if (correlate(c, K_ELE, n, pros, dx)) return -1;
    addfilt_kernel<<<tb, 256>>>(c->combined, 1.0, c->working, es, c->vol, n3);
    addfilt_kernel<<<tb, 256>>>(c->lcEle, 1.0, c->working, es, c->vol, n3);
    if (vrep(c, c->working, 0, es, kBT, stat + 1 * NSTAT)) return -1;

    if (correlate(c, K_VDW, n, pros, dx)) return -1;
    addfilt_kernel<<<tb, 256>>>(c->combined, 1.0, c->working, vscl, c->vol, n3);
    addfilt_kernel<<<tb, 256>>>(c->lcVdw, 1.0, c->working, vscl, c->vol, n3);
    if (vrep(c, c->working, 0, vscl, kBT, stat + 2 * NSTAT)) return -1;

    if (correlate(c, K_V12, n, pros, dx)) return -1;
    addfilt_kernel<<<tb, 256>>>(c->combined, 1.0, c->working, vscl, c->vol, n3);
    addfilt_kernel<<<tb, 256>>>(c->lcVdw, 1.0, c->working, vscl, c->vol, n3);
    if (vrep(c, c->working, 0, vscl, kBT, stat + 3 * NSTAT)) return -1;
    if (vrep(c, c->combined, 0, 1.0, kBT, stat + 4 * NSTAT)) return -1;
    CK(cudaGetLastError());

    if (c->tern == 2) quant_kernel<<<tb, 256>>>(c->out, c->lcEle, c->lcVdw, c->vol, c->l, 2);
    else              quant_kernel<<<tb, 256>>>(c->out, c->combined, c->combined, c->vol, c->l, 1);
    CK(cudaGetLastError());
    CK(cudaMemcpy(out, c->out, n3 * c->tern, cudaMemcpyDeviceToHost));
    return 0;
}

// Cells of the last orientation below ecut that are not excluded: linear cell index (k + l*(j + l*i)) and value.
// Returns the total count; only the first `cap` are stored (call again with a larger cap if count > cap).
int fg_select(fg_ctx *c, double ecut, int cap, int32_t *cell, double *val) {
    int32_t *dc; double *dv;
    CK(cudaMemset(c->dsel, 0, 4));
    CK(cudaMalloc(&dc, (size_t)(cap > 0 ? cap : 1) * 4)); CK(cudaMalloc(&dv, (size_t)(cap > 0 ? cap : 1) * 8));
    select_kernel<<<blocks_for(c->n3, 256), 256>>>(c->combined, c->vol, ecut, c->n3, c->dsel, dc, dv, cap);
    unsigned int cnt;
    CK(cudaMemcpy(&cnt, c->dsel, 4, cudaMemcpyDeviceToHost));
    int m = (int)cnt < cap ? (int)cnt : cap;
    if (m > 0) { CK(cudaMemcpy(cell, dc, (size_t)m * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(val, dv, (size_t)m * 8, cudaMemcpyDeviceToHost)); }
    cudaFree(dc); cudaFree(dv);
    return (int)cnt;
}

}  // extern "C"

// ---- pair energies of listed poses (fmapdd): one thread per (pose, atom) ----
// side 1: ligand atom looping over the receptor; side 2: receptor atom looping over the ligand (the per-atom sums need no atomics and are
// reproducible). out[c + 3 * (atom + na * pose)], c = (vol, ele, vdw), unscaled. Atoms are 7 arrays: x y z q r Asq Bsq.
__global__ void pair_kernel(double *out, int side, const double *rec, int nr, const double *lig, int nl, const double *Rm, const double *offs,
                            int p0, int np, double kap, double rEup2, double rup2, double rlow2, double L) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int na = side == 1 ? nl : nr;
    if (t >= (long long)na * np) return;
    int a = (int)(t % na), p = (int)(t / na);
    int gp = p0 + p;
    double hL = 0.5 * L, cut2 = fmax(rEup2, rup2);
    const double *R = Rm + 9 * gp;
    double o1 = offs[3 * gp], o2 = offs[3 * gp + 1], o3 = offs[3 * gp + 2];
    double sv = 0.0, se = 0.0, sw = 0.0;
    int n = side == 1 ? nr : nl;
    for (int b = 0; b < n; b++) {
        int i = side == 1 ? b : a, j = side == 1 ? a : b;
        double px = lig[j], py = lig[nl + j], pz = lig[2 * nl + j];
        double lx = R[0] * px + R[1] * py + R[2] * pz + o1;
        double ly = R[3] * px + R[4] * py + R[5] * pz + o2;
        double lz = R[6] * px + R[7] * py + R[8] * pz + o3;
        double dx = rec[i] - lx; dx = dx > hL ? dx - L : (dx < -hL ? dx + L : dx);
        double dy = rec[nr + i] - ly; dy = dy > hL ? dy - L : (dy < -hL ? dy + L : dy);
        double dz = rec[2 * nr + i] - lz; dz = dz > hL ? dz - L : (dz < -hL ? dz + L : dz);
        double d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > cut2) continue;
        double rs = rec[4 * nr + i] + lig[4 * nl + j];
        double v = d2 > rs * rs ? 0.0 : 1.0;
        double e = 0.0;
        if (d2 <= rEup2) {
            double dd = sqrt(d2 > rlow2 ? d2 : rlow2);
            e = rec[3 * nr + i] * lig[3 * nl + j] * exp(-1.0 * dd * kap) / dd;
        }
        double w = 0.0, d2m = d2 > rlow2 ? d2 : rlow2;
        if (d2m <= rup2) {
            double d6 = d2m * d2m * d2m;
            w = rec[5 * nr + i] * lig[5 * nl + j] / (d6 * d6) - rec[6 * nr + i] * lig[6 * nl + j] / d6;
        }
        sv += v; se += e; sw += w;
    }
    out[3 * (a + (size_t)na * p)] = sv; out[3 * (a + (size_t)na * p) + 1] = se; out[3 * (a + (size_t)na * p) + 2] = sw;
}
// sums over the atoms of each pose: sums[3 * (p0 + p) + c]
__global__ void pose_sum_kernel(double *sums, const double *out, int na, int p0) {
    int p = blockIdx.x, c = threadIdx.x;          // 3 threads per pose
    double s = 0.0;
    for (int a = 0; a < na; a++) s += out[3 * (a + (size_t)na * p) + c];
    sums[3 * (p0 + p) + c] += s;
}
// per-atom sums over the poses of a chunk: acc[3 * a + c]
__global__ void atom_sum_kernel(double *acc, const double *out, int na, int np) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= na) return;
    for (int c = 0; c < 3; c++) {
        double s = 0.0;
        for (int p = 0; p < np; p++) s += out[3 * (a + (size_t)na * p) + c];
        acc[3 * a + c] += s;
    }
}

extern "C" int fg_pair(int nr, const double *rec, int nl, const double *lig, int np, const double *Rm, const double *offs, double kap, double rEup2,
                       double rup2, double rlow2, double L, int atm, double *sums, double *accr, double *accl) {
    const int chunk = 256;
    double *d_rec, *d_lig, *d_R, *d_off, *d_out1, *d_out2 = NULL, *d_sums, *d_accr = NULL, *d_accl = NULL;
    CK(cudaMalloc(&d_rec, (size_t)nr * 56)); CK(cudaMalloc(&d_lig, (size_t)nl * 56));
    CK(cudaMalloc(&d_R, (size_t)np * 72)); CK(cudaMalloc(&d_off, (size_t)np * 24));
    CK(cudaMemcpy(d_rec, rec, (size_t)nr * 56, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_lig, lig, (size_t)nl * 56, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_R, Rm, (size_t)np * 72, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_off, offs, (size_t)np * 24, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_out1, (size_t)chunk * nl * 24)); CK(cudaMalloc(&d_sums, (size_t)np * 24)); CK(cudaMemset(d_sums, 0, (size_t)np * 24));
    if (atm) {
        CK(cudaMalloc(&d_out2, (size_t)chunk * nr * 24)); CK(cudaMalloc(&d_accr, (size_t)nr * 24)); CK(cudaMalloc(&d_accl, (size_t)nl * 24));
        CK(cudaMemset(d_accr, 0, (size_t)nr * 24)); CK(cudaMemset(d_accl, 0, (size_t)nl * 24));
    }
    for (int p0 = 0; p0 < np; p0 += chunk) {
        int nc = np - p0 < chunk ? np - p0 : chunk;
        pair_kernel<<<(unsigned)(((long long)nl * nc + 127) / 128), 128>>>(d_out1, 1, d_rec, nr, d_lig, nl, d_R, d_off, p0, nc, kap, rEup2, rup2, rlow2, L);
        CK(cudaGetLastError());
        pose_sum_kernel<<<nc, 3>>>(d_sums, d_out1, nl, p0);
        if (atm) {
            atom_sum_kernel<<<(nl + 127) / 128, 128>>>(d_accl, d_out1, nl, nc);
            pair_kernel<<<(unsigned)(((long long)nr * nc + 127) / 128), 128>>>(d_out2, 2, d_rec, nr, d_lig, nl, d_R, d_off, p0, nc, kap, rEup2, rup2, rlow2, L);
            CK(cudaGetLastError());
            atom_sum_kernel<<<(nr + 127) / 128, 128>>>(d_accr, d_out2, nr, nc);
        }
        CK(cudaGetLastError());
    }
    CK(cudaMemcpy(sums, d_sums, (size_t)np * 24, cudaMemcpyDeviceToHost));
    if (atm) { CK(cudaMemcpy(accr, d_accr, (size_t)nr * 24, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(accl, d_accl, (size_t)nl * 24, cudaMemcpyDeviceToHost)); }
    cudaFree(d_rec); cudaFree(d_lig); cudaFree(d_R); cudaFree(d_off); cudaFree(d_out1); cudaFree(d_sums);
    if (atm) { cudaFree(d_out2); cudaFree(d_accr); cudaFree(d_accl); }
    return 0;
}
