// CUDA port of sim2iq (grid_mapping.c, fft.c, binning.c, saxs_functions.c, main.c).
// Host-side PQR / trajectory / argument handling reuse the original C sources.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cufft.h>

extern "C" {
#include "saxs_calculator.h"
#include "args.h"
#include "pqr_reader.h"
#include "pd6d.h"
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)
#define CKF(x) do { cufftResult r_ = (x); if (r_ != CUFFT_SUCCESS) { \
    fprintf(stderr, "cuFFT error %d at %s:%d\n", (int)r_, __FILE__, __LINE__); exit(1);} } while (0)


// Cromer-Mann parameters indexed by ElementType (0 unknown, 1 HOH, 2 H, 3 C, 4 N, 5 O, 6 P, 7 S)
__constant__ double d_a[8][4], d_b[8][4], d_vdw[8];
__constant__ int d_elec[8];

static const double h_a[8][4] = {
    {0,0,0,0},
    {5.9565474326684456, -1.0902225877160387, 4.455816562289012, -9.253357957505079},
    {0.48964434865164597, 0.25795506418162684, 0.19862808650933614, 0.05398838736648019},
    {3.049502465329157, 0.9469490189353504, 0.9205181585530644, 1.0895088896753868},
    {5.551459140691302, 2.480555683008789, 2.773120764746902, -3.8187500178972695},
    {3.2844165713667874, 2.1863494590744375, 1.734158144526276, 0.7918335350536696},
    {7.138789042249852, 4.142606814307699, 2.128293691849062, 1.5929377824602673},
    {7.1088513543231375, 5.215028849933757, 2.091806378705691, 1.583611951864076}};
static const double h_b[8][4] = {
    {0,0,0,0},
    {18.395696887124288, 1.4191113040744165, 1.4200667487158216, 69.96135092707375},
    {20.659384838283998, 7.740482800259867, 49.551905131184874, 2.2010683273966882},
    {16.122830991045564, 0.879328843206143, 0.2466316869059288, 51.88436420676904},
    {0.3145270000398675, 7.917446957141198, 24.43261890668131, 0.3145232450476127},
    {13.427747243117944, 5.212997411416355, 0.24282907358098274, 32.87575302824983},
    {1.7548326959642166, 26.16163488546785, 0.12230615790259426, 68.26204711582082},
    {1.4377552638137123, 22.18043778213814, 0.10418335440325559, 56.17384058069765}};
static const double h_vdw[8] = {0, 0.0, 1.2, 1.7, 1.55, 1.52, 1.9, 1.8};
static const int h_elec[8] = {0, 0, 1, 6, 7, 8, 15, 16};

struct DAtom { double x, y, z, vol; int etype; int pad; };

// ---------------------------------------------------------------- grid mapping
// One block per atom: pass 1 reduces the total contribution, pass 2 adds normalised values.
__device__ __forceinline__ double contribution(int mode, double r2, int et, double vol,
                                               double B, double rho0) {
    if (mode == NORM_NONE) return 1.0;
    if (mode == NORM_VOLUME) {
        double Bt = 4.0 * M_PI * pow(vol, 2.0 / 3.0);
        double pre = 8.0 * rho0 * pow(M_PI, 1.5) * vol / pow(Bt, 1.5);
        return pre * exp(-4.0 * M_PI * M_PI * r2 / Bt);
    }
    double ff = 0.0;
    for (int i = 0; i < 4; i++) {
        double Bt = d_b[et][i] + B;
        ff += pow(4.0 * M_PI / Bt, 1.5) * d_a[et][i] * exp(-4.0 * M_PI * M_PI * r2 / Bt);
    }
    return ff;
}

__global__ void map_atoms_kernel(float* grid, int N, double spacing, const DAtom* atoms,
                                 int mode, double cutoff_offset, double B, double rho0) {
    const DAtom a = atoms[blockIdx.x];
    const int et = a.etype;
    if (et == 0) return;
    double cutoff;
    if (mode == NORM_ELECTRON_COUNT || mode == NORM_DEFAULT) cutoff = 3.0;
    else if (mode == NORM_VOLUME) cutoff = 2.0 * d_vdw[et];
    else cutoff = cutoff_offset + d_vdw[et];
    const int small = 2 * (int)ceil(cutoff / spacing) + 1;
    const int ox = (int)floor(a.x / spacing) - small / 2;
    const int oy = (int)floor(a.y / spacing) - small / 2;
    const int oz = (int)floor(a.z / spacing) - small / 2;
    const double rx = a.x / spacing - ox, ry = a.y / spacing - oy, rz = a.z / spacing - oz;
    const double cut2 = cutoff * cutoff;
    const int total_pts = small * small * small;

    __shared__ double red[32];
    __shared__ double s_total;
    double local = 0.0;
    for (int p = threadIdx.x; p < total_pts; p += blockDim.x) {
        int x = p / (small * small), y = (p / small) % small, z = p % small;
        double dx = (x - rx) * spacing, dy = (y - ry) * spacing, dz = (z - rz) * spacing;
        double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 <= cut2) local += contribution(mode, r2, et, a.vol, B, rho0);
    }
    for (int o = 16; o > 0; o >>= 1) local += __shfl_down_sync(0xffffffff, local, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        double t = 0.0;
        for (int w = 0; w < (blockDim.x + 31) / 32; w++) t += red[w];
        s_total = t;
    }
    __syncthreads();
    const double total = s_total;
    if (!(total > 0.0)) return;
    double nf = 1.0;
    if (mode == NORM_ELECTRON_COUNT) nf = d_elec[et] > 0 ? d_elec[et] / total : 1.0 / total;
    else if (mode == NORM_VOLUME) nf = a.vol * rho0 / total;
    else if (mode == NORM_DEFAULT) nf = 1.0 / total;

    for (int p = threadIdx.x; p < total_pts; p += blockDim.x) {
        int x = p / (small * small), y = (p / small) % small, z = p % small;
        double dx = (x - rx) * spacing, dy = (y - ry) * spacing, dz = (z - rz) * spacing;
        double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 <= cut2) {
            double c = contribution(mode, r2, et, a.vol, B, rho0) * nf;
            if (c > 0.0) {
                size_t fx = (size_t)(((ox + x) % N + N) % N);
                size_t fy = (size_t)(((oy + y) % N + N) % N);
                size_t fz = (size_t)(((oz + z) % N + N) % N);
                atomicAdd(&grid[fx + fy * N + fz * (size_t)N * N], (float)c);
            }
        }
    }
}

// ---------------------------------------------------------------- grid helpers
__global__ void add2grid_kernel(const float* src, float* dst, size_t n, double scale) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        dst[i] = (float)((double)dst[i] + (double)src[i] * scale);
}

__global__ void addU2grid_kernel(const float* src, float* dst, size_t n, double add,
                                 unsigned long long* count) {
    unsigned long long c = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (src[i] > 0.5f) { dst[i] = (float)((double)dst[i] + add); c++; }
    for (int o = 16; o > 0; o >>= 1) c += __shfl_down_sync(0xffffffff, c, o);
    if ((threadIdx.x & 31) == 0 && c) atomicAdd(count, c);
}

__global__ void real2complex_kernel(const float* src, cufftComplex* dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        dst[i] = make_cuFloatComplex(src[i], 0.0f);
}

// ---------------------------------------------------------------- binning
__global__ void bin_kernel(const cufftComplex* vac, const cufftComplex* other, int N, double factor,
                           double qmax, double bw, int nb, double b_factor,
                           double* g_sums, double* g_qsums, unsigned long long* g_counts) {
    extern __shared__ double sh[];
    double* s_sums = sh;
    double* s_q = sh + nb;
    unsigned long long* s_cnt = (unsigned long long*)(sh + 2 * nb);
    for (int i = threadIdx.x; i < nb; i += blockDim.x) { s_sums[i] = 0; s_q[i] = 0; s_cnt[i] = 0; }
    __syncthreads();
    const size_t total = (size_t)N * N * N;
    const int half = N / 2;
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        if (idx == 0) continue;
        int i = (int)(idx % N), j = (int)((idx / N) % N), k = (int)(idx / ((size_t)N * N));
        double qx = (i <= half ? i : i - N) * factor;
        double qy = (j <= half ? j : j - N) * factor;
        double qz = (k <= half ? k : k - N) * factor;
        double q = sqrt(qx * qx + qy * qy + qz * qz);
        if (q > 0 && q < qmax) {
            int b = (int)(q / bw);
            if (b < nb) {
                double bs = b_factor > 0.0 ? exp(b_factor * pow(q / (4.0 * M_PI), 2)) : 1.0;
                double re = (double)vac[idx].x * bs, im = (double)vac[idx].y * bs;
                if (other) { re += (double)other[idx].x; im += (double)other[idx].y; }
                atomicAdd(&s_sums[b], re * re + im * im);
                atomicAdd(&s_q[b], q);
                atomicAdd(&s_cnt[b], 1ULL);
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < nb; i += blockDim.x) {
        if (s_cnt[i]) {
            atomicAdd(&g_sums[i], s_sums[i]);
            atomicAdd(&g_qsums[i], s_q[i]);
            atomicAdd(&g_counts[i], s_cnt[i]);
        }
    }
}

// ---------------------------------------------------------------- driver
static double now_s() { struct timeval t; gettimeofday(&t, NULL); return t.tv_sec + t.tv_usec * 1e-6; }
static double u2B(double u) { return 8.0 * M_PI * M_PI * u * u; }

static void map_step(float* grid, size_t gtotal, int N, double spacing, const DAtom* d_atoms, size_t natoms,
                     int mode, double cutoff_offset, double B, double rho0) {
    CK(cudaMemset(grid, 0, gtotal * sizeof(float)));
    // grid.x limit is 2^31-1; chunk anyway to keep launches modest
    const size_t chunk = 1u << 24;
    for (size_t s = 0; s < natoms; s += chunk) {
        unsigned n = (unsigned)((natoms - s) < chunk ? (natoms - s) : chunk);
        map_atoms_kernel<<<n, 128>>>(grid, N, spacing, d_atoms + s, mode, cutoff_offset, B, rho0);
    }
    CK(cudaGetLastError());
}

static QIData calculate_saxs_cuda(Structure* st, int N, double spacing, int steps, double cutoff_offset,
                                  double se_w, double ou_w, double in_w, double B) {
    QIData result = {0};
    double t0 = now_s();
    size_t gtotal = (size_t)N * N * N;
    const int TPB = 256, NBLK = 1024;

    CK(cudaMemcpyToSymbol(d_a, h_a, sizeof(h_a)));
    CK(cudaMemcpyToSymbol(d_b, h_b, sizeof(h_b)));
    CK(cudaMemcpyToSymbol(d_vdw, h_vdw, sizeof(h_vdw)));
    CK(cudaMemcpyToSymbol(d_elec, h_elec, sizeof(h_elec)));

    DAtom* h_atoms = (DAtom*)malloc(st->count * sizeof(DAtom));
    for (size_t i = 0; i < st->count; i++) {
        h_atoms[i].x = st->atoms[i].x; h_atoms[i].y = st->atoms[i].y; h_atoms[i].z = st->atoms[i].z;
        h_atoms[i].vol = st->atoms[i].vol;
        h_atoms[i].etype = get_element_type(st->atoms[i].element);
    }
    DAtom* d_atoms;
    CK(cudaMalloc(&d_atoms, st->count * sizeof(DAtom)));
    CK(cudaMemcpy(d_atoms, h_atoms, st->count * sizeof(DAtom), cudaMemcpyHostToDevice));
    free(h_atoms);

    float *d_vac, *d_other = NULL, *d_work = NULL;
    CK(cudaMalloc(&d_vac, gtotal * sizeof(float)));
    if (steps > 1) {
        CK(cudaMalloc(&d_other, gtotal * sizeof(float)));
        CK(cudaMalloc(&d_work, gtotal * sizeof(float)));
        CK(cudaMemset(d_other, 0, gtotal * sizeof(float)));
    }
    unsigned long long* d_count;
    CK(cudaMalloc(&d_count, sizeof(unsigned long long)));

    printf("[%.3f s] Mapping atoms to grid...\n", now_s() - t0);
    map_step(d_vac, gtotal, N, spacing, d_atoms, st->count, NORM_ELECTRON_COUNT, 0.0, B, 0.334);

    if (steps > 1) {
        printf("[%.3f s] Mapping solvent excluded volume...\n", now_s() - t0);
        map_step(d_work, gtotal, N, spacing, d_atoms, st->count, NORM_VOLUME, 0.0, B, 0.334);
        add2grid_kernel<<<NBLK, TPB>>>(d_work, d_other, gtotal, se_w);
    }
    const double dV = spacing * spacing * spacing;
    for (int s = 3; s <= steps; s++) {
        printf("[%.3f s] Mapping %s shell...\n", now_s() - t0, s == 3 ? "outer" : "inner");
        map_step(d_work, gtotal, N, spacing, d_atoms, st->count, NORM_NONE, s == 3 ? cutoff_offset : 0.0, B, 0.334);
        CK(cudaMemset(d_count, 0, sizeof(unsigned long long)));
        addU2grid_kernel<<<NBLK, TPB>>>(d_work, d_other, gtotal, (s == 3 ? ou_w : in_w) * dV, d_count);
        unsigned long long cnt;
        CK(cudaMemcpy(&cnt, d_count, sizeof(cnt), cudaMemcpyDeviceToHost));
        printf("Volume: %.12e\n", (double)cnt * dV);
    }
    if (d_work) CK(cudaFree(d_work));
    CK(cudaFree(d_atoms));

    printf("[%.3f s] Calculating FFTs...\n", now_s() - t0);
    cufftComplex *d_fv, *d_fo = NULL;
    CK(cudaMalloc(&d_fv, gtotal * sizeof(cufftComplex)));
    real2complex_kernel<<<NBLK, TPB>>>(d_vac, d_fv, gtotal);
    CK(cudaFree(d_vac));
    cufftHandle plan;
    CKF(cufftPlan3d(&plan, N, N, N, CUFFT_C2C));
    CKF(cufftExecC2C(plan, d_fv, d_fv, CUFFT_FORWARD));
    if (steps > 1) {
        CK(cudaMalloc(&d_fo, gtotal * sizeof(cufftComplex)));
        real2complex_kernel<<<NBLK, TPB>>>(d_other, d_fo, gtotal);
        CK(cudaFree(d_other));
        CKF(cufftExecC2C(plan, d_fo, d_fo, CUFFT_FORWARD));
    }
    CKF(cufftDestroy(plan));
    CK(cudaDeviceSynchronize());

    printf("[%.3f s] Binning intensity...\n", now_s() - t0);
    const double factor = 2.0 * M_PI / (N * spacing);
    const double qmax = (double)(N / 2 - 1) * factor - 1e-8;
    const double bw = factor - 1e-8;
    const int nb = (int)(qmax / bw) + 1;
    double *d_s, *d_q; unsigned long long* d_c;
    CK(cudaMalloc(&d_s, nb * sizeof(double))); CK(cudaMemset(d_s, 0, nb * sizeof(double)));
    CK(cudaMalloc(&d_q, nb * sizeof(double))); CK(cudaMemset(d_q, 0, nb * sizeof(double)));
    CK(cudaMalloc(&d_c, nb * sizeof(unsigned long long))); CK(cudaMemset(d_c, 0, nb * sizeof(unsigned long long)));
    size_t shbytes = nb * (2 * sizeof(double) + sizeof(unsigned long long));
    bin_kernel<<<NBLK, TPB, shbytes>>>(d_fv, d_fo, N, factor, qmax, bw, nb, B, d_s, d_q, d_c);
    CK(cudaGetLastError());

    cufftComplex o0[1] = {{0, 0}}, v0[1];
    CK(cudaMemcpy(v0, d_fv, sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    if (d_fo) CK(cudaMemcpy(o0, d_fo, sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    double re0 = (double)(v0[0].x + o0[0].x), im0 = (double)(v0[0].y + o0[0].y);
    // C sums in float before widening: sum_at_origin is fftwf_complex
    double Iq0 = re0 * re0 + im0 * im0;

    double* sums = (double*)malloc(nb * sizeof(double));
    double* qs = (double*)malloc(nb * sizeof(double));
    unsigned long long* cnt = (unsigned long long*)malloc(nb * sizeof(unsigned long long));
    CK(cudaMemcpy(sums, d_s, nb * sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(qs, d_q, nb * sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(cnt, d_c, nb * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CK(cudaFree(d_fv)); if (d_fo) CK(cudaFree(d_fo));
    CK(cudaFree(d_s)); CK(cudaFree(d_q)); CK(cudaFree(d_c)); CK(cudaFree(d_count));

    result = qi_data_create(nb + 1);
    result.data[0].q = 0.0; result.data[0].intensity = Iq0; result.count = 1;
    for (int i = 0; i < nb; i++) {
        if (cnt[i] > 0) {
            result.data[result.count].q = i == 0 ? 0.0 : qs[i] / cnt[i];
            result.data[result.count].intensity = sums[i] / cnt[i];
            result.count++;
        }
    }
    free(sums); free(qs); free(cnt);
    printf("[%.3f s] SAXS calculation complete.\n", now_s() - t0);
    return result;
}

// Replicate the structure for every trajectory entry (main.c: process_trajectory)
static Structure process_trajectory(Structure* orig, const char* traj, bool txt) {
    int n = PD6DCountAtoms((char*)traj, txt);
    double (*tr)[3] = (double(*)[3])malloc(n * sizeof(*tr));
    double (*xr)[3] = (double(*)[3])malloc(n * sizeof(*xr));
    FILE* f = fopen(traj, "r");
    int got = PD6DReadxyz(f, n, tr, xr, txt);
    fclose(f);
    if (got != n) fprintf(stderr, "Warning: Read %d proteins from trajectory file, expected %d\n", got, n);
    printf("Successfully read %d proteins from trajectory file\n", got);
    size_t n0 = orig->count;
    Structure s; s.count = n0 * n; s.capacity = s.count;
    s.atoms = (Atom*)malloc(s.count * sizeof(Atom));
    double (*o)[3] = (double(*)[3])malloc(n0 * sizeof(*o));
    double (*w)[3] = (double(*)[3])malloc(n0 * sizeof(*w));
    for (size_t i = 0; i < n0; i++) { o[i][0] = orig->atoms[i].x; o[i][1] = orig->atoms[i].y; o[i][2] = orig->atoms[i].z; }
    for (int t = 0; t < n; t++) {
        pd6dmmTR(tr[t], xr[t], (int)n0, o, w);
        for (size_t i = 0; i < n0; i++) {
            Atom a = orig->atoms[i];
            a.x = w[i][0]; a.y = w[i][1]; a.z = w[i][2];
            s.atoms[t * n0 + i] = a;
        }
    }
    free(tr); free(xr); free(o); free(w);
    printf("Trajectory processing complete - created structure with %ld atoms\n", s.count);
    return s;
}

int main(int argc, char** argv) {
    CommandArgs args;
    if (!parse_args(argc, argv, &args)) return EXIT_FAILURE;
    Structure orig = read_pqr_file(args.input_file);
    Structure* use = &orig;
    Structure trans;
    if (args.has_traj) {
        printf("Processing trajectory file: %s\n", args.traj_file);
        trans = process_trajectory(&orig, args.traj_file, args.traj_is_txt);
        use = &trans;
    }
    double B = u2B(0.25 * args.grid_spacing);
    if (args.b_factor > 0.0) B = args.b_factor;
    printf("Actual B-factor: %.6f\n", B);
    int dev = 0; cudaDeviceProp prop;
    CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s\n", prop.name);

    QIData r = calculate_saxs_cuda(use, args.grid_size, args.grid_spacing, args.steps, args.cutoff_offset,
                                   args.se_weight, args.ou_weight, args.in_weight, B);
    if (r.count == 0) { fprintf(stderr, "Error: No data points generated\n"); return EXIT_FAILURE; }
    FILE* fo = fopen(args.output_file, "w");
    if (!fo) { fprintf(stderr, "Error: Could not open output file: %s\n", args.output_file); return EXIT_FAILURE; }
    for (size_t i = 0; i < r.count; i++) fprintf(fo, "%.12e\t%.12e\n", r.data[i].q, r.data[i].intensity);
    fclose(fo);
    printf("Wrote %zu data points to %s\n", r.count, args.output_file);
    qidata_free(&r);
    if (args.has_traj) structure_free(&trans);
    structure_free(&orig);
    return EXIT_SUCCESS;
}
