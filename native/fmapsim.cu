// fmapsim.cu - Monte Carlo of rigid molecules with FMAP pair energies on the GPU, see fmapsim.h. Port of the Julia/CUDA.jl engine (sim/gpu: mc_core,
// mix_kernels, gemc_box) and of the C reference (fmap_energy.cu): one block of 256 threads advances one box, the 256 threads share the sum over
// the other molecules; the steps of a box are sequential. Boxes run on their own streams.
#include "fmapsim.h"
#include "fmap_core.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

static char g_err[512];
static int fail(const char *what, const char *msg) { snprintf(g_err, sizeof g_err, "%s: %s", what, msg); return -1; }
#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) return fail(#call, cudaGetErrorString(e_)); } while (0)
#define CKN(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { fail(#call, cudaGetErrorString(e_)); return NULL; } } while (0)

struct fsim_tab {
    fmap_tab h;                        // pointers are device pointers
    int16_t *d_aidx, *d_amap; double *d_rotm; float *d_rotm_f;
    std::vector<int16_t> aidx;         // host copy (rotation index of orientations on the host)
    int8_t *d_sp[FMAP_MAX_CTX];
    double scl;
    int device;
};

struct fsim_box {
    fsim_tab *tab;
    int cap;
    double *pos, *bins;
    int *ridx, *ptype;
    unsigned long long *lcg;
    double *mc, *ins, *res;
    double box[3], kT, es, vs;
    cudaStream_t stream;
    int mode;                          // pending result: bit 0 mc, 1 ins, 2 res
};

// ---- device helpers ----
__device__ __forceinline__ fmap_v3 ld3(const double *a, int i) { fmap_v3 v; v.x = a[3 * i]; v.y = a[3 * i + 1]; v.z = a[3 * i + 2]; return v; }
__device__ __forceinline__ void st3(double *a, int i, fmap_v3 v) { a[3 * i] = v.x; a[3 * i + 1] = v.y; a[3 * i + 2] = v.z; }

// tree reduction of one value per thread (256 threads); every thread returns the sum
__device__ __forceinline__ double block_sum(double val, double *sdata, int tid) {
    __syncthreads();
    sdata[tid] = val;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    double r = sdata[0];
    __syncthreads();
    return r;
}

struct BoxArgs { double *pos, *bins; int *ridx, *ptype; int cap; fmap_v3 box; double es, vs; };

// Energy of `point` (rotation `rot`, type self_type) against slots [0, n) except self_idx; slots below self_idx are the first molecule of the pair.
__device__ double sum_pair_energy(const fmap_tab &t, const BoxArgs &b, fmap_v3 point, int rot, int self_idx, int self_type, int n, double *sdata, int tid) {
    double val = 0.0;
    for (int j = tid; j < n; j += 256) {
        if (j == self_idx) continue;
        fmap_v3 pj = ld3(b.pos, j);
        int rj = b.ridx[j], tj = b.ptype[j];
        int ctx = fmap_ctx_index(self_type, tj, t.n_types);
        if (j < self_idx) val += fmap_pair_energy_f_pbc(&t, ctx, pj, rj, point, rot, b.box, b.es, b.vs);
        else              val += fmap_pair_energy_f_pbc(&t, ctx, point, rot, pj, rj, b.box, b.es, b.vs);
    }
    return block_sum(val, sdata, tid);
}

// Energy change of a displacement: e_new - e_old summed in one pass (as the C kernel), which rounds differently from two separate sums.
__device__ double sum_pair_energy_diff(const fmap_tab &t, const BoxArgs &b, fmap_v3 trial, int trial_rot, fmap_v3 cur, int cur_rot, int self_idx,
                                       int self_type, int n, double *sdata, int tid) {
    double val = 0.0;
    const bool same_pos = trial.x == cur.x && trial.y == cur.y && trial.z == cur.z;      // rotation move: one minimum image serves both
    for (int j = tid; j < n; j += 256) {
        if (j == self_idx) continue;
        fmap_v3 pj = ld3(b.pos, j);
        int rj = b.ridx[j], tj = b.ptype[j];
        int ctx = fmap_ctx_index(self_type, tj, t.n_types);
        const bool first = j < self_idx;                      // pj is the first molecule of the pair
        // minimum-image separation (first - second) of the trial and of the current position
        fmap_v3 q = trial;
        double nx = first ? fmap_pbc(pj.x - q.x, b.box.x) : fmap_pbc(q.x - pj.x, b.box.x);
        double ny = first ? fmap_pbc(pj.y - q.y, b.box.y) : fmap_pbc(q.y - pj.y, b.box.y);
        double nz = first ? fmap_pbc(pj.z - q.z, b.box.z) : fmap_pbc(q.z - pj.z, b.box.z);
        double ox = nx, oy = ny, oz = nz;
        if (!same_pos) {
            q = cur;
            ox = first ? fmap_pbc(pj.x - q.x, b.box.x) : fmap_pbc(q.x - pj.x, b.box.x);
            oy = first ? fmap_pbc(pj.y - q.y, b.box.y) : fmap_pbc(q.y - pj.y, b.box.y);
            oz = first ? fmap_pbc(pj.z - q.z, b.box.z) : fmap_pbc(q.z - pj.z, b.box.z);
        }
        double e_new, e_old;
        if (first) {
            e_new = fmap_pair_energy_f(&t, ctx, nx, ny, nz, rj, trial_rot, b.es, b.vs);
            e_old = fmap_pair_energy_f(&t, ctx, ox, oy, oz, rj, cur_rot, b.es, b.vs);
        } else {
            e_new = fmap_pair_energy_f(&t, ctx, nx, ny, nz, trial_rot, rj, b.es, b.vs);
            e_old = fmap_pair_energy_f(&t, ctx, ox, oy, oz, cur_rot, rj, b.es, b.vs);
        }
        val += e_new - e_old;
    }
    return block_sum(val, sdata, tid);
}

// Energy of `point` appended after every other slot (they are the first molecule), skipping slot `skip`.
__device__ double sum_pair_energy_excl(const fmap_tab &t, const BoxArgs &b, fmap_v3 point, int rot, int skip, int self_type, int n, double *sdata, int tid) {
    double val = 0.0;
    for (int j = tid; j < n; j += 256) {
        if (j == skip) continue;
        fmap_v3 pj = ld3(b.pos, j);
        int ctx = fmap_ctx_index(self_type, b.ptype[j], t.n_types);
        val += fmap_pair_energy_f_pbc(&t, ctx, pj, b.ridx[j], point, rot, b.box, b.es, b.vs);
    }
    return block_sum(val, sdata, tid);
}

// Order-preserving removal of slot s from a compact box of n molecules (the energy depends on the order of a pair). Block-wide.
__device__ void compact_remove_slot(const BoxArgs &b, int s, int n, int tid) {
    for (int base = s; base < n - 1; base += 256) {
        int k = base + tid;
        bool live = k < n - 1;
        double p1 = 0, p2 = 0, p3 = 0, b1 = 0, b2 = 0, b3 = 0; int r = 0, t = 0;
        if (live) {
            int o = (k + 1) * 3;
            p1 = b.pos[o]; p2 = b.pos[o + 1]; p3 = b.pos[o + 2];
            b1 = b.bins[o]; b2 = b.bins[o + 1]; b3 = b.bins[o + 2];
            r = b.ridx[k + 1]; t = b.ptype[k + 1];
        }
        __syncthreads();
        if (live) {
            int o = k * 3;
            b.pos[o] = p1; b.pos[o + 1] = p2; b.pos[o + 2] = p3;
            b.bins[o] = b1; b.bins[o + 1] = b2; b.bins[o + 2] = b3;
            b.ridx[k] = r; b.ptype[k] = t;
        }
        __syncthreads();
    }
}

// ---- displacement moves: translation and rotation are independent trial moves (50/50), each tallied separately ----
__global__ void mc_kernel(fmap_tab t, BoxArgs b, unsigned long long *lcg, double *result, int n, int batch, double kT, double js, double rs) {
    __shared__ double sdata[256];
    __shared__ unsigned long long s_lcg;
    __shared__ double s_delta, s_cnt[5];
    __shared__ int s_pid, s_active, s_is_trans, s_type, s_cur_rot, s_trial_r;
    __shared__ fmap_v3 s_cur, s_trial, s_trial_b;
    int tid = threadIdx.x;
    if (tid == 0) { s_delta = 0.0; for (int k = 0; k < 5; k++) s_cnt[k] = 0.0; s_lcg = lcg[0]; }
    __syncthreads();
    for (int step = 0; step < batch; step++) {
        if (tid == 0) {
            unsigned long long state = s_lcg;
            double r = fmap_lcg(&state);
            int pid = (int)(ceil(r * n) - 1.0);
            pid = pid < 0 ? 0 : (pid >= n ? n - 1 : pid);
            s_pid = pid;
            bool active = n > 0;
            s_active = active ? 1 : 0;
            if (active) {
                s_type = b.ptype[pid];
                fmap_v3 cur = ld3(b.pos, pid), curb = ld3(b.bins, pid);
                s_cur = cur; s_cur_rot = b.ridx[pid];
                double rm = fmap_lcg(&state);
                bool is_trans = rm < 0.5;
                s_is_trans = is_trans ? 1 : 0;
                if (is_trans) s_cnt[1] += 1.0; else s_cnt[3] += 1.0;
                double tp[3] = {cur.x, cur.y, cur.z}, tb[3] = {curb.x, curb.y, curb.z};
                const double bx[3] = {b.box.x, b.box.y, b.box.z};
                for (int d = 0; d < 3; d++) {
                    double r1 = fmap_lcg(&state);
                    if (is_trans) tp[d] = fmap_cwrap(((d == 0 ? cur.x : d == 1 ? cur.y : cur.z)) + (r1 - 0.5) * js, bx[d]);
                    else          tb[d] = fmap_cwrap(tb[d] + (r1 - 0.5) * rs, (double)t.nbin);
                }
                fmap_v3 trial = {tp[0], tp[1], tp[2]}, trial_b = {tb[0], tb[1], tb[2]};
                s_trial = trial; s_trial_b = trial_b;
                s_trial_r = is_trans ? s_cur_rot : fmap_rot_index(&t, tb[0], tb[1], tb[2]);
            }
            s_lcg = state;
        }
        __syncthreads();
        if (s_active) {
            int pid = s_pid;
            double delta_sum = sum_pair_energy_diff(t, b, s_trial, s_trial_r, s_cur, s_cur_rot, pid, s_type, n, sdata, tid);
            if (tid == 0) {
                double delta = delta_sum;
                bool accept = delta <= 0.0;
                if (!accept) {
                    unsigned long long state = s_lcg;
                    double r = fmap_lcg(&state);
                    s_lcg = state;
                    accept = r < exp(-delta / kT);
                }
                if (accept) {
                    st3(b.pos, pid, s_trial); st3(b.bins, pid, s_trial_b);
                    b.ridx[pid] = s_trial_r;
                    s_delta += delta;
                    s_cnt[0] += 1.0;
                    if (s_is_trans) s_cnt[2] += 1.0; else s_cnt[4] += 1.0;
                }
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        result[0] = s_delta;
        for (int k = 0; k < 5; k++) result[k + 1] = s_cnt[k];
        lcg[0] = s_lcg;
    }
}

// ---- same-box insertion: a random molecule is removed and re-inserted at a random place, appended at the highest slot ----
__global__ void insert_kernel(fmap_tab t, BoxArgs b, unsigned long long *lcg, double *result, int n, int batch, double kT) {
    __shared__ double sdata[256];
    __shared__ unsigned long long s_lcg;
    __shared__ double s_de, s_nacc, s_removal, s_insertion;
    __shared__ int s_valid, s_s, s_acc, s_ptype, s_trial_r;
    __shared__ fmap_v3 s_trial, s_trial_b;
    int tid = threadIdx.x;
    if (tid == 0) { s_lcg = lcg[0]; s_de = 0.0; s_nacc = 0.0; }
    __syncthreads();
    for (int step = 0; step < batch; step++) {
        if (tid == 0) {
            int valid = 0;
            if (n > 0) {
                unsigned long long state = s_lcg;
                double r = fmap_lcg(&state);
                int r_idx = (int)(r * n);
                r_idx = r_idx >= n ? n - 1 : r_idx;
                valid = 1;
                s_s = r_idx;
                s_ptype = b.ptype[r_idx];
                double tp[3], tb[3];
                const double bx[3] = {b.box.x, b.box.y, b.box.z};
                for (int d = 0; d < 3; d++) {
                    double ra = fmap_lcg(&state); tp[d] = ra * bx[d];
                    double rb = fmap_lcg(&state); tb[d] = rb * t.nbin;
                }
                s_trial.x = tp[0]; s_trial.y = tp[1]; s_trial.z = tp[2];
                s_trial_b.x = tb[0]; s_trial_b.y = tb[1]; s_trial_b.z = tb[2];
                s_trial_r = fmap_rot_index(&t, tb[0], tb[1], tb[2]);
                s_lcg = state;
            }
            s_valid = valid;
        }
        __syncthreads();
        bool valid = s_valid != 0;
        if (valid) {
            int si = s_s;
            double removal = sum_pair_energy(t, b, ld3(b.pos, si), b.ridx[si], si, s_ptype, n, sdata, tid);
            double insertion = sum_pair_energy_excl(t, b, s_trial, s_trial_r, si, s_ptype, n, sdata, tid);
            if (tid == 0) { s_removal = removal; s_insertion = insertion; }
        }
        __syncthreads();
        if (tid == 0) {
            s_acc = 0;
            if (valid) {
                double ln_acc = -(s_insertion - s_removal) / kT;
                bool accept = ln_acc >= 0.0;
                if (!accept) {
                    unsigned long long state = s_lcg;
                    double r = fmap_lcg(&state);
                    s_lcg = state;
                    accept = r < exp(ln_acc);
                }
                if (accept) { s_acc = 1; s_de += s_insertion - s_removal; s_nacc += 1.0; }
            }
        }
        __syncthreads();
        if (s_acc) {
            int ptype = s_ptype, trr = s_trial_r; fmap_v3 tr = s_trial, trb = s_trial_b;
            compact_remove_slot(b, s_s, n, tid);
            __syncthreads();
            if (tid == 0) { st3(b.pos, n - 1, tr); st3(b.bins, n - 1, trb); b.ridx[n - 1] = trr; b.ptype[n - 1] = ptype; }
        }
        __syncthreads();
    }
    if (tid == 0) {
        result[0] = s_de; result[1] = 0.0; result[2] = (double)batch; result[3] = s_nacc; result[4] = (double)n;
        result[5] = 0.0; result[6] = 0.0; result[7] = 0.0;
        lcg[0] = s_lcg;
    }
}

// ---- reservoir insertion into slots n .. committed-1: accept if the insertion energy <= 0, else Metropolis on exp(-e/kT) ----
__global__ void reservoir_kernel(fmap_tab t, BoxArgs b, unsigned long long *lcg, double *result, int n_init, int committed, int batch, int max_accepts, double kT) {
    __shared__ double sdata[256];
    __shared__ unsigned long long s_lcg;
    __shared__ double s_de, s_nacc, s_t0d;
    __shared__ int s_nact, s_valid, s_ptype, s_trial_r;
    __shared__ fmap_v3 s_trial, s_trial_b;
    int tid = threadIdx.x;
    if (tid == 0) { s_lcg = lcg[0]; s_de = 0.0; s_nacc = 0.0; s_t0d = 0.0; s_nact = n_init; }
    __syncthreads();
    for (int step = 0; step < batch; step++) {
        if (tid == 0) {
            int valid = 0;
            int tt = s_nact;
            if (tt < committed && tt < b.cap && (int)llround(s_nacc) < max_accepts) {
                valid = 1;
                s_ptype = b.ptype[tt];
                unsigned long long state = s_lcg;
                double tp[3], tb[3];
                const double bx[3] = {b.box.x, b.box.y, b.box.z};
                for (int d = 0; d < 3; d++) {
                    double ra = fmap_lcg(&state); tp[d] = ra * bx[d];
                    double rb = fmap_lcg(&state); tb[d] = rb * t.nbin;
                }
                s_trial.x = tp[0]; s_trial.y = tp[1]; s_trial.z = tp[2];
                s_trial_b.x = tb[0]; s_trial_b.y = tb[1]; s_trial_b.z = tb[2];
                s_trial_r = fmap_rot_index(&t, tb[0], tb[1], tb[2]);
                s_lcg = state;
            }
            s_valid = valid;
        }
        __syncthreads();
        if (s_valid) {
            int tt = s_nact;
            double ins = sum_pair_energy(t, b, s_trial, s_trial_r, tt, s_ptype, tt, sdata, tid);
            if (tid == 0) {
                bool accept = ins <= 0.0;
                if (!accept) {
                    unsigned long long state = s_lcg;
                    double r = fmap_lcg(&state);
                    s_lcg = state;
                    accept = r < exp(-ins / kT);
                }
                if (accept) {
                    st3(b.pos, tt, s_trial); st3(b.bins, tt, s_trial_b);
                    b.ridx[tt] = s_trial_r;
                    s_nact = tt + 1;
                    s_de += ins; s_nacc += 1.0;
                    if (s_ptype == 0) s_t0d += 1.0;
                }
            }
        }
        __syncthreads();
    }
    if (tid == 0) { result[0] = s_de; result[1] = s_nacc; result[2] = (double)s_nact; result[3] = s_t0d; lcg[0] = s_lcg; }
}

// ---- energies for placement and the total energy ----
__global__ void ins_energy_kernel(fmap_tab t, BoxArgs b, double *out, fmap_v3 tp, int tridx, int ttype, int n) {
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    double e = sum_pair_energy(t, b, tp, tridx, n, ttype, n, sdata, tid);
    if (tid == 0) out[0] = e;
}
// block i: pairs (j < i, i) with the lower slot first
__global__ void total_energy_kernel(fmap_tab t, BoxArgs b, double *out) {
    __shared__ double sdata[256];
    int tid = threadIdx.x, i = blockIdx.x;
    double e = sum_pair_energy(t, b, ld3(b.pos, i), b.ridx[i], i, b.ptype[i], i, sdata, tid);
    if (tid == 0) out[i] = e;
}

// ---- SPT=1 table from a fmap.ve.sp2p file (see Spt.jl for the file layout) ----
__device__ __forceinline__ unsigned ld32(const int8_t *w, long long byteoff) { unsigned v; __builtin_memcpy(&v, w + byteoff, 4); return v; }
__device__ __forceinline__ int cnt_mixed(unsigned x) { return __popc((x >> 1) & ~x & 0x55555555u); }
#define SP_PH 16

__global__ void sp1_scan_kernel(uint8_t *cls, int *stored, const int8_t *w2p, int nrot, long long nblk, int mapb) {
    long long blk = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= nblk) return;
    bool any2 = false; long long wo = 4 * (blk >> 4); int sh = (int)((blk & 15) << 1);
    for (int r = 0; r < nrot; r++) {
        long long bm = ld32(w2p, SP_PH + 16 * r);
        unsigned c = (ld32(w2p, bm + wo) >> sh) & 3u;
        any2 |= c == 2u;
        cls[blk * mapb + (r >> 2)] |= (uint8_t)(c << ((r & 3) << 1));
    }
    stored[blk] = any2 ? 1 : 0;
}

__global__ void sp1_fill_kernel(int8_t *sp, const int8_t *w2p, const int *blkof, long long nst, int nrot, long long sp_data) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nst * nrot) return;
    int r = (int)(t % nrot); long long d = t / nrot;
    long long blk = blkof[d];
    long long uh = SP_PH + 16 * r;
    long long bm = ld32(w2p, uh), widx = blk >> 4; int sh = (int)((blk & 15) << 1);
    unsigned word = ld32(w2p, bm + 4 * widx);
    unsigned c = (word >> sh) & 3u;
    if (c == 1u) {
        for (int cib = 0; cib < 64; cib++) {
            long long o = sp_data + ((d * 64 + cib) * nrot + r) * 2;
            sp[o] = 127; sp[o + 1] = 127;
        }
    } else if (c == 2u) {
        long long g = widx >> 2;
        long long rk = ld32(w2p, ld32(w2p, uh + 4) + 4 * g);
        for (long long q = g << 2; q < widx; q++) rk += cnt_mixed(ld32(w2p, bm + 4 * q));
        rk += cnt_mixed(word & ((1u << sh) - 1u));
        long long off = (long long)ld32(w2p, uh + 12) + ld32(w2p, (long long)ld32(w2p, uh + 8) + 4 * rk);
        int nd = (uint8_t)w2p[off];
        int bits = nd > 1 ? 32 - __clz((unsigned)(nd - 1)) : 0;
        for (int cib = 0; cib < 64; cib++) {
            int idx = 0;
            if (nd > 1) {
                int bp = cib * bits; long long by = off + 1 + 2 * nd + (bp >> 3);
                int v = (uint8_t)w2p[by] | ((uint8_t)w2p[by + 1] << 8);
                idx = (v >> (bp & 7)) & ((1 << bits) - 1);
            }
            long long pe = off + 1 + 2 * idx;
            long long o = sp_data + ((d * 64 + cib) * nrot + r) * 2;
            sp[o] = w2p[pe]; sp[o + 1] = w2p[pe + 1];
        }
    }
}

// returns 0 and sets *sp (device) and the grid parameters
static int load_sp2p(const char *path, int nrot_expect, int spt, int8_t **sp, int *blen, double *spacing, int *tern_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return fail("fsim_tab_open", path);
    uint8_t h[48];
    if (fread(h, 1, 48, f) != 48 || memcmp(h, "SPTV", 4) != 0) { fclose(f); return fail("fsim_tab_open", "not a fmap.ve.sp2p file"); }
    uint32_t mode; int32_t nrot, bl; double sp_; uint64_t nbytes; uint32_t tern;
    memcpy(&mode, h + 8, 4); memcpy(&nrot, h + 12, 4); memcpy(&bl, h + 16, 4); memcpy(&sp_, h + 20, 8); memcpy(&nbytes, h + 28, 8); memcpy(&tern, h + 36, 4);
    if (tern == 0) tern = 2;
    if (mode != 34) { fclose(f); return fail("fsim_tab_open", "only the per-rotation sparse table (SPT=2p) is supported"); }
    if (nrot != nrot_expect) { fclose(f); return fail("fsim_tab_open", "nrot of the grid differs from the rotation tables"); }
    if (tern != 1 && tern != 2) { fclose(f); return fail("fsim_tab_open", "unknown cell format in the grid file"); }
    *tern_out = (int)tern;
    std::vector<uint8_t> blob(nbytes);
    if (fread(blob.data(), 1, nbytes, f) != nbytes) { fclose(f); return fail("fsim_tab_open", "truncated grid file"); }
    fclose(f);
    int8_t *w2p; CK(cudaMalloc(&w2p, nbytes + 16)); CK(cudaMemset(w2p, 0, nbytes + 16)); CK(cudaMemcpy(w2p, blob.data(), nbytes, cudaMemcpyHostToDevice));
    if (spt == 2) { *sp = w2p; *blen = bl; *spacing = sp_; return 0; }     // SPT=2p: the blob itself is the table (padded by 16 bytes)
    long long nb = (bl + 3) / 4, nblk = nb * nb * nb; int mapb = (nrot + 3) >> 2;
    long long sp_map = 16 + 4 * nblk, sp_data = (sp_map + (long long)mapb * nblk + 15) / 16 * 16;
    uint8_t *cls; int *stored;
    CK(cudaMalloc(&cls, nblk * mapb)); CK(cudaMemset(cls, 0, nblk * mapb)); CK(cudaMalloc(&stored, nblk * 4));
    sp1_scan_kernel<<<(unsigned)((nblk + 127) / 128), 128>>>(cls, stored, w2p, nrot, nblk, mapb);
    CK(cudaGetLastError());
    std::vector<int> hs(nblk), dir(nblk), blkof;
    CK(cudaMemcpy(hs.data(), stored, nblk * 4, cudaMemcpyDeviceToHost));
    long long nst = 0;
    for (long long k = 0; k < nblk; k++) { if (hs[k]) { dir[k] = (int)nst; blkof.push_back((int)k); nst++; } else dir[k] = -1; }
    int8_t *tab; size_t tab_bytes = (size_t)sp_data + (size_t)nst * 64 * nrot * 2 + 4;
    if (cudaMalloc(&tab, tab_bytes) != cudaSuccess) return fail("fsim_tab_open", "out of device memory (energy table)");
    CK(cudaMemset(tab, 0, tab_bytes));
    int32_t hdr[4] = {(int32_t)nst, 0, 0, 0};
    CK(cudaMemcpy(tab, hdr, 16, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(tab + 16, dir.data(), nblk * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(tab + sp_map, cls, (size_t)nblk * mapb, cudaMemcpyDeviceToDevice));
    if (nst > 0) {
        int *d_blkof; CK(cudaMalloc(&d_blkof, nst * 4)); CK(cudaMemcpy(d_blkof, blkof.data(), nst * 4, cudaMemcpyHostToDevice));
        sp1_fill_kernel<<<(unsigned)((nst * nrot + 255) / 256), 256>>>(tab, w2p, d_blkof, nst, nrot, sp_data);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        cudaFree(d_blkof);
    }
    cudaFree(w2p); cudaFree(cls); cudaFree(stored);
    *sp = tab; *blen = bl; *spacing = sp_;
    return 0;
}

static BoxArgs box_args(const fsim_box *b) {
    BoxArgs a; a.pos = b->pos; a.bins = b->bins; a.ridx = b->ridx; a.ptype = b->ptype; a.cap = b->cap;
    a.box.x = b->box[0]; a.box.y = b->box[1]; a.box.z = b->box[2]; a.es = b->es; a.vs = b->vs;
    return a;
}

extern "C" {

const char *fsim_error(void) { return g_err; }
int fsim_device_count(void) { int n = 0; return cudaGetDeviceCount(&n) == cudaSuccess ? n : 0; }

fsim_tab *fsim_tab_open(int device, int n_types, int nrot, int nbin, const int16_t *aidx, const int16_t *amap, const double *rotm,
                        int n_ctx, const char *const *paths, int spt, int use_float) {
    if (spt != 1 && spt != 2) { fail("fsim_tab_open", "spt must be 1 or 2"); return NULL; }
    if (n_types < 1 || n_types > 5 || n_ctx != n_types * (n_types + 1) / 2) { fail("fsim_tab_open", "n_ctx must be n_types*(n_types+1)/2, n_types <= 5"); return NULL; }
    CKN(cudaSetDevice(device));
    fsim_tab *t = new fsim_tab();
    memset(&t->h, 0, sizeof t->h);
    t->device = device;
    t->aidx.assign(aidx, aidx + (size_t)nbin * nbin * nbin);
    CKN(cudaMalloc(&t->d_aidx, (size_t)nbin * nbin * nbin * 2)); CKN(cudaMemcpy(t->d_aidx, aidx, (size_t)nbin * nbin * nbin * 2, cudaMemcpyHostToDevice));
    CKN(cudaMalloc(&t->d_amap, (size_t)nrot * nrot * 2)); CKN(cudaMemcpy(t->d_amap, amap, (size_t)nrot * nrot * 2, cudaMemcpyHostToDevice));
    CKN(cudaMalloc(&t->d_rotm, (size_t)nrot * 9 * 8)); CKN(cudaMemcpy(t->d_rotm, rotm, (size_t)nrot * 9 * 8, cudaMemcpyHostToDevice));
    t->h.spt = spt; t->d_rotm_f = NULL;
    if (use_float) {
        std::vector<float> rf((size_t)nrot * 9); for (size_t i = 0; i < rf.size(); i++) rf[i] = (float)rotm[i];
        CKN(cudaMalloc(&t->d_rotm_f, rf.size() * 4)); CKN(cudaMemcpy(t->d_rotm_f, rf.data(), rf.size() * 4, cudaMemcpyHostToDevice));
        t->h.rotm_f = t->d_rotm_f;
    }
    t->h.n_types = n_types; t->h.n_ctx = n_ctx; t->h.nrot = nrot; t->h.nbin = nbin;
    t->h.aidx = t->d_aidx; t->h.amap = t->d_amap; t->h.rotm = t->d_rotm;
    double spacing0 = 0.0;
    for (int c = 0; c < n_ctx; c++) {
        int blen, ct; double spacing;
        if (load_sp2p(paths[c], nrot, spt, &t->d_sp[c], &blen, &spacing, &ct)) return NULL;
        if (c == 0) t->h.tern = ct;
        else if (ct != t->h.tern) { fail("fsim_tab_open", "cell format differs between contexts"); return NULL; }
        if (c == 0) spacing0 = spacing;
        else if (spacing != spacing0) { fail("fsim_tab_open", "grid spacing differs between contexts"); return NULL; }
        t->h.sp[c] = t->d_sp[c]; t->h.blen[c] = blen;
        double rc = blen * 0.5 * spacing; t->h.rcut2[c] = rc * rc;
    }
    t->h.scl = 1.0 / spacing0; t->scl = t->h.scl;
    return t;
}

void fsim_tab_close(fsim_tab *t) {
    if (!t) return;
    cudaFree(t->d_aidx); cudaFree(t->d_amap); cudaFree(t->d_rotm); if (t->d_rotm_f) cudaFree(t->d_rotm_f);
    for (int c = 0; c < t->h.n_ctx; c++) cudaFree(t->d_sp[c]);
    delete t;
}
double fsim_tab_scl(const fsim_tab *t) { return t->scl; }
int fsim_tab_tern(const fsim_tab *t) { return t->h.tern; }
int fsim_tab_blen(const fsim_tab *t, int ctx) { return t->h.blen[ctx]; }
int fsim_rot_index(const fsim_tab *t, double b1, double b2, double b3) {
    fmap_tab h = t->h; h.aidx = t->aidx.data();
    return fmap_rot_index(&h, b1, b2, b3);
}

fsim_box *fsim_box_open(fsim_tab *t, int cap, const double box[3], double kT, double es, double vs) {
    CKN(cudaSetDevice(t->device));
    fsim_box *b = new fsim_box();
    b->tab = t; b->cap = cap; b->kT = kT; b->es = es; b->vs = vs; b->mode = 0;
    for (int d = 0; d < 3; d++) b->box[d] = box[d];
    size_t c = cap > 0 ? cap : 1;
    CKN(cudaMalloc(&b->pos, c * 24)); CKN(cudaMalloc(&b->bins, c * 24)); CKN(cudaMalloc(&b->ridx, c * 4)); CKN(cudaMalloc(&b->ptype, c * 4));
    CKN(cudaMemset(b->pos, 0, c * 24)); CKN(cudaMemset(b->bins, 0, c * 24)); CKN(cudaMemset(b->ridx, 0, c * 4)); CKN(cudaMemset(b->ptype, 0, c * 4));
    CKN(cudaMalloc(&b->lcg, 8)); CKN(cudaMalloc(&b->mc, 6 * 8)); CKN(cudaMalloc(&b->ins, 8 * 8)); CKN(cudaMalloc(&b->res, 4 * 8));
    CKN(cudaMemset(b->mc, 0, 48)); CKN(cudaMemset(b->ins, 0, 64)); CKN(cudaMemset(b->res, 0, 32));
    CKN(cudaStreamCreateWithFlags(&b->stream, cudaStreamNonBlocking));
    return b;
}
void fsim_box_close(fsim_box *b) {
    if (!b) return;
    cudaStreamSynchronize(b->stream); cudaStreamDestroy(b->stream);
    cudaFree(b->pos); cudaFree(b->bins); cudaFree(b->ridx); cudaFree(b->ptype); cudaFree(b->lcg); cudaFree(b->mc); cudaFree(b->ins); cudaFree(b->res);
    delete b;
}

int fsim_box_set(fsim_box *b, int n, const double *pos, const double *bins, const int *ridx, const int *ptype, uint64_t lcg) {
    if (n > b->cap) return fail("fsim_box_set", "n exceeds the capacity");
    CK(cudaStreamSynchronize(b->stream));
    if (n > 0) {
        CK(cudaMemcpy(b->pos, pos, (size_t)n * 24, cudaMemcpyHostToDevice)); CK(cudaMemcpy(b->bins, bins, (size_t)n * 24, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(b->ridx, ridx, (size_t)n * 4, cudaMemcpyHostToDevice));
    }
    if (ptype) CK(cudaMemcpy(b->ptype, ptype, (size_t)b->cap * 4, cudaMemcpyHostToDevice));   // cap entries: types of the reserved slots too
    unsigned long long s = lcg;
    CK(cudaMemcpy(b->lcg, &s, 8, cudaMemcpyHostToDevice));
    return 0;
}
int fsim_box_get(fsim_box *b, int n, double *pos, double *bins, int *ridx, int *ptype, uint64_t *lcg) {
    CK(cudaStreamSynchronize(b->stream));
    if (n > 0) {
        if (pos) CK(cudaMemcpy(pos, b->pos, (size_t)n * 24, cudaMemcpyDeviceToHost));
        if (bins) CK(cudaMemcpy(bins, b->bins, (size_t)n * 24, cudaMemcpyDeviceToHost));
        if (ridx) CK(cudaMemcpy(ridx, b->ridx, (size_t)n * 4, cudaMemcpyDeviceToHost));
        if (ptype) CK(cudaMemcpy(ptype, b->ptype, (size_t)n * 4, cudaMemcpyDeviceToHost));
    }
    if (lcg) { unsigned long long s; CK(cudaMemcpy(&s, b->lcg, 8, cudaMemcpyDeviceToHost)); *lcg = s; }
    return 0;
}

int fsim_total_energy(fsim_box *b, int n, double *energy) {
    *energy = 0.0;
    if (n < 2) return 0;
    double *d_out; CK(cudaMalloc(&d_out, (size_t)n * 8));
    CK(cudaStreamSynchronize(b->stream));
    total_energy_kernel<<<n, 256, 0, b->stream>>>(b->tab->h, box_args(b), d_out);
    CK(cudaGetLastError());
    std::vector<double> h(n);
    CK(cudaMemcpyAsync(h.data(), d_out, (size_t)n * 8, cudaMemcpyDeviceToHost, b->stream));
    CK(cudaStreamSynchronize(b->stream));
    cudaFree(d_out);
    double s = 0.0; for (int i = 0; i < n; i++) s += h[i];
    *energy = s;
    return 0;
}

int fsim_mc_launch(fsim_box *b, int n, int batch, double js, double rs) {
    mc_kernel<<<1, 256, 0, b->stream>>>(b->tab->h, box_args(b), b->lcg, b->mc, n, batch, b->kT, js, rs * FMAP_ROT_DEFAULT);
    CK(cudaGetLastError());
    b->mode |= 1;
    return 0;
}
int fsim_insert_launch(fsim_box *b, int n, int batch) {
    insert_kernel<<<1, 256, 0, b->stream>>>(b->tab->h, box_args(b), b->lcg, b->ins, n, batch, b->kT);
    CK(cudaGetLastError());
    b->mode |= 2;
    return 0;
}
int fsim_reservoir_launch(fsim_box *b, int n, int committed, int batch, int max_accepts) {
    reservoir_kernel<<<1, 256, 0, b->stream>>>(b->tab->h, box_args(b), b->lcg, b->res, n, committed, batch, max_accepts, b->kT);
    CK(cudaGetLastError());
    b->mode |= 4;
    return 0;
}
int fsim_wait(fsim_box *b, double *mc, double *ins, double *res) {
    CK(cudaStreamSynchronize(b->stream));
    if (mc) CK(cudaMemcpy(mc, b->mc, 48, cudaMemcpyDeviceToHost));
    if (ins) CK(cudaMemcpy(ins, b->ins, 64, cudaMemcpyDeviceToHost));
    if (res) CK(cudaMemcpy(res, b->res, 32, cudaMemcpyDeviceToHost));
    b->mode = 0;
    return 0;
}

int fsim_place(fsim_tab *t, double es, double vs, const double box[3], int n_exist, int n_new, const int *types,
               double *pos, double *bins, int *ridx, int *ptype, uint64_t *lcg) {
    CK(cudaSetDevice(t->device));
    int total = n_exist + n_new; size_t c = total > 0 ? total : 1;
    BoxArgs a; a.cap = total; a.box.x = box[0]; a.box.y = box[1]; a.box.z = box[2]; a.es = es; a.vs = vs;
    CK(cudaMalloc(&a.pos, c * 24)); CK(cudaMalloc(&a.bins, c * 24)); CK(cudaMalloc(&a.ridx, c * 4)); CK(cudaMalloc(&a.ptype, c * 4));
    double *d_out; CK(cudaMalloc(&d_out, 8));
    if (n_exist > 0) {
        CK(cudaMemcpy(a.pos, pos, (size_t)n_exist * 24, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(a.ridx, ridx, (size_t)n_exist * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(a.ptype, ptype, (size_t)n_exist * 4, cudaMemcpyHostToDevice));
    }
    fmap_tab h = t->h;
    std::vector<int16_t> &ai = t->aidx; h.aidx = ai.data();       // host rotation index
    unsigned long long state = *lcg;
    int placed = 0, placed_total = n_exist;
    for (int k = 0; k < n_new; k++) {
        int self_type = types[k];
        bool ok = false;
        fmap_v3 tp = {0, 0, 0}, tb = {0, 0, 0}; int tridx = 0;
        for (int attempt = 0; attempt < 1000; attempt++) {
            double mp[3], mb[3];
            for (int d = 0; d < 3; d++) { double r1 = fmap_lcg(&state); mp[d] = r1 * box[d]; double r2 = fmap_lcg(&state); mb[d] = r2 * t->h.nbin; }
            tp.x = mp[0]; tp.y = mp[1]; tp.z = mp[2]; tb.x = mb[0]; tb.y = mb[1]; tb.z = mb[2];
            tridx = fmap_rot_index(&h, mb[0], mb[1], mb[2]);
            if (placed_total == 0) { ok = true; break; }
            ins_energy_kernel<<<1, 256>>>(t->h, a, d_out, tp, tridx, self_type, placed_total);
            double e; CK(cudaMemcpy(&e, d_out, 8, cudaMemcpyDeviceToHost));
            if (e < 999.0) { ok = true; break; }
        }
        if (!ok) break;
        int s = n_exist + placed;
        pos[3 * s] = tp.x; pos[3 * s + 1] = tp.y; pos[3 * s + 2] = tp.z;
        bins[3 * s] = tb.x; bins[3 * s + 1] = tb.y; bins[3 * s + 2] = tb.z;
        ridx[s] = tridx; ptype[s] = self_type;
        CK(cudaMemcpy(a.pos + 3 * placed_total, pos + 3 * s, 24, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(a.ridx + placed_total, ridx + s, 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(a.ptype + placed_total, ptype + s, 4, cudaMemcpyHostToDevice));
        placed++; placed_total++;
    }
    *lcg = state;
    cudaFree(a.pos); cudaFree(a.bins); cudaFree(a.ridx); cudaFree(a.ptype); cudaFree(d_out);
    return placed;
}

}  // extern "C"
