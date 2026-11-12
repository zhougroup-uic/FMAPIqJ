/* fmap_core.h - the FMAP pair energy as header-only host + device code (CUDA, or plain C/C++).
 *
 * Everything a simulation needs to evaluate the protein-protein energy from the tables, with no dependence on a framework: the rotation
 * index of an orientation, the lookup of an energy cell in the block-sparse table (SPT=1, built by fmapsim.cu from a fmap.ve.sp2p file),
 * and the pair energy of two rigid molecules given their positions and rotation indices. The Julia engine (through libfmapsim.so) and
 * the plugin engines (FEASST, hoomd) use the same definitions; a plugin only has to provide the table arrays.
 *
 * Conventions (as in the C reference, fmap_energy.c): the pair energy is NOT symmetric, the cell of the separation is taken in the frame of
 * the FIRST molecule; orientation bins are b = (alpha, (1 - cos beta) nbin / 2, gamma) in [0, nbin); energies are in kcal/mol, 9999 marks
 * an overlap.
 */
#ifndef FMAP_CORE_H
#define FMAP_CORE_H

#include <stdint.h>
#include <math.h>

#if defined(__CUDACC__)
#define FMAP_HD __host__ __device__
#else
#define FMAP_HD
#endif

#define FMAP_TERN_MAX 127
#define FMAP_TERN_SCL 8.0
#define FMAP_ROT_DEFAULT 3.0
#define FMAP_MAX_CTX 15            /* contexts: sorted pairs of up to 5 molecule types */

typedef struct { double x, y, z; } fmap_v3;

/* The tables of one run. Pointers are host or device memory, whichever the code that reads them runs on. */
typedef struct {
    int n_types, n_ctx;            /* contexts: n_types * (n_types + 1) / 2, one per sorted type pair */
    int nrot, nbin;                /* orientation samples; bins per rotation period of the index table */
    int tern;                      /* cell format: 2 = [elec, vdw] (scaled live), 1 = summed value in the elec byte */
    double scl;                    /* 1 / grid spacing */
    const int16_t *aidx;           /* [nbin^3] orientation bin -> rotation index (alpha, beta, gamma slowest to fastest) */
    const int16_t *amap;           /* [nrot^2] relative orientation index of rotation i seen from rotation j, i * nrot + j */
    const double *rotm;            /* [nrot * 9] rotation matrices, row major */
    const int8_t *sp[FMAP_MAX_CTX];/* block-sparse energy table per context */
    int blen[FMAP_MAX_CTX];        /* grid length per context */
    double rcut2[FMAP_MAX_CTX];    /* squared cut-off per context: (blen / 2 / scl)^2 */
} fmap_tab;

/* ---- helpers ---- */

/* the C library's linear congruential generator: state = (state * 1103515245 + 12345) & 0x7fffffff, uniform in [0, 1] */
FMAP_HD static inline double fmap_lcg(unsigned long long *state) {
    *state = (*state * 1103515245ULL + 12345ULL) & 0x7fffffffULL;
    return (double)*state / 2147483647.0;
}

/* x mod m with the sign of the C fmod corrected into [0, m) */
FMAP_HD static inline double fmap_cwrap(double x, double m) {
    x = fmod(x, m);
    return x < 0.0 ? x + m : x;
}

FMAP_HD static inline double fmap_pbc(double x, double box) {
    double h = box * 0.5;
    return x > h ? x - box : (x < -h ? x + box : x);
}

/* sorted type pair (t1, t2) -> context index (0-based) */
FMAP_HD static inline int fmap_ctx_index(int t1, int t2, int n_types) {
    int i = t1 < t2 ? t1 : t2, j = t1 < t2 ? t2 : t1;
    return i * (2 * n_types - 1 - i) / 2 + j;
}

/* Rotation index of orientation bins: alpha and gamma are rounded (half away from zero, wrapped), beta uses floor clamped to the last bin
 * (the index tables have half-bin beta centres, bin j covers b2 in [j, j+1)). */
FMAP_HD static inline int fmap_rot_index(const fmap_tab *t, double b1, double b2, double b3) {
    int n = t->nbin;
    int i = (int)round(b1); i = i < 0 ? i + n : (i == n ? 0 : i);
    int j = (int)floor(b2); j = j < 0 ? 0 : (j >= n ? n - 1 : j);
    int k = (int)round(b3); k = k < 0 ? k + n : (k == n ? 0 : k);
    return (int)t->aidx[((long long)i * n + j) * n + k];
}

/* ---- the block-sparse table (SPT=1) ----
 * [16 B header: i32 stored blocks] [dir: i32 per block, index of its stored block or -1] [class map: ceil(nrot/4) bytes per block, 2 bits per
 * orientation: 0 all zero, 1 all overlap] [stored blocks: [cell in 4x4x4 block][orientation][elec, vdw]]; block = (bi * nb + bj) * nb + bk,
 * nb = ceil(blen / 4). */
FMAP_HD static inline void fmap_sp_ve(const int8_t *w, int blen, int nrot, int i, int j, int k, int a3, int *e0, int *e1) {
    long long nb = (blen + 3) >> 2, nblk = nb * nb * nb, mapb = (nrot + 3) >> 2;
    long long sp_map = 16 + 4 * nblk;
    long long blk = ((i >> 2) * nb + (j >> 2)) * nb + (k >> 2);
    int32_t d;
    __builtin_memcpy(&d, w + 16 + 4 * blk, 4);
    if (d >= 0) {
        long long sp_data = (sp_map + mapb * nblk + 15) / 16 * 16;
        long long off = sp_data + ((((long long)d * 64 + ((i & 3) * 4 + (j & 3)) * 4 + (k & 3)) * nrot) + a3) * 2;
        *e0 = w[off]; *e1 = w[off + 1];
        return;
    }
    unsigned mb = (uint8_t)w[sp_map + blk * mapb + (a3 >> 2)];
    unsigned c = (mb >> ((a3 & 3) << 1)) & 3u;
    *e0 = *e1 = (c == 0u) ? 0 : FMAP_TERN_MAX;
}

/* Pair energy (kcal/mol) of molecule 1 at p1 with rotation index r1 and molecule 2 at p2 with r2, in a periodic box; ctx selects the table of
 * the pair's types. es, vs scale the electrostatic and van der Waals channels of a two-channel grid (1 for a summed grid). */
FMAP_HD static inline double fmap_pair_energy(const fmap_tab *t, int ctx, fmap_v3 p1, int r1, fmap_v3 p2, int r2, fmap_v3 box, double es, double vs) {
    double rx = fmap_pbc(p1.x - p2.x, box.x), ry = fmap_pbc(p1.y - p2.y, box.y), rz = fmap_pbc(p1.z - p2.z, box.z);
    double d2 = rx * rx + ry * ry + rz * rz;
    if (d2 > t->rcut2[ctx]) return 0.0;
    int a3 = (int)t->amap[(long long)r1 * t->nrot + r2];
    const double *R = t->rotm + (long long)r1 * 9;
    double x = R[0] * rx + R[1] * ry + R[2] * rz;
    double y = R[3] * rx + R[4] * ry + R[5] * rz;
    double z = R[6] * rx + R[7] * ry + R[8] * rz;
    int blen = t->blen[ctx], h = blen / 2;
    int i = (int)(x * t->scl + h + 0.5), j = (int)(y * t->scl + h + 0.5), k = (int)(z * t->scl + h + 0.5);
    if (i < 0 || i >= blen || j < 0 || j >= blen || k < 0 || k >= blen) return 0.0;
    int e0, e1;
    fmap_sp_ve(t->sp[ctx], blen, t->nrot, i, j, k, a3, &e0, &e1);
    if (e0 == FMAP_TERN_MAX) return 9999.0;
    if (t->tern == 2) return ((double)e0 * es + (double)e1 * vs) / FMAP_TERN_SCL;
    return (double)e0 / FMAP_TERN_SCL;
}

#endif
