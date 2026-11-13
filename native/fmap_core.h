/* fmap_core.h - the FMAP pair energy as header-only host + device code (CUDA, or plain C/C++).
 *
 * Everything a simulation needs to evaluate the protein-protein energy from the tables, with no dependence on a framework: the rotation
 * index of an orientation, the lookup of an energy cell in the fmap.ve.sp2p table (SPT=2p, read in place) or in the block-sparse table built from it (SPT=1),
 * and the pair energy of two rigid molecules given their positions and rotation indices. The Julia engine (through libfmapsim.so) and
 * the plugin engines (FEASST, hoomd) use the same definitions; a plugin only has to provide the table arrays.
 *
 * Table layouts (fmap_tab.spt): SPT=2p (default: spt = 2, or 0 when unset), the blob of the fmap.ve.sp2p file itself, read in place (7x smaller for the N=200 grid, about 3% slower), or
 * SPT=1 (spt = 1), the block-sparse table that fmapsim.cu builds from a fmap.ve.sp2p file (fastest, larger). Precision: fmap_pair_energy works in double,
 * fmap_pair_energy_f does the lookup arithmetic in float (needs rotm_f; falls back to double when rotm_f is NULL) and decides the cell in
 * double near a cell boundary. Both layouts and both precisions return the same energies.
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

#define FMAP_TERN_MAX 127            /* cell value reserved for excluded volume (hard-core overlap); real values lie in [-127, 126] */
#define FMAP_TERN_SCL 8.0            /* cell values are energies in units of 1/8 kcal/mol (0.125 kcal/mol) */
#define FMAP_CLASH_ENERGY 9999.0     /* kcal/mol returned for a pair in overlap */
#define FMAP_ROT_DEFAULT 3.0
#define FMAP_MAX_CTX 15            /* contexts: sorted pairs of up to 5 molecule types */

typedef struct { double x, y, z; } fmap_v3;

/* The tables of one run. Pointers are host or device memory, whichever the code that reads them runs on. */
typedef struct {
    int n_types, n_ctx;            /* contexts: n_types * (n_types + 1) / 2, one per sorted type pair */
    int nrot, nbin;                /* orientation samples; bins per rotation period of the index table */
    int tern;                      /* cell format: 2 = [elec, vdw] (scaled live), 1 = summed value in the elec byte */
    int spt;                       /* layout of sp[]: 2 = the fmap.ve.sp2p blob after its 48 byte header (SPT=2p); 1 = block-sparse SPT=1 table;
                                    * 0 = not set (zero-initialised struct), the same as 2. There is no uncompressed layout: SPT=1 reads one block-directory entry and
                                    * one cell per lookup, nearly the access pattern of a dense table, at a fraction of the memory. */
    double scl;                    /* 1 / grid spacing */
    const int16_t *aidx;           /* [nbin^3] orientation bin -> rotation index (alpha, beta, gamma slowest to fastest) */
    const int16_t *amap;           /* [nrot^2] relative orientation index of rotation i seen from rotation j, i * nrot + j */
    const double *rotm;            /* [nrot * 9] rotation matrices, row major */
    const float *rotm_f;           /* [nrot * 9] the same in float, only for fmap_pair_energy_f (NULL: that function uses double) */
    const int8_t *sp[FMAP_MAX_CTX];/* energy table per context, layout spt */
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

/* ---- the per-rotation sparse table (SPT=2p), used unless fmap_tab.spt == 1 ----
 * `w` is the blob of a fmap.ve.sp2p file after its 48 byte header, 4-byte aligned and padded by 16 bytes at the end; it is read in place.
 * Per relative orientation (unit, 16 B at 16 + 16 a3): [class bitmap of the 4^3-cell blocks, 2 bits each (0 all zero, 1 all overlap,
 * 2 mixed)] [rank prefix per 4 words] [slice offsets of the mixed blocks] [pool of palette slices: nd, nd [elec, vdw] pairs, 64 cell
 * indices of ceil(log2 nd) bits]. */
FMAP_HD static inline uint32_t fmap_ld32(const int8_t *w, long long off) {
#if defined(__CUDA_ARCH__)
    return __ldg(reinterpret_cast<const uint32_t *>(w + off));
#else
    uint32_t v;
    __builtin_memcpy(&v, w + off, 4);
    return v;
#endif
}
FMAP_HD static inline int fmap_cnt_mixed(uint32_t x) {
    x = (x >> 1) & ~x & 0x55555555u;
#if defined(__CUDA_ARCH__)
    return __popc(x);
#else
    return __builtin_popcount(x);
#endif
}
FMAP_HD static inline void fmap_sp2p_ve(const int8_t *w, int blen, int i, int j, int k, int a3, int *e0, int *e1) {
    long long nb = (blen + 3) >> 2;
    long long blk = ((i >> 2) * nb + (j >> 2)) * nb + (k >> 2);
    long long uh = 16 + 16 * (long long)a3;
    long long bm = fmap_ld32(w, uh), widx = blk >> 4;
    int sh = (int)((blk & 15) << 1);
    uint32_t word = fmap_ld32(w, bm + 4 * widx);
    unsigned c = (word >> sh) & 3u;
    if (c == 0u) { *e0 = *e1 = 0; return; }
    if (c == 1u) { *e0 = *e1 = FMAP_TERN_MAX; return; }
    long long g = widx >> 2;
    long long r = fmap_ld32(w, (long long)fmap_ld32(w, uh + 4) + 4 * g);
    for (long long q = g << 2; q < widx; q++) r += fmap_cnt_mixed(fmap_ld32(w, bm + 4 * q));
    r += fmap_cnt_mixed(word & ((1u << sh) - 1u));
    long long off = (long long)fmap_ld32(w, uh + 12) + fmap_ld32(w, (long long)fmap_ld32(w, uh + 8) + 4 * r);
    int nd = (uint8_t)w[off], idx = 0;
    if (nd > 1) {
        unsigned x = (unsigned)(nd - 1);
#if defined(__CUDA_ARCH__)
        int bits = 32 - __clz(x);
#else
        int bits = 32 - __builtin_clz(x);
#endif
        int bp = (((i & 3) * 4 + (j & 3)) * 4 + (k & 3)) * bits;
        long long by = off + 1 + 2 * nd + (bp >> 3);
        int v = (uint8_t)w[by] | ((uint8_t)w[by + 1] << 8);
        idx = (v >> (bp & 7)) & ((1 << bits) - 1);
    }
    long long pe = off + 1 + 2 * idx;
    *e0 = w[pe]; *e1 = w[pe + 1];
}

/* energy cell (i, j, k) of context ctx for relative orientation a3, in the layout of t->spt (uniform over the threads of a warp) */
FMAP_HD static inline void fmap_cell(const fmap_tab *t, int ctx, int i, int j, int k, int a3, int *e0, int *e1) {
    if (t->spt == 1) fmap_sp_ve(t->sp[ctx], t->blen[ctx], t->nrot, i, j, k, a3, e0, e1);
    else fmap_sp2p_ve(t->sp[ctx], t->blen[ctx], i, j, k, a3, e0, e1);
}

/* energy of a looked-up cell pair [e0, e1] */
FMAP_HD static inline double fmap_cell_energy(const fmap_tab *t, int e0, int e1, double es, double vs) {
    if (e0 == FMAP_TERN_MAX) return FMAP_CLASH_ENERGY;
    if (t->tern == 2) return ((double)e0 * es + (double)e1 * vs) / FMAP_TERN_SCL;
    return (double)e0 / FMAP_TERN_SCL;
}

/* Pair energy (kcal/mol) of two molecules with rotation indices r1, r2, from the separation (rx,ry,rz) = pos(molecule 1) - pos(molecule 2), which the
 * caller must already have reduced to the minimum image (no periodic wrapping here). ctx selects the table of the pair's types. es, vs scale the
 * electrostatic and van der Waals channels of a two-channel grid (1 for a summed grid). The *_pbc functions below wrap the separation first. */
FMAP_HD static inline double fmap_pair_energy(const fmap_tab *t, int ctx, double rx, double ry, double rz, int r1, int r2, double es, double vs) {
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
    fmap_cell(t, ctx, i, j, k, a3, &e0, &e1);
    return fmap_cell_energy(t, e0, e1, es, vs);
}

/* Same as fmap_pair_energy with the lookup arithmetic in float (t->rotm_f); within 2e-4 of a cell boundary the cell is decided in double (t->rotm), so
 * the result equals fmap_pair_energy except for separations that close to a boundary in double itself. Without rotm_f it is fmap_pair_energy. */
FMAP_HD static inline double fmap_pair_energy_f(const fmap_tab *t, int ctx, double rx, double ry, double rz, int r1, int r2, double es, double vs) {
    if (!t->rotm_f) return fmap_pair_energy(t, ctx, rx, ry, rz, r1, r2, es, vs);
    float dx = (float)rx, dy = (float)ry, dz = (float)rz;
    if (dx * dx + dy * dy + dz * dz > (float)t->rcut2[ctx]) return 0.0;
    int a3 = (int)t->amap[(long long)r1 * t->nrot + r2];
    const float *R = t->rotm_f + (long long)r1 * 9;
    float scl = (float)t->scl;
    int blen = t->blen[ctx], h = blen / 2;
    float tx = (R[0] * dx + R[1] * dy + R[2] * dz) * scl + (float)h + 0.5f;
    float ty = (R[3] * dx + R[4] * dy + R[5] * dz) * scl + (float)h + 0.5f;
    float tz = (R[6] * dx + R[7] * dy + R[8] * dz) * scl + (float)h + 0.5f;
    int i = (int)tx, j = (int)ty, k = (int)tz;
    const float tol = 2.0e-4f;      /* float rounding of t is below ~5e-5 */
    float fx = tx - floorf(tx), fy = ty - floorf(ty), fz = tz - floorf(tz);
    if (fx < tol || fx > 1.0f - tol || fy < tol || fy > 1.0f - tol || fz < tol || fz > 1.0f - tol) {
        const double *Rd = t->rotm + (long long)r1 * 9;
        i = (int)((Rd[0] * rx + Rd[1] * ry + Rd[2] * rz) * t->scl + h + 0.5);
        j = (int)((Rd[3] * rx + Rd[4] * ry + Rd[5] * rz) * t->scl + h + 0.5);
        k = (int)((Rd[6] * rx + Rd[7] * ry + Rd[8] * rz) * t->scl + h + 0.5);
    }
    if (i < 0 || i >= blen || j < 0 || j >= blen || k < 0 || k >= blen) return 0.0;
    int e0, e1;
    fmap_cell(t, ctx, i, j, k, a3, &e0, &e1);
    return fmap_cell_energy(t, e0, e1, es, vs);
}


/* Periodic-box wrappers: minimum image of p1 - p2 in the box, then the functions above. */
FMAP_HD static inline double fmap_pair_energy_pbc(const fmap_tab *t, int ctx, fmap_v3 p1, int r1, fmap_v3 p2, int r2, fmap_v3 box, double es, double vs) {
    return fmap_pair_energy(t, ctx, fmap_pbc(p1.x - p2.x, box.x), fmap_pbc(p1.y - p2.y, box.y), fmap_pbc(p1.z - p2.z, box.z), r1, r2, es, vs);
}
FMAP_HD static inline double fmap_pair_energy_f_pbc(const fmap_tab *t, int ctx, fmap_v3 p1, int r1, fmap_v3 p2, int r2, fmap_v3 box, double es, double vs) {
    return fmap_pair_energy_f(t, ctx, fmap_pbc(p1.x - p2.x, box.x), fmap_pbc(p1.y - p2.y, box.y), fmap_pbc(p1.z - p2.z, box.z), r1, r2, es, vs);
}

#endif
