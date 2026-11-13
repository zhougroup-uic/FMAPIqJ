/* fmapsim.h - Monte Carlo of rigid molecules with FMAP pair energies on the GPU (CUDA), plain C interface.
 *
 * The device-side physics of the NVT engine: displacement moves (translation / rotation), same-box insertion moves (a molecule is removed and
 * re-inserted at a random place), reservoir insertion that fills a start-up deficit, random placement, total energy. The caller (any host
 * language) keeps the run logic: step-size adaptation, logs, trajectory files. A wrapper only passes arrays and calls:
 *
 *     fsim_tab *t  = fsim_tab_open(0, 1, nrot, nbin, aidx, amap, rotm, 1, paths, 1, 0);     // tables + energy grid, once per run
 *     fsim_box *b  = fsim_box_open(t, cap, box, kT, 1.0, 1.0);                             // one per replica (own stream and RNG)
 *     fsim_box_set(b, n, pos, bins, ridx, ptype, seed);
 *     fsim_mc_launch(b, n, steps, js, rs);   fsim_wait(b, mc6, NULL, NULL);               // many boxes can be launched, then waited
 *
 * The pair energy itself is native/fmap_core.h (host + device), shared with the plugin engines. Arrays are host memory, double precision;
 * positions and orientation bins are particle-major (x y z of particle 0, then particle 1, ...). Orientation bins: (alpha, (1-cos beta) nbin / 2,
 * gamma) in [0, nbin). Slots [0, n) of a box are the active particles, compact; a particle removed from the middle shifts the later ones down
 * (the energy depends on the order of the two molecules of a pair, so the order of the molecules is state).
 * Return values >= 0 on success, -1 on error (fsim_error()). A box is driven from one thread at a time.
 */
#ifndef FMAPSIM_H
#define FMAPSIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fsim_tab fsim_tab;
typedef struct fsim_box fsim_box;

const char *fsim_error(void);
int fsim_device_count(void);

/* Tables on `device`: the rotation tables (aidx [nbin^3] i16, amap [nrot^2] i16, rotm [nrot*9] f64, all host arrays that are copied) and one
 * energy grid file (fmap.ve.sp2p) per context, n_ctx = n_types*(n_types+1)/2 sorted type pairs. The cell format (1 summed channel, 2 elec + vdw) is read from the
 * files and must be the same in all of them (fsim_tab_tern). The files are expanded on the device into the lookup table (spt = 1, fastest, larger) or, with spt = 2 (the Julia driver's default), used in place (SPT=2p, 7x smaller for the N=200
 * grid, about 3% slower); use_float = 1 does the lookup arithmetic in float (same energies, see fmap_core.h). */
fsim_tab *fsim_tab_open(int device, int n_types, int nrot, int nbin, const int16_t *aidx, const int16_t *amap, const double *rotm,
                        int n_ctx, const char *const *sp2p_paths, int spt, int use_float);
void fsim_tab_close(fsim_tab *t);
double fsim_tab_scl(const fsim_tab *t);
int fsim_tab_tern(const fsim_tab *t);                     /* cell format of the grids, from the files: 1 summed channel, 2 elec + vdw */                   /* 1 / grid spacing */
int fsim_tab_blen(const fsim_tab *t, int ctx);
int fsim_rot_index(const fsim_tab *t, double b1, double b2, double b3);   /* rotation index of orientation bins (host) */

/* One box (replica): capacity `cap` molecules, edge lengths box[3], temperature in kcal/mol (kT), scales of the electrostatic and van der Waals
 * channels (1, 1 for a summed grid). Own CUDA stream and RNG state, so boxes run concurrently. */
fsim_box *fsim_box_open(fsim_tab *t, int cap, const double box[3], double kT, double elec_scale, double vdw_scale);
void fsim_box_close(fsim_box *b);
/* Upload n molecules (slots 0..n-1); ptype also gives the types of the slots n.. cap-1 that are reserved for reservoir insertion. lcg: RNG state. */
int fsim_box_set(fsim_box *b, int n, const double *pos, const double *bins, const int *ridx, const int *ptype, uint64_t lcg);
/* Download n molecules and the RNG state (pass NULL to skip an array). Waits for pending work. */
int fsim_box_get(fsim_box *b, int n, double *pos, double *bins, int *ridx, int *ptype, uint64_t *lcg);

/* Energy: sum over pairs of the n molecules of the box (lower slot = first molecule), kcal/mol. */
int fsim_total_energy(fsim_box *b, int n, double *energy);

/* Moves, asynchronous on the box's stream; fsim_wait synchronizes and copies the results (any pointer may be NULL):
 *   mc  [6]: delta energy, accepted, tries translation, accepted translation, tries rotation, accepted rotation
 *   ins [8]: delta energy, 0, attempts, accepted, n, 0, 0, 0
 *   res [4]: delta energy, accepted, n active, accepted moves of type 0
 * fsim_mc_launch: `batch` steps of n molecules; translation and rotation 50/50, maximum displacement js (translation, Angstrom) and rs (rotation,
 * in units of rot_default = 3 bins). fsim_insert_launch: `batch` remove-and-reinsert moves. fsim_reservoir_launch: up to `batch` insertions into slots
 * n .. committed-1 (types preset by fsim_box_set), stopping after max_accepts accepted. */
int fsim_mc_launch(fsim_box *b, int n, int batch, double js, double rs);
int fsim_insert_launch(fsim_box *b, int n, int batch);
int fsim_reservoir_launch(fsim_box *b, int n, int committed, int batch, int max_accepts);
int fsim_wait(fsim_box *b, double *mc, double *ins, double *res);

/* Random placement with overlap rejection (up to 1000 tries per molecule, accept when the insertion energy < 999): appends up to n_new molecules
 * of the types given in `types[n_new]` after the n_exist already in pos/bins/ridx/ptype (capacity n_exist + n_new), using and updating *lcg.
 * Returns the number placed (fewer than n_new when a molecule could not be placed). Uses the box's table and scales, not its state. */
int fsim_place(fsim_tab *t, double elec_scale, double vdw_scale, const double box[3], int n_exist, int n_new, const int *types,
               double *pos, double *bins, int *ridx, int *ptype, uint64_t *lcg);

#ifdef __cplusplus
}
#endif
#endif
