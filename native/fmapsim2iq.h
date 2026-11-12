/* fmapsim2iq.h - small-angle scattering I(q) of an atomic structure on the GPU (CUDA + cuFFT), plain C interface.
 *
 * Core of sim2iq: the atoms are mapped to a density grid (vacuum electron density, solvent excluded volume, outer and inner shell),
 * the grid is Fourier transformed and |F(q)|^2 is averaged in spherical shells of q. Everything numeric is on the device; reading
 * structures (PQR, trajectories) is left to the caller, so any host language can use the library with a thin wrapper:
 *
 *     fs2i_ctx *c = fs2i_open(620, 0);                         // grid 620^3, device 0 (buffers and FFT plan, reused by every call)
 *     n = fs2i_iq(c, natoms, x, y, z, vol, et, 1.0, 4, 2.8, -1.0, 0.011, -0.011, fs2i_u2B(0.25), q, iq, maxpts, shell_vol);
 *     fs2i_close(c);
 *
 * Arrays are host memory, double precision, atoms as separate coordinate arrays (x y z in Angstrom, vol in Angstrom^3, et = element type).
 * Return values are >= 0 on success, -1 on error (fs2i_error() holds the message). The context is not thread safe: use one per thread.
 */
#ifndef FMAPSIM2IQ_H
#define FMAPSIM2IQ_H

#ifdef __cplusplus
extern "C" {
#endif

/* element types of the Cromer-Mann table; atoms of type FS2I_UNKNOWN are ignored */
enum { FS2I_UNKNOWN = 0, FS2I_HOH = 1, FS2I_H = 2, FS2I_C = 3, FS2I_N = 4, FS2I_O = 5, FS2I_P = 6, FS2I_S = 7 };

typedef struct fs2i_ctx fs2i_ctx;

const char *fs2i_error(void);                 /* message of the last failed call */
int fs2i_device_count(void);
int fs2i_element_type(const char *element);   /* first letter, case-insensitive: H C N O P S, else FS2I_UNKNOWN */
double fs2i_u2B(double u);                    /* B factor of a displacement u: 8 pi^2 u^2 */

/* Grid of n x n x n cells on `device`. Device memory: 24 n^3 bytes (n = 620: 5.7 GB, n = 984: 23 GB). */
fs2i_ctx *fs2i_open(int n, int device);
void fs2i_close(fs2i_ctx *c);

/* One I(q).
 *   natoms, x, y, z, vol, et : the structure (coordinates relative to the grid origin; the box is periodic with side n*spacing)
 *   spacing                  : grid spacing, Angstrom
 *   steps                    : 1 vacuum density, 2 + solvent excluded volume, 3 + outer shell, 4 + inner shell
 *   cutoff_offset            : outer shell offset, Angstrom (2.8)
 *   se_w, ou_w, in_w         : weights of excluded volume (-1.0), outer shell (0.011) and inner shell (-0.011)
 *   bfactor                  : B factor of the Gaussian smearing (fs2i_u2B(0.25 * spacing))
 *   q, iq                    : output, at most maxpts points (n / 2 + 1 are enough); q = 0 first
 *   shell_vol                : optional, 2 doubles: volume of the outer and inner shell (Angstrom^3; unset when steps < 3 / < 4), or NULL
 * Returns the number of points written. */
int fs2i_iq(fs2i_ctx *c, int natoms, const double *x, const double *y, const double *z, const double *vol, const int *et,
            double spacing, int steps, double cutoff_offset, double se_w, double ou_w, double in_w, double bfactor,
            double *q, double *iq, int maxpts, double *shell_vol);

#ifdef __cplusplus
}
#endif
#endif
