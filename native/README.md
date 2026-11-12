# native: the numerical core as plain C libraries

CUDA C + cuFFT shared libraries with a plain C interface. They hold the heavy numerics; a host language only needs a thin wrapper
(Julia `ccall`, Python `ctypes`, a C++ plugin). All arrays are host memory, double precision; the library owns the device memory.

| library | header | what it does | Julia wrapper |
|---|---|---|---|
| `libfmapgrid.so` | (`fmapgrid.cu`, `extern "C"` block) | energy grid of one orientation: crowder grids kept as FFTs, probe mapping, cross-correlation, quantisation, pose selection; pair energies of listed poses (`fmapdd`) | `src/grid/gpu/native.jl` |
| `libfmapsim.so` | `fmapsim.h`, `fmap_core.h` | NVT Monte Carlo of rigid molecules: displacement and same-box insertion moves, reservoir fill, random placement, total energy; the energy table is expanded on the device from `fmap.ve.sp2p` | `src/sim/gpu/nvt_native.jl` |
| `libfmapsim2iq.so` | `fmapsim2iq.h` | I(q) of an atomic structure: density grid (vacuum, excluded volume, shells), FFT, shell binning | `src/sim2iq/gpu/Sim2IqGPU.jl` |

```
module load CUDA && make -C native          # both; ARCHS="80 89" to narrow, NVCC=... to pick the compiler
```
The libraries link `libcudart` and `libcufft` dynamically with an rpath to the toolkit that built them; running needs the NVIDIA driver only.
Julia finds them next to the sources, or through `FMAPGRID_LIB` / `FMAPSIM2IQ_LIB`.

Conventions: a context (`fs2i_open`, `fg_open`) owns device buffers and FFT plans and is reused by every call (open once per grid size);
calls return `>= 0` on success and `-1` on error with the message in `fs2i_error()` / `fg_error()`; a context is not thread safe.

Example (C, I(q) of one structure):
```c
fs2i_ctx *c = fs2i_open(620, 0);
int n = fs2i_iq(c, natoms, x, y, z, vol, et, 1.0, 4, 2.8, -1.0, 0.011, -0.011, fs2i_u2B(0.25), q, iq, 311, shell_vol);
fs2i_close(c);
```
Checks: `julia --project=. test/native_grid.jl [blen] [norient]` (grid library against the CPU code), `bash test/test_sim2iq_gpu.sh`
(I(q) against the C reference output, max relative difference about 1e-7).

`fmap_core.h` is header-only host + device code (the rotation index of an orientation, the block-sparse table lookup, the pair energy) with no
dependence on a framework, so a plugin engine (FEASST, hoomd) can include it and only has to provide the table arrays.
