# FMAPIqJ — FMAPIq in Julia (CPU + GPU)

Small-angle X-ray scattering I(q) of concentrated protein solutions from FMAP energy maps,
Monte Carlo (NVT) sampling and a Debye-type scattering calculation. Everything is Julia
(the GPU paths are CUDA C libraries in `native/` called from Julia); the original C/CUDA sources are kept in `reference/` for comparison.

Pipeline (one command, `bin/fmapiq.jl`):

1. **energy maps** `fmappre` → `fmap` → `fmapstat`: protein–crowder energy grids for 72 orientations,
   written directly as `fmap.ve.sp2p`, the spatially compressed grid (`src/spt/Spt.jl`; no intermediate files, no zlib).
2. **simulation** `simulation_nvt` / `simulation_nvt_gpu`: independent NVT Monte Carlo runs (box with
   N proteins), all replicas at once on the GPU.
3. **scattering** `sim2iq`: P(q) of one protein and I(q) of every configuration (FFT grid), then
   `Iqs.txt` (per-protein I(q)), `Iqs.stat.txt` (mean ± SEM over runs, normalised to P(0)) and `iq_plot.svg`.

## Requirements
- Julia ≥ 1.9 (developed and tested with 1.12).
- GPU mode: an NVIDIA GPU with driver and the CUDA toolkit once, to build the libraries in `native/` (`module load CUDA && make -C native`); Julia itself needs no CUDA package. Memory: about 6 GB for the
  energy maps and the N=200 example; the N=800 example needs about 23 GB for its 984³ I(q) grid and was run on a 46 GB L40.
- CPU mode works without a GPU and takes about 6 times longer for the N=200 example (12.5 min against 2 min). Measured on 8 cores (N=200): energy maps about 9.5 min,
  Monte Carlo of 20000 cycles about 36 s per replica (2 threads each), then about 40 s per I(q); the whole example takes 12.5 min.
- Disk: about 0.4 GB per work directory (`fmap.ve.sp2p`). Both simulation engines (GPU and CPU) read the file in place (SPT=2p),
  about 0.4 GB of memory for the N=200 example instead of the 5.7 GB dense grid; a dense grid is never made.
  `FMAP_SPT=1` (environment) expands it into the block-sparse lookup table instead, about 2.7 GB, about 3% faster (GPU, N=800).
- Python (numpy) is only needed for the optional `tools/traj2pdb.py`; no C extension anywhere.

## Native GPU libraries (no CUDA.jl)
Every GPU stage calls a CUDA C library in `native/` through `ccall`; Julia needs only the shared libraries and the NVIDIA driver:
`libfmapgrid.so` (energy maps, `fmapdd`), `libfmapsim.so` (the NVT Monte Carlo engine) and `libfmapsim2iq.so` (I(q)). Build once on a
node with the toolkit: `module load CUDA && make -C native` (`ARCHS="80 89"` to narrow; `FMAPGRID_LIB`, `FMAPSIM_LIB`,
`FMAPSIM2IQ_LIB` relocate a library). The pair energy itself is the header `native/fmap_core.h` (host + device), the interface of each
library is in `native/*.h`; see `native/README.md`. FFTW is loaded only by the CPU stages (energy maps, I(q)), so a GPU run needs
no FFTW; the notebook adds it to its own environment for CPU runs only. Checks: `test/native_grid.jl`, `test/native_sim.jl`, `test/test_sim2iq_gpu.sh`.

## Install
```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl          # sim2iq (CPU) against the C reference output, about 1 min
```

## Run
```
julia --project=. bin/fmapiq.jl protein.pqr --work-dir run --gpu \
      --n-proteins 200 --box-size 619.65 --ion 0.016 --temp 25 --vscl 0.16 --n-reps 3 --n-cycles 20000
```
Input is a PQR with the original atom names (centred or not). `--help` lists all options; `--gpu` runs
every stage on the GPU, without it the CPU engines are used. Options in short:

| option | meaning | default |
|---|---|---|
| `--n-proteins N`, `--box-size L` | proteins per box, box edge (Å) | 200, 619.65 |
| `--ion C`, `--temp T` | ionic strength (M), temperature (°C) | 0.016, 25 |
| `--vscl X` | van der Waals scale (negative: from mass) | -1 |
| `--n-reps R`, `--n-cycles C` | independent runs, MC cycles per run | 3, 20000 |
| `--spacing X` | I(q) grid spacing (Å) | 1.0 |
| `--threads`, `--sim-threads` | CPU threads (total / per CPU simulation) | cores/2, 2 |
| `--pack` | archive the results (`<work>.tar.gz`) | off |

## Examples
| | `examples/n200_test` | `examples/n800_production` |
|---|---|---|
| system | N=200, L=619.65 Å | N=800, L=984.15 Å (N800 L24.3) |
| runs | 64 × 20000 cycles on GPU (3 on CPU) | 64 × 20000 cycles |
| time (1 × L40, GPU) | about 4 min | about 8 min |
| check | P(0) = 2.186e6 | P(0) = 2.185e6 |

```
examples/n200_test/run.sh [workdir]         # test example; MODE=cpu to force the CPU engines
examples/n800_production/run.sh [workdir]   # production example
# notebooks (Julia kernel; MODE gpu/64 runs or cpu/3 runs):
#   examples/n200_test/n200_test.ipynb         runs the pipeline
#   examples/n200_test/n200_test.cached.ipynb  same notebook with stored outputs of a Colab T4 run (64 runs, ~10 min, PASS)
```
Both use `examples/cenA.pqr` (1959 atoms), build the maps, simulate, compute I(q) and compare
`Iqs.stat.txt` with `expected/Iqs.stat.n<runs>.txt` via `tools/check_result.jl` (prints PASS/FAIL). The N=800
result agrees with the published FMAPIq curve for the same system to 1.0001 on average (0.984–1.015
point by point) over q = 0.01–0.4 Å⁻¹.

## Notebooks
`examples/n200_test/n200_test.ipynb` needs a Julia Jupyter kernel. One command
installs IJulia and a kernel `julia` (started with `--project=<repo>`) and launches Jupyter:
```
julia tools/notebook.jl            # install + launch;   julia tools/notebook.jl install   # kernel only
```
If you already have your own Julia kernel, pick it in Jupyter's *Kernel → Change kernel* menu.

### Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zhougroup-uic/FMAPIqJ/blob/main/examples/n200_test/n200_test.ipynb)
The notebook declares the `julia` kernel, which Colab starts itself; its first code cell clones this repository into `/content`.
Choose a GPU runtime (T4, 16 GB): the N=200 example fits; the N=800 production grid (984³) needs a larger GPU.
[`n200_test.cached.ipynb`](examples/n200_test/n200_test.cached.ipynb) is the same notebook with the outputs of a Colab T4 run stored (64 runs, about 10 min, check PASS), for reading without a GPU.

## Layout
```
bin/         command-line tools: fmapiq fmappre fmap fmapstat simulation_nvt simulation_nvt_gpu
             sim2iqpre sim2iq sim2iq_gpu sim2iq_gpu_batch
src/grid/    energy maps (cpu/, gpu/)        src/sim/     NVT Monte Carlo (cpu/, gpu/)
src/sim2iq/  scattering (cpu/, gpu/)         src/fmapiq/  pipeline driver
data/        rotation tables, mean volumes
examples/    N200 test and N800 production examples, input PQR
test/        regression test against the C reference (CPU runtests.jl, GPU test_sim2iq_gpu.sh)
tools/       check_result.jl, traj2pdb.py (configuration → PDB, needs numpy)
reference/   original C/CUDA code (fmapb2, maeppi, sim2iq, sim2iq_cuda) and python wrappers
docs/        notes on the pipeline options
```

## Notes
- `TERN_BYTES` (environment, default 1 in the pipeline) selects the cell format when the grid is made; the simulation
  tools read it from the `fmap.ve.sp2p` header, so nothing has to be set for them.
- `FMAP_SPT` (environment, default 2) selects the table layout of the simulation engines, 2 = `fmap.ve.sp2p` read in place, 1 = block-sparse
  lookup table; `FMAP_FLOAT=1` does the GPU lookup arithmetic in float. All settings give the same energies.
- GPU results are reproducible to numerical noise (atomic adds in the grid mapping), not bit-for-bit.
- Orientation lookup: the beta bin index is `floor(b2)` clamped to `nbin-1` (the `angijk` tables use half-bin beta
  centres); earlier versions used `round(b2)` with wrap, which sent beta near pi to beta near 0. The result
  shifts I(q) by at most ~2% (within the run-to-run error); the `expected/` files were regenerated with this
  fix (GPU, 64 runs) and differ from the old ones by that amount. Same change in `reference/maeppi/fmap_energy.c`.
- The 10-point second-order charge spreading (Qin & Zhou, JCTC 2014, 10, 2824) is computed from its closed form, per atom
  (`map_L2nd!` in `src/grid/cpu/mapping.jl`, the CUDA kernel in `native/fmapgrid.cu`, `Map2GrdL2nd` in `reference/fmapb2/map.c`); the
  table `data/lattice.txt` (1000 sub-grid positions, weights from a least-squares solve) is removed. The formulas are the same ones the
  table sampled, so the weights now use the exact position of the atom inside its cell instead of its 0.1-cell, and they satisfy
  the moment conditions exactly (the table of earlier versions also missed the y*z condition: error up to 0.34). Energy maps
  therefore change slightly: cenA B2 (0.15 M, 25 °C, 72 orientations, CPU and GPU) -1.560 -> -1.558 (error bar 0.08); the `expected/` I(q) files
  were regenerated (mean/P0 differs from the previous ones by 0.1% on average; the largest single-point differences, 6-7% at q < 0.025, are
  within the statistical noise of the runs: rms difference / combined SEM about 1.1-1.3 for 64 runs). Grid generation time is unchanged
  (the spreading itself is 3x faster on the CPU and 30% on the GPU, but it is under 0.1% of one orientation). The generator and the closed form
  are in the FMAPlattice repository.
- Rotation system and the pair table `angle_mapping` (`angij2k`, nrot^2 16-bit entries: 42.5 MB at 4608 orientations, read at every pair
  energy). The simulations (CPU and GPU) take it in one of three forms, `AMAP_FOLD` (environment): `0` plain table (default), `1` the
  symmetry-folded, bit-packed table read in place (5.9 MB at 4608, same entries as `0` built from the Euler angles, so identical results),
  `2` no table: the index is looked up at every pair from `R_i' R_j` and the `angijk` bin table (approximate: same sample as the exact table
  for 89% of the pairs, mean excess angle 0.09 deg, worst 8 deg; for sample lists whose n^2 table does not fit). The tables are built by the
  Rotij package (`src/sim/rotij`, GPL-2.0-or-later; the same code as `rotij/julia`, with the folded tables and the lookup), see
  `src/sim/rotsys.jl`. There is no `fmap.rot.zlib` any more: the rotation system (bin table, rotation matrices, pair table) is built when
  a simulation starts from the `rot.eul` of the sample set (`data/sspNF<nrot>/rot.eul` is the only rotation file; the pipeline copies it to the
  work directory and `fmap.paths` lists `rot.eul` and `fmap.ve.sp2p`). The build takes a few seconds (the folded search, about 6 s at 4608
  orientations with 8 threads). 
  Tables built from the Euler angles (modes 0/1) differ from the exhaustive-search table only at near-ties
  (72 samples: 7% of the entries, the returned sample at most 1.2e-4 deg farther than the nearest one; 4608: 0.6%). n200_test (GPU, 64 runs) passes
  with every mode on the GPU and the CPU (a plain table read from an old `rot.zlib` reproduces C05 exactly; lookup fails one point of the 3-run CPU example, which has large noise).
- License: GPL-3.0 (see `LICENSE`).
