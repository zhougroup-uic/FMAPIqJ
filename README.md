# FMAPIqJ — FMAPIq in Julia (CPU + GPU)

Small-angle X-ray scattering I(q) of concentrated protein solutions from FMAP energy maps,
Monte Carlo (NVT) sampling and a Debye-type scattering calculation. Everything is Julia
(CUDA.jl for the GPU paths); the original C/CUDA sources are kept in `reference/` for comparison.

Pipeline (one command, `bin/fmapiq.jl`):

1. **energy maps** `fmappre` → `fmap` → `fmapstat`: protein–crowder energy grids for 72 orientations,
   written directly as `fmap.ve.zlib` / `fmap.rot.zlib` (no intermediate files).
2. **simulation** `simulation_nvt` / `simulation_nvt_gpu`: independent NVT Monte Carlo runs (box with
   N proteins), all replicas at once on the GPU.
3. **scattering** `sim2iq`: P(q) of one protein and I(q) of every configuration (FFT grid), then
   `Iqs.txt` (per-protein I(q)), `Iqs.stat.txt` (mean ± SEM over runs, normalised to P(0)) and `iq_plot.svg`.

## Requirements
- Julia ≥ 1.9 (developed and tested with 1.12).
- GPU mode: an NVIDIA GPU with driver (CUDA.jl downloads its own toolkit). Memory: about 6 GB for the
  energy maps and the N=200 example; the N=800 example was run on a 46 GB L40 (984³ FFT grid; smaller cards untested).
- CPU mode works without a GPU but is much slower. Measured on 8 cores (N=200): energy maps about 19 min,
  then about 40 s per I(q); the Monte Carlo run time of 20000 cycles per replica on the CPU was not benchmarked.
- Disk: about 0.3 GB per work directory (`fmap.ve.zlib`).
- Python (numpy) is only needed for the optional `tools/traj2pdb.py`; no C extension anywhere.

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
bin/         command-line tools: fmapiq fmappre fmap fmapstat fmap_pack_zlib simulation_nvt simulation_nvt_gpu
             sim2iqpre sim2iq sim2iq_gpu sim2iq_gpu_batch
src/grid/    energy maps (cpu/, gpu/)        src/sim/     NVT Monte Carlo (cpu/, gpu/)
src/sim2iq/  scattering (cpu/, gpu/)         src/fmapiq/  pipeline driver
data/        rotation tables, lattice, mean volumes
examples/    N200 test and N800 production examples, input PQR
test/        regression test against the C reference (CPU runtests.jl, GPU test_sim2iq_gpu.sh)
tools/       check_result.jl, traj2pdb.py (configuration → PDB, needs numpy)
reference/   original C/CUDA code (fmapb2, maeppi, sim2iq, sim2iq_cuda) and python wrappers
docs/        notes on the pipeline options
```

## Notes
- `TERN_BYTES` (environment, default 1 in the pipeline) selects the grid cell format; stand-alone tools
  default to 2, so export `TERN_BYTES=1` when running `simulation_nvt*` on a pipeline grid.
- GPU results are reproducible to numerical noise (atomic adds in the grid mapping), not bit-for-bit.
- License: GPL-3.0 (see `LICENSE`).
