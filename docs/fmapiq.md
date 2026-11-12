# fmapiq (Julia pipeline)

Command line pipeline `bin/fmapiq.jl` (replaces the python `pipeline.py` in `reference/python`, no GUI). Three stages, each skipped when its output exists:

| stage | what | tools |
|---|---|---|
| 1 | energy maps: `fmappre`, `fmap` (CPU or `--gpu`), `fmapstat stats --ern` (the grid is written directly as `fmap.ve.sp2p`) | `src/grid` |
| 2 | `n_reps` NVT runs (displacement + same-box insertion) -> `config_NNN.dat`, log `sim_NNN.log` | `src/sim` (`bin/simulation_nvt.jl`) |
| 3 | `sim2iqpre` (`subA.pqr` -> `subA.vol`), P(q) -> `Pq.dat`, `iq_NNN.dat`, `Iqs.txt` | `src/sim2iq` |

then `Iqs.stat.txt` (q, P(q)/P(0), mean, SEM), `iq_plot.svg` and, with `--pack`, `<workdir>.tar.gz`.

    julia --project=. bin/fmapiq.jl subA.pqr --work-dir run1 --ion 0.15 --temp 25 --n-reps 3 [--gpu] [--pack]
    julia --project=. bin/fmapiq.jl --load run1 --pack --skip-energy-maps

Same options and defaults as `pipeline.py` (`--n-proteins 200 --box-size 619.65 --ion 0.016 --temp 25 --nrot 72
--escl 1 --vscl -1 --n-reps 3 --n-cycles 20000 --spacing 1 --threads --sim-threads 2`), plus `--gpu`: all stages on the GPU (CUDA C libraries in `native/` called through `ccall`: energy maps `libfmapgrid.so`, `src/sim/gpu/main_nvt_gpu.jl` with all replicas in one process on `libfmapsim.so`, `src/sim2iq/gpu` on `libfmapsim2iq.so` (all runs in one process); build with `make -C native`).
`TERN_BYTES` (default 1, pre-summed grid cells like the python pipeline),
`FMAPIQ_ROOT` (location of `src/ data/`).

Differences to `pipeline.py`: the simulation is the NVT engine of `src/sim` (no shared-memory server; every process builds its
block-sparse lookup table from `fmap.ve.sp2p` itself, no dense grid or cache); the plot is an SVG; a work directory is reused when
`parms.txt`, `fmap.ern` and `fmap.ve.sp2p` exist (older `ve/` or `.zlib` grids are not read any more). `--pack` without `--skip-*` packs
everything, as in the python CLI.
