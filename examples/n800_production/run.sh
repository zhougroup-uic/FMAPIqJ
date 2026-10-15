#!/bin/bash
# Production example: N=800 proteins, L=984.15 A (N800 L24.3), 64 independent runs, 20000 cycles each
# (about 8 min on one NVIDIA L40 with 46 GB; the 984^3 I(q) grid needs a large GPU).
#   examples/n800_production/run.sh [workdir]     NREPS=64 NCYCLES=20000 THREADS=8 override the defaults.
# Expected: P(0) = 2.185e6; Iqs.stat.txt agrees with expected/Iqs.stat.n64.txt (and the published curve) within ~2%.
set -e
source "$(dirname "$0")/../common.sh"
W=${1:-$PWD/work_n800}
rm -rf "$W"; mkdir -p "$W/cache"; cd "$W"
export FMAPIQ_CACHE=$W/cache
cp "$ROOT/examples/cenA.pqr" subA.pqr
time julia --project="$ROOT" "$ROOT/bin/fmapiq.jl" subA.pqr --work-dir run $GPUFLAG --threads $THREADS --sim-threads 2 \
     --n-reps ${NREPS:-64} --n-cycles ${NCYCLES:-20000} --ion 0.016 --temp 25 --vscl 0.16 \
     --n-proteins 800 --box-size 984.15
julia --project="$ROOT" "$ROOT/tools/check_result.jl" run/Iqs.stat.txt "$ROOT/examples/n800_production/expected/Iqs.stat.n64.txt" 0.03
