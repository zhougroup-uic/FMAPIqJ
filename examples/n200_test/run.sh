#!/bin/bash
# Test example: N=200 proteins, L=619.65 A, 20000 cycles per run.
#   GPU (default if nvidia-smi works): 64 independent runs, about 2-3 min on one GPU.
#   CPU (MODE=cpu): 3 independent runs (much slower: the energy maps alone take about 20 min).
#   examples/n200_test/run.sh [workdir]     NREPS=.. NCYCLES=.. THREADS=.. MODE=cpu|gpu override the defaults.
# Expected: P(0) = 2.186e6 and Iqs.stat.txt within tolerance of expected/Iqs.stat.n<NREPS>.txt.
set -e
source "$(dirname "$0")/../common.sh"
if [ "$MODE" = gpu ]; then NREPS=${NREPS:-64}; else NREPS=${NREPS:-3}; fi
W=${1:-$PWD/work_n200}
rm -rf "$W"; mkdir -p "$W/cache"; cd "$W"
export FMAPIQ_CACHE=$W/cache
cp "$ROOT/examples/cenA.pqr" subA.pqr
time julia --project="$ROOT" "$ROOT/bin/fmapiq.jl" subA.pqr --work-dir run $GPUFLAG --threads $THREADS --sim-threads 2 \
     --n-reps $NREPS --n-cycles ${NCYCLES:-20000} --ion 0.016 --temp 25 --vscl 0.16 \
     --n-proteins 200 --box-size 619.65
EXP="$ROOT/examples/n200_test/expected/Iqs.stat.n$NREPS.txt"
if [ -f "$EXP" ] && [ "${NCYCLES:-20000}" = 20000 ]; then
  julia --project="$ROOT" "$ROOT/tools/check_result.jl" run/Iqs.stat.txt "$EXP" 0.05
else
  echo "no expected file for NREPS=$NREPS / NCYCLES=${NCYCLES:-20000}: skipping the check"
fi
