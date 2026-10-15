#!/bin/bash
# GPU sim2iq (CUDA.jl) against the C reference output test/data/sim2iq.dat; expect max rel diff ~1e-7.
# Requirements: julia >= 1.9, NVIDIA GPU + driver (CUDA.jl fetches its own toolkit), ~4 CPU cores.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
D=$ROOT/test/data
OUT=${1:-$(mktemp -d)}
julia --project="$ROOT" -e 'using Pkg; Pkg.instantiate()'
SP=$(awk 'BEGIN{printf "%.15g", 619.65/620}')
time julia --project="$ROOT" -t 4 "$ROOT/bin/sim2iq_gpu.jl" $D/input.pqr 620 $SP $OUT/out_gpu.dat -steps 4 -traj $D/snap.txt -txt
awk 'function a(x){return x<0?-x:x}
 NR==FNR{for(i=1;i<=NF;i++)r[FNR,i]=$i;next}
 {for(i=1;i<=NF;i++){u=$i;v=r[FNR,i];m=a(u)>a(v)?a(u):a(v);if(m>0){d=a(u-v)/m;if(d>mx)mx=d}}}
 END{print "lines",FNR,"max rel diff",mx}' $D/sim2iq.dat $OUT/out_gpu.dat
