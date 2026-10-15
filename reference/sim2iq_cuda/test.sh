#!/bin/bash
# Build the original CUDA-C sim2iq and compare with the C reference output (max relative difference).
# Requirements: nvcc (CUDA toolkit), make, gcc, an NVIDIA GPU.  Run inside this directory.
set -e
cd "$(dirname "$0")"
D=../../test/data
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
make clean && make ARCH=$ARCH
N=620; L=619.65
SP=$(awk -v l=$L -v n=$N 'BEGIN{printf "%.15g", l/n}')
time ./sim2iq_cuda $D/input.pqr $N $SP out.dat -steps 4 -traj $D/snap.txt -txt
awk 'function a(x){return x<0?-x:x}
 NR==FNR{for(i=1;i<=NF;i++)r[FNR,i]=$i;next}
 {for(i=1;i<=NF;i++){u=$i;v=r[FNR,i];m=a(u)>a(v)?a(u):a(v);if(m>0){d=a(u-v)/m;if(d>mx)mx=d}}}
 END{print "lines",FNR,"max rel diff",mx}' $D/sim2iq.dat out.dat
