# libfmapgrid.so: GPU energy-map generation, libfmapsim2iq.so: GPU I(q) (both CUDA runtime + cuFFT, plain C interface, loaded by Julia with ccall).
# Build on a node with the CUDA toolkit:  module load CUDA && make -C native     (ARCHS="75" for one GPU, e.g. a T4; default: several)
NVCC  ?= nvcc
ARCHS ?= 70 75 80 86 89 90
GENCODE = $(foreach a,$(ARCHS),-gencode arch=compute_$(a),code=sm_$(a)) -gencode arch=compute_$(firstword $(ARCHS)),code=compute_$(firstword $(ARCHS))
CUDALIB := $(abspath $(dir $(shell which $(NVCC)))/../lib64)

all: libfmapgrid.so libfmapsim2iq.so libfmapsim.so

libfmapgrid.so: fmapgrid.cu
	$(NVCC) -O3 -shared -Xcompiler -fPIC $(GENCODE) -o $@ $< -lcufft -Xlinker -rpath,$(CUDALIB)

libfmapsim2iq.so: fmapsim2iq.cu fmapsim2iq.h
	$(NVCC) -O3 -shared -Xcompiler -fPIC $(GENCODE) -o $@ $< -lcufft -Xlinker -rpath,$(CUDALIB)

libfmapsim.so: fmapsim.cu fmapsim.h fmap_core.h
	$(NVCC) -O3 -shared -Xcompiler -fPIC $(GENCODE) -o $@ $< -lcudart -Xlinker -rpath,$(CUDALIB)

clean:
	rm -f libfmapgrid.so libfmapsim2iq.so libfmapsim.so
