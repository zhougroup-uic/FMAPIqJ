# usage: make ARCH=80   (compute capability; nvidia-smi --query-gpu=compute_cap)
ARCH ?= 80
CC = gcc
NVCC = nvcc
CFLAGS = -O3 -fPIC -I.
all: sim2iq_cuda
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
sim2iq_cuda.o: sim2iq_cuda.cu
	$(NVCC) -O3 -arch=sm_$(ARCH) -I. -c $< -o $@
sim2iq_cuda: sim2iq_cuda.o pqr_reader.o pd6d.o args.o utils.o
	$(NVCC) -arch=sm_$(ARCH) $^ -lcufft -lm -o $@
clean:
	rm -f *.o sim2iq_cuda
.PHONY: all clean
