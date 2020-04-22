CC=g++
CCUDA=nvcc
CFLAGS=-march=native -O2 -fopenmp -I. -DNOPRINT -Wall
CUDAFLAGS=-arch=sm_60 -Xcompiler "-fopenmp" -L/usr/lib/x86_64-linux-gnu/ -lcudart -lcudadevrt
# NOTE: need to use sm_35 on cims machines 

all: vec_dot mat_vec jacobi2D 

vec_dot: vec_dot.cu vec_dot.o
	$(CCUDA) $(CUDAFLAGS) vec_dot.o -o vec_dot

vec_dot.o: vec_dot.cu
	$(CCUDA) $(CUDAFLAGS) -dc vec_dot.cu -o vec_dot.o

mat_vec: mat_vec.cu mat_vec.o
	$(CCUDA) $(CUDAFLAGS) mat_vec.o -o mat_vec

mat_vec.o: mat_vec.cu
	$(CCUDA) $(CUDAFLAGS) -dc mat_vec.cu -o mat_vec.o

jacobi2D: jacobi2D.cu jacobi2D.o
	$(CCUDA) $(CUDAFLAGS) jacobi2D.o -o jacobi2D

jacobi2D.o: jacobi2D.cu 
	$(CCUDA) $(CUDAFLAGS) -dc jacobi2D.cu -o jacobi2D.o

clean:
	rm -f vec_dot mat_vec jacobi2D *.o *.out
