CC=g++
CFLAGS=-march=native -O2 -fopenmp -I. -DNOPRINT -Wall
#CFLAGS=-march=native -O2 -I. -DNOPRINT -Wall


all: val_test01_solved val_test02_solved MMult1 omp_solved2 \
			omp_solved3  omp_solved4  omp_solved5 omp_solved6 \
      jacobi2D-omp gs2D-omp

val_test01_solved: val_test01_solved.cpp
	$(CC) $(CFLAGS) -o val_test01_solved val_test01_solved.cpp

val_test02_solved: val_test02_solved.cpp
	$(CC) $(CFLAGS) -o val_test02_solved val_test02_solved.cpp

MMult1: MMult1.cpp
	$(CC) $(CFLAGS) -o MMult1 MMult1.cpp

omp_solved2: omp_solved2.c
	$(CC) $(CFLAGS) -fopenmp -o omp_solved2 omp_solved2.c	

omp_solved3: omp_solved3.c
	$(CC) $(CFLAGS) -fopenmp -o omp_solved3 omp_solved3.c
	
omp_solved4: omp_solved4.c
	$(CC) $(CFLAGS) -fopenmp -o omp_solved4 omp_solved4.c	

omp_solved5: omp_solved5.c
	$(CC) $(CFLAGS) -fopenmp -o omp_solved5 omp_solved5.c	

omp_solved6: omp_solved6.c
	$(CC) $(CFLAGS) -fopenmp -o omp_solved6 omp_solved6.c	

jacobi2D-omp: jacobi2D-omp.cpp common.o
	$(CC) $(CFLAGS) -o jacobi2D-omp jacobi2D-omp.cpp common.o

gs2D-omp: gs2D-omp.cpp common.o
	$(CC) $(CFLAGS) -o gs2D-omp gs2D-omp.cpp common.o

common.o: common.cpp
	$(CC) $(CFLAGS) -c common.cpp	

clean:
	rm -f val_test01_solved val_test02_solved MMult1 \
        omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 \
        jacobi2D-omp gs2D-omp *.o *.out
