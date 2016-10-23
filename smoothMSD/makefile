FLAG= -fopenmp -std=c++11 -O3 -w
CC=g++-4.8

all:
	$(CC) $(FLAG) -o smoothMSD smoothMSD.cpp

test:
	$(MAKE) -C ../../data test
	cp ../../data/test.loguai2 ./
	cp ../../data/test.loguai2.h5 ./

gamma=1e-3
m=100000

smooth:
	./smoothMSD -p loguai -g $(gamma) -m $(m) test.loguai2

rho=0.01
pred:
	./predict -p loguai -s 3 -o $(rho) -e 1 -m 100000 test.loguai2
