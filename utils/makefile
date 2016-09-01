all: loguai_compress opengm2uai opengm2loguai multilabel2uai

loguai_compress:
	g++-4.8 -o loguai_compress loguai_compress.cpp -std=c++11 -O3

opengm2uai:
	#on server:
	g++ opengm2uai.cxx -o opengm2uai -I /scratch/cluster/xrhuang/opengm/include/ -I /scratch/cluster/xrhuang/hdf5/include/ -L/scratch/cluster/xrhuang/hdf5/lib/ /scratch/cluster/xrhuang/hdf5/lib/libhdf5.a /scratch/cluster/xrhuang/hdf5/lib/libhdf5_hl.a -ldl -lm -lz
	#on MAC:
	#g++-4.8 opengm2uai.cxx -o opengm2uai -lhdf5 -ldl -lm -lz

opengm2loguai:
	#on server:
	g++ opengm2loguai.cxx -o opengm2loguai -I /scratch/cluster/xrhuang/opengm/include/ -I /scratch/cluster/xrhuang/hdf5/include/ -L/scratch/cluster/xrhuang/hdf5/lib/ /scratch/cluster/xrhuang/hdf5/lib/libhdf5.a /scratch/cluster/xrhuang/hdf5/lib/libhdf5_hl.a -ldl -lm -lz
	#on MAC: 
	#g++-4.8 opengm2loguai.cxx -o opengm2loguai -lhdf5 -ldl -lm -lz

#./multilabel2uai 
multilabel2uai:
	g++-4.8 -O3 -std=c++11 multilabel2uai.cpp -o multilabel2uai 

uai_test_file=EurLex.subtest.uai		
AD3.log: ad3_multi $(uai_test_file)
	./ad3_multi --format=uai --file_graphs=$(uai_test_file) --file_posteriors=AD3.out --algorithm=ad3 --max_iterations=100000 > $@
