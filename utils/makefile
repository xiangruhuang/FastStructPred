all:
	g++-4.8 -o loguai_compress loguai_compress.cpp -std=c++11 -O3
	g++-4.8 opengm2loguai.cxx -o opengm2loguai -lhdf5 -ldl -lm -lz
	g++-4.8 opengm2uai.cxx -o opengm2uai -lhdf5 -ldl -lm -lz
	g++-4.8 -O3 -std=c++11 multilabel2uai.cpp -o multilabel2uai 
	
