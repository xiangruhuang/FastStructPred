FLAG= -fopenmp -std=c++11 -O3 -w
CC=g++-4.8

all:
	$(CC) $(FLAG) -o predict predict.cpp

data_dir=/home/xiangru/Projects/MAP-sublinear/data/
model_dir=/home/xiangru/Projects/MAP-sublinear/model/
s=2
m=100000
q=1
e=3
a=0.1
opt=
c=1
t=0.1
scale=test
rho=0.01
eta=1
infea_tol=1e-3
testno=1

.PHONY:pos 107network

penguin-gm:
	$(eval test_file := ../penguin-gm.h5.loguai2)
	./predict -p loguai -s $(s) -o $(rho) -e $(eta) -m $(m) $(test_file)

tsu-gm:
	$(eval test_file := ../tsu-gm.h5.uai)
	./predict -p uai -s $(s) -o $(rho) -e $(eta) -m $(m) $(test_file)

clownfish:	
	$(eval test_file := ../clownfish-small.h5.uai)
	./predict -p uai -s $(s) -o $(rho) -e $(eta) -m $(m) $(test_file)

107network:
	$(eval test_file := $(data_dir)/107network)
	./predict -p network -s $(s) -o $(rho) -e $(eta) -m $(m) $(test_file)
	#./predict -p network -s 2 $(test_file)

107network.loguai2:
	$(eval test_file := $(data_dir)/$@)
	./predict -p loguai -s $(s) -o $(rho) -e $(eta) -m $(m) $(test_file)

pos:
	$(eval test_file := $(data_dir)/POS/wsj.pos.crf.$(scale))
	$(eval model := model)
	./predict -p chain -s $(s) -o $(rho) -e $(eta) -m $(m) $(test_file) $(model)

tmp:
	$(eval test_file := hehe)
	$(eval model := $(model_dir)/POS/pos.model)
	./predict -p chain -s $(s) -o $(rho) -e $(eta) -m $(m) $(test_file) $(model) 

OCR.fea:
	$(eval test_file := $(data_dir)/ChineseOCR/data.fea.$(scale))
	$(eval model := $(@).model)
	./predict -p chain -s $(s) -o $(rho) -e $(eta) -m $(m) $(test_file) $(model) 

yeast:
	$(eval test_file := $(data_dir)/yeast_test.svm)
	$(eval model := yeast.model)
	./predict $(opt) -p multilabel -s $(s) -o $(rho) -e $(eta) -m $(m)  $(test_file) $(model)

rcv1:
	$(eval test_file := $(data_dir)/multilabel/rcv1_regions.test)
	$(eval model := $(model_dir)/multilabel/rcv1.model)
	./predict $(opt) -p multilabel -s $(s) -o $(rho) -e $(eta) -m $(m)  $(test_file) $(model)

EurLex: 
	$(eval test_file := $(data_dir)/EurLex.subtest.1)
	$(eval model := $(model_dir)/EurLex.model)
	./predict $(opt) -p multilabel -s $(s) -o $(rho) -e $(eta) -m $(m) $(test_file) $(model)

protein:
	$(eval test_file := $(data_dir)/2BBN.uai)
	./predict $(opt) -p uai -s $(s) -o $(rho) -e $(eta) -m $(m)  $(test_file)

rcv1.uai:
	$(eval test_file := $(data_dir)/multilabel/rcv1_regions.uai)
	./predict $(opt) -p uai -s $(s) -o $(rho) -e $(eta) -m $(m)  $(test_file) $(model)

test.loguai2:
	$(eval test_file := $(data_dir)/$@)
	./predict $(opt) -p loguai -s $(s) -o $(rho) -e $(eta) -m $(m)  $(test_file)

clownfish-small.h5.loguai2:
	$(eval test_file := $(data_dir)/$@)
	./predict $(opt) -p loguai -s $(s) -o $(rho) -e $(eta) -m $(m)  $(test_file)

acro.uai:
	$(eval test_file := $(data_dir)/acro.uai)
	./predict $(opt) -p uai -s $(s) -o $(rho) -e $(eta) -m $(m)  $(test_file) $(model)

EurLex.uai:
	$(eval test_file := $(data_dir)/multilabel/EurLex.uai)
	./predict $(opt) -p uai -s $(s) -o $(rho) -e $(eta) -m $(m)  $(test_file) $(model)

family-gm.h5.loguai2:
	./predict -p loguai -s 2 -o $(rho) -e $(eta) -m $(m) $(data_dir)/photo-montage/$@

test.uai:
	$(eval uai_file := $(data_dir)/test.loguai2.h5.uai)
	./predict -p loguai -s $(s) -o $(rho) -e $(eta) -m $(m) $(uai_file)	

	
#
#speech:
#	$(eval train_file := /scratch/cluster/ianyen/data/speech/speech.$(scale))
#	$(eval heldout_file := /scratch/cluster/ianyen/data/speech/speech.tmp)
#	$(eval model := model)
#	./train -s $(s) -t $(t) -c $(c) -q $(q) -a $(a) -m $(m) $(opt) -e $(e) -h $(heldout_file) $(train_file) $(model) 

