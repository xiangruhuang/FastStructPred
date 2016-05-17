FLAG= -fopenmp -std=c++11 -O3 -w
CC=g++-4.8

all:
	$(CC) $(FLAG) -o predict predict.cpp

data_dir=.#/scratch/cluster/xrhuang/data/
s=2
m=10000
q=1
e=3
a=0.1
opt=
c=1
t=0.1
scale=test
rho=1
eta=1
infea_tol=1e-3

.PHONY:pos 107network

107network:
	$(eval test_file := $(data_dir)/107network/107network)
	./predict -p network -s $(s) -o $(rho) -e $(eta) -m $(m) $(test_file)

pos:
	$(eval test_file := $(data_dir)/POS/wsj.pos.crf.$(scale))
	$(eval model := model)
	./predict -p chain -s $(s) -o $(rho) -e $(eta) -m $(m) $(test_file) $(model)

tmp:
	$(eval test_file := $(data_dir)/POS/pos.tmp)
	$(eval model := model)
	./predict -p chain -s $(s) -o $(rho) -e $(eta) -m $(m) $(test_file) $(model) 

OCR.fea:
	$(eval test_file := $(data_dir)/ChineseOCR/data.fea.$(scale))
	$(eval model := $(@).model)
	./predict -p chain -s $(s) -o $(rho) -e $(eta) -m $(m) $(test_file) $(model) 
	
OCR.fea.tmp:
	$(eval test_file := $(data_dir)/ChineseOCR/OCR.fea.test)
	$(eval model := OCR.fea.model)
	./predict -p chain -s $(s) -o $(rho) -e $(eta) -m $(m) $(test_file) $(model) 

yeast:
	$(eval test_file := $(data_dir)/yeast_test.svm)
	$(eval model := yeast.model)
	./predict $(opt) -p multilabel -s $(s) -o $(rho) -e $(eta) -m $(m)  $(test_file) $(model)

rcv1:
	$(eval test_file := $(data_dir)/rcv1_regions_test.03k)
	$(eval model := rcv1.model)
	./predict $(opt) -p multilabel -s $(s) -o $(rho) -e $(eta) -m $(m)  $(test_file) $(model)


#
#speech:
#	$(eval train_file := /scratch/cluster/ianyen/data/speech/speech.$(scale))
#	$(eval heldout_file := /scratch/cluster/ianyen/data/speech/speech.tmp)
#	$(eval model := model)
#	./train -s $(s) -t $(t) -c $(c) -q $(q) -a $(a) -m $(m) $(opt) -e $(e) -h $(heldout_file) $(train_file) $(model) 

