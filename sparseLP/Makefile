FLAG= -fopenmp -std=c++11 -O3 -w
CC=g++

all:
	$(CC) $(FLAG) -o predict predict.cpp

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

pos:
	$(eval test_file := /scratch/cluster/ianyen/data/POS/wsj.pos.crf.$(scale))
	$(eval model := model)
	./predict -s $(s) -o $(rho) -e $(eta) -m $(m) -t $(infea_tol) $(test_file) $(model)

OCR.fea:
	$(eval test_file := /scratch/cluster/ianyen/data/ChineseOCR/data.fea.$(scale))
	$(eval model := $(@).model)
	./predict -s $(s) -o $(rho) -e $(eta) -m $(m) -t $(infea_tol) $(test_file) $(model) 
	
OCR.fea.tmp:
	$(eval test_file := OCR.fea.test)
	$(eval model := OCR.fea.model)
	./predict -s $(s) -o $(rho) -e $(eta) -m $(m) -t $(infea_tol) $(test_file) $(model) 

#tmp:
#	$(eval test_file := pos.tmp)
#	$(eval model := model)
#	./predict -s $(s) -o $(rho) -e $(eta) -m $(m) $(test_file) $(model) 
#
#speech:
#	$(eval train_file := /scratch/cluster/ianyen/data/speech/speech.$(scale))
#	$(eval heldout_file := /scratch/cluster/ianyen/data/speech/speech.tmp)
#	$(eval model := model)
#	./train -s $(s) -t $(t) -c $(c) -q $(q) -a $(a) -m $(m) $(opt) -e $(e) -h $(heldout_file) $(train_file) $(model) 

