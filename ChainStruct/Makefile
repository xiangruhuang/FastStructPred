FLAG= -fopenmp -std=c++11 -O3 -w

all:
	g++ $(FLAG) -o train train.cpp

s=1
m=10
q=1
e=3
a=0.1
opt=
c=1
t=0.1
scale=test

pos:
	$(eval train_file := /scratch/cluster/ianyen/data/POS/wsj.pos.crf.$(scale))
	$(eval heldout_file := /scratch/cluster/ianyen/data/POS/wsj.pos.crf.test)
	$(eval model := model)
	./train -s $(s) -t $(t) -c $(c) -q $(q) -a $(a) -m $(m) $(opt) -e $(e) -h $(heldout_file) $(train_file) $(model) 

speech:
	$(eval train_file := /scratch/cluster/ianyen/data/speech/speech.$(scale))
	$(eval heldout_file := /scratch/cluster/ianyen/data/speech/speech.tmp)
	$(eval model := model)
	./train -s $(s) -t $(t) -c $(c) -q $(q) -a $(a) -m $(m) $(opt) -e $(e) -h $(heldout_file) $(train_file) $(model) 

OCR:
	$(eval train_file := /scratch/cluster/ianyen/data/ChineseOCR/data.bin20.$(scale))
	$(eval heldout_file := /scratch/cluster/ianyen/data/ChineseOCR/data.bin20.test)
	$(eval model := model)
	./train -s $(s) -t $(t) -c $(c) -q $(q) -a $(a) -m $(m) $(opt) -e $(e) -h $(heldout_file) $(train_file) $(model) 

OCR.fea:
	$(eval train_file := /scratch/cluster/ianyen/data/ChineseOCR/data.fea.$(scale))
	$(eval heldout_file := /scratch/cluster/ianyen/data/ChineseOCR/data.fea.test)
	$(eval model := model)
	./train -s $(s) -t $(t) -c $(c) -q $(q) -a $(a) -m $(m) $(opt) -e $(e) -h $(heldout_file) $(train_file) $(model) 
