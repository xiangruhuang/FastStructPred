all:
	g++ -O3 -std=c++11 convert2uai.cpp -o convert2uai 

data_dir=/scratch/cluster/xrhuang/data
model_dir=/scratch/cluster/xrhuang/model
uai_test_file=EurLex.subtest.uai
test_file=$(data_dir)/multilabel/Eur-Lex-Small.subtest.1

Solvers=AD3 MPLP GDMM #PSDD LPsparse Soft-BCFW Soft-BCFW-accurate

.PRECIOUS: %.log

predict:
	cp ../../../sparseLP/predict ./

mplp:
	cp ../../../ComparedSolvers/mplp ./

ad3_multi:
	cp ../../../ComparedSolvers/ad3_multi ./

AD3.log: ad3_multi $(uai_test_file)
	./ad3_multi --format=uai --file_graphs=EurLex.subtest.uai --file_posteriors=AD3.out --algorithm=ad3 --max_iterations=100000 > $@

PSDD.log: ad3_multi $(uai_test_file) should_not_run_again
	./ad3_multi --format=uai --file_graphs=$(uai_test_file) --file_posteriors=PSDD.out --algorithm=psdd --max_iterations=1000000 > $@

MPLP.log: mplp $(uai_test_file) 
	./mplp $(uai_test_file)  > $@

LPsparse.log: predict $(uai_test_file)
	((./predict -p uai -o 0.01 -e 1 -m 100000 -s 1 $(uai_test_file)) 2>&1) > $@

GDMM.log: predict $(test_file)
	((./predict -p multilabel -o 0.01 -e 1 -m 100000 -s 2 $(test_file) $(model_dir)/multilabel/EurLex.model) 2>&1) > $@

Soft-BCFW.log: predict $(uai_test_file)
	((./predict -p uai -o 1 -e 0 -m 100000 -s 3 $(uai_test_file)) 2>&1) > $@

Soft-BCFW-accurate.log: predict $(uai_test_file)
	((./predict -p uai -o 10 -e 0 -m 100000 -s 3 $(uai_test_file)) 2>&1) > $@

.PHONY: time_obj

time_obj:
	mkdir -p time_obj

obj:
	for i in $(Solvers); do \
		make time_obj/$$i ; \
	done

time_obj/PSDD: time_obj 
	echo "Time Objective" > $@
	cat PSDD.log | grep 'Primal rel obj' | sed 's/.*Primal rel obj = //' | sed 's/ sec.//' | sed 's/Primal residual.* //' | awk '{print $$2 " " $$1}' >> $@

time_obj/MPLP: time_obj
	echo "Time Objective" > $@
	cat MPLP.log | grep 'Decoded=' | sed 's/.*Decoded=//' | sed 's/ .*Elapsed Time=/ /' | awk '{print $$2 " " $$1}' >> $@

time_obj/GDMM: time_obj
	echo "Time Objective" > $@
	cat GDMM.log | grep 'best_decoded' | sed 's/.* best_decoded=//' | sed 's/,.* overall time=/ /' | awk '{print $$2 " " $$1}' >> $@

time_obj/AD3: time_obj
	echo "Time Objective" > $@
	cat AD3.log | grep 'Best primal obj' | sed 's/.*Best primal obj = //' | sed 's/ sec.//' | sed 's/[\t| ].*[\t| ]/ /' | awk '{print $$2 " " $$1}' >> $@

time_obj/LPsparse: time_obj 
	echo "Time Objective" > $@
	cat LPsparse.log | grep 'obj=' | sed 's/.*obj=-//' | sed 's/,.*time=/ /' | sed 's/,.*$$//' | awk ' { print $$2 " " $$1 ; } ' >> $@

time_obj/Soft-BCFW: time_obj
	echo "Time Objective" > $@
	cat Soft-BCFW.log | grep 'score_t' | sed 's/.*score_t=-//' | sed 's/,.*overall time=/ /' | awk '{ print $$2 " " $$1 ; }' >> $@

time_obj/Soft-BCFW-accurate: time_obj
	echo "Time Objective" > $@
	cat Soft-BCFW-accurate.log | grep 'score_t' | sed 's/.*score_t=-//' | sed 's/,.*overall time=/ /' | awk '{ print $$2 " " $$1 ; }' >> $@

