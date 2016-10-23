data_dir=../../../data/
uai_file=$(data_dir)/test.loguai2.h5.uai
h5_file=$(data_dir)/test.loguai2.h5
loguai2_file=$(data_dir)/test.loguai2
logparser_dir=../../utils/
players=TRWS AD3 MPLP PSDD GDMM Soft-BCFW LPsparse smoothMSD

.PHONY: $(addsuffix .log, $(players))

test:
	$(MAKE) -C ../../../data/ test
	#$(MAKE) TRWS.log
	#$(MAKE) GDMM.log 
	#tail -10 GDMM.log
	#tail -3 TRWS.log

$(uai_file):
	cd $(data_dir) && $(MAKE) $(notdir $@)

$(h5_file):
	cd $(data_dir) && $(MAKE) $(notdir $@)

rho=0.01
eta=1

smoothMSD:
	cp ../../ComparedSolvers/$@ $@	

ad3:
	cp ../../ComparedSolvers/$@ $@

mplp:
	cp ../../ComparedSolvers/$@ ../../../mplp_ver2/no.evid ./

trws:
	cp ../../ComparedSolvers/$@ ./

predict:
	cp ../../sparseLP/$@ ./

TRWS.log: trws $(h5_file)
	((./trws $(h5_file)) 2>& 1) > $@

AD3.log: ad3 $(uai_file)
	./ad3 --format=uai --file_graphs=$(uai_file) --file_posteriors=AD3.out --algorithm=ad3 --max_iterations=100000 > $@

PSDD.log: ad3 $(uai_file)
	./ad3 --format=uai --file_graphs=$(uai_file) --file_posteriors=PSDD.out --algorithm=psdd --max_iterations=100000 > $@
	
MPLP.log: mplp $(uai_file)
	./mplp $(uai_file) > $@	

GDMM.log: $(loguai2_file) predict
	((./predict -p loguai -o $(rho) -e $(eta) -m 100000 -s 2 $<) 2>&1) > $@

gamma=1e-5
smoothMSD.log: $(loguai2_file) smoothMSD
	((./smoothMSD -p loguai -g $(gamma) -m 100000 $<) 2>&1) > $@

softrho=1

Soft-BCFW.log: $(uai_file) predict
	((./predict -p uai -o $(softrho) -e 0 -m 100000 -s 3 $<) 2>&1) > $@

Soft-BCFW-acc.log: predict $(uai_file)
	((./predict -p uai -o $(softrho) -e 0 -m 100000 -s 3 $(uai_file)) 2>&1) > $@

Soft-BCFW-temp.log: predict $(uai_file)
	((./predict -p uai -o $(softrho) -e 0 -m 100000 -s 3 $(uai_file)) 2>&1) > $@

LP_eta=1
LPsparse.log: predict $(uai_file)
	((./predict -p uai -o $(LP_eta) -e 0 -m 100000 -s 1 $(uai_file)) 2>&1) > $@

.PHONY: time_obj
time_obj:
	echo .$@
	$(MAKE) -C ./$@ $(addsuffix .$@, $(players))
