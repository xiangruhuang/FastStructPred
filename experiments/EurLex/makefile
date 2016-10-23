data_dir=../../../data/
model_dir=../../../model/
uai_file=$(data_dir)/EurLex.uai
loguai_file=$(data_dir)/EurLex.loguai
h5_file=$(data_dir)/EurLex.loguai.h5
test_file=$(data_dir)/EurLex.subtest.1
model_file=$(model_dir)/EurLex.model
logparser_dir=../../utils/
EurLex_players=TRWS AD3 MPLP PSDD GDMM Soft-BCFW smoothMSD #Soft-BCFW-acc #LPsparse

$(uai_file):
	$(MAKE) -C $(data_dir) $(notdir $@)

$(loguai2_file):
	$(MAKE) -C $(data_dir) $(notdir $@)

$(h5_file):
	$(MAKE) -C $(data_dir) $(notdir $@)

rho=0.25
eta=0.1

ad3:
	cp ../../ComparedSolvers/$@ $@

mplp:
	cp ../../ComparedSolvers/$@ ../../../mplp_ver2/no.evid ./

trws:
	cp ../../ComparedSolvers/$@ ./

smoothMSD:
	cp ../../ComparedSolvers/$@ ./

predict:
	cp ../../sparseLP/$@ ./

TRWS.log: $(h5_file) trws
	((./trws $<) 2>&1) > $@

AD3.log: $(uai_file) ad3
	./ad3 --format=uai --file_graphs=$< --file_posteriors=AD3.out --algorithm=ad3 --max_iterations=100000 > $@

PSDD.log: $(uai_file) ad3
	./ad3 --format=uai --file_graphs=$< --file_posteriors=PSDD.out --algorithm=psdd --max_iterations=100000 > $@
	
MPLP.log: $(uai_file) mplp
	./mplp $< > $@ 

MPLP_with_tighten.log: $(uai_file) mplp_with_tighten
	./mplp_with_tighten $< > $@

GDMM.log: $(test_file) predict
	((./predict -p multilabel -o $(rho) -e $(eta) -m 100000 -s 2 $< $(model_file) ) 2>&1) > $@

gamma=1e-2
smoothMSD.log: $(loguai_file) smoothMSD
	((./smoothMSD -p loguai -g $(gamma) -m 100000 $<) 2>&1) > $@

softrho=1

Soft-BCFW.log: $(loguai_file) predict
	((./predict -p loguai -o $(softrho) -e 0 -m 100000 -s 3 $<) 2>&1) > $@

softrho_acc=10

Soft-BCFW-acc.log: $(loguai_file) predict
	((./predict -p loguai -o $(softrho_acc) -e 0 -m 100000 -s 3 $<) 2>&1) > $@


LPrho=1
LPsparse.log: $(loguai_file) predict 
	((./predict -p loguai -o $(LPrho) -e 1 -m 100000 -s 1 $<) 2>&1) > $@

.PHONY: time_obj
time_obj:
	echo .$@
	$(MAKE) -C ./$@ $(addsuffix .$@, $(EurLex_players))
