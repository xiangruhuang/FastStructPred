data_dir=../../../data/
uai_file=$(data_dir)/107network.loguai2.h5.uai
loguai2_file=$(data_dir)/107network.loguai2
h5_file=$(data_dir)/107network.loguai2.h5
logparser_dir=../../utils/
107network_players=TRWS AD3 MPLP PSDD GDMM Soft-BCFW smoothMSD

$(uai_file):
	$(MAKE) -C $(data_dir) $(notdir $@)

$(loguai2_file):
	$(MAKE) -C $(data_dir) $(notdir $@)

$(h5_file):
	$(MAKE) -C $(data_dir) $(notdir $@)

rho=0.045
eta=1

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

GDMM.log: $(loguai2_file) predict
	((./predict -p loguai -o $(rho) -e $(eta) -m 100000 -s 2 $< ) 2>&1) > $@

gamma=1e-2
smoothMSD.log: $(loguai2_file) smoothMSD
	((./smoothMSD -p loguai -g $(gamma) -m 100000 $<) 2>&1) > $@

softrho=1

Soft-BCFW.log: $(loguai2_file) predict
	((./predict -p loguai -o $(softrho) -e 0 -m 100000 -s 3 $<) 2>&1) > $@

softrho_acc=10

Soft-BCFW-acc.log: $(loguai2_file) predict
	((./predict -p loguai -o $(softrho_acc) -e 0 -m 100000 -s 3 $<) 2>&1) > $@

LPsparse.log: $(loguai2_file) predict MLE 
	((./predict -p loguai -o 1 -e 1 -m 100000 -s 1 $<) 2>&1) > $@

.PHONY: time_obj
time_obj:
	echo .$@
	$(MAKE) -C ./$@ $(addsuffix .$@, $(107network_players))
