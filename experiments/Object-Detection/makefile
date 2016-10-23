data_dir=../../../data/
uai_file=$(data_dir)/deer_rescaled_0034.K20.F100.model.uai
h5_file=$(uai_file).h5
loguai2_file=$(h5_file).loguai2
logparser_dir=../../utils/
Protein_players=TRWS AD3 MPLP PSDD GDMM Soft-BCFW LPsparse smoothMSD

$(uai_file): 
	$(MAKE) -C $(data_dir) $(notdir $@)

$(h5_file): $(uai_file)
	$(MAKE) -C $(data_dir) $(notdir $@)

$(loguai2_file): $(h5_file)
	$(MAKE) -C $(data_dir) $(notdir $@)


rho=1
eta=1

ad3:
	cp ../../ComparedSolvers/$@ $@

mplp:
	cp ../../ComparedSolvers/$@ ../../../mplp_ver2/no.evid ./

trws:
	cp ../../ComparedSolvers/$@ ./

predict:
	cp ../../sparseLP/$@ ./

smoothMSD:
	cp ../../ComparedSolvers/$@ ./

TRWS.log: trws $(h5_file)
	((./trws $(h5_file)) 2>&1) > $@

AD3.log: ad3 $(uai_file)
	./ad3 --format=uai --file_graphs=$(uai_file) --file_posteriors=AD3.out --algorithm=ad3 --max_iterations=1000000 > $@

PSDD.log: ad3 $(uai_file)
	./ad3 --format=uai --file_graphs=$(uai_file) --file_posteriors=PSDD.out --algorithm=psdd --max_iterations=1000000 > $@
	
MPLP.log: mplp $(uai_file)
	./mplp $(uai_file) > $@	

GDMM.log: $(loguai2_file) predict
	((./predict -p loguai -o $(rho) -e $(eta) -m 10000000 -s 2 $<) 2>&1) > $@

gamma=0.01
smoothMSD.log: $(uai_file) smoothMSD
	((./smoothMSD -p uai -g $(gamma) -m 100000 $<) 2>&1) > $@

softrho=1

Soft-BCFW.log: predict $(uai_file)
	((./predict -p uai -o $(softrho) -e 0 -m 100000 -s 3 $(uai_file)) 2>&1) > $@

softrho_acc=100

Soft-BCFW-acc.log: predict $(uai_file)
	((./predict -p uai -o $(softrho_acc) -e 0 -m 100000 -s 3 $(uai_file)) 2>&1) > $@

LPsparse.log: predict $(uai_file)
	((./predict -p uai -o 1 -e 1 -m 100000 -s 1 $(uai_file)) 2>&1) > $@

.PHONY: time_obj
time_obj:
	echo .$@
	$(MAKE) -C ./$@ $(addsuffix .$@, $(Protein_players))
