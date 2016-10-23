data_dir=../../../data/
uai_file=$(data_dir)/clownfish-small.h5.uai
h5_file=$(data_dir)/3096.bmp.h5
loguai2_file=$(data_dir)/3096.bmp.h5.loguai.loguai2
logparser_dir=../../utils/
players=TRWS AD3 MPLP PSDD GDMM Soft-BCFW LPsparse

$(uai_file):
	$(MAKE) -C $(data_dir) $(notdir $@)

$(loguai2_file):
	$(MAKE) -C $(data_dir) $(notdir $@)

$(h5_file):
	$(MAKE) -C $(data_dir) $(notdir $@)

rho=0.1
eta=1

ad3:
	cp ../../ComparedSolvers/$@ $@

mplp:
	cp ../../ComparedSolvers/$@ ../../../mplp_ver2/no.evid ./

trws:
	cp ../../ComparedSolvers/$@ ./

predict:
	cp ../../sparseLP/$@ ./

TRWS.log: trws $(h5_file)
	((./trws $(h5_file)) 2>&1) > $@

AD3.log: ad3 $(uai_file)
	./ad3 --format=uai --file_graphs=$(uai_file) --file_posteriors=AD3.out --algorithm=ad3 --max_iterations=100000 > $@

PSDD.log: ad3 $(uai_file)
	./ad3 --format=uai --file_graphs=$(uai_file) --file_posteriors=PSDD.out --algorithm=psdd --max_iterations=100000 > $@
	
MPLP.log: mplp $(uai_file)
	./mplp $(uai_file) > $@	

GDMM.log: $(loguai2_file) predict 
	((./predict -p loguai -o $(rho) -e $(eta) -m 100000 -s 2 $<) 2>&1) > $@

softrho=1

Soft-BCFW.log: predict $(uai_file)
	((./predict -p uai -o $(softrho) -e 0 -m 100000 -s 3 $(uai_file)) 2>&1) > $@

Soft-BCFW-acc.log: predict $(uai_file)
	((./predict -p uai -o $(softrho) -e 0 -m 100000 -s 3 $(uai_file)) 2>&1) > $@

Soft-BCFW-temp.log: predict $(uai_file)
	((./predict -p uai -o $(softrho) -e 0 -m 100000 -s 3 $(uai_file)) 2>&1) > $@

LPsparse.log: predict $(uai_file)
	((./predict -p uai -o 1 -e 1 -m 100000 -s 1 $(uai_file)) 2>&1) > $@

.PHONY: time_obj
time_obj:
	echo .$@
	$(MAKE) -C ./$@ $(addsuffix .$@, $(players))
