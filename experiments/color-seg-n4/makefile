data_dir=../../../data/color-seg-n4/
h5_file=$(data_dir)/palm-small.h5
uai_file=$(h5_file).uai
loguai_file=$(h5_file).loguai2
logparser_dir=../../utils/
players=TRWS AD3 MPLP PSDD GDMM Soft-BCFW LPsparse

$(uai_file):
	cd $(data_dir) && $(MAKE)

$(h5_file):
	cd $(data_dir) && $(MAKE)

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

GDMM.log: predict $(loguai_file)
	((./predict -p loguai -o $(rho) -e $(eta) -m 100000 -s 2 $(loguai_file)) 2>&1) > $@

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
