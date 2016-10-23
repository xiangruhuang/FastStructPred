opengm2uai:
	cd ../utils/ && $(MAKE) $@
	cp ../utils/$@ ./

opengm2loguai:
	cd ../utils/ && $(MAKE) $@
	cp ../utils/$@ ./

loguai_compress:
	cd ../utils/ && $(MAKE) $@
	cp ../utils/$@ ./

multilabel2uai:
	cd ../utils/ && $(MAKE) $@
	cp ../utils/$@ ./

%.uai: % opengm2uai
	./opengm2uai $< $@
	
%.loguai: % opengm2loguai
	./opengm2loguai $< $@

%.loguai2: %.loguai loguai_compress
	./loguai_compress $< $@

h5_files: penguin-gm.h5 house-gm.h5 tsu-gm.h5 clownfish-small.h5

penguin-gm.h5:
	wget http://hciweb2.iwr.uni-heidelberg.de/opengm/benchmarks/models/zip/mrf-inpainting.zip
	unzip mrf-inpainting.zip
	mv mrf-inpainting/penguin-gm.h5 ./
	rm -rf mrf-inpainting/
	rm -f mrf-inpainting.zip

house-gm.h5:
	wget http://hciweb2.iwr.uni-heidelberg.de/opengm/benchmarks/models/zip/mrf-inpainting.zip
	unzip mrf-inpainting.zip
	mv mrf-inpainting/house-gm.h5 ./
	rm -rf mrf-inpainting/
	rm -f mrf-inpainting.zip

tsu-gm.h5:
	wget http://hciweb2.iwr.uni-heidelberg.de/opengm/benchmarks/models/zip/mrf-stereo.zip
	unzip mrf-stereo.zip
	mv mrf-stereo/tsu-gm.h5 ./
	rm -rf mrf-stereo/
	rm -f mrf-stereo.zip

clownfish-small.h5:
	wget http://hciweb2.iwr.uni-heidelberg.de/opengm/benchmarks/models/zip/color-seg-n.zip
	unzip color-seg-n.zip
	mv color-seg-n8/clownfish-small.h5 ./
	rm -rf color-seg-n8/ color-seg-n4/
	rm -f color-seg-n.zip


modeldir=/scratch/cluster/xrhuang/model/multilabel/EurLex.model

EurLex575.uai: multilabel2uai
	./multilabel2uai -p multilabel -o EurLex575.uai EurLex575.test $(modeldir)


