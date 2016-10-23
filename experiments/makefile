#include Protein/makefile
solvers=GDMM TRWS AD3 PSDD MPLP Soft-BCFW LPsparse smoothMSD

.PHONY: Protein EurLex 107network ImageAlignment Segmentation

Protein: Protein.py
	$(MAKE) -C $@/time_obj/ clean
	$(MAKE) -C $@ time_obj
	python $@.py $(addprefix $@/time_obj/,$(addsuffix .time_obj, $(solvers)))
	scp $@.eps xrhuang@narsil-7.cs.utexas.edu:./public_html/figures/
	evince $@.eps

EurLex: EurLex.py
	$(MAKE) -C $@/time_obj/ clean
	$(MAKE) -C $@ time_obj
	python $@.py $(addprefix $@/time_obj/,$(addsuffix .time_obj, $(solvers)))
	scp $@.eps xrhuang@narsil-7.cs.utexas.edu:./public_html/figures/
	evince $@.eps

107network: 107network.py
	$(MAKE) -C $@/time_obj/ clean
	$(MAKE) -C $@ time_obj
	python $@.py $(addprefix $@/time_obj/,$(addsuffix .time_obj, $(solvers)))
	scp $@.eps xrhuang@narsil-7.cs.utexas.edu:./public_html/figures/
	evince $@.eps

ImageAlignment: ImageAlignment.py
	$(MAKE) -C $@/time_obj/ clean
	$(MAKE) -C $@ time_obj
	python $@.py $(addprefix $@/time_obj/,$(addsuffix .time_obj, $(solvers)))
	scp $@.eps xrhuang@narsil-7.cs.utexas.edu:./public_html/figures/
	evince $@.eps

Segmentation: Segmentation.py
	$(MAKE) -C $@/time_obj/ clean
	$(MAKE) -C $@ time_obj
	python $@.py $(addprefix $@/time_obj/,$(addsuffix .time_obj, $(solvers)))
	scp $@.eps xrhuang@narsil-7.cs.utexas.edu:./public_html/figures/
	evince $@.eps
