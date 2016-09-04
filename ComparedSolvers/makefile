all: ad3 mplp trws

ad3:
	cd ../../AD3-2.0.2/ && $(MAKE) ad3_multi
	cp ../../AD3-2.0.2/ad3_multi ./ad3

mplp:
	cd ../../mplp_ver2/ && $(MAKE)
	cp ../../mplp_ver2/mplp ./mplp

trws:
	$(eval trws_dir := /home/xiangru/opengm-build/src/unittest/inference/)
	cd $(trws_dir) && $(MAKE) test-trwsi
	cp $(trws_dir)/test-trwsi ./trws

