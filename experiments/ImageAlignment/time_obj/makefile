%.time_obj: ../../time_obj/%.sh ../%.log
	$(eval algo := $(basename $@))
	sh ../../time_obj/$(algo).sh ../$(algo).log $@

clean:
	rm -f *.time_obj
