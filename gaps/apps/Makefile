#
# Makefile for GAPS apps
#

opt:
	$(MAKE) target "TARGET=$@"

debug:
	$(MAKE) target "TARGET=$@"

mesa:
	$(MAKE) target "TARGET=$@"

clean:
	$(MAKE) target "TARGET=$@"

release:
	mkdir -p ../release/apps
	cp Makefile ../release/apps
	$(MAKE) target "TARGET=$@"

target: 
	cd img2img; $(MAKE) $(TARGET)
	cd msh2df; $(MAKE) $(TARGET)
	cd msh2msh; $(MAKE) $(TARGET)
	cd msh2grd; $(MAKE) $(TARGET)
	cd msh2pts; $(MAKE) $(TARGET)
	cd msh2prp; $(MAKE) $(TARGET)
	cd mshalign; $(MAKE) $(TARGET)
	cd mshinfo; $(MAKE) $(TARGET)
	cd mshview; $(MAKE) $(TARGET)
	cd grdview; $(MAKE) $(TARGET)
	cd grd2grd; $(MAKE) $(TARGET)
	cd grd2msh; $(MAKE) $(TARGET)
	cd grd2pts; $(MAKE) $(TARGET)
	cd grd2txt; $(MAKE) $(TARGET)
	cd prp2prp; $(MAKE) $(TARGET)
	cd prpview; $(MAKE) $(TARGET)
	cd scninfo; $(MAKE) $(TARGET)
	cd scn2scn; $(MAKE) $(TARGET)
	cd scn2cam; $(MAKE) $(TARGET)
	cd scn2img; $(MAKE) $(TARGET)
	cd scnview; $(MAKE) $(TARGET)
	cd confview; $(MAKE) $(TARGET)
	cd conf2img; $(MAKE) $(TARGET)








