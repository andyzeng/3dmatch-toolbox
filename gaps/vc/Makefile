#
# Makefile for GAPS visual studio release
#


#
# Set up options, etc.
#

RELEASE_DIR=../release



#
# Make targets
#

opt:

debug:

mesa:

release:
	mkdir -p $(RELEASE_DIR)/vc
	cp vc.sln $(RELEASE_DIR)/vc
	cp -r glut $(RELEASE_DIR)/vc
	mkdir -p $(RELEASE_DIR)/bin
	mkdir -p $(RELEASE_DIR)/lib
	mkdir -p $(RELEASE_DIR)/bin/win32
	mkdir -p $(RELEASE_DIR)/lib/win32
	cp glut/glut32.dll $(RELEASE_DIR)/bin/win32
	cp glut/glut32.lib $(RELEASE_DIR)/lib/win32
	cp Makefile $(RELEASE_DIR)/vc
	cp README.txt $(RELEASE_DIR)/vc

clean:
	- rm -f -r Debug Release ../bin/win32/*.exe ../bin/win32/*.ilk ../bin/win32/*.pdb ../lib/win32/*.lib vc.ncb vc.suo vc.sdf *~ 







