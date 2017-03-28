#
# Makefile for GAPS docs
#

opt:
	./makeall

debug:
	./makeall

release:
	mkdir -p ../release/docs
	cp Makefile ../release/docs
	cp makeall ../release/docs
	cp makepkg ../release/docs
	cp makepkg.awk ../release/docs
	cp -r headers ../release/docs
	cp ./*.html ../release/docs

clean:
	- rm -f *~ *.html







