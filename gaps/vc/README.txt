To compile, you should be able to open vc.sln and "Rebuild Solution.

------------------------------------------------------------------------

This file contains notes about settings made when creating the vc.sln file.

All:
  General: Output Directory = $(SolutionDir)/$(ConfigurationName)/$(ProjectName)
  General: Intermediate Directory = $(SolutionDir)/$(ConfigurationName)/$(ProjectName)
  Input: Additional Dependencies = R3Graphics.lib R3Shapes.lib R2Shapes.lib RNBasics.lib jpeg.lib glut32.lib glu32.lib opengl32.lib
  C/C++: General: Additional Include Directories = ../../pkgs
  C/C++: Advanced: Comple as C++ Code (Set TP explicitly) --- except jpeg, png, fglut
  C/C++: Advanced: Disable Specific warnings 4244, 4267, 4996

For pkgs:
   Librarian: General: Output File = ../../lib/win32/${ProjectName}.lib

For apps:
   Linker: General:Additional Library Directories = ../../lib/win32
   Linker: General: Output file = ../../bin/win32/$(ProjectName).exe
