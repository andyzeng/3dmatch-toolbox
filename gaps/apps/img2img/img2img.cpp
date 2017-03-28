// Source file for the GLUT programs


// RING include files

#include "R2Shapes/R2Shapes.h"



int main(int argc, char **argv)
{
  // Check arguments
  if (argc != 3) {
    fprintf(stderr, "Usage: img2img inputfile outputfile\n");
    return 1;
  }

  // Read and write image file 
  R2Image image;
  if (!image.Read(argv[1])) return -1;
  if (!image.Write(argv[2])) return -1;
  return 0;
}






