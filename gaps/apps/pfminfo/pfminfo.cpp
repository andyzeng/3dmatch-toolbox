// Source file for the image info program



// Include files 

#include "R2Shapes/R2Shapes.h"



// Program variables

static char *input_name = NULL;
static int print_verbose = 0;



static int
PrintBasicInfo(const R2Grid *grid)
{
  printf("Basic info:\n");
  printf("  Resolution = %d %d\n", grid->XResolution(), grid->YResolution());
  printf("  Spacing = %g\n", grid->GridToWorldScaleFactor());
  printf("  Cardinality = %d\n", grid->Cardinality());
  RNInterval grid_range = grid->Range();
  printf("  Minimum = %g\n", grid_range.Min());
  printf("  Maximum = %g\n", grid_range.Max());
  printf("  L1Norm = %g\n", grid->L1Norm());
  printf("  L2Norm = %g\n", grid->L2Norm());
  fflush(stdout);

  // Return success
  return 1;
}



static int 
ParseArgs(int argc, char **argv)
{
  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) print_verbose = 1; 
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
    }
    else {
      if (!input_name) input_name = *argv;
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
    }
    argv++; argc--;
  }

  // Check input filename
  if (!input_name) {
    fprintf(stderr, "Usage: pfminfo inputfile [options]\n");
    return 0;
  }

  // Return OK status 
  return 1;
}



int 
main(int argc, char **argv)
{
  // Parse program arguments
  if (!ParseArgs(argc, argv)) exit(-1);

  // Read grid
  R2Grid *grid = new R2Grid();
  if (!grid->Read(input_name)) exit(-1);

  // Print info
  if (!PrintBasicInfo(grid)) exit(-1);

  // Return success
  return 0;
}
