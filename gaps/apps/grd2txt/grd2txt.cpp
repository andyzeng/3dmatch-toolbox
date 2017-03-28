// Source file for the grid conversion to ascii text



// Include files 

#include "R3Shapes/R3Shapes.h"



// Program variables

static char *grid_name = NULL;
static char *txt_name = NULL;
static int print_verbose = 0;



static R3Grid *
ReadGrid(char *grid_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate a grid
  R3Grid *grid = new R3Grid();
  if (!grid) {
    RNFail("Unable to allocate grid");
    return NULL;
  }

  // Read grid
  int status = grid->ReadFile(grid_name);
  if (!status) {
    RNFail("Unable to read grid file %s", grid_name);
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("Read grid ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Voxels = %d %d% d\n", grid->XResolution(), grid->YResolution(), grid->ZResolution());
    printf("  L2Norm = %g\n", grid->L2Norm());
    fflush(stdout);
  }

  // Return success
  return grid;
}



static int
WriteText(R3Grid *grid, const char *txt_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Open file
  FILE *fp = stdout;
  if (txt_name) {
    fp = fopen(txt_name, "w");
    if (!fp) {
      fprintf(stderr, "Unable to open TXT file %s\n", txt_name);
      return 0;
    }  
  }

  // Write values
  grid->Print(fp);

  // Close file
  if (txt_name) fclose(fp);

  // Print statistics
  if (print_verbose) {
    printf("Wrote text ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Values = %d\n", grid->NEntries());
    fflush(stdout);
  }

  // Return number of values written
  return grid->NEntries();
}



static int 
ParseArgs(int argc, char **argv)
{
  // Check number of arguments
  if (argc < 2) {
    printf("Usage: grd2txt gridfile [txtfile] [-v]\n");
    exit(0);
  }

  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) { 
        print_verbose = 1; 
      }
      else { 
        fprintf(stderr, "Invalid program argument: %s", *argv); 
        exit(1); 
      }
    }
    else {
      if (!grid_name) grid_name = *argv;
      else if (!txt_name) txt_name = *argv;
      else { 
        fprintf(stderr, "Invalid program argument: %s", *argv); 
        exit(1); 
      }
    }
    argv++; argc--;
  }

  // Check grid filename
  if (!grid_name) {
    fprintf(stderr, "You did not specify a grid file.\n");
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

  // Read grid file
  R3Grid *grid = ReadGrid(grid_name);
  if (!grid) exit(-1);

  // Write ascii text
  int status = WriteText(grid, txt_name);
  if (!status) exit(-1);

  // Return success
  return 0;
}
