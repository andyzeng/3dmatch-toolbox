// Source file for the grid conversion to ascii text



// Include files 

#include "R3Shapes/R3Shapes.h"



// Type definitions

typedef enum {
  NOP_OPERATION,
  SQUARE_OPERATION,
  SQRT_OPERATION,
  NEGATE_OPERATION,
  INVERT_OPERATION,
  NORMALIZE_OPERATION,
  EDGE_DETECT_OPERATION,
  SIGNED_DISTANCE_OPERATION,
  SQUARED_DISTANCE_OPERATION,
  VORONOI_OPERATION,
  CLEAR_OPERATION,
  ADD_OPERATION,
  SUBTRACT_OPERATION,
  MULTIPLY_OPERATION,
  DIVIDE_OPERATION,
  POW_OPERATION,
  DILATE_OPERATION,
  ERODE_OPERATION,
  BLUR_OPERATION,
  THRESHOLD_OPERATION,
  RESAMPLE_OPERATION,
  ADD_GRID_OPERATION,
  SUBTRACT_GRID_OPERATION,
  MULTIPLY_GRID_OPERATION,
  DIVIDE_GRID_OPERATION,
  MASK_GRID_OPERATION,
  NUM_OPERATIONS
} OperationType;

struct Operation {
  int type;
  char *operand1;
  char *operand2;
  char *operand3;
};



// Program variables

static char *input_name = NULL;
static char *output_name = NULL;
static const int max_operations = 100;
static Operation operations[max_operations];
static int noperations = 0;
static int print_verbose = 0;
static int print_debug = 0;



static R3Grid *
ReadGrid(char *input_name)
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
  int status = grid->ReadFile(input_name);
  if (!status) {
    RNFail("Unable to read grid file %s", input_name);
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("Read grid ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Resolution = %d %d %d\n", grid->XResolution(), grid->YResolution(), grid->ZResolution());
    printf("  Spacing = %g\n", grid->GridToWorldScaleFactor());
    printf("  Cardinality = %d\n", grid->Cardinality());
    printf("  Volume = %g\n", grid->Volume());
    RNInterval grid_range = grid->Range();
    printf("  Minimum = %g\n", grid_range.Min());
    printf("  Maximum = %g\n", grid_range.Max());
    printf("  L1Norm = %g\n", grid->L1Norm());
    printf("  L2Norm = %g\n", grid->L2Norm());
    fflush(stdout);
  }

  // Return success
  return grid;
}



static int 
WriteGrid(R3Grid *grid, const char *grid_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Write grid
  int status = grid->WriteFile(grid_name);

  // Print statistics
  if (print_verbose) {
    printf("Wrote grid ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Bytes = %d\n", (int) (grid->NEntries() * sizeof(RNScalar)));
    fflush(stdout);
  }

  // Return status
  return status;
}



static int
ApplyOperations(R3Grid *grid, Operation *operations, int noperations)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Apply operations
  for (int i = 0; i < noperations; i++) {
    Operation *operation = &operations[i];

    // Read grid operand
    R3Grid *grid1 = NULL;
    switch (operation->type) {
    case ADD_GRID_OPERATION: 
    case SUBTRACT_GRID_OPERATION: 
    case MULTIPLY_GRID_OPERATION: 
    case DIVIDE_GRID_OPERATION: 
    case MASK_GRID_OPERATION: 
      grid1 = ReadGrid(operation->operand1);
      if (!grid1) {
        fprintf(stderr, "Unable to read grid file (%s) for operation %d\n", operation->operand1, i);
        return 0;
      }
    }

    // Print debug message
    if (print_debug) {
      printf("Applying operation: %d %s %s %s\n", operation->type, 
             (operation->operand1) ? operation->operand1 : "-", 
             (operation->operand2) ? operation->operand2 : "-", 
             (operation->operand3) ? operation->operand3 : "-");
    }

    // Apply operation
    switch (operation->type) {
    case NOP_OPERATION: break;
    case SQUARE_OPERATION: grid->Square(); break;
    case SQRT_OPERATION: grid->Sqrt(); break;
    case NEGATE_OPERATION: grid->Negate(); break;
    case INVERT_OPERATION: grid->Invert(); break;
    case NORMALIZE_OPERATION: grid->Normalize(); break;
    case EDGE_DETECT_OPERATION: grid->DetectEdges(); break;
    case SIGNED_DISTANCE_OPERATION: grid->SignedDistanceTransform(); break;
    case SQUARED_DISTANCE_OPERATION: grid->SquaredDistanceTransform(); break;
    case VORONOI_OPERATION: grid->Voronoi(); break;
    case CLEAR_OPERATION: grid->Clear(atof(operation->operand1)); break;
    case ADD_OPERATION: grid->Add(atof(operation->operand1)); break;
    case SUBTRACT_OPERATION: grid->Subtract(atof(operation->operand1)); break;
    case MULTIPLY_OPERATION: grid->Multiply(atof(operation->operand1)); break;
    case DIVIDE_OPERATION: grid->Divide(atof(operation->operand1)); break;
    case POW_OPERATION: grid->Pow(atof(operation->operand1)); break;
    case DILATE_OPERATION: grid->Dilate(atof(operation->operand1)); break;
    case ERODE_OPERATION: grid->Erode(atof(operation->operand1)); break;
    case BLUR_OPERATION: grid->Blur(atof(operation->operand1)); break;
    case RESAMPLE_OPERATION: grid->Resample(atoi(operation->operand1), atoi(operation->operand2), atoi(operation->operand3)); break;
    case ADD_GRID_OPERATION: grid->Add(*grid1); break;
    case SUBTRACT_GRID_OPERATION: grid->Subtract(*grid1); break;
    case MULTIPLY_GRID_OPERATION: grid->Multiply(*grid1); break;
    case DIVIDE_GRID_OPERATION: grid->Divide(*grid1); break;
    case MASK_GRID_OPERATION: grid->Divide(*grid1); break;
    case THRESHOLD_OPERATION: {
      RNScalar value1 = (strcmp(operation->operand2, "keep")) ? atof(operation->operand2) : R3_GRID_KEEP_VALUE;
      RNScalar value2 = (strcmp(operation->operand3, "keep")) ? atof(operation->operand3) : R3_GRID_KEEP_VALUE;
      grid->Threshold(atof(operation->operand1), value1, value2); 
      break; }
    default: 
      fprintf(stderr, "Unknown operation type (%d) in operation %d\n", operation->type, i); 
      return 0; 
    }
  }

  // Print statistics
  if (print_verbose) {
    printf("Applied operations ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Operations = %d\n", noperations);
    printf("  Resolution = %d %d %d\n", grid->XResolution(), grid->YResolution(), grid->ZResolution());
    printf("  Spacing = %g\n", grid->GridToWorldScaleFactor());
    printf("  Cardinality = %d\n", grid->Cardinality());
    printf("  Volume = %g\n", grid->Volume());
    RNInterval grid_range = grid->Range();
    printf("  Minimum = %g\n", grid_range.Min());
    printf("  Maximum = %g\n", grid_range.Max());
    printf("  L1Norm = %g\n", grid->L1Norm());
    printf("  L2Norm = %g\n", grid->L2Norm());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int 
ParseArgs(int argc, char **argv)
{
  // Check number of arguments
  if (argc < 2) {
    printf("Usage: grd2grd inputfile outputfile [-v]\n");
    exit(0);
  }

  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) {
        print_verbose = 1; 
      }
      else if (!strcmp(*argv, "-debug")) {
        print_debug = 1; 
      }
      else if (!strcmp(*argv, "-square")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = SQUARE_OPERATION;
      }
      else if (!strcmp(*argv, "-sqrt")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = SQRT_OPERATION;
      }
      else if (!strcmp(*argv, "-negate")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = NEGATE_OPERATION;
      }
      else if (!strcmp(*argv, "-invert")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = INVERT_OPERATION;
      }
      else if (!strcmp(*argv, "-normalize")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = NORMALIZE_OPERATION;
      }
      else if (!strcmp(*argv, "-edge_detect")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = EDGE_DETECT_OPERATION;
      }
      else if (!strcmp(*argv, "-signed_distance")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = SIGNED_DISTANCE_OPERATION;
      }
      else if (!strcmp(*argv, "-squared_distance")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = SQUARED_DISTANCE_OPERATION;
      }
      else if (!strcmp(*argv, "-voronoi")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = VORONOI_OPERATION;
      }
      else if (!strcmp(*argv, "-clear")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = CLEAR_OPERATION;
        argc--; argv++; operation->operand1 = *argv; 
      }
      else if (!strcmp(*argv, "-add")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = ADD_OPERATION;
        argc--; argv++; operation->operand1 = *argv; 
      }
      else if (!strcmp(*argv, "-subtract")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = SUBTRACT_OPERATION;
        argc--; argv++; operation->operand1 = *argv; 
      }
      else if (!strcmp(*argv, "-multiply")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = MULTIPLY_OPERATION;
        argc--; argv++; operation->operand1 = *argv; 
      }
      else if (!strcmp(*argv, "-divide")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = DIVIDE_OPERATION;
        argc--; argv++; operation->operand1 = *argv; 
      }
      else if (!strcmp(*argv, "-pow")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = POW_OPERATION;
        argc--; argv++; operation->operand1 = *argv; 
      }
      else if (!strcmp(*argv, "-dilate")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = DILATE_OPERATION;
        argc--; argv++; operation->operand1 = *argv; 
      }
      else if (!strcmp(*argv, "-erode")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = ERODE_OPERATION;
        argc--; argv++; operation->operand1 = *argv; 
      }
      else if (!strcmp(*argv, "-blur")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = BLUR_OPERATION;
        argc--; argv++; operation->operand1 = *argv; 
      }
      else if (!strcmp(*argv, "-threshold")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = THRESHOLD_OPERATION;
        argc--; argv++; operation->operand1 = *argv; 
        argc--; argv++; operation->operand2 = *argv; 
        argc--; argv++; operation->operand3 = *argv; 
      }
      else if (!strcmp(*argv, "-resample")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = RESAMPLE_OPERATION;
        argc--; argv++; operation->operand1 = *argv; 
        argc--; argv++; operation->operand2 = *argv; 
        argc--; argv++; operation->operand3 = *argv; 
      }
      else if (!strcmp(*argv, "-add_grid")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = ADD_GRID_OPERATION;
        argc--; argv++; operation->operand1 = *argv;
      }
      else if (!strcmp(*argv, "-subtract_grid")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = SUBTRACT_GRID_OPERATION;
        argc--; argv++; operation->operand1 = *argv;
      }
      else if (!strcmp(*argv, "-multiply_grid")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = MULTIPLY_GRID_OPERATION;
        argc--; argv++; operation->operand1 = *argv;
      }
      else if (!strcmp(*argv, "-divide_grid")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = DIVIDE_GRID_OPERATION;
        argc--; argv++; operation->operand1 = *argv;
      }
      else if (!strcmp(*argv, "-mask_grid")) {
        assert(noperations < max_operations);
        Operation *operation = &operations[noperations++];
        operation->type = MASK_GRID_OPERATION;
        argc--; argv++; operation->operand1 = *argv;
      }
      else { 
        fprintf(stderr, "Invalid program argument: %s", *argv); 
        exit(1); 
      }
    }
    else {
      if (!input_name) input_name = *argv;
      else if (!output_name) output_name = *argv;
      else { 
        fprintf(stderr, "Invalid program argument: %s", *argv); 
        exit(1); 
      }
    }
    argv++; argc--;
  }

  // Check input filename
  if (!input_name) {
    fprintf(stderr, "You did not specify an input file.\n");
    return 0;
  }

  // Check output filename
  if (!output_name) {
    fprintf(stderr, "You did not specify an output file.\n");
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
  R3Grid *grid = ReadGrid(input_name);
  if (!grid) exit(-1);

  // Apply operations
  int status1 = ApplyOperations(grid, operations, noperations);
  if (!status1) exit(-1);

  // Write grid file
  int status2 = WriteGrid(grid, output_name);
  if (!status2) exit(-1);

  // Return success
  return 0;
}
