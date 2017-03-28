// Source file for the program to generate a set of points from a grid



// Include files 

#include "R3Shapes/R3Shapes.h"



// Program variables

static RNArray<char *> grid_names;
static char *points_name = NULL;
static RNLength min_spacing = 0;
static RNScalar sigma = 0;
static RNBoolean local_maxima_only = FALSE;
static int npoints = 1024;
static RNBoolean ascii = FALSE;
static RNBoolean print_verbose = FALSE;



// Type definitions

struct Point {
  Point(void) : position(0,0,0), normal(0,0,0), value(0) {};
  Point(const R3Point& position, const R3Vector& normal, RNScalar value) : position(position), normal(normal), value(value) {};
  R3Point position;
  R3Vector normal;
  RNScalar value;
};



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
ReadGrids(RNArray<R3Grid *>& grids, const RNArray<char *>& grid_names)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Read grid files
  for (int i = 0; i < grid_names.NEntries(); i++) {
    // Get grid filename
    char *grid_name = grid_names[i];

    // Read grid
    R3Grid *grid = ReadGrid(grid_name);
    if (!grid) return 0;

    // Add grid to array
    grids.Insert(grid);
  }

  // Print statistics
  if (print_verbose) {
    printf("Read %d grids ...\n", grid_names.NEntries());
    printf("  Total Time = %.2f seconds\n", start_time.Elapsed());
    fflush(stdout);
  }

  // Return number of grids
  return grid_names.NEntries();
}



static R3Grid *
CreateGrid(const RNArray<R3Grid *>& grids, RNLength sigma)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Build a single grid representing all grids
  R3Grid *grid = new R3Grid(*(grids[0]));
  for (int i = 1; i < grids.NEntries(); i++) grid->Add(*(grids[i]));

  // Blur grid
  RNScalar grid_sigma = sigma * grid->WorldToGridScaleFactor();
  if (grid_sigma > 0) grid->Blur(grid_sigma);

  // Print statistics
  if (print_verbose) {
    printf("Created combined grid ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Resolution = %d %d %d\n", grid->XResolution(), grid->YResolution(), grid->ZResolution());
    const R3Box bbox = grid->WorldBox();
    printf("  World Box = ( %g %g %g ) ( %g %g %g )\n", bbox[0][0], bbox[0][1], bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]);
    printf("  Spacing = %g\n", grid->GridToWorldScaleFactor());
    printf("  Cardinality = %d\n", grid->Cardinality());
    printf("  Volume = %g\n", grid->Volume());
    RNInterval grid_range = grid->Range();
    printf("  Minimum = %g\n", grid_range.Min());
    printf("  Maximum = %g\n", grid_range.Max());
    printf("  L1Norm = %g\n", grid->L1Norm());
    printf("  Sigma = %g\n", sigma);
    fflush(stdout);
  }

  // Return grid
  return grid;
}



static R3Grid *
CreateMask(R3Grid *grid, RNBoolean local_maxima_only)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Create mask
  R3Grid *mask = new R3Grid(*grid);
  if (!mask) {
    fprintf(stderr, "Unable to allocate mask\n");
    return NULL;
  }

  // Keep only local maxima, make others zero
  if (local_maxima_only) {
    for (int k = 0; k < grid->ZResolution(); k++) {
      for (int j = 0; j < grid->YResolution(); j++) {
        for (int i = 0; i < grid->XResolution(); i++) {
          RNScalar grid_value = grid->GridValue(i, j, k);
          for (int t = k-1; t <= k+1; t++) {
            if ((t < 0) || (t >= grid->ZResolution())) continue;
            for (int s = j-1; s <= j+1; s++) {
              if ((s < 0) || (s >= grid->YResolution())) continue;
              for (int r = i-1; r <= i+1; r++) {
                if ((r < 0) || (r >= grid->XResolution())) continue;
                if ((r == i) && (s == j) && (t == k)) continue;
                RNScalar neighbor_value = grid->GridValue(r, s, t);
                if (neighbor_value > grid_value) {
                  mask->SetGridValue(i, j, k, 0);
                  goto exitinteriorloops;
                }
              }
            }
          }
        exitinteriorloops:;
        }
      }
    }
  }

  // Print statistics
  if (print_verbose) {
    printf("Created mask ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Resolution = %d %d %d\n", mask->XResolution(), mask->YResolution(), mask->ZResolution());
    const R3Box bbox = mask->WorldBox();
    printf("  World Box = ( %g %g %g ) ( %g %g %g )\n", bbox[0][0], bbox[0][1], bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]);
    printf("  Spacing = %g\n", mask->GridToWorldScaleFactor());
    printf("  Cardinality = %d\n", mask->Cardinality());
    printf("  Volume = %g\n", mask->Volume());
    RNInterval grid_range = mask->Range();
    printf("  Minimum = %g\n", grid_range.Min());
    printf("  Maximum = %g\n", grid_range.Max());
    printf("  L1Norm = %g\n", mask->L1Norm());
    printf("  Local Maxima Only = %d\n", local_maxima_only);
    fflush(stdout);
  }

  // Return mask
  return mask;
}



static RNArray<Point *> *
CreatePoints(R3Grid *grid, R3Grid *mask, int npoints, RNLength min_spacing)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate array of points
  RNArray<Point *> *points = new RNArray<Point *>();
  if (!points) {
    fprintf(stderr, "Unable to allocate array of points\n");
    return NULL;
  }

  // Compute points
  for (int i = 0; i < npoints; i++) {
    // Find maximum value
    int maximum_index = -1;
    RNScalar maximum_value = 0;
    const RNScalar *mask_valuesp = mask->GridValues();
    for (int j = 0; j < mask->NEntries(); j++) {
      if (*mask_valuesp > maximum_value) { maximum_value = *mask_valuesp; maximum_index = j; }
      mask_valuesp++;
    }

    // Check if found value greater than zero
    if (maximum_index < 0) break;

    // Determine world position and normal
    int x, y, z;
    mask->IndexToIndices(maximum_index, x, y, z);
    R3Point position = grid->WorldPosition(x, y, z);
    R3Vector gradient = R3zero_vector; // grid->WorldGradientVector(x, y, z);
    gradient.Normalize();

    // Create point
    Point *point = new Point(position, gradient, maximum_value);
    assert(point);

    // Add point to list
    points->Insert(point);

    // Mask area around point 
    if (min_spacing > 0) mask->RasterizeWorldSphere(position, min_spacing, -maximum_value);
    else mask->SetGridValue(x, y, z, 0);
  }

  // Print message
  if (print_verbose) {
    printf("Created points ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Points = %d\n", points->NEntries());
    printf("  Requested Points = %d\n", npoints);
    printf("  Min Spacing = %g\n", min_spacing);
    fflush(stdout);
  }

  // Return points
  return points;
}



RNBoolean 
WritePoints(const RNArray<Point *>& points, const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Check if ascii output
  if (ascii) {
    // Open file
    FILE *fp = stdout;
    if (filename) {
      fp = fopen(filename, "w");
      if (!fp) {
        fprintf(stderr, "Unable to open output file %s\n", filename);
        return FALSE;
      }
    }

    // Write points
    float coordinates[6];
    for (int i = 0; i < points.NEntries(); i++) {
      Point *point = points[i];
      coordinates[0] = point->position.X();
      coordinates[1] = point->position.Y();
      coordinates[2] = point->position.Z();
      coordinates[3] = point->normal.X();
      coordinates[4] = point->normal.Y();
      coordinates[5] = point->normal.Z();
      fprintf(fp, "%12.6g %12.6g %12.6g\n", coordinates[0], coordinates[1], coordinates[2]);
      // fprintf(fp, "%12.6g %12.6g %12.6g  %12.6g %12.6g %12.6g\n", 
      //         coordinates[0], coordinates[1], coordinates[2], 
      //         coordinates[3], coordinates[4], coordinates[5]);
    }

    // Close file
    if (filename) fclose(fp);
  }
  else {
    // Open file
    FILE *fp = stdout;
    if (filename) {
      fp = fopen(filename, "wb");
      if (!fp) {
        fprintf(stderr, "Unable to open output file %s\n", filename);
        return FALSE;
      }
    }

    // Write points
    float coordinates[6];
    for (int i = 0; i < points.NEntries(); i++) {
      Point *point = points[i];
      coordinates[0] = point->position.X();
      coordinates[1] = point->position.Y();
      coordinates[2] = point->position.Z();
      coordinates[3] = point->normal.X();
      coordinates[4] = point->normal.Y();
      coordinates[5] = point->normal.Z();
      if (fwrite(coordinates, sizeof(float), 6, fp) != (unsigned int) 6) {
        fprintf(stderr, "Unable to write point to output file %s\n", filename);
        return FALSE;
      }
    }
  }

  // Print message
  if (print_verbose) {
    fprintf(stdout, "Wrote points to %s ...\n", (filename) ? filename : "stdout");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Points = %d\n", points.NEntries());
    printf("  Ascii = %d\n", ascii);
    fflush(stdout);
  }

  // Return success
  return TRUE;
}



static int 
ParseArgs(int argc, char **argv)
{
  // Check number of arguments
  if ((argc == 2) && (*argv[1] == '-')) {
    printf("Usage: grd2pts gridfile [gridfiles] pointfile\n");
    exit(0);
  }

  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) print_verbose = TRUE; 
      else if (!strcmp(*argv, "-ascii")) ascii = TRUE; 
      else if (!strcmp(*argv, "-npoints")) { argv++; argc--; npoints = atoi(*argv); }
      else if (!strcmp(*argv, "-min_spacing")) { argv++; argc--; min_spacing = atof(*argv); }
      else if (!strcmp(*argv, "-sigma")) { argv++; argc--; sigma = atof(*argv); }
      else if (!strcmp(*argv, "-local_maxima_only")) local_maxima_only = TRUE; 
      else { 
        fprintf(stderr, "Invalid program argument: %s\n", *argv); 
        exit(1); 
      }
    }
    else {
      grid_names.Insert(*argv);
    }
    argv++; argc--;
  }

  // Check grid filenames
  if (grid_names.IsEmpty()) {
    fprintf(stderr, "You did not specify any files.\n");
    return 0;
  }

  // Check points filename
  if (grid_names.NEntries() < 2) {
    fprintf(stderr, "You did not specify a points file.\n");
    return 0;
  }

  // Get points filename
  points_name = grid_names.Tail();
  grid_names.RemoveTail();

  // Return OK status 
  return 1;
}



int 
main(int argc, char **argv)
{
  // Parse program arguments
  if (!ParseArgs(argc, argv)) exit(-1);

  // Read grid files
  RNArray<R3Grid *> grids;
  int status = ReadGrids(grids, grid_names);
  if (!status) exit(-1);

  // Create combined grid
  R3Grid *grid = CreateGrid(grids, sigma);
  if (!grid) exit(-1);

  // Create mask
  R3Grid *mask = CreateMask(grid, local_maxima_only);
  if (!mask) exit(-1);

  // Create points
  RNArray<Point *> *points = CreatePoints(grid, mask, npoints, min_spacing);
  if (!points) exit(-1);

  // Write points
  if (!WritePoints(*points, points_name)) exit(-1);

  // Return success 
  return 0;
}



