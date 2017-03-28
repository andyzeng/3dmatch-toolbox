// Source file for the program to extract a isosurface from a grid



// Include files 

#include "R3Shapes/R3Shapes.h"



// Program variables

static char *grid_name = NULL;
static char *mesh_name = NULL;
static RNScalar threshold = 0;
static RNBoolean print_verbose = FALSE;



static R3Grid *
ReadGrid(char *grid_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate grid
  R3Grid *grid = new R3Grid();
  if (!grid) {
    RNFail("Unable to allocated grid");
    return NULL;
  }

  // Read grid 
  if (!grid->ReadFile(grid_name)) {
    RNFail("Unable to read grid file %s", grid_name);
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("Read grid ...\n");
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
    fflush(stdout);
  }

  // Return success
  return grid;
}



static R3Mesh *
CreateMesh(R3Grid *grid, RNScalar threshold)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Create mesh
  R3Mesh *mesh = new R3Mesh();
  if (!mesh) {
    fprintf(stderr, "Unable to allocate mesh\n");
    return NULL;
  }

  // Extract isosurface from grid
  const int max_points = 16 * 1024 * 1024;
  static R3Point points[max_points];
  int npoints = grid->GenerateIsoSurface(threshold, points, max_points);
  if (npoints == 0) {
    fprintf(stderr, "Empty isosurface for threshold: %g\n", threshold);
    return NULL;
  }

  // Create mesh
  R3Point *pointsp = points;
  for (int i = 0; i < npoints; i += 3) {
    R3MeshVertex *v1 = mesh->CreateVertex(grid->WorldPosition(*(pointsp++)));
    R3MeshVertex *v2 = mesh->CreateVertex(grid->WorldPosition(*(pointsp++)));
    R3MeshVertex *v3 = mesh->CreateVertex(grid->WorldPosition(*(pointsp++)));
    mesh->CreateFace(v3, v2, v1);
  }

  // Print statistics
  if (print_verbose) {
    printf("Created mesh ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Faces = %d\n", mesh->NFaces());
    printf("  # Edges = %d\n", mesh->NEdges());
    printf("  # Vertices = %d\n", mesh->NVertices());
    fflush(stdout);
  }

  // Return mesh
  return mesh;
}



static int
WriteMesh(R3Mesh *mesh, char *mesh_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Write mesh to file
  if (!mesh->WriteFile(mesh_name)) return 0;

  // Print statistics
  if (print_verbose) {
    printf("Wrote mesh ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Faces = %d\n", mesh->NFaces());
    printf("  # Edges = %d\n", mesh->NEdges());
    printf("  # Vertices = %d\n", mesh->NVertices());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int 
ParseArgs(int argc, char **argv)
{
  // Check number of arguments
  if ((argc == 2) && (*argv[1] == '-')) {
    printf("Usage: grd2off gridfile meshfile -threshold <real> [-v]\n");
    exit(0);
  }

  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) print_verbose = TRUE; 
      else if (!strcmp(*argv, "-threshold")) { argc--; argv++; threshold = atof(*argv); }
      else { 
        fprintf(stderr, "Invalid program argument: %s\n", *argv); 
        exit(1); 
      }
      argv++; argc--;
    }
    else {
      if (!grid_name) grid_name = *argv;
      else if (!mesh_name) mesh_name = *argv;
      else { 
        fprintf(stderr, "Invalid program argument: %s\n", *argv); 
        exit(1); 
      }
      argv++; argc--;
    }
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
  R3Grid *grid = ReadGrid(grid_name);
  if (!grid) exit(-1);

  // Create isosurface
  R3Mesh *mesh = CreateMesh(grid, threshold);
  if (!mesh) exit(-1);

  // Write mesh
  int status = WriteMesh(mesh, mesh_name);
  if (!status) exit(-1);

  // Return success 
  return 0;
}



