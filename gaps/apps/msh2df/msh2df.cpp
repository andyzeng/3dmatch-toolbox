// Source file for the mesh to grid conversion program

// To run it, on a mesh, execute this command:
//     msh2df foo.off foo.raw -v [options]
//      Options include:
//         -grid_spacing <real> (default is 0.01) = width of a grid cell
//         -border_distance <int> (default is 40) = how many grid cells are between surface and boundary of grid
//         -truncation_distance <real> (default is infinity) = truncation threshold of values in distance function
//         -max_resolution <int> (default is 512) = maximum number of grid cells any dimension

// The program will create three files for a NX by NY by NZ grid:
//     foo.raw = a binary file with NX * NY * NZ floats representing distance from the surface in grid units
//        The file does not have any header (with Z varying in the outermost loop ... and X varying in the innermost loop)
//     foo.size = an ASCII file containing one line: "Float32(NX,NY,NZ,1)" (with NX, NY, NZ filled in with integer values)
//     foo.xf = an ASCII file containing a 4x4 transformation matrix for converting mesh coordinates to grid coordinates

// Include files 

#include "R3Shapes/R3Shapes.h"



// Program variables

static char *input_mesh_name = NULL;
static char *output_grid_name = NULL;
static double grid_spacing = 0.01;  // in world units
static double truncation_distance = RN_INFINITY; // in grid units
static double border_distance = 40; // in grid units
static int max_resolution = 512; // in grid units
static int print_verbose = 0;



static R3Mesh *
ReadMesh(char *input_mesh_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate mesh
  R3Mesh *mesh = new R3Mesh();
  if (!mesh) {
    fprintf(stderr, "Unable to allocate mesh for %s\n", input_mesh_name);
    return NULL;
  }

  // Read mesh from file
  if (!mesh->ReadFile(input_mesh_name)) {
    delete mesh;
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("Read mesh ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Faces = %d\n", mesh->NFaces());
    printf("  # Edges = %d\n", mesh->NEdges());
    printf("  # Vertices = %d\n", mesh->NVertices());
    fflush(stdout);
  }

  // Return success
  return mesh;
}



static R3Grid *
AllocateGrid(R3Mesh *mesh)
{
  // Compute bounding box
  R3Box bbox = mesh->BBox();
  if (RNIsZero(bbox.Volume())) return NULL;

  // Compute grid spacing 
  RNScalar spacing = grid_spacing;
  if (spacing == 0) spacing = 0.01;
  if (bbox.XLength()/spacing > max_resolution) spacing = bbox.XLength()/max_resolution;
  if (bbox.YLength()/spacing > max_resolution) spacing = bbox.YLength()/max_resolution;
  if (bbox.ZLength()/spacing > max_resolution) spacing = bbox.ZLength()/max_resolution;

  // Add border to bbox
  bbox[0] -= spacing * border_distance * R3ones_vector;
  bbox[1] += spacing * border_distance * R3ones_vector;

  // Compute grid resolution
  if (RNIsZero(spacing)) return 0;
  int xres = (bbox.XLength() / spacing + 0.5);
  int yres = (bbox.YLength() / spacing + 0.5);
  int zres = (bbox.ZLength() / spacing + 0.5);

  // Allocate grid
  R3Grid *grid = new R3Grid(xres, yres, zres, bbox);
  if (!grid) {
    fprintf(stderr, "Unable to allocate grid\n");
    exit(-1);
  }

  // Return grid
  return grid;
}



static int
RasterizeMesh(R3Grid *grid, R3Mesh *mesh)
{
  // Rasterize each triangle into grid
  for (int i = 0; i < mesh->NFaces(); i++) {
    R3MeshFace *face = mesh->Face(i);
    const R3Point& p0 = mesh->VertexPosition(mesh->VertexOnFace(face, 0));
    const R3Point& p1 = mesh->VertexPosition(mesh->VertexOnFace(face, 1));
    const R3Point& p2 = mesh->VertexPosition(mesh->VertexOnFace(face, 2));
    grid->RasterizeWorldTriangle(p0, p1, p2, 1.0);
  }

  // Truncate to avoid double rasterization
  grid->Threshold(0.5, 0.0, 1.0);

  // Return success
  return 1;
}




static int
ProcessGrid(R3Grid *grid)
{
  // Compute distance transform
  grid->SquaredDistanceTransform();
  grid->Sqrt();

  // Truncate distance transform
  if (truncation_distance < RN_INFINITY) {
    grid->Threshold(truncation_distance, R3_GRID_KEEP_VALUE, truncation_distance);
  }
  
  // Return success
  return 1;
}




static R3Grid *
CreateGrid(R3Mesh *mesh)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate grid
  R3Grid *grid = AllocateGrid(mesh);
  if (!grid) return NULL;

  // Rasterize mesh
  if (!RasterizeMesh(grid, mesh)) {
    delete grid;
    return NULL;
  }

  // Process grid
  if (!ProcessGrid(grid)) {
    delete grid;
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("Created grid ...\n");
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

  // Return grid
  return grid;
}



static int 
WriteGrid(R3Grid *grid, const char *output_grid_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Write grid
  if (!grid->WriteFile(output_grid_name)) return 0;

  // Print statistics
  if (print_verbose) {
    printf("Wrote grid ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # MBytes = %g\n", (double) (grid->NEntries() * sizeof(RNScalar)) / (1024.0 * 1024.0));
    fflush(stdout);
  }

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
      else if (!strcmp(*argv, "-grid_spacing")) { argc--; argv++; grid_spacing = atof(*argv); }
      else if (!strcmp(*argv, "-truncation_distance")) { argc--; argv++; truncation_distance = atof(*argv); }
      else if (!strcmp(*argv, "-border_distance")) { argc--; argv++; border_distance = atof(*argv); }
      else if (!strcmp(*argv, "-max_resolution")) { argc--; argv++; max_resolution = atoi(*argv); }
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
    }
    else {
      if (!input_mesh_name) input_mesh_name = *argv;
      else if (!output_grid_name) output_grid_name = *argv;
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
    }
    argv++; argc--;
  }

  // Check filenames
  if (!input_mesh_name || !output_grid_name) {
    fprintf(stderr, "Usage: msh2tdf inputmeshfile outputgridfile [options]\n");
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

  // Read mesh file
  R3Mesh *mesh = ReadMesh(input_mesh_name);
  if (!mesh) exit(-1);

  // Create grid from mesh
  R3Grid *grid = CreateGrid(mesh);
  if (!grid) exit(-1);

  // Write grid
  int status = WriteGrid(grid, output_grid_name);
  if (!status) exit(-1);

  // Return success
  return 0;
}
