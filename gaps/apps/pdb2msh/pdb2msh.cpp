// Source file for the pdb viewer program



// Include files 

#include "R3Shapes/R3Shapes.h"
#include "PDB/PDB.h"



// Program variables

static char *pdb_name = NULL;
static char *mesh_name = NULL;
static RNScalar grid_spacing = 0.25;
static int print_verbose = 0;



static PDBFile *
ReadPDB(char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate PDBFile
  PDBFile *file = new PDBFile(filename);
  if (!file) {
    RNFail("Unable to allocate PDB file for %s", filename);
    return NULL;
  }

  // Read PDB file
  if (!file->ReadFile(filename)) return 0;

  // Print statistics
  if (print_verbose) {
    printf("Read PDB file ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    for (int i =0; i < file->NModels(); i++) {
      PDBModel *model = file->Model(i);
      printf("  Model %s ...\n", model->Name());
      printf("  # Chains = %d\n", model->NChains());
      printf("  # Residues = %d\n", model->NResidues());
      printf("  # Atoms = %d\n", model->NAtoms());
    }
    fflush(stdout);
  }

  // Return success
  return file;
}


/* Union current grid (containing signed distance function) with signed
   distance function for sphere.  The sphere's distance function is
   rendered only in an enclosing box region where on the border of the
   box the SDF becomes negative. */
void UnionGridSphere(R3Grid *g, R3Point center, RNScalar r) {
  RNScalar x = center.X(), y = center.Y(), z = center.Z();
  int nx = g->XResolution();
  int ny = g->YResolution();
  int nz = g->ZResolution();
  int xmin = (int) (x - r - 1), xmax = (int) (x + r + 2);
  int ymin = (int) (y - r - 1), ymax = (int) (y + r + 2);
  int zmin = (int) (z - r - 1), zmax = (int) (z + r + 2);
  if (xmin < 0) { xmin = 0; }
  if (ymin < 0) { ymin = 0; }
  if (zmin < 0) { zmin = 0; }
  if (xmax > nx) { xmax = nx; }
  if (ymax > ny) { ymax = ny; }
  if (zmax > nz) { zmax = nz; }
  for (int iz = zmin; iz < zmax; iz++) {
    for (int iy = ymin; iy < ymax; iy++) {
      for (int ix = xmin; ix < xmax; ix++) {
        RNScalar dx = ix - x;
        RNScalar dy = iy - y;
        RNScalar dz = iz - z;
        RNScalar a = r - ((RNScalar) sqrt(dx*dx+dy*dy+dz*dz));
        RNScalar b = g->GridValue(ix, iy, iz);
        RNScalar c = a > b ? a: b;
        g->SetGridValue(ix, iy, iz, c);
      }
    }
  }
}

/* Same as UnionGridSphere, but operate in world coords.
   Assumes world <=> grid transform uses uniform scaling */
void UnionWorldSphere(R3Grid *g, R3Point center, RNScalar r) {
  UnionGridSphere(g, g->GridPosition(center), r * g->WorldToGridScaleFactor());
}


static R3Grid *
CreateGrid(PDBFile *file, RNScalar grid_spacing)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Get model from PDB file
  if (file->NModels() == 0) return NULL;
  PDBModel *model = file->Model(0);

  // Determine bounding box (expanded a bit)
  R3Box world_box = model->BBox();
  R3Point world_center = world_box.Centroid();
  R3Point world_max = world_box.Max();
  R3Vector diagonal = world_max - world_center;
  world_box.Reset(world_center - 1.1 * diagonal, world_center + 1.1 * diagonal);

  // Determine grid resolution
  const int max_resolution[3] = { 256, 256, 256 };
  int xres = (int) (world_box.XLength() / grid_spacing + 1);
  int yres = (int) (world_box.YLength() / grid_spacing + 1);
  int zres = (int) (world_box.ZLength() / grid_spacing + 1);
  if (xres > max_resolution[0]) xres = max_resolution[0];
  if (yres > max_resolution[1]) yres = max_resolution[1];
  if (zres > max_resolution[2]) zres = max_resolution[2];
  if (xres < 1) xres = 1;
  if (yres < 1) yres = 1;
  if (zres < 1) zres = 1;

  // Allocate grid
  R3Grid *grid = new R3Grid(xres, yres, zres);
  grid->Clear(-1.0);
  if (!grid) {
    fprintf(stderr, "Unable to allocate grid\n");
    exit(-1);
  }

  // Set world to grid transformation
  grid->SetWorldToGridTransformation(world_box);

  // Rasterize atoms into grid
  for (int i = 0; i < model->NAtoms(); i++) {
    PDBAtom *atom = model->Atom(i);
    UnionWorldSphere(grid, atom->Position(), atom->Radius());
  }

  // Print statistics
  if (print_verbose) {
    printf("Created grid ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Grid Resolution = %d %d% d\n", grid->XResolution(), grid->YResolution(), grid->ZResolution());
    printf("  World Center = %g %g %g\n", world_center[0], world_center[1], world_center[2]);
    printf("  World Spacing = %g\n", grid->GridToWorldScaleFactor());
    fflush(stdout);
  }

  // Return grid
  return grid;
}



static R3Mesh *
CreateMesh(R3Grid *grid)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Extract isosurface from grid
  const int max_points = 8 * 1024 * 1024;
  static R3Point points[max_points];
  int npoints = grid->GenerateIsoSurface(0.0, points, max_points);
  if (npoints == 0) return NULL;

  // Allocate mesh
  R3Mesh *mesh = new R3Mesh();
  assert(mesh);

  // Create mesh
  R3Point *pointsp = points;
  for (int i = 0; i < npoints; i += 3) {
    R3MeshVertex *v1 = mesh->CreateVertex(grid->WorldPosition(*(pointsp++)));
    R3MeshVertex *v2 = mesh->CreateVertex(grid->WorldPosition(*(pointsp++)));
    R3MeshVertex *v3 = mesh->CreateVertex(grid->WorldPosition(*(pointsp++)));
    mesh->CreateFace(v1, v3, v2);
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
  if (argc < 3) {
    printf("Usage: pdb2off pdbfile offfile [-spacing d] [-v]\n");
    exit(0);
  }

  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) { 
        print_verbose = 1; 
      }
      else if (!strcmp(*argv, "-spacing")) { 
        argc--; argv++; grid_spacing = atof(*argv); 
      }
      else { 
        fprintf(stderr, "Invalid program argument: %s", *argv); 
        return 0;
      }
    }
    else {
      if (!pdb_name) pdb_name = *argv;
      else if (!mesh_name) mesh_name = *argv;
      else { 
        fprintf(stderr, "Invalid program argument: %s", *argv); 
        return 0;
      }
    }
    argv++; argc--;
  }

  // Check pdb filename
  if (!pdb_name) {
    fprintf(stderr, "You did not specify a pdb file.\n");
    return 0;
  }

  // Check mesh filename
  if (!mesh_name) {
    fprintf(stderr, "You did not specify a mesh file.\n");
    return 0;
  }

  // Check grid spacing
  if (grid_spacing <= 0) {
    fprintf(stderr, "Invalid spacing: %g\n", grid_spacing);
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

  // Read PDB file
  PDBFile *file = ReadPDB(pdb_name);
  if (!file) exit(-1);

  // Create grid 
  R3Grid *grid = CreateGrid(file, grid_spacing);
  if (!grid) exit(-1);

  // Create mesh
  R3Mesh *mesh = CreateMesh(grid);
  if (!mesh) exit(-1);

  // Write mesh
  int status = WriteMesh(mesh, mesh_name);
  if (!status) exit(-1);

  // Return success
  return 0;
}
