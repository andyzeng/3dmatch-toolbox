// Source file for the scene converter program



// Include files 

#include "R3Graphics/R3Graphics.h"



// Program arguments

static const char *input_scene_name = NULL;
static const char *output_grid_name = NULL;
static double grid_spacing = 0.01;
static double grid_boundary_radius = 0.05;
static int grid_max_resolution = 256;
static int print_verbose = 0;



////////////////////////////////////////////////////////////////////////
// I/O STUFF
////////////////////////////////////////////////////////////////////////

static R3Scene *
ReadScene(const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate scene
  R3Scene *scene = new R3Scene();
  if (!scene) {
    fprintf(stderr, "Unable to allocate scene for %s\n", filename);
    return NULL;
  }

  // Read scene from file
  if (!scene->ReadFile(filename)) {
    delete scene;
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("Read scene from %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Nodes = %d\n", scene->NNodes());
    fflush(stdout);
  }

  // Return scene
  return scene;
}



static int 
WriteGrid(R3Grid *grid, const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Write grid
  if (!grid->WriteFile(filename)) return 0;

  // Print statistics
  if (print_verbose) {
    printf("Wrote grid to %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Resolution = %d %d %d\n", grid->XResolution(), grid->YResolution(), grid->ZResolution());
    printf("  Spacing = %g\n", grid->GridToWorldScaleFactor());
    printf("  Cardinality = %d\n", grid->Cardinality());
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



////////////////////////////////////////////////////////////////////////
// GRID CREATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

static void
RasterizeTriangles(R3Grid *grid, R3Scene *scene, R3SceneNode *node, const R3Affine& parent_transformation)
{
  // Update transformation
  R3Affine transformation = R3identity_affine;
  transformation.Transform(parent_transformation);
  transformation.Transform(node->Transformation());
  
  // Rasterize triangles
  for (int i = 0; i < node->NElements(); i++) {
    R3SceneElement *element = node->Element(i);
    for (int j = 0; j < element->NShapes(); j++) {
      R3Shape *shape = element->Shape(j);
      if (shape->ClassID() == R3TriangleArray::CLASS_ID()) {
        R3TriangleArray *triangles = (R3TriangleArray *) shape;
        for (int k = 0; k < triangles->NTriangles(); k++) {
          R3Triangle *triangle = triangles->Triangle(k);
          R3TriangleVertex *v0 = triangle->V0();
          R3TriangleVertex *v1 = triangle->V1();
          R3TriangleVertex *v2 = triangle->V2();
          R3Point p0 = v0->Position();
          R3Point p1 = v1->Position();
          R3Point p2 = v2->Position();
          p0.Transform(transformation);
          p1.Transform(transformation);
          p2.Transform(transformation);
          grid->RasterizeWorldTriangle(p0, p1, p2, 1.0);
        }
      }
    }
  }

  // Rasterize children
  for (int i = 0; i < node->NChildren(); i++) {
    R3SceneNode *child = node->Child(i);
    RasterizeTriangles(grid, scene, child, transformation);
  }
}



static R3Grid *
CreateGrid(R3Scene *scene)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Get bounding box
  R3Box bbox = scene->BBox();
  if (grid_boundary_radius > 0) {
    bbox[0] -= R3Vector(grid_boundary_radius, grid_boundary_radius, grid_boundary_radius);
    bbox[1] += R3Vector(grid_boundary_radius, grid_boundary_radius, grid_boundary_radius);
  }

  // Compute grid spacing
  RNLength diameter = bbox.LongestAxisLength();
  RNLength min_grid_spacing = (grid_max_resolution > 0) ? diameter / grid_max_resolution : RN_EPSILON;
  if (grid_spacing == 0) grid_spacing = diameter / 64;
  if (grid_spacing < min_grid_spacing) grid_spacing = min_grid_spacing;

  // Compute grid resolution
  int xres = (int) (bbox.XLength() / grid_spacing + 0.5); if (xres == 0) xres = 1;
  int yres = (int) (bbox.YLength() / grid_spacing + 0.5); if (yres == 0) yres = 1;
  int zres = (int) (bbox.ZLength() / grid_spacing + 0.5); if (zres == 0) zres = 1;
    
  // Allocate grid
  R3Grid *grid = new R3Grid(xres, yres, zres, bbox);
  if (!grid) {
    fprintf(stderr, "Unable to allocate grid\n");
    return NULL;
  }

  // Rasterize scene into grid
  RasterizeTriangles(grid, scene, scene->Root(), R3identity_affine);

  // Threshold grid (to compensate for possible double rasterization)
  grid->Threshold(0.5, 0.0, 1.0);
 
  // Return grid
  return grid;
}


////////////////////////////////////////////////////////////////////////
// PROGRAM ARGUMENT PARSING
////////////////////////////////////////////////////////////////////////

static int
ParseArgs(int argc, char **argv)
{
  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) print_verbose = 1;
      else if (!strcmp(*argv, "-spacing")) { argc--; argv--; grid_spacing = atof(*argv); }
      else if (!strcmp(*argv, "-boundary_radius")) { argc--; argv--; grid_boundary_radius = atof(*argv); }
      else if (!strcmp(*argv, "-max_resolution")) { argc--; argv--; grid_max_resolution = atoi(*argv); }
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
    else {
      if (!input_scene_name) input_scene_name = *argv;
      else if (!output_grid_name) output_grid_name = *argv;
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
  }

  // Check input filename
  if (!input_scene_name || !output_grid_name) {
    fprintf(stderr, "Usage: scn2grd inputscenefile outputgridfile [options]\n");
    return 0;
  }

  // Return OK status 
  return 1;
}



////////////////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
  // Check number of arguments
  if (!ParseArgs(argc, argv)) exit(1);

  // Read scene
  R3Scene *scene = ReadScene(input_scene_name);
  if (!scene) exit(-1);

  // Create grid
  R3Grid *grid = CreateGrid(scene);
  if (!grid) exit(-1);

  // Write grid
  if (!WriteGrid(grid, output_grid_name)) exit(-1);

  // Return success 
  return 0;
}

