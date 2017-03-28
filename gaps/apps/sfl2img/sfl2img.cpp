// Source file for the surfel scene processing program



////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// Program arguments
////////////////////////////////////////////////////////////////////////

static char *input_scene_name = NULL;
static char *input_database_name = NULL;
static char *output_directory_name = NULL;
static double pixel_spacing = 0.5; 
static int max_resolution = 32768;
static int write_base_grids = 0;
static int write_color_grids = 0;
static int write_slice_grids = 0;
static int write_height_grids = 0;
static int write_graph_grids = 0;
static int write_planar_grids = 0;
static int write_pixel_grids = 0;
static int write_horizontal_grids = 0;
static double chunk_size = 20;
static int print_verbose = 0;
static int print_debug = 0;





////////////////////////////////////////////////////////////////////////
// Surfel scene I/O Functions
////////////////////////////////////////////////////////////////////////

static R3SurfelScene *
OpenScene(const char *input_scene_name, const char *input_database_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate scene
  R3SurfelScene *scene = new R3SurfelScene();
  if (!scene) {
    fprintf(stderr, "Unable to allocate scene\n");
    return NULL;
  }

  // Open scene files
  if (!scene->OpenFile(input_scene_name, input_database_name, "r", "r")) {
    delete scene;
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("Opened scene ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Objects = %d\n", scene->NObjects());
    printf("  # Labels = %d\n", scene->NLabels());
    printf("  # Assignments = %d\n", scene->NLabelAssignments());
    printf("  # Features = %d\n", scene->NFeatures());
    printf("  # Nodes = %d\n", scene->Tree()->NNodes());
    printf("  # Blocks = %d\n", scene->Tree()->Database()->NBlocks());
    printf("  # Surfels = %d\n", scene->Tree()->Database()->NSurfels());
    fflush(stdout);
  }

  // Return scene
  return scene;
}



static int
CloseScene(R3SurfelScene *scene)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Print statistics
  if (print_verbose) {
    printf("Closing scene ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Objects = %d\n", scene->NObjects());
    printf("  # Labels = %d\n", scene->NLabels());
    printf("  # Assignments = %d\n", scene->NLabelAssignments());
    printf("  # Features = %d\n", scene->NFeatures());
    printf("  # Nodes = %d\n", scene->Tree()->NNodes());
    printf("  # Blocks = %d\n", scene->Tree()->Database()->NBlocks());
    printf("  # Surfels = %d\n", scene->Tree()->Database()->NSurfels());
    fflush(stdout);
  }

  // Close scene files
  if (!scene->CloseFile()) {
    delete scene;
    return 0;
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////////

static void
InitializeOverheadGrid(R2Grid& grid, R3SurfelScene *scene, RNScalar pixel_spacing, int max_resolution, RNScalar initial_value)
{
  // Compute bounding box
  const R3Box& scene_bbox = scene->BBox();
  R2Box bbox(scene_bbox[0][0], scene_bbox[0][1], scene_bbox[1][0], scene_bbox[1][1]);

  // Compute pixel resolution
  int nxpixels = (int) (bbox.XLength() / pixel_spacing);
  int nypixels = (int) (bbox.YLength() / pixel_spacing);
  if (nxpixels == 0) nxpixels = 1;
  if (nypixels == 0) nypixels = 1;
  if (nxpixels > max_resolution) {
    pixel_spacing = bbox.XLength() / max_resolution;
    nxpixels = (int) (bbox.XLength() / pixel_spacing);
    nypixels = (int) (bbox.YLength() / pixel_spacing);
  }
  if (nypixels > max_resolution) {
    pixel_spacing = bbox.YLength() / max_resolution;
    nxpixels = (int) (bbox.XLength() / pixel_spacing);
    nypixels = (int) (bbox.YLength() / pixel_spacing);
  }

  // Set grid resolution
  grid.Resample(nxpixels, nypixels);

  // Set grid transformation
  grid.SetWorldToGridTransformation(bbox);

  // Set initial values
  grid.Clear(initial_value);
}



static R2Grid *
ReadGrid(const char *directory_name, const char *category_name, const char *field_name)
{
  // Create filename
  char filename[1024];
  sprintf(filename, "%s/%s_%s.grd", directory_name, category_name, field_name);

  // Allocate grid
  R2Grid *grid = new R2Grid();
  if (!grid) {
    fprintf(stderr, "Unable to allocate grid\n");
    return NULL;
  }

  // Read grid
  if (!grid->ReadFile(filename)) { 
    delete grid; 
    return NULL; 
  }

  // Return grid
  return grid;
}


static int
WriteGrid(const R2Grid& grid, 
  const char *directory_name, const char *category_name, const char *field_name)
{
  // Create filename
  char filename[1024];
  sprintf(filename, "%s/%s_%s.grd", directory_name, category_name, field_name);

  // Write grid
  return grid.WriteFile(filename);
}


static int
WritePlanarGrid(const R3PlanarGrid& grid, 
  const char *directory_name, const char *category_name, const char *field_name)
{
  // Create filename
  char filename[1024];
  sprintf(filename, "%s/%s_%s.grd", directory_name, category_name, field_name);

  // Write grid
  return grid.WriteFile(filename);
}


static int
WriteImage(const R2Grid& red, const R2Grid& green, const R2Grid& blue, 
  const char *directory_name, const char *category_name, const char *field_name)
{
  // Create image
  R2Image image(red.XResolution(), red.YResolution());

  // Fill image
  for (int j = 0; j < red.YResolution(); j++) {
    for (int i = 0; i < red.XResolution(); i++) {
      RNScalar r = red.GridValue(i, j);
      RNScalar g = green.GridValue(i, j);
      RNScalar b = blue.GridValue(i, j);
      image.SetPixelRGB(i, j, RNRgb(r, g, b));
    }
  }


  // Create filename
  char filename[1024];
  sprintf(filename, "%s/%s_%s.jpg", directory_name, category_name, field_name);

  // Write image
  return image.Write(filename);
}



////////////////////////////////////////////////////////////////////////
// Base image functions
////////////////////////////////////////////////////////////////////////

static int
WriteBaseGrids(R3SurfelScene *scene, const char *directory_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Creating base images ...\n");
    fflush(stdout);
  }
    
  // Create grids
  R2Grid count_grid; 
  InitializeOverheadGrid(count_grid, scene, pixel_spacing, max_resolution, R2_GRID_UNKNOWN_VALUE);
  R2Grid zmin_grid(count_grid);
  R2Grid zmax_grid(count_grid);
  R2Grid zmean_grid(count_grid);
  R2Grid radius_grid(count_grid);
  R2Grid nx_grid(count_grid);
  R2Grid ny_grid(count_grid);
  R2Grid nz_grid(count_grid);
  R2Grid horizontal_grid(count_grid);

  // Fill grids
  int node_count = 0;
  R3SurfelTree *tree = scene->Tree();
  if (!tree) return 0;
  R3SurfelDatabase *database = tree->Database();
  if (!database) return 0;
  RNArray<R3SurfelNode *> stack;
  stack.Insert(tree->RootNode());
  while (!stack.IsEmpty()) {
    R3SurfelNode *node = stack.Tail();
    stack.RemoveTail();

    // Print debug statement
    node_count++;
    if (print_debug) {
      static int next_node_count = 1;
      if (node_count == next_node_count) {
        int node_step = tree->NNodes() / 10;
        if (node_step < 100) node_step = 100;
        next_node_count += node_step;
        printf("%.0f%%\n", 100.0 * node_count / (double) tree->NNodes());
        fflush(stdout);
      }
    }
      
    // Check if node is not a leaf
    if (node->NParts() > 0) { // (node->NBlocks() == 0) || (node->Resolution() < 10 / pixel_spacing)) {
      // Decend into children
      for (int i = 0; i < node->NParts(); i++) {
        R3SurfelNode *part = node->Part(i);
        stack.Insert(part);
      }
    }
    else {
      // Prcoess blocks
      for (int i = 0; i < node->NBlocks(); i++) {
        R3SurfelBlock *block = node->Block(i);
        const R3Point& origin = block->Origin();

        // Read block
        database->ReadBlock(block);

        // Process surfels
        for (int j = 0; j < block->NSurfels(); j++) {
          const R3Surfel *surfel = block->Surfel(j);

          // Get world coordinates
          double px = origin.X() + surfel->X();
          double py = origin.Y() + surfel->Y();
          double pz = origin.Z() + surfel->Z();

          // Get normal
          float nx = surfel->NX();
          float ny = surfel->NY();
          float nz = surfel->NZ();

          // Get radius
          float radius = surfel->Radius();

          // Get grid coordinates
          R2Point grid_position = count_grid.GridPosition(R2Point(px, py));
          int ix = (int) (grid_position.X() + 0.5);
          int iy = (int) (grid_position.Y() + 0.5);
          if ((ix < 0) || (ix >= count_grid.XResolution())) continue;
          if ((iy < 0) || (iy >= count_grid.YResolution())) continue;

          // Update count grid
          count_grid.AddGridValue(ix, iy, 1);

          // Update zmin grid
          RNScalar zmin = zmin_grid.GridValue(ix, iy);
          if ((zmin == R2_GRID_UNKNOWN_VALUE) || (pz < zmin)) {
            zmin_grid.SetGridValue(ix, iy, pz);
          }
 
          // Update zmax grid (and rgb)
          RNScalar zmax = zmax_grid.GridValue(ix, iy);
          if ((zmax == R2_GRID_UNKNOWN_VALUE) || (pz > zmax)) {
            zmax_grid.SetGridValue(ix, iy, pz);
          }

          // Update other grids
          zmean_grid.AddGridValue(ix, iy, pz);
          nx_grid.AddGridValue(ix, iy, nx);
          ny_grid.AddGridValue(ix, iy, ny);
          nz_grid.AddGridValue(ix, iy, nz);
          radius_grid.AddGridValue(ix, iy, radius);
          horizontal_grid.AddGridValue(ix, iy, fabs(nz));
        }

        // Release block
        database->ReleaseBlock(block);
      }
    }
  }

  // Divide by counts to get averages
  zmean_grid.Divide(count_grid);
  nx_grid.Divide(count_grid);
  ny_grid.Divide(count_grid);
  nz_grid.Divide(count_grid);
  radius_grid.Divide(count_grid);
  horizontal_grid.Divide(count_grid);

  // Write grids
  if (!WriteGrid(count_grid, directory_name, "Base", "Count")) return 0;
  if (!WriteGrid(zmin_grid, directory_name, "Base", "ZMin")) return 0;
  if (!WriteGrid(zmax_grid, directory_name, "Base", "ZMax")) return 0;
  if (!WriteGrid(zmean_grid, directory_name, "Base", "ZMean")) return 0;
  if (!WriteGrid(nx_grid, directory_name, "Base", "NX")) return 0;
  if (!WriteGrid(ny_grid, directory_name, "Base", "NY")) return 0;
  if (!WriteGrid(nz_grid, directory_name, "Base", "NZ")) return 0;
  if (!WriteGrid(radius_grid, directory_name, "Base", "Radius")) return 0;
  if (!WriteGrid(horizontal_grid, directory_name, "Base", "Horizontal")) return 0;

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Resolution = %d %d\n", count_grid.XResolution(), count_grid.YResolution());
    printf("  Spacing = %g\n", count_grid.WorldToGridScaleFactor());
    fflush(stdout);
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Color image functions
////////////////////////////////////////////////////////////////////////

static int
WriteColorGrids(R3SurfelScene *scene, const char *directory_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Creating color images ...\n");
    fflush(stdout);
  }
    
  // Create grids
  R2Grid zmax_grid; 
  InitializeOverheadGrid(zmax_grid, scene, pixel_spacing, max_resolution, R2_GRID_UNKNOWN_VALUE);
  R2Grid red_grid(zmax_grid);
  R2Grid green_grid(zmax_grid);
  R2Grid blue_grid(zmax_grid);

  // Fill grids
  int node_count = 0;
  R3SurfelTree *tree = scene->Tree();
  if (!tree) return 0;
  R3SurfelDatabase *database = tree->Database();
  if (!database) return 0;
  RNArray<R3SurfelNode *> stack;
  stack.Insert(tree->RootNode());
  while (!stack.IsEmpty()) {
    R3SurfelNode *node = stack.Tail();
    stack.RemoveTail();

    // Print debug statement
    node_count++;
    if (print_debug) {
      static int next_node_count = 1;
      if (node_count == next_node_count) {
        int node_step = tree->NNodes() / 10;
        if (node_step < 100) node_step = 100;
        next_node_count += node_step;
        printf("%.0f%%\n", 100.0 * node_count / (double) tree->NNodes());
        fflush(stdout);
      }
    }
      
    // Check if node is not a leaf
    if (node->NParts() > 0) { // (node->NBlocks() == 0) || (node->Resolution() < 10 / pixel_spacing)) {
      // Decend into children
      for (int i = 0; i < node->NParts(); i++) {
        R3SurfelNode *part = node->Part(i);
        stack.Insert(part);
      }
    }
    else {
      // Prcoess blocks
      for (int i = 0; i < node->NBlocks(); i++) {
        R3SurfelBlock *block = node->Block(i);
        const R3Point& origin = block->Origin();

        // Read block
        database->ReadBlock(block);

        // Process surfels
        for (int j = 0; j < block->NSurfels(); j++) {
          const R3Surfel *surfel = block->Surfel(j);

          // Get surfel color
          RNRgb rgb = surfel->Rgb();

          // Get world coordinates
          double px = origin.X() + surfel->X();
          double py = origin.Y() + surfel->Y();
          double pz = origin.Z() + surfel->Z();

          // Get grid coordinates
          R2Point grid_position = zmax_grid.GridPosition(R2Point(px, py));
          int ix = (int) (grid_position.X() + 0.5);
          int iy = (int) (grid_position.Y() + 0.5);
          if ((ix < 0) || (ix >= zmax_grid.XResolution())) continue;
          if ((iy < 0) || (iy >= zmax_grid.YResolution())) continue;

          // Update zmax grid (and rgb)
          RNScalar zmax = zmax_grid.GridValue(ix, iy);
          if ((zmax == R2_GRID_UNKNOWN_VALUE) || (pz > zmax)) {
            zmax_grid.SetGridValue(ix, iy, pz);
            red_grid.SetGridValue(ix, iy, rgb.R());
            green_grid.SetGridValue(ix, iy, rgb.G());
            blue_grid.SetGridValue(ix, iy, rgb.B());
          }
        }

        // Release block
        database->ReleaseBlock(block);
      }
    }
  }

  // Write grids
  if (!WriteGrid(red_grid, directory_name, "Color", "Red")) return 0;
  if (!WriteGrid(green_grid, directory_name, "Color", "Green")) return 0;
  if (!WriteGrid(blue_grid, directory_name, "Color", "Blue")) return 0;
  if (!WriteImage(red_grid, green_grid, blue_grid, directory_name, "Color", "Rgb")) return 0;

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Resolution = %d %d\n", zmax_grid.XResolution(), zmax_grid.YResolution());
    printf("  Spacing = %g\n", zmax_grid.WorldToGridScaleFactor());
    fflush(stdout);
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Color image functions
////////////////////////////////////////////////////////////////////////

static int
WriteSliceGrids(R3SurfelScene *scene, const char *directory_name)
{
  // Parameters
  const RNScalar slice_spacing = 1.0;

  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Creating slice images ...\n");
    fflush(stdout);
  }

  // Get slice parameters
  const R3Box& bbox = scene->BBox();
  if (bbox.ZLength() == 0) return 0;
  int nslices = (int) (bbox.ZLength() / slice_spacing) + 1;
  if (nslices <= 1) return 0;
  RNScalar zmin = bbox.ZMin();
  RNScalar zscale = 1.0 / slice_spacing;

  // Create grids
  R2Grid *slice_grids = new R2Grid [nslices]; 
  for (int i = 0; i < nslices; i++) {
    InitializeOverheadGrid(slice_grids[i], scene, pixel_spacing, max_resolution, R2_GRID_UNKNOWN_VALUE);
  }

  // Fill grids
  int node_count = 0;
  R3SurfelTree *tree = scene->Tree();
  if (!tree) return 0;
  R3SurfelDatabase *database = tree->Database();
  if (!database) return 0;
  RNArray<R3SurfelNode *> stack;
  stack.Insert(tree->RootNode());
  while (!stack.IsEmpty()) {
    R3SurfelNode *node = stack.Tail();
    stack.RemoveTail();

    // Print debug statement
    node_count++;
    if (print_debug) {
      static int next_node_count = 1;
      if (node_count == next_node_count) {
        int node_step = tree->NNodes() / 10;
        if (node_step < 100) node_step = 100;
        next_node_count += node_step;
        printf("%.0f%%\n", 100.0 * node_count / (double) tree->NNodes());
        fflush(stdout);
      }
    }
      
    // Check if node is not a leaf
    if (node->NParts() > 0) { // (node->NBlocks() == 0) || (node->Resolution() < 10 / pixel_spacing)) {
      // Decend into children
      for (int i = 0; i < node->NParts(); i++) {
        R3SurfelNode *part = node->Part(i);
        stack.Insert(part);
      }
    }
    else {
      // Prcoess blocks
      for (int i = 0; i < node->NBlocks(); i++) {
        R3SurfelBlock *block = node->Block(i);
        const R3Point& origin = block->Origin();

        // Read block
        database->ReadBlock(block);

        // Process surfels
        for (int j = 0; j < block->NSurfels(); j++) {
          const R3Surfel *surfel = block->Surfel(j);

          // Get world coordinates
          double px = origin.X() + surfel->X();
          double py = origin.Y() + surfel->Y();
          double pz = origin.Z() + surfel->Z();

          // Get grid coordinates
          R2Point grid_position = slice_grids[0].GridPosition(R2Point(px, py));
          int ix = (int) (grid_position.X() + 0.5);
          int iy = (int) (grid_position.Y() + 0.5);
          if ((ix < 0) || (ix >= slice_grids[0].XResolution())) continue;
          if ((iy < 0) || (iy >= slice_grids[0].YResolution())) continue;

          // Get slice
          int slice = (int) (zscale * (pz - zmin));
          if (slice < 0) slice = 0;
          if (slice >= nslices) slice = nslices-1;

          // Update slice grid 
          slice_grids[slice].AddGridValue(ix, iy, 1.0);
        }

        // Release block
        database->ReleaseBlock(block);
      }
    }
  }

  // Write grids
  for (int i = 0; i < nslices; i++) {
    char buffer[4096];
    sprintf(buffer, "%d", i);
    if (!WriteGrid(slice_grids[i], directory_name, "Slice", buffer)) return 0;
  }

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Resolution = %d %d\n", slice_grids[0].XResolution(), slice_grids[0].YResolution());
    printf("  Spacing = %g %g\n", slice_grids[0].WorldToGridScaleFactor(), slice_spacing);
    printf("  # Slices = %d\n", nslices);
    fflush(stdout);
  }

  // Delete grids
  delete [] slice_grids;

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Chunk grid functions
////////////////////////////////////////////////////////////////////////

static int 
WritePixelGrids(R3SurfelScene *scene, const char *directory_name)
{
  // Parameters
  if (pixel_spacing == 0) pixel_spacing = 1;
  RNLength pixel_spacing_squared = pixel_spacing * pixel_spacing;
  RNLength region_radius = 2;
  RNLength chunk_diameter = 4 * region_radius;
  const int max_neighbors = 16;
  RNLength max_neighbor_distance = 1;
  if (max_neighbor_distance < pixel_spacing) max_neighbor_distance = pixel_spacing;
  R3Point cloud [ max_neighbors + 1];
  int min_points_per_pixel = 4;

  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Creating pixel images ...\n");
    printf("NOTE THAT THIS IS REALLY SLOW ...\n");
    fflush(stdout);
  }
    
  // Create grids
  R2Grid pixel_count_grid; 
  InitializeOverheadGrid(pixel_count_grid, scene, pixel_spacing, max_resolution, R2_GRID_UNKNOWN_VALUE);
  R2Grid graph_count_grid(pixel_count_grid); 
  R2Grid region_count_grid(pixel_count_grid); 
  R2Grid graph_radius_grid(pixel_count_grid);
  R2Grid zmin_grid(pixel_count_grid); 
  R2Grid zmax_grid(pixel_count_grid);
  R2Grid zmean_grid(pixel_count_grid);
  R2Grid zmedian_grid(pixel_count_grid);
  R2Grid zstddev_grid(pixel_count_grid);
  R2Grid zsupport_grid(pixel_count_grid);
  R2Grid pca0_grid(pixel_count_grid);
  R2Grid pca1_grid(pixel_count_grid);
  R2Grid pca2_grid(pixel_count_grid);
  R2Grid normal_x_grid(pixel_count_grid);
  R2Grid normal_y_grid(pixel_count_grid);
  R2Grid normal_z_grid(pixel_count_grid);

  // Compute chunking variables
  int nxchunks = (int) (scene->BBox().XLength() / chunk_diameter);
  int nychunks = (int) (scene->BBox().YLength() / chunk_diameter);
  if (nxchunks == 0) nxchunks = 1;
  if (nychunks == 0) nychunks = 1;
  int nxpixels = pixel_count_grid.XResolution() / nxchunks;
  int nypixels = pixel_count_grid.YResolution() / nychunks;
  if (nxpixels == 0) nxpixels = 1;
  if (nypixels == 0) nypixels = 1;

  // Compute grids chunk-by-chunk
  R3SurfelPointSet *last_chunk_pointset = NULL;
  for (int jc = 0; jc < nychunks; jc++) {
    for (int ic = 0; ic < nxchunks; ic++) {
      // Compute chunk bounding box
      R2Point p1 = pixel_count_grid.WorldPosition(    ic * nxpixels,     jc * nxpixels);
      R2Point p2 = pixel_count_grid.WorldPosition((ic+1) * nxpixels, (jc+1) * nxpixels);
      R3Box chunk_bbox(p1.X() - region_radius, p1.Y() - region_radius, -FLT_MAX, 
                       p2.X() + region_radius, p2.Y() + region_radius,  FLT_MAX);
      
      // Create chunk pointset
      R3SurfelBoxConstraint chunk_constraint(chunk_bbox);
      R3SurfelPointSet *chunk_pointset = CreatePointSet(scene, NULL, &chunk_constraint);
      if (!chunk_pointset) continue;
      if (chunk_pointset->NPoints() < min_points_per_pixel) { delete chunk_pointset; continue; }
      if (last_chunk_pointset) { delete last_chunk_pointset; last_chunk_pointset = NULL; }
      last_chunk_pointset = chunk_pointset;

      // Print debug statement
      if (print_debug) {
        printf("%4d/%4d %4d/%4d : %9.3f %9.3f : %12d\n", jc, nychunks, ic, nxchunks, 
          p1.X(), p1.Y(), chunk_pointset->NPoints());
        fflush(stdout);
      }

      // Compute grids within chunk pixel-by-pixel
      for (int iy = 0; iy < nypixels; iy++) {
        int grid_y = jc * nypixels + iy;
        for (int ix = 0; ix < nxpixels; ix++) {
          int grid_x = ic * nxpixels + ix;
          R2Point grid_position(grid_x, grid_y);
          R2Point twod_position = pixel_count_grid.WorldPosition(grid_position);
          R3Point world_position(twod_position[0], twod_position[1], chunk_bbox.Centroid().Z());

          // Create region pointset
          R3SurfelCylinderConstraint region_constraint(world_position, region_radius);
          R3SurfelPointSet *region_pointset = CreatePointSet(chunk_pointset, &region_constraint);
          if (!region_pointset) continue;
          if (region_pointset->NPoints() < min_points_per_pixel) { 
            delete region_pointset; 
            continue; 
          }

          // Create graph pointset
          R3SurfelCylinderConstraint graph_constraint(world_position, max_neighbor_distance);
          R3SurfelPointSet *graph_pointset = CreatePointSet(region_pointset, &graph_constraint);
          if (!graph_pointset) { 
            delete region_pointset; 
            continue; 
          }
          if (graph_pointset->NPoints() < min_points_per_pixel) { 
            delete region_pointset; 
            delete graph_pointset; 
            continue; 
          }

          // Create pixel pointset
          R3SurfelCylinderConstraint pixel_constraint(world_position, pixel_spacing);
          R3SurfelPointSet *pixel_pointset = CreatePointSet(graph_pointset, &pixel_constraint);
          if (!pixel_pointset) { 
            delete region_pointset; 
            delete graph_pointset;
            continue; 
          }
          if (pixel_pointset->NPoints() < min_points_per_pixel) { 
            delete region_pointset; 
            delete graph_pointset;
            delete pixel_pointset; 
            continue; 
          }

          // Create pixel graph
          R3SurfelPointGraph *pixel_graph = new R3SurfelPointGraph(*graph_pointset, max_neighbors, max_neighbor_distance);
          if (!pixel_graph) { 
            delete region_pointset; 
            delete graph_pointset; 
            delete pixel_pointset; 
            continue; 
          }
          if (pixel_graph->NPoints() < min_points_per_pixel) { 
            delete region_pointset; 
            delete graph_pointset; 
            delete pixel_pointset; 
            delete pixel_graph;
            continue; 
          }

          // Print debug statement
          // if (print_debug) {
          //   printf("    %4d/%-4d %4d/%-4d : %d %d : %12d %12d %12d\n", iy, nypixels, ix, nxpixels, grid_x, grid_y,
          //     region_pointset->NPoints(), graph_pointset->NPoints(), pixel_pointset->NPoints());
          //    fflush(stdout);
          // }

          // Gather height statistics
          RNCoord zsum = 0;
          RNCoord zmax = -FLT_MAX;
          RNCoord zmin = FLT_MAX;
          for (int i = 0; i < pixel_pointset->NPoints(); i++) {
            R3SurfelPoint *point = pixel_pointset->Point(i);
            R3Point position = point->Position();
            if (position.Z() > zmax) zmax = position.Z();
            if (position.Z() < zmin) zmin = position.Z();
            zsum += position.Z();
          }

          // Gather ssd statistics
          RNScalar zssd = 0;
          RNScalar zmean = zsum / pixel_pointset->NPoints();
          for (int i = 0; i < pixel_pointset->NPoints(); i++) {
            R3SurfelPoint *point = pixel_pointset->Point(i);
            R3Point position = point->Position();
            RNScalar delta = position.Z() - zmean;
            zssd += delta * delta;
          }

          // Compute variance statistics
          RNScalar zvariance = zssd / pixel_pointset->NPoints();
          RNScalar zstddev = sqrt(zvariance);

          // Compute median statistics
          RNScalar *zcoords = new RNScalar [ pixel_pointset->NPoints() ];
          for (int i = 0; i < pixel_pointset->NPoints(); i++) zcoords[i] = pixel_pointset->Point(i)->Z();
          qsort(zcoords, pixel_pointset->NPoints(), sizeof(RNScalar), RNCompareScalars);
          RNScalar zmedian = zcoords[pixel_pointset->NPoints() / 2];
          delete [] zcoords;

          // Compute support elevation
          RNCoord zsupport = 0;
          R3Plane plane = EstimateSupportPlane(region_pointset);
          if (plane[2] == 1) zsupport = -plane[3];
          else if (plane[2] != 0) zsupport = (plane[0]*world_position[0] + plane[1]*world_position[1] + plane[3]) / plane[2];

          // Sum graph stuff over all points in pixel
          int graph_count = 0;
          RNScalar graph_radius = 0;
          RNScalar graph_variances[3] = { 0, 0, 0 };
          R3Vector graph_normal(0, 0, 0);
          for (int i = 0; i < pixel_graph->NPoints(); i++) {
            if (pixel_graph->NNeighbors(i) < 2) continue;
            R3SurfelPoint *point = pixel_graph->Point(i);
            R3Point position = point->Position();
            RNScalar dx = twod_position[0] - position[0];
            RNScalar dy = twod_position[1] - position[1];
            if (dx*dx + dy*dy > pixel_spacing_squared) continue;
            graph_count++;

            // Create cloud of points in neighborhood
            cloud[0] = position;
            for (int j = 0; j < pixel_graph->NNeighbors(i); j++) {
              const R3SurfelPoint *neighbor = pixel_graph->Neighbor(i, j);
              cloud[j+1] = neighbor->Position();
            }

            // Increment stuff
            RNScalar variances[3];
            R3Point centroid = R3Centroid(pixel_graph->NNeighbors(i)+1, cloud);
            R3Triad triad = R3PrincipleAxes(centroid, pixel_graph->NNeighbors(i)+1, cloud, NULL, variances);
            graph_variances[0] += variances[0];
            graph_variances[1] += variances[1];
            graph_variances[2] += variances[2];
            graph_normal[0] += fabs(triad[2][0]);
            graph_normal[1] += fabs(triad[2][1]);
            graph_normal[2] += fabs(triad[2][2]);
            graph_radius += R3Distance(world_position, cloud[pixel_graph->NNeighbors(i)]);
          }

          // Divide graph stuff by count to get means
          graph_radius = (graph_count > 0) ? graph_radius / graph_count : R2_GRID_UNKNOWN_VALUE;
          graph_variances[0] = (graph_count > 0) ? graph_variances[0] / graph_count : R2_GRID_UNKNOWN_VALUE;
          graph_variances[1] = (graph_count > 0) ? graph_variances[1] / graph_count : R2_GRID_UNKNOWN_VALUE;
          graph_variances[2] = (graph_count > 0) ? graph_variances[2] / graph_count : R2_GRID_UNKNOWN_VALUE;
          graph_normal[0] = (graph_count > 0) ? graph_normal[0] / graph_count : R2_GRID_UNKNOWN_VALUE;
          graph_normal[1] = (graph_count > 0) ? graph_normal[1] / graph_count : R2_GRID_UNKNOWN_VALUE;
          graph_normal[2] = (graph_count > 0) ? graph_normal[2] / graph_count : R2_GRID_UNKNOWN_VALUE;

          // Update grid values
          pixel_count_grid.SetGridValue(grid_x, grid_y, pixel_pointset->NPoints());
          region_count_grid.SetGridValue(grid_x, grid_y, region_pointset->NPoints());
          zmin_grid.SetGridValue(grid_x, grid_y, zmin);
          zmax_grid.SetGridValue(grid_x, grid_y, zmax);
          zmean_grid.SetGridValue(grid_x, grid_y, zmean);
          zmedian_grid.SetGridValue(grid_x, grid_y, zmedian);
          zstddev_grid.SetGridValue(grid_x, grid_y, zstddev);
          zsupport_grid.SetGridValue(grid_x, grid_y, zsupport);
          pca0_grid.SetGridValue(grid_x, grid_y, graph_variances[0]);
          pca1_grid.SetGridValue(grid_x, grid_y, graph_variances[1]);
          pca2_grid.SetGridValue(grid_x, grid_y, graph_variances[2]);
          normal_x_grid.SetGridValue(grid_x, grid_y, graph_normal[0]);
          normal_y_grid.SetGridValue(grid_x, grid_y, graph_normal[1]);
          normal_z_grid.SetGridValue(grid_x, grid_y, graph_normal[2]);
          graph_count_grid.SetGridValue(grid_x, grid_y, graph_count);
          graph_radius_grid.SetGridValue(grid_x, grid_y, graph_radius);
          
          // Delete pixel graph
          delete pixel_graph;

          // Delete pixel point set
          delete pixel_pointset;

          // Delete graph point set
          delete graph_pointset;

          // Delete neighborhood pointset
          delete region_pointset;
        }
      }
    }
  }

  // Delete last chunk pointset
  if (last_chunk_pointset) { 
    delete last_chunk_pointset; 
    last_chunk_pointset = NULL; 
  }

  // Write grids
  if (!WriteGrid(pixel_count_grid, directory_name, "Pixel", "PixelCount")) return 0;
  if (!WriteGrid(graph_count_grid, directory_name, "Pixel", "GraphCount")) return 0;
  if (!WriteGrid(region_count_grid, directory_name, "Pixel", "RegionCount")) return 0;
  if (!WriteGrid(graph_radius_grid, directory_name, "Graph", "GraphRadius")) return 0;
  if (!WriteGrid(zmin_grid, directory_name, "Pixel", "ZMin")) return 0;
  if (!WriteGrid(zmax_grid, directory_name, "Pixel", "ZMax")) return 0;
  if (!WriteGrid(zmean_grid, directory_name, "Pixel", "ZMean")) return 0;
  if (!WriteGrid(zmedian_grid, directory_name, "Pixel", "ZMedian")) return 0;
  if (!WriteGrid(zstddev_grid, directory_name, "Pixel", "ZStddev")) return 0;
  if (!WriteGrid(zsupport_grid, directory_name, "Pixel", "ZSupport")) return 0;
  if (!WriteGrid(pca0_grid, directory_name, "Pixel", "PCA0")) return 0;
  if (!WriteGrid(pca1_grid, directory_name, "Pixel", "PCA1")) return 0;
  if (!WriteGrid(pca2_grid, directory_name, "Pixel", "PCA2")) return 0;
  if (!WriteGrid(normal_x_grid, directory_name, "Pixel", "NormalX")) return 0;
  if (!WriteGrid(normal_y_grid, directory_name, "Pixel", "NormalY")) return 0;
  if (!WriteGrid(normal_z_grid, directory_name, "Pixel", "NormalZ")) return 0;

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Resolution = %d %d\n", pixel_count_grid.XResolution(), pixel_count_grid.YResolution());
    printf("  Spacing = %g\n", pixel_count_grid.WorldToGridScaleFactor());
    fflush(stdout);
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Heights grid functions
////////////////////////////////////////////////////////////////////////

static int 
WriteHeightGrids(R3SurfelScene *scene, const char *directory_name)
{
  // Parameters
  if (pixel_spacing == 0) pixel_spacing = 1;
  RNLength neighborhood_radius = 2;
  RNLength chunk_diameter = 4 * neighborhood_radius;
  int min_points_per_pixel = 4;

  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Creating height images ...\n");
    printf("NOTE THAT THIS IS REALLY SLOW ...\n");
    fflush(stdout);
  }
    
  // Create grids
  R2Grid pixel_count_grid; 
  InitializeOverheadGrid(pixel_count_grid, scene, pixel_spacing, max_resolution, R2_GRID_UNKNOWN_VALUE);
  R2Grid neighborhood_count_grid(pixel_count_grid); 
  R2Grid zmin_grid(pixel_count_grid); 
  R2Grid zmax_grid(pixel_count_grid);
  R2Grid zmean_grid(pixel_count_grid);
  R2Grid zmedian_grid(pixel_count_grid);
  R2Grid zstddev_grid(pixel_count_grid);
  R2Grid zsupport_grid(pixel_count_grid);

  // Compute chunking variables
  int nxchunks = (int) (scene->BBox().XLength() / chunk_diameter);
  int nychunks = (int) (scene->BBox().YLength() / chunk_diameter);
  if (nxchunks == 0) nxchunks = 1;
  if (nychunks == 0) nychunks = 1;
  int nxpixels = pixel_count_grid.XResolution() / nxchunks;
  int nypixels = pixel_count_grid.YResolution() / nychunks;
  if (nxpixels == 0) nxpixels = 1;
  if (nypixels == 0) nypixels = 1;

  // Compute grids chunk-by-chunk
  R3SurfelPointSet *last_chunk_pointset = NULL;
  for (int jc = 0; jc < nychunks; jc++) {
    for (int ic = 0; ic < nxchunks; ic++) {
      // Compute chunk bounding box
      R2Point p1 = pixel_count_grid.WorldPosition(    ic * nxpixels,     jc * nxpixels);
      R2Point p2 = pixel_count_grid.WorldPosition((ic+1) * nxpixels, (jc+1) * nxpixels);
      R3Box chunk_bbox(p1.X(), p1.Y(), -FLT_MAX, p2.X(), p2.Y(), FLT_MAX);

      // Create chunk pointset
      R3SurfelBoxConstraint chunk_constraint(chunk_bbox);
      R3SurfelPointSet *chunk_pointset = CreatePointSet(scene, NULL, &chunk_constraint);
      if (!chunk_pointset) continue;
      if (chunk_pointset->NPoints() < min_points_per_pixel) { delete chunk_pointset; continue; }
      if (last_chunk_pointset) { delete last_chunk_pointset; last_chunk_pointset = NULL; }
      last_chunk_pointset = chunk_pointset;

      // Print debug statement
      if (print_debug) {
        printf("%4d/%4d %4d/%4d : %9.3f %9.3f : %12d\n", jc, nychunks, ic, nxchunks, 
          p1.X(), p1.Y(), chunk_pointset->NPoints());
        fflush(stdout);
      }

      // Compute grids within chunk pixel-by-pixel
      for (int iy = 0; iy < nypixels; iy++) {
        int grid_y = jc * nypixels + iy;
        for (int ix = 0; ix < nxpixels; ix++) {
          int grid_x = ic * nxpixels + ix;
          R2Point grid_position(grid_x, grid_y);
          R2Point twod_position = pixel_count_grid.WorldPosition(grid_position);
          R3Point world_position(twod_position[0], twod_position[1], chunk_bbox.Centroid().Z());

          // Create neighborhood pointset
          R3SurfelCylinderConstraint neighborhood_constraint(world_position, neighborhood_radius);
          R3SurfelPointSet *neighborhood_pointset = CreatePointSet(chunk_pointset, &neighborhood_constraint);
          if (!neighborhood_pointset) continue;
          if (neighborhood_pointset->NPoints() < min_points_per_pixel) { 
            delete neighborhood_pointset; 
            continue; 
          }

          // Create pixel pointset
          RNLength pixel_radius = pixel_count_grid.GridToWorldScaleFactor();
          R3SurfelCylinderConstraint pixel_constraint(world_position, pixel_radius);
          R3SurfelPointSet *pixel_pointset = CreatePointSet(neighborhood_pointset, &pixel_constraint);
          if (!pixel_pointset) { 
            delete neighborhood_pointset; 
            continue; 
          }
          if (pixel_pointset->NPoints() < min_points_per_pixel) { 
            delete neighborhood_pointset; 
            delete pixel_pointset; 
            continue; 
          }

          // Print debug statement
          // if (print_debug) {
          //   printf("  %4d/%-4d %4d/%-4d : %d %d : %12d %12d\n", iy, nypixels, ix, nxpixels, grid_x, grid_y,
          //     neighborhood_pointset->NPoints(), pixel_pointset->NPoints());
          //   fflush(stdout);
          // }

          // Gather count statistics
          RNScalar pixel_count = pixel_pointset->NPoints();
          RNScalar neighborhood_count = neighborhood_pointset->NPoints();

          // Gather height statistics
          RNCoord zsum = 0;
          RNCoord zmax = -FLT_MAX;
          RNCoord zmin = FLT_MAX;
          for (int i = 0; i < pixel_pointset->NPoints(); i++) {
            R3SurfelPoint *point = pixel_pointset->Point(i);
            R3Point position = point->Position();
            if (position.Z() > zmax) zmax = position.Z();
            if (position.Z() < zmin) zmin = position.Z();
            zsum += position.Z();
          }

          // Gather ssd statistics
          RNScalar zssd = 0;
          RNScalar zmean = zsum / pixel_pointset->NPoints();
          for (int i = 0; i < pixel_pointset->NPoints(); i++) {
            R3SurfelPoint *point = pixel_pointset->Point(i);
            R3Point position = point->Position();
            RNScalar delta = position.Z() - zmean;
            zssd += delta * delta;
          }

          // Compute variance statistics
          RNScalar zvariance = zssd / pixel_pointset->NPoints();
          RNScalar zstddev = sqrt(zvariance);

          // Compute median statistics
          RNScalar *zcoords = new RNScalar [ pixel_pointset->NPoints() ];
          for (int i = 0; i < pixel_pointset->NPoints(); i++) zcoords[i] = pixel_pointset->Point(i)->Z();
          qsort(zcoords, pixel_pointset->NPoints(), sizeof(RNScalar), RNCompareScalars);
          RNScalar zmedian = zcoords[pixel_pointset->NPoints() / 2];
          delete [] zcoords;

          // Compute support elevation
          RNCoord zsupport = 0;
          R3Plane plane = EstimateSupportPlane(neighborhood_pointset);
          if (plane[2] == 1) zsupport = -plane[3];
          else if (plane[2] != 0) zsupport = (plane[0]*world_position[0] + plane[1]*world_position[1] + plane[3]) / plane[2];

          // Assign grid values
          pixel_count_grid.SetGridValue(grid_x, grid_y, pixel_count);
          neighborhood_count_grid.SetGridValue(grid_x, grid_y, neighborhood_count);
          zmin_grid.SetGridValue(grid_x, grid_y, zmin);
          zmax_grid.SetGridValue(grid_x, grid_y, zmax);
          zmean_grid.SetGridValue(grid_x, grid_y, zmean);
          zmedian_grid.SetGridValue(grid_x, grid_y, zmedian);
          zstddev_grid.SetGridValue(grid_x, grid_y, zstddev);
          zsupport_grid.SetGridValue(grid_x, grid_y, zsupport);

          // Delete pixel point set
          delete pixel_pointset;

          // Delete neighborhood pointset
          delete neighborhood_pointset;
        }
      }
    }
  }

  // Delete last chunk pointset
  if (last_chunk_pointset) { 
    delete last_chunk_pointset; 
    last_chunk_pointset = NULL; 
  }

  // Write grids
  if (!WriteGrid(pixel_count_grid, directory_name, "Height", "Count")) return 0;
  if (!WriteGrid(neighborhood_count_grid, directory_name, "Height", "NeighborhoodCount")) return 0;
  if (!WriteGrid(zmin_grid, directory_name, "Height", "ZMin")) return 0;
  if (!WriteGrid(zmax_grid, directory_name, "Height", "ZMax")) return 0;
  if (!WriteGrid(zmean_grid, directory_name, "Height", "ZMean")) return 0;
  if (!WriteGrid(zmedian_grid, directory_name, "Height", "ZMedian")) return 0;
  if (!WriteGrid(zstddev_grid, directory_name, "Height", "ZStddev")) return 0;
  if (!WriteGrid(zsupport_grid, directory_name, "Height", "ZSupport")) return 0;

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Resolution = %d %d\n", pixel_count_grid.XResolution(), pixel_count_grid.YResolution());
    printf("  Spacing = %g\n", pixel_count_grid.WorldToGridScaleFactor());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int 
WriteGraphGrids(R3SurfelScene *scene, const char *directory_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Creating graph images ...\n");
    printf("NOTE THAT THIS IS REALLY SLOW ...\n");
    fflush(stdout);
  }
    
  // Parameters
  int tile_xpixels = 20;
  int tile_ypixels = 20;
  RNScalar horizontal_tolerance = cos(10.0*RN_PI/180.0);
  RNScalar vertical_tolerance = sin(10.0*RN_PI/180.0);
  const int max_neighbor_count = 16;
  RNLength max_neighbor_distance = 1;
  R3Point cloud [ max_neighbor_count + 1];

  // Allocate grids
  R2Grid count_grid; InitializeOverheadGrid(count_grid, scene, pixel_spacing, max_resolution, 0);
  R2Grid zmax_grid(count_grid);  zmax_grid.Clear(R2_GRID_UNKNOWN_VALUE);
  R2Grid zmin_grid(count_grid);  zmin_grid.Clear(R2_GRID_UNKNOWN_VALUE);
  R2Grid zsum_grid(count_grid);
  R2Grid pca0_grid(count_grid);
  R2Grid pca1_grid(count_grid);
  R2Grid pca2_grid(count_grid);
  R2Grid normal_x_grid(count_grid);
  R2Grid normal_y_grid(count_grid);
  R2Grid normal_z_grid(count_grid);
  R2Grid horizontal_grid(count_grid);
  R2Grid vertical_grid(count_grid);
  R2Grid oblique_grid(count_grid);
  R2Grid planar_grid(count_grid);
  R2Grid neighbor_radius_grid(count_grid);

  // Compute grid by tiles
  R3SurfelPointSet *last_pointset = NULL;
  int tile_xcount = count_grid.XResolution() / tile_xpixels;
  int tile_ycount = count_grid.YResolution() / tile_ypixels;
  if (tile_xcount * tile_xpixels < count_grid.XResolution()) tile_xcount++;
  if (tile_ycount * tile_ypixels < count_grid.YResolution()) tile_ycount++;
  for (int tile_i = 0; tile_i < tile_xcount; tile_i++) {
    for (int tile_j = 0; tile_j < tile_ycount; tile_j++) {
      int tile_ix1 = tile_i*tile_xpixels;
      int tile_iy1 = tile_j*tile_ypixels;
      int tile_ix2 = (tile_i+1)*tile_xpixels;
      int tile_iy2 = (tile_j+1)*tile_ypixels;
      if (tile_ix2 > count_grid.XResolution()) tile_ix2 = count_grid.XResolution();
      if (tile_iy2 > count_grid.YResolution()) tile_iy2 = count_grid.YResolution();
      R2Point tile_ip1(tile_ix1, tile_iy1);
      R2Point tile_ip2(tile_ix2, tile_iy2);
      R2Point tile_p1 = count_grid.WorldPosition(tile_ip1);
      R2Point tile_p2 = count_grid.WorldPosition(tile_ip2);
      R3Box tile_box(tile_p1[0], tile_p1[1], -FLT_MAX, tile_p2[0], tile_p2[1], FLT_MAX);
      R3Vector tile_overlap(max_neighbor_distance, max_neighbor_distance, 0);
      R3Box expanded_box(tile_box.Min() - tile_overlap, tile_box.Max() + tile_overlap);
      R3SurfelBoxConstraint box_constraint(expanded_box);
      R3SurfelPointSet *tile_pointset = CreatePointSet(scene, NULL, &box_constraint);
      if (!tile_pointset || (tile_pointset->NPoints() == 0)) continue;
      R3SurfelPointGraph *tile_graph = new R3SurfelPointGraph(*tile_pointset, max_neighbor_count, max_neighbor_distance);
      if (!tile_graph || (tile_graph->NPoints() == 0)) { delete tile_pointset; continue; }
      if (print_debug) printf("%3d/%3d %3d/%3d %9d\n", tile_i, tile_xcount, tile_j, tile_ycount, tile_graph->NPoints());
      for (int i = 0; i < tile_graph->NPoints(); i++) {
        R3SurfelPoint *point = tile_graph->Point(i);
        R3Point position(point->Position());
        if (!R3Contains(tile_box, position)) continue;
        if (tile_graph->NNeighbors(i) < 3) continue;
        R2Point grid_position = count_grid.GridPosition(R2Point(position[0], position[1]));
        int ix = (int) (grid_position.X() + 0.5);
        int iy = (int) (grid_position.Y() + 0.5);

        // Create cloud of points in neighborhood
        cloud[0] = position;
        for (int j = 0; j < tile_graph->NNeighbors(i); j++) {
          const R3SurfelPoint *neighbor = tile_graph->Neighbor(i, j);
          cloud[j+1] = neighbor->Position();
        }

        // Compute normal with PCA of neighborhood
        RNScalar variances[3] = { 0, 0, 0 };
        R3Point centroid = R3Centroid(tile_graph->NNeighbors(i)+1, cloud);
        R3Triad triad = R3PrincipleAxes(centroid, tile_graph->NNeighbors(i)+1, cloud, NULL, variances);
        R3Vector normal = triad[2];
        
        // Compute neighborhood radius
        RNScalar neighborhood_radius = R3Distance(cloud[0], cloud[tile_graph->NNeighbors(i)]);

        // Update grids
        count_grid.AddGridValue(ix, iy, 1);
        zsum_grid.AddGridValue(ix, iy, point->Z());
        if ((zmax_grid.GridValue(ix, iy) == R2_GRID_UNKNOWN_VALUE) || (point->Z() > zmax_grid.GridValue(ix, iy)))
          zmax_grid.SetGridValue(ix, iy, point->Z());
        if ((zmin_grid.GridValue(ix, iy) == R2_GRID_UNKNOWN_VALUE) || (point->Z() < zmin_grid.GridValue(ix, iy)))
          zmin_grid.SetGridValue(ix, iy, point->Z());
        pca0_grid.AddGridValue(ix, iy, variances[0]);
        pca1_grid.AddGridValue(ix, iy, variances[1]);
        pca2_grid.AddGridValue(ix, iy, variances[2]);
        normal_x_grid.AddGridValue(ix, iy, fabs(normal.X()));
        normal_y_grid.AddGridValue(ix, iy, fabs(normal.Y()));
        normal_z_grid.AddGridValue(ix, iy, fabs(normal.Z()));
        neighbor_radius_grid.AddGridValue(ix, iy, neighborhood_radius);
        if ((variances[1] > 0.04) && (variances[2] < 0.01)) {
          if (normal.Z() > horizontal_tolerance) horizontal_grid.AddGridValue(ix, iy, 1);
          else if (fabs(normal.Z()) < vertical_tolerance) vertical_grid.AddGridValue(ix, iy, 1);
          else oblique_grid.AddGridValue(ix, iy, 1);
          planar_grid.AddGridValue(ix, iy, 1);
        }
      }

      // Delete temporary memory
      delete tile_graph;
      if (last_pointset) delete last_pointset;
      last_pointset = tile_pointset;
    }
  }

  // Delete last pointset
  if (last_pointset) delete last_pointset;

  // Divide by counts to produce averages
  R2Grid denominator_grid(count_grid);
  denominator_grid.Threshold(RN_EPSILON, R2_GRID_UNKNOWN_VALUE, R2_GRID_KEEP_VALUE);
  zsum_grid.Divide(denominator_grid); 
  pca0_grid.Divide(denominator_grid);
  pca1_grid.Divide(denominator_grid);
  pca2_grid.Divide(denominator_grid);
  normal_x_grid.Divide(denominator_grid);
  normal_y_grid.Divide(denominator_grid);
  normal_z_grid.Divide(denominator_grid);
  horizontal_grid.Divide(denominator_grid);
  vertical_grid.Divide(denominator_grid);
  oblique_grid.Divide(denominator_grid);
  oblique_grid.Divide(denominator_grid);
  neighbor_radius_grid.Divide(denominator_grid);

  // Write grids
  if (!WriteGrid(count_grid, directory_name, "Graph", "Count")) return 0;
  if (!WriteGrid(zmin_grid, directory_name, "Graph", "ZMin")) return 0;
  if (!WriteGrid(zmax_grid, directory_name, "Graph", "ZMax")) return 0;
  if (!WriteGrid(zsum_grid, directory_name, "Graph", "ZSum")) return 0;
  if (!WriteGrid(pca0_grid, directory_name, "Graph", "PCA0")) return 0;
  if (!WriteGrid(pca1_grid, directory_name, "Graph", "PCA1")) return 0;
  if (!WriteGrid(pca2_grid, directory_name, "Graph", "PCA2")) return 0;
  if (!WriteGrid(normal_x_grid, directory_name, "Graph", "NormalX")) return 0;
  if (!WriteGrid(normal_y_grid, directory_name, "Graph", "NormalY")) return 0;
  if (!WriteGrid(normal_z_grid, directory_name, "Graph", "NormalZ")) return 0;
  if (!WriteGrid(horizontal_grid, directory_name, "Graph", "Horizontal")) return 0;
  if (!WriteGrid(vertical_grid, directory_name, "Graph", "Vertical")) return 0;
  if (!WriteGrid(oblique_grid, directory_name, "Graph", "Oblique")) return 0;
  if (!WriteGrid(planar_grid, directory_name, "Graph", "Planar")) return 0;
  if (!WriteGrid(neighbor_radius_grid, directory_name, "Graph", "NeighborRadius")) return 0;
  if (!WriteGrid(denominator_grid, directory_name, "Graph", "Denominator")) return 0;

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Resolution = %d %d\n", zmin_grid.XResolution(), zmin_grid.YResolution());
    printf("  Spacing = %g\n", zmin_grid.WorldToGridScaleFactor());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int 
WritePlanarGrids(R3SurfelScene *scene, const char *directory_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Creating planar grids ...\n");
    fflush(stdout);
  }
  if (print_debug) {
    printf("  ");
    fflush(stdout);
  }

  // Get convenient variables
  const R3Box& scene_bbox = scene->BBox();
  if (chunk_size <= 0) chunk_size = scene_bbox.DiagonalLength() / 10;
  if (chunk_size <= 0) return 0;
  int nxchunks = (int) (scene_bbox.XLength() / chunk_size) + 1;
  int nychunks = (int) (scene_bbox.YLength() / chunk_size) + 1;
  RNLength xchunk = scene_bbox.XLength() / nxchunks;
  RNLength ychunk = scene_bbox.YLength() / nychunks;
  char name[256];

  // Fill array of grids chunk-by-chunk
  int grid_count = 0;
  for (int j = 0; j < nychunks; j++) {
    for (int i = 0; i < nxchunks; i++) {
      // Compute chunk bounding box
      R3Box chunk_bbox = scene_bbox;
      chunk_bbox[0][0] = scene_bbox.XMin() + i     * xchunk;
      chunk_bbox[1][0] = scene_bbox.XMin() + (i+1) * xchunk;
      chunk_bbox[0][1] = scene_bbox.YMin() + j     * ychunk;
      chunk_bbox[1][1] = scene_bbox.YMin() + (j+1) * ychunk;

      // Compute grids for chunk
      R3SurfelBoxConstraint box_constraint(chunk_bbox);
      RNArray<R3PlanarGrid *> *chunk_grids = CreatePlanarGrids(scene, NULL, &box_constraint);
      if (!chunk_grids) continue;

      // Write/delete  grids for chunk
      for (int k = 0; k < chunk_grids->NEntries(); k++) {
        R3PlanarGrid *grid = chunk_grids->Kth(k);
        sprintf(name, "%d", grid_count);
        if (!WritePlanarGrid(*grid, directory_name, "Planar", name)) return 0;
        if (print_debug) {
          const R3Plane& plane = grid->Plane();
          printf("  %9d : %4d/%4d %4d/%4d %4d/%4d : %6.3f %6.3f %6.3f %12.3f : %12.3f\n", 
                 grid_count, i, nxchunks, j, nychunks, k, chunk_grids->NEntries(),
                 plane[0], plane[1], plane[2], plane[3], grid->L1Norm());
        }
        delete grid;
        grid_count++;
      }

      // Delete grids for chunk
      delete chunk_grids;
    }
  }

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Grids = %d\n", grid_count);
    fflush(stdout);
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Horizontal grids
////////////////////////////////////////////////////////////////////////

struct HorizontalCluster {
  HorizontalCluster *parent;
  RNArray<struct HorizontalClusterPair *> pairs;
  RNArray<const RNScalar *> pixels;
  RNScalar total_x, total_y, total_z;
  RNScalar total_dzdx, total_dzdy;
};

struct HorizontalClusterPair {
  HorizontalCluster *clusters[2];
  RNScalar affinity;
  HorizontalClusterPair **heapentry;
};


static RNScalar
HorizontalClusterAffinity(HorizontalCluster *cluster0, HorizontalCluster *cluster1,
  RNScalar min_dzdxy,
  RNScalar max_dzdxy,
  RNScalar max_z_difference,
  RNScalar max_xy_difference,
  RNScalar max_dzdxy_difference,
  RNScalar max_area)
{
  // Get convenient variables
  RNScalar n0 = cluster0->pixels.NEntries();
  RNScalar n1 = cluster1->pixels.NEntries();
  if ((n0 == 0) || (n1 == 0)) return FLT_MAX;

  // Check number of pixels
  if (max_area > 0) {
    if (n0 + n1 > max_area) return FLT_MAX;
  }

  // Compute means for cluster0
  RNScalar mean_x0 = cluster0->total_x / n0;
  RNScalar mean_y0 = cluster0->total_y / n0;
  RNScalar mean_z0 = cluster0->total_z / n0;
  RNScalar mean_dzdx0 = cluster0->total_dzdx / n0;
  RNScalar mean_dzdy0 = cluster0->total_dzdy / n0;

  // Compute means for cluster1
  RNScalar mean_x1 = cluster1->total_x / n1;
  RNScalar mean_y1 = cluster1->total_y / n1;
  RNScalar mean_z1 = cluster1->total_z / n1;
  RNScalar mean_dzdx1 = cluster1->total_dzdx / n1;
  RNScalar mean_dzdy1 = cluster1->total_dzdy / n1;

  // Compute dzdxy 
  if ((min_dzdxy > 0) || (max_dzdxy > 0)) {
    RNScalar t0 = n0 / (n0 + n1);
    RNScalar t1 = n1 / (n0 + n1);
    RNScalar dzdx = t0*mean_dzdx0 + t1*mean_dzdx1;
    RNScalar dzdy = t0*mean_dzdy0 + t1*mean_dzdy1;
    RNScalar dzdxy = sqrt(dzdx*dzdx + dzdy*dzdy);
    if ((min_dzdxy > 0) && (dzdxy < min_dzdxy)) return FLT_MAX;
    if ((max_dzdxy > 0) && (dzdxy > max_dzdxy)) return FLT_MAX;
  }

  // Compute dzdxy difference
  RNScalar dzdxy_difference_factor = 1;
  if (max_dzdxy_difference > 0) {
    RNScalar dx = mean_dzdx0 - mean_dzdx1;
    RNScalar dy = mean_dzdy0 - mean_dzdy1;
    RNScalar dzdxy_difference = sqrt(dx*dx + dy*dy);
    if (dzdxy_difference > max_dzdxy_difference) return FLT_MAX;
    dzdxy_difference_factor = 1.0 - dzdxy_difference / max_dzdxy_difference;
  }

  // Compute z difference
  RNScalar z_difference_factor = 1;
  if (max_z_difference > 0) {
    RNScalar z_difference = fabs(mean_z0 - mean_z1);
    if (z_difference > max_z_difference) return FLT_MAX;
    z_difference_factor = 1.0 - z_difference / max_z_difference;
  }

  // Compute xy difference
  RNScalar xy_difference_factor = 1;
  if (max_xy_difference > 0) {
    RNScalar dx = mean_x0 - mean_x1;
    RNScalar dy = mean_y0 - mean_y1;
    RNScalar xy_difference = sqrt(dx*dx + dy*dy);
    if (xy_difference > max_xy_difference) return FLT_MAX;
    xy_difference_factor = 1.0 - xy_difference / max_xy_difference;
  }

  // Compute affinity
  RNScalar affinity = 1;
  affinity *= z_difference_factor;
  affinity *= xy_difference_factor;
  affinity *= dzdxy_difference_factor;

  // Return affinity
  return affinity;
}



static int
CreateHorizontalClusterGrids(const R2Grid& input_grid,
  R2Grid& cluster_index_grid, R2Grid& cluster_size_grid,
  RNScalar min_affinity,
  RNScalar min_dzdxy,
  RNScalar max_dzdxy,
  RNScalar max_z_difference,
  RNScalar max_xy_difference,
  RNScalar max_dzdxy_difference,
  RNScalar max_area)
{
  ////////////////////////////////////////////////////////////////////////

  // printf("HEREA\n");

  // Create gradient grids
  R2Grid *dx_grid = new R2Grid(input_grid);
  R2Grid *dy_grid = new R2Grid(input_grid);
  dx_grid->Gradient(RN_X);
  dy_grid->Gradient(RN_Y);

  // Create clusters
  HorizontalCluster *clusters = new HorizontalCluster [ input_grid.NEntries() ];
  const RNScalar *input_grid_values = input_grid.GridValues();
  for (int iy = 0; iy < input_grid.YResolution(); iy++) {
    for (int ix = 0; ix < input_grid.XResolution(); ix++) {
      int index;
      input_grid.IndicesToIndex(ix, iy, index);
      const RNScalar *pixel = &input_grid_values[index];
      HorizontalCluster *cluster = &clusters[index];
      cluster->parent = NULL;
      cluster->pixels.Insert(pixel);
      cluster->total_x = ix;
      cluster->total_y = iy;
      cluster->total_z = *pixel;
      cluster->total_dzdx = dx_grid->GridValue(ix, iy);
      cluster->total_dzdy = dy_grid->GridValue(ix, iy);
    }
  }

  // Delete gradient grids
  delete dx_grid;
  delete dy_grid;

  ////////////////////////////////////////////////////////////////////////
  
  // printf("HEREB %d\n", input_grid.NEntries());

  // Create cluster pairs 
  HorizontalClusterPair tmp;
  RNHeap<HorizontalClusterPair *> heap(&tmp, &tmp.affinity, &tmp.heapentry, FALSE);;
  for (int iy0 = 0; iy0 < input_grid.YResolution(); iy0++) {
    for (int ix0 = 0; ix0 < input_grid.XResolution(); ix0++) {
      int index0;
      input_grid.IndicesToIndex(ix0, iy0, index0);
      HorizontalCluster *cluster0 = &clusters[index0];
      for (int iy1 = iy0-1; iy1 <= iy0+1; iy1++) {
        if ((iy1 < 0) || (iy1 >= input_grid.YResolution())) continue;
        for (int ix1 = ix0-1; ix1 <= ix0+1; ix1++) {
          if ((ix1 < 0) || (ix1 >= input_grid.XResolution())) continue;
          if ((ix1 == ix0) && (iy1 == iy0)) continue;
          int index1;
          input_grid.IndicesToIndex(ix1, iy1, index1);
          HorizontalCluster *cluster1 = &clusters[index1];

          // Compute affinity
          RNScalar affinity = HorizontalClusterAffinity(cluster0, cluster1,
             min_dzdxy, max_dzdxy, 
             max_z_difference, max_xy_difference, max_dzdxy_difference, 
             max_area);
          if (affinity == FLT_MAX) continue;

          // Create pair
          HorizontalClusterPair *pair = new HorizontalClusterPair();
          pair->clusters[0] = cluster0;
          pair->clusters[1] = cluster1;
          pair->affinity = affinity;
          pair->heapentry = NULL;
          heap.Push(pair);

          // Add pair to clusters
          cluster0->pairs.Insert(pair);
          cluster1->pairs.Insert(pair);
        }
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////

  // printf("HEREC %d\n", heap.NEntries());

  // Merge clusters hierarchically
  int merge_count = 0;
  while (!heap.IsEmpty()) {
    // Get pair
    HorizontalClusterPair *pair01 = heap.Pop();

    // Check if we are done
    if (pair01->affinity < min_affinity) break;

    // Get clusters
    HorizontalCluster *cluster0 = pair01->clusters[0];
    HorizontalCluster *cluster1 = pair01->clusters[1];

    if (print_debug) {
      static unsigned long count = 0;
      if ((count++ % 1000) == 0) {
        printf("  %15.12f : %9d %9d : %15d\n", pair01->affinity, 
               cluster0->pixels.NEntries(), cluster1->pixels.NEntries(), 
               heap.NEntries());
      }
    }
    
    // Check if clusters have already been merged
    if (cluster0->parent || cluster1->parent) {
      // Find clusters
      HorizontalCluster *ancestor0 = cluster0;
      HorizontalCluster *ancestor1 = cluster1;
      while (ancestor0->parent) ancestor0 = ancestor0->parent;
      while (ancestor1->parent) ancestor1 = ancestor1->parent;
      if (ancestor0 != ancestor1) {
        // Find pair
        HorizontalClusterPair *pair = NULL;
        for (int j = 0; j < ancestor0->pairs.NEntries(); j++) {
          HorizontalClusterPair *tmp = ancestor0->pairs.Kth(j);
          if (tmp->clusters[0] == ancestor1) { pair = tmp; break; }
          if (tmp->clusters[1] == ancestor1) { pair = tmp; break; }
        }      
        
        // Create pair
        if (!pair) {
          RNScalar affinity = HorizontalClusterAffinity(ancestor0, ancestor1,
             min_dzdxy, max_dzdxy, 
             max_z_difference, max_xy_difference, max_dzdxy_difference, 
             max_area);
          if (affinity < FLT_MAX) {
            pair = new HorizontalClusterPair();
            pair->clusters[0] = ancestor0;
            pair->clusters[1] = ancestor1;
            pair->affinity = affinity;
            pair->heapentry = NULL;
            ancestor0->pairs.Insert(pair);
            ancestor1->pairs.Insert(pair);
            heap.Push(pair);
          }
        }
      }
    }
    else {
      // Merge cluster1 into cluster0
      cluster0->total_x += cluster1->total_x;
      cluster0->total_y += cluster1->total_y;
      cluster0->total_z += cluster1->total_z;
      cluster0->total_dzdx += cluster1->total_dzdx;
      cluster0->total_dzdy += cluster1->total_dzdy;
      cluster0->pixels.Append(cluster1->pixels);
      cluster1->pixels.Empty(TRUE);
      cluster1->parent = cluster0;

      // Update statistics
      merge_count++;
    }

    // Delete pair
    cluster0->pairs.Remove(pair01);
    cluster1->pairs.Remove(pair01);
    delete pair01;
  }

  ////////////////////////////////////////////////////////////////////////

  // printf("HERED\n");

  // Create output grids
  cluster_index_grid.Resample(input_grid.XResolution(), input_grid.YResolution());
  cluster_size_grid.Resample(input_grid.XResolution(), input_grid.YResolution());
  cluster_index_grid.SetWorldToGridTransformation(input_grid.WorldToGridTransformation());
  cluster_size_grid.SetWorldToGridTransformation(input_grid.WorldToGridTransformation());
  cluster_index_grid.Clear(R2_GRID_UNKNOWN_VALUE);
  cluster_size_grid.Clear(R2_GRID_UNKNOWN_VALUE);

  // Fill output grids 
  int nclusters = 0;
  for (int iy = 0; iy < input_grid.YResolution(); iy++) {
    for (int ix = 0; ix < input_grid.XResolution(); ix++) {
      int pixel_index;
      input_grid.IndicesToIndex(ix, iy, pixel_index);
      HorizontalCluster *cluster = &clusters[pixel_index];
      if (cluster->parent) continue;
      int cluster_size = cluster->pixels.NEntries();
      for (int i = 0; i < cluster->pixels.NEntries(); i++) {
        const RNScalar *pixel = cluster->pixels.Kth(i);
        int pixel_index = pixel - input_grid_values;
        cluster_index_grid.SetGridValue(pixel_index, nclusters);
        cluster_size_grid.SetGridValue(pixel_index, cluster_size);
      }
      nclusters++;
    }
  }
  
  // printf("HEREE %d\n", nclusters);

  // Delete clusters
  delete [] clusters;

  // Return number of clusters
  return nclusters;
}



static int 
WriteHorizontalGrids(R3SurfelScene *scene, const char *directory_name,
  RNScalar min_affinity = 1E-6,
  RNScalar min_dzdxy = 0,
  RNScalar max_dzdxy = 1,
  RNScalar max_z_difference = 0,
  RNScalar max_xy_difference = 0,
  RNScalar max_dzdxy_difference = 1,
  RNScalar min_area = 10, RNScalar max_area = 0)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Creating horizontal images ...\n");
    fflush(stdout);
  }

  // Read zmax grid
  R2Grid *zmax_grid = ReadGrid(directory_name, "Base", "ZMax");
  if (!zmax_grid) return 0;

  // Scale parameters (convert from meters to pixels)
  RNScalar scale = zmax_grid->WorldToGridScaleFactor();
  if (scale > 0) {
    min_area *= scale * scale;
    max_area *= scale * scale;
    max_xy_difference *= scale;
    min_dzdxy /= scale;
    max_dzdxy /= scale;
    max_dzdxy_difference /= scale;
  }

  // Create horizontal cluster grids
  R2Grid cluster_index_grid;
  R2Grid cluster_size_grid;
  int nclusters = CreateHorizontalClusterGrids(*zmax_grid, cluster_index_grid, cluster_size_grid, 
    min_affinity, min_dzdxy, max_dzdxy, max_z_difference, max_xy_difference, max_dzdxy_difference, max_area);
  if (nclusters == 0) return 0;

  // Find the large cluster(s)
  R2Grid large_cluster_z_grid(*zmax_grid);
  RNScalar min_cluster_size = cluster_size_grid.Maximum();
  if (min_cluster_size > 10000) min_cluster_size = 10000;
  R2Grid *large_cluster_mask = new R2Grid(cluster_size_grid);
  large_cluster_mask->Threshold(min_cluster_size - 0.5, R2_GRID_UNKNOWN_VALUE, 1);
  large_cluster_z_grid.Mask(*large_cluster_mask);
  delete large_cluster_mask;

  // Extrapolate z coordinate of large horizontal clusters
  R2Grid *ground_mask = new R2Grid(large_cluster_z_grid);
  ground_mask->FillHoles();
  ground_mask->Subtract(*zmax_grid);
  ground_mask->Abs();
  ground_mask->Threshold(0.5, 1, R2_GRID_UNKNOWN_VALUE);
  R2Grid ground_grid(*zmax_grid);
  ground_grid.Mask(*ground_mask);
  ground_grid.FillHoles();
  delete ground_mask;

  // Mask out buildings
  R2Grid building_grid(*zmax_grid);
  building_grid.Subtract(ground_grid);
  building_grid.Threshold(3, R2_GRID_UNKNOWN_VALUE, 1);
  building_grid.ConnectedComponentSizeFilter(0.5);
  building_grid.Threshold(scale * scale * 25, R2_GRID_UNKNOWN_VALUE, 1);
  building_grid.Substitute(R2_GRID_UNKNOWN_VALUE, 0);
  R2Grid non_building_grid(building_grid);
  non_building_grid.Threshold(0.5, 1, 0);

  // Create grid with Z coordinate of all horizontal surfaces
  R2Grid horizontal_z_grid(*zmax_grid);
  R2Grid horizontal_mask(cluster_size_grid);
  if (min_area > 0) horizontal_mask.Threshold(min_area, R2_GRID_UNKNOWN_VALUE, 1);
  if (max_area > 0) horizontal_mask.Threshold(max_area, 1, R2_GRID_UNKNOWN_VALUE);
  horizontal_z_grid.Mask(horizontal_mask);

#if 0
  // Allocate cluster statistics
  RNScalar *cluster_dog_sum = new RNScalar [ nclusters ];
  RNScalar *cluster_dog_min = new RNScalar [ nclusters ];
  RNScalar *cluster_dog_max = new RNScalar [ nclusters ];
  for (int i = 0; i < nclusters; i++) cluster_dog_sum[i] = 0;
  for (int i = 0; i < nclusters; i++) cluster_dog_min[i] = -FLT_MAX;
  for (int i = 0; i < nclusters; i++) cluster_dog_max[i] = FLT_MAX;

  // Compute cluster statistics
  RNLength dog_radius = 4;
  R2Grid *dog_grid = new R2Grid(*zmax_grid);
  dog_grid->Blur(scale * dog_radius);
  dog_grid->Subtract(*zmax_grid);
  dog_grid->Negate();
  for (int i = 0; i < dog_grid->NEntries(); i++) {
    RNScalar dog = dog_grid->GridValue(i);
    if (dog == R2_GRID_UNKNOWN_VALUE) continue;
    RNScalar cluster_index = cluster_index_grid.GridValue(i);
    if (cluster_index == R2_GRID_UNKNOWN_VALUE) continue;
    RNScalar cluster_size = cluster_size_grid.GridValue(i);
    if (cluster_size == R2_GRID_UNKNOWN_VALUE) continue;
    int k = (int) (cluster_index + 0.5);
    if ((k < 0) || (k >= nclusters)) continue;
    if (dog < cluster_dog_min[k]) cluster_dog_min[k] = dog;
    if (dog > cluster_dog_max[k]) cluster_dog_max[k] = dog;
    cluster_dog_sum[k] += dog;
  }
  delete dog_grid;

  // Create support grid
  R2Grid support_grid(*zmax_grid);
  support_grid.Clear(R2_GRID_UNKNOWN_VALUE);
  for (int i = 0; i < support_grid.NEntries(); i++) {
    RNScalar cluster_index = cluster_index_grid.GridValue(i);
    if (cluster_index != R2_GRID_UNKNOWN_VALUE) {
      int k = (int) (cluster_index + 0.5);
      if ((k >= 0) && (k < nclusters)) {
        if (cluster_dog_max[k] < max_dzdxy) {
          if (cluster_dog_sum[k] < 0) {
            RNScalar zmax = zmax_grid->GridValue(i);
            support_grid.SetGridValue(i, zmax);
          }
        }
      }
    }
  }

  // Delete cluster statistics
  delete [] cluster_dog_sum;
  delete [] cluster_dog_min;
  delete [] cluster_dog_max;
#endif

  // Write grids
  if (!WriteGrid(horizontal_z_grid, directory_name, "Horizontal", "Z")) return 0;
  if (!WriteGrid(horizontal_mask, directory_name, "Horizontal", "Mask")) return 0;
  if (!WriteGrid(large_cluster_z_grid, directory_name, "Horizontal", "LargeClusterZ")) return 0;
  if (!WriteGrid(cluster_index_grid, directory_name, "Horizontal", "Index")) return 0;
  if (!WriteGrid(cluster_size_grid, directory_name, "Horizontal", "Size")) return 0;
  if (!WriteGrid(ground_grid, directory_name, "Horizontal", "Ground")) return 0;
  if (!WriteGrid(building_grid, directory_name, "Horizontal", "Building")) return 0;
  if (!WriteGrid(non_building_grid, directory_name, "Horizontal", "NonBuilding")) return 0;
  // if (!WriteGrid(support_grid, directory_name, "Horizontal", "Support")) return 0;

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Resolution = %d %d\n", zmax_grid->XResolution(), zmax_grid->YResolution());
    printf("  Spacing = %g\n", zmax_grid->WorldToGridScaleFactor());
    fflush(stdout);
  }

  // Delete zmax grid
  delete zmax_grid;

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// GRID WRITING
////////////////////////////////////////////////////////////////////////

static int
WriteGrids(R3SurfelScene *scene, const char *directory_name)
{
  // Create directory
  char buffer[1024];
  sprintf(buffer, "mkdir -p %s", directory_name);
  system(buffer);

  // Write grids
  if (write_base_grids && !WriteBaseGrids(scene, directory_name)) return 0;
  if (write_color_grids && !WriteColorGrids(scene, directory_name)) return 0;
  if (write_slice_grids && !WriteSliceGrids(scene, directory_name)) return 0;
  if (write_height_grids && !WriteHeightGrids(scene, directory_name)) return 0;
  if (write_graph_grids && !WriteGraphGrids(scene, directory_name)) return 0;
  if (write_planar_grids && !WritePlanarGrids(scene, directory_name)) return 0;
  if (write_pixel_grids && !WritePixelGrids(scene, directory_name)) return 0;
  if (write_horizontal_grids && !WriteHorizontalGrids(scene, directory_name)) return 0;

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// PROGRAM ARGUMENT PARSING
////////////////////////////////////////////////////////////////////////

static int 
ParseArgs(int argc, char **argv)
{
  // Detects if default set of grids should be computed
  int default_grids = 1;

  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) print_verbose = 1;
      else if (!strcmp(*argv, "-debug")) print_debug = 1;
      else if (!strcmp(*argv, "-base")) { write_base_grids = 1; default_grids = 0; }
      else if (!strcmp(*argv, "-color")) { write_color_grids = 1; default_grids = 0; }
      else if (!strcmp(*argv, "-slice")) { write_slice_grids = 1; default_grids = 0; }
      else if (!strcmp(*argv, "-pixel"))  { write_pixel_grids = 1; default_grids = 0; }
      else if (!strcmp(*argv, "-height")) { write_height_grids = 1; default_grids = 0; }
      else if (!strcmp(*argv, "-graph"))  { write_graph_grids = 1; default_grids = 0; }
      else if (!strcmp(*argv, "-planar"))  { write_planar_grids = 1; default_grids = 0; }
      else if (!strcmp(*argv, "-horizontal"))  { write_horizontal_grids = 1; default_grids = 0; }
      else if (!strcmp(*argv, "-pixel_spacing")) { argc--; argv++; pixel_spacing = atof(*argv); }
      else if (!strcmp(*argv, "-max_resolution")) { argc--; argv++; max_resolution = atoi(*argv); }
      else if (!strcmp(*argv, "-chunk_size")) { argc--; argv++; chunk_size = atof(*argv); }
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
    else {
      if (!input_scene_name) input_scene_name = *argv;
      else if (!input_database_name) input_database_name = *argv;
      else if (!output_directory_name) output_directory_name = *argv;
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
  }

  // Check file names
  if (!input_scene_name || !input_database_name || !output_directory_name) {
    fprintf(stderr, "Usage: sfl2img scenefile databasefile [options]\n");
    return FALSE;
  }

  // Set grid selection if nothing else specified
  if (default_grids) {
    write_base_grids = 1;
    write_color_grids = 1;
  }

  // Return OK status 
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
  // Parse program arguments
  if (!ParseArgs(argc, argv)) exit(-1);

  // Open tree
  R3SurfelScene *scene = OpenScene(input_scene_name, input_database_name);
  if (!scene) exit(-1);

  // Write grids
  if (!WriteGrids(scene, output_directory_name)) exit(-1);

  // Close scene
  if (!CloseScene(scene)) exit(-1);

  // Return success 
  return 0;
}



