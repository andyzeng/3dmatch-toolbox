// Source file for the rgbd loader program



////////////////////////////////////////////////////////////////////////
// Include files 
////////////////////////////////////////////////////////////////////////

#include "R3Graphics/R3Graphics.h"
#include "RGBD/RGBD.h"



////////////////////////////////////////////////////////////////////////
// Program arguments
////////////////////////////////////////////////////////////////////////

static const char *input_configuration_filename = NULL;
static const char *input_mesh_filename = NULL;
static const char *output_vertex_image_overlap_filename = NULL;
static const char *output_image_vertex_overlap_filename = NULL;
static const char *output_image_image_vertex_overlap_filename = NULL;
static const char *output_image_image_vertex_overlap_matrix = NULL;
static const char *output_image_image_vertex_iou_matrix = NULL;
static const char *output_image_image_pixel_overlap_filename = NULL;
static const char *output_image_image_pixel_overlap_matrix = NULL;
static const char *output_image_image_pixel_iou_matrix = NULL;
static const char *output_image_image_grid_overlap_filename = NULL;
static const char *output_image_image_grid_overlap_matrix = NULL;
static const char *output_image_image_grid_iou_matrix = NULL;
static int load_every_kth_image = 1;
static int max_reprojection_distance = 32; // in pixels
static double max_depth_inconsistency = 0.1; // as fraction of depth
static double max_depth = 0;
static double grid_spacing = 0.05;
static double vertex_spacing = 0;
static int pixel_spacing = 1;
static int print_verbose = 0;
static int print_debug = 0;



////////////////////////////////////////////////////////////////////////
// Input functions
////////////////////////////////////////////////////////////////////////

static RGBDConfiguration *
ReadConfigurationFile(const char *filename) 
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Reading configuration from %s ...\n", filename);
    fflush(stdout);
  }

  // Allocate configuration
  RGBDConfiguration *configuration = new RGBDConfiguration();
  if (!configuration) {
    fprintf(stderr, "Unable to allocate configuration for %s\n", filename);
    return NULL;
  }

  // Read file
  if (!configuration->ReadFile(filename, load_every_kth_image)) {
    fprintf(stderr, "Unable to read configuration from %s\n", filename);
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Images = %d\n", configuration->NImages());
    fflush(stdout);
  }

  // Return configuration
  return configuration;
}



static R3Mesh *
ReadMeshFile(const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate mesh
  R3Mesh *mesh = new R3Mesh();
  if (!mesh) {
    fprintf(stderr, "Unable to allocate mesh for %s\n", filename);
    return NULL;
  }

  // Read mesh from file
  if (!mesh->ReadFile(filename)) {
    fprintf(stderr, "Unable to read mesh from %s\n", filename);
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

  // Return mesh
  return mesh;
}



////////////////////////////////////////////////////////////////////////
// Compute functions
////////////////////////////////////////////////////////////////////////

static int
CountVertexOverlaps(R3Mesh *mesh, const RNArray<R3MeshVertex *>& a0, const RNArray<R3MeshVertex *>& a1)
{
  // Note: this assumes that the inputs are sorted by VertexID
  
  // Initialize the count
  int count = 0;

  // Count overlaps
  int i0 = 0, i1 = 0;
  while ((i0 < a0.NEntries()) && (i1 < a1.NEntries())) {
    if (mesh->VertexID(a0[i0]) < mesh->VertexID(a1[i1])) i0++;
    else if (mesh->VertexID(a0[i0]) > mesh->VertexID(a1[i1])) i1++;
    else { count++; i0++; i1++; }
  }
  
  // Return the count
  return count;
}



static int
CountPixelOverlaps(RGBDImage *dst_image, RGBDImage *src_image)
{
  // Convenient variables
  RNScalar max_reprojection_distance_squared = max_reprojection_distance * max_reprojection_distance;
  
  // Initialize count
  int count = 0;

  // Get depth channels
  R2Grid *src_depth_channel = src_image->DepthChannel();
  if (!src_depth_channel) return 0; 
  R2Grid *dst_depth_channel = dst_image->DepthChannel();
  if (!dst_depth_channel) return 0; 

  // Check all pixels via reverse mapping
  for (int dst_iy = 0; dst_iy < dst_depth_channel->YResolution(); dst_iy += pixel_spacing) {
    for (int dst_ix = 0; dst_ix < dst_depth_channel->XResolution(); dst_ix += pixel_spacing) {
      // Get the dst image depth
      RNScalar dst_depth = dst_depth_channel->GridValue(dst_ix, dst_iy);
      if (dst_depth == R2_GRID_UNKNOWN_VALUE) continue;

      // Map point from dst_image coordinates into dst world coordinates
      R3Point dst_world_position;
      R2Point dst_image_position(dst_ix+0.5, dst_iy+0.5);
      if (!RGBDTransformImageToWorld(dst_image_position, dst_world_position, dst_image)) continue;
      
      // Map from dst world coordinates into src image coordinates      
      R2Point src_image_position;
      if (!RGBDTransformWorldToImage(dst_world_position, src_image_position, src_image)) continue;

      // Check src image coordinates
      int src_ix = src_image_position.X() + 0.5;
      int src_iy = src_image_position.Y() + 0.5;
      if ((src_ix < 0) || (src_ix >= src_depth_channel->XResolution())) continue;
      if ((src_iy < 0) || (src_iy >= src_depth_channel->YResolution())) continue;
      
      // Check for depth consistency in src image
      if (max_depth_inconsistency > 0) {
        RNScalar src_depth = src_depth_channel->GridValue(src_ix, src_iy);
        if (src_depth == R2_GRID_UNKNOWN_VALUE) continue;
        if (RNIsPositive(src_depth)) {
          RNScalar reprojected_src_depth = (dst_world_position - src_image->WorldViewpoint()).Dot(src_image->WorldTowards());
          if (reprojected_src_depth <= 0) continue;
          RNScalar depth_difference = fabs(src_depth - reprojected_src_depth);
          RNScalar depth_inconsistency = depth_difference / src_depth;
          if (depth_inconsistency > max_depth_inconsistency) continue;
        }
      }

      // Map from src image coordinates to src world coordinates
      R3Point src_world_position;
      if (!RGBDTransformImageToWorld(src_image_position, src_world_position, src_image)) continue;

      // Compute reprojected depth in dst image
      RNScalar reprojected_dst_depth = (src_world_position - dst_image->WorldViewpoint()).Dot(dst_image->WorldTowards());
      if (reprojected_dst_depth <= 0) continue;

      // Check for depth consistency in dst image
      if (max_depth_inconsistency > 0) {
        RNScalar dst_depth = dst_depth_channel->GridValue(dst_ix, dst_iy);
        if (RNIsPositive(dst_depth)) {
          RNScalar depth_difference = fabs(dst_depth - reprojected_dst_depth);
          RNScalar depth_inconsistency = depth_difference / dst_depth;
          if (depth_inconsistency > max_depth_inconsistency) continue;
        }
      }

      // Check for reprojection consistency
      if (max_reprojection_distance > 0) {
        R2Point reprojected_dst_image_position;
        if (!RGBDTransformWorldToImage(src_world_position, reprojected_dst_image_position, dst_image)) continue;
        if (R2SquaredDistance(reprojected_dst_image_position, dst_image_position) > max_reprojection_distance_squared) continue;
      }
    
      // Update counter
      count++;
    }
  }

  // Return number of pixels in overlap
  return count;
}



static int
ComputePixelOverlaps(RGBDConfiguration *configuration, R2Grid& pixel_overlap_matrix) 
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Computing pixel overlaps ...\n");
    fflush(stdout);
  }

  // Initialize matrix
  pixel_overlap_matrix = R2Grid(configuration->NImages(), configuration->NImages());

  // Consider every image
  for (int i0 = 0; i0 < configuration->NImages(); i0++) {
    RGBDImage *dst_image = configuration->Image(i0);

    // Read dst image depth channel
    dst_image->ReadDepthChannel();
    R2Grid *dst_depth_channel = dst_image->DepthChannel();

    // Compute dst image info
    RNInterval dst_depth_range = dst_depth_channel->Range();
    const R3Box& dst_bbox = dst_image->WorldBBox();
    R3Frustum dst_frustum(dst_image->WorldViewpoint(), dst_image->WorldTowards(), dst_image->WorldUp(),
      dst_image->XFov(), dst_image->YFov(), dst_depth_range.Min(), dst_depth_range.Max());

    // Consider every other image
    for (int i1 = 0; i1 < configuration->NImages(); i1++) {
      RGBDImage *src_image = configuration->Image(i1);

      // Check src bounding box intersection
      const R3Box& src_bbox = src_image->WorldBBox();
      if (!dst_bbox.Intersects(src_bbox)) continue;
      if (!dst_frustum.Intersects(src_bbox)) continue;

      // Read src image depth channel
      src_image->ReadDepthChannel();
      R2Grid *src_depth_channel = src_image->DepthChannel();

      // Check src view frustum intersection
      RNInterval src_depth_range = src_depth_channel->Range();
      R3Frustum src_frustum(src_image->WorldViewpoint(), src_image->WorldTowards(), src_image->WorldUp(),
        src_image->XFov(), src_image->YFov(), src_depth_range.Min(), src_depth_range.Max());
      if (!src_frustum.Intersects(dst_bbox)) continue;

      // Compute overlap
      int overlaps = CountPixelOverlaps(dst_image, src_image);
      pixel_overlap_matrix.SetGridValue(i0, i1, overlaps);

      // Print debug statement
      if (print_debug) printf("%d %d  %d\n", i0, i1, overlaps);

      // Release src image depth channel
      src_image->ReleaseDepthChannel();
    }

    // Release dst image depth channel
    dst_image->ReleaseDepthChannel();
  }
  
  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Images = %d\n", configuration->NImages());
    printf("  Avg Overlaps = %.0f\n", pixel_overlap_matrix.Mean());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
ComputeGridOverlaps(RGBDConfiguration *configuration, R2Grid& image_image_grid_overlap_matrix)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Computing grid overlaps ...\n");
    fflush(stdout);
  }

  // Initialize matrix
  image_image_grid_overlap_matrix = R2Grid(configuration->NImages(), configuration->NImages());

  // Consider every image
  for (int i0 = 0; i0 < configuration->NImages(); i0++) {
    RGBDImage *image0 = configuration->Image(i0);

    // Read image0 depth channel
    image0->ReadDepthChannel();
    R2Grid *depth_channel0 = image0->DepthChannel();
    if (!depth_channel0) { image0->ReleaseDepthChannel(); continue; }
    const R3Box& bbox0 = image0->WorldBBox();

    // Create grid
    R3Grid grid0(bbox0, grid_spacing, 1, 1024);
    for (int iy0 = 0; iy0 < depth_channel0->YResolution(); iy0 += pixel_spacing) {
      for (int ix0 = 0; ix0 < depth_channel0->XResolution(); ix0 += pixel_spacing) {
        R3Point world_position0;
        if (!RGBDTransformImageToWorld(R2Point(ix0+0.5, iy0+0.5), world_position0, image0)) continue;
        grid0.RasterizeWorldPoint(world_position0, 1.0);
      }
    }
  
    // Release image0 depth channel
    image0->ReleaseDepthChannel();
    if (grid0.Cardinality() == 0) continue;

    // Consider every other image
    for (int i1 = 0; i1 < configuration->NImages(); i1++) {
      RGBDImage *image1 = configuration->Image(i1);
      if (!R3Intersects(bbox0, image1->WorldBBox())) continue;

      // Read image1 depth channel
      image1->ReadDepthChannel();
      R2Grid *depth_channel1 = image1->DepthChannel();
      if (!depth_channel1) continue;
      const R3Box& bbox1 = image1->WorldBBox();
      if (!R3Intersects(bbox0, bbox1)) { image1->ReleaseDepthChannel(); continue; }

      // Count pixels from image1 overlapping grid0
      int count = 0;
      for (int iy1 = 0; iy1 < depth_channel1->YResolution(); iy1 += pixel_spacing) {
        for (int ix1 = 0; ix1 < depth_channel1->XResolution(); ix1 += pixel_spacing) {
          R3Point world_position1;
          R2Point image_position1(ix1+0.5, iy1+0.5);
          if (!RGBDTransformImageToWorld(image_position1, world_position1, image1)) continue;
          RNScalar grid_value = grid0.WorldValue(world_position1);
          if (grid_value == 0) continue;
          // Check normal?
          count++;
        }
      }
      
      // Update overlap matrix
      image_image_grid_overlap_matrix.SetGridValue(i0, i1, count);

      // Print debug info
      if (print_debug) {
        if (count > 0) printf("%d %d : %d \n", i0, i1, count);
      }

      // Release image1 depth channel
      image1->ReleaseDepthChannel();
    }
  }

  // Print statistics
  if (print_verbose) {
    printf("Computed grid overlap matrix ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Overlaps = %d\n", image_image_grid_overlap_matrix.Cardinality());
    fflush(stdout);
  }

  // Return success
  return 1;
}

  

static int
ComputeMeshOverlaps(RGBDConfiguration *configuration, R3Mesh *mesh,
  RNArray<R3MeshVertex *> **image_to_vertex_overlaps = NULL,
  RNArray<RGBDImage *> **vertex_to_image_overlaps = NULL) 
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  unsigned long overlap_count = 0;
  if (print_verbose) {
    printf("Computing mesh overlaps ...\n");
    fflush(stdout);
  }

  // Create a sampled list of vertices to check
  RNArray<R3MeshVertex *> vertices;
  RNScalar sampling_fraction = (vertex_spacing > 0) ?  mesh->AverageEdgeLength() / vertex_spacing : 1.0;
  for (int i = 0; i < mesh->NVertices(); i++) {
    if (RNRandomScalar() > sampling_fraction) continue;
    vertices.Insert(mesh->Vertex(i));
  }

  // Consider every image
  for (int i0 = 0; i0 < configuration->NImages(); i0++) {
    RGBDImage *image = configuration->Image(i0);

    // Read image depth channel
    if (!image->ReadDepthChannel()) continue;
    const R3Box& image_bbox = image->WorldBBox();
    if (image_bbox.IsEmpty()) { image->ReleaseDepthChannel(); continue; }
    if (!R3Intersects(image_bbox, mesh->BBox())) { image->ReleaseDepthChannel(); continue; }
    
    // Consider every sample vertex
    for (int i1 = 0; i1 < vertices.NEntries(); i1++) {
      R3MeshVertex *vertex = vertices.Kth(i1);
      const R3Point& vertex_position = mesh->VertexPosition(vertex);
      if (!R3Intersects(image_bbox, vertex_position)) continue;
      
      // Compute mapping from vertex into image
      R2Point image_position;
      if (!RGBDTransformWorldToImage(vertex_position, image_position, image)) continue;

      // Check depth
      if (max_depth > 0) {
        RNScalar pixel_depth = image->DepthChannel()->GridValue(image_position);
        if (pixel_depth > max_depth) continue;
      }
      
      // Check for depth consistency
      if (max_depth_inconsistency > 0) {
        RNScalar vertex_depth = (vertex_position - image->WorldViewpoint()).Dot(image->WorldTowards());
        RNScalar pixel_depth = image->DepthChannel()->GridValue(image_position);
        RNScalar depth_difference = fabs(vertex_depth - pixel_depth);
        RNScalar depth_maximum = (pixel_depth > vertex_depth) ? pixel_depth : vertex_depth;
        if (depth_maximum <= 0) continue;
        RNScalar depth_inconsistency = depth_difference / depth_maximum;
        if (depth_inconsistency > max_depth_inconsistency) continue;
      }

      // Insert overlap
      if (image_to_vertex_overlaps) (*image_to_vertex_overlaps)[image->ConfigurationIndex()].Insert(vertex);
      if (vertex_to_image_overlaps) (*vertex_to_image_overlaps)[mesh->VertexID(vertex)].Insert(image);

      // Update statistics
      overlap_count++;
    }

    // Release image depth channel
    image->ReleaseDepthChannel();
  }
  
  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Images = %d\n", configuration->NImages());
    printf("  # Vertices = %d\n", mesh->NVertices());
    printf("  # Overlaps = %lu\n", overlap_count);
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
ComputeImageImageOverlapMatrix(RGBDConfiguration *configuration, R3Mesh *mesh,
  const RNArray<R3MeshVertex *> *image_to_vertex_overlaps, R2Grid& image_image_vertex_overlap_matrix)
{
  // Initialize matrix
  image_image_vertex_overlap_matrix = R2Grid(configuration->NImages(), configuration->NImages());

  // Compute matrix
  for (int i0 = 0; i0 < configuration->NImages(); i0++) {
    for (int i1 = 0; i1 < configuration->NImages(); i1++) {
      if (image_to_vertex_overlaps[i0].IsEmpty()) continue;
      if (image_to_vertex_overlaps[i1].IsEmpty()) continue;
      int overlap_count = CountVertexOverlaps(mesh, image_to_vertex_overlaps[i0], image_to_vertex_overlaps[i1]);
      image_image_vertex_overlap_matrix.SetGridValue(i0, i1, overlap_count);
    }
  }

  // Return success
  return 1;
}




////////////////////////////////////////////////////////////////////////
// Input functions
////////////////////////////////////////////////////////////////////////

static int
WriteImageVertexOverlapFile(RGBDConfiguration *configuration, R3Mesh *mesh,
  const RNArray<R3MeshVertex *> *image_to_vertex_overlaps, const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Writing image->vertex overlaps to %s ...\n", filename);
    fflush(stdout);
  }
  
  // Open output file
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "Unable to open overlap file %s\n", filename);
    return 0;
  }
  
  // Write header
  fprintf(fp, "C %s\n", input_configuration_filename);
  fprintf(fp, "M %s\n", input_mesh_filename);
  fprintf(fp, "\n");
  
  // Write overlaps
  for (int i0 = 0; i0 < configuration->NImages(); i0++) {
    if (image_to_vertex_overlaps[i0].IsEmpty()) continue;
    fprintf(fp, "IV %d  %d  ", i0, image_to_vertex_overlaps[i0].NEntries());
    for (int i1 = 0; i1 < image_to_vertex_overlaps[i0].NEntries(); i1++) {
      R3MeshVertex *vertex = image_to_vertex_overlaps[i0].Kth(i1);
      fprintf(fp, "%d ", mesh->VertexID(vertex));
    }
    fprintf(fp, "\n");
  }
  
  // Close info file
  fclose(fp);
  
  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Images = %d\n", configuration->NImages());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
WriteVertexImageOverlapFile(RGBDConfiguration *configuration, R3Mesh *mesh,
  const RNArray<RGBDImage *> *vertex_to_image_overlaps, const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Writing vertex->image overlaps to %s ...\n", filename);
    fflush(stdout);
  }
  
  // Open output file
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "Unable to open overlap file %s\n", filename);
    return 0;
  }
  
  // Write header
  fprintf(fp, "C %s\n", input_configuration_filename);
  fprintf(fp, "M %s\n", input_mesh_filename);
  fprintf(fp, "\n");
  
  // Write overlaps
  for (int i0 = 0; i0 < mesh->NVertices(); i0++) {
    R3MeshVertex *vertex = mesh->Vertex(i0);
    if (vertex_to_image_overlaps[i0].IsEmpty()) continue;
    const R3Point& position = mesh->VertexPosition(vertex);
    const R3Vector& normal = mesh->VertexNormal(vertex);
    fprintf(fp, "VI  %d  %g %g %g  %g %g %g  %d  ", i0,
      position.X(), position.Y(), position.Z(), normal.X(), normal.Y(), normal.Z(),
      vertex_to_image_overlaps[i0].NEntries());
    for (int i1 = 0; i1 < vertex_to_image_overlaps[i0].NEntries(); i1++) {
      RGBDImage *image = vertex_to_image_overlaps[i0].Kth(i1);
      fprintf(fp, "%d ", image->ConfigurationIndex());
    }
    fprintf(fp, "\n");
  }
  
  // Close info file
  fclose(fp);
  
  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Vertices = %d\n", mesh->NVertices());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
WriteImageImageOverlapFile(RGBDConfiguration *configuration, const R2Grid& overlap_matrix, const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  int count = 0;
  if (print_verbose) {
    printf("Writing image->image overlaps to %s ...\n", filename);
    fflush(stdout);
  }
  
  // Open output file
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "Unable to open overlap file %s\n", filename);
    return 0;
  }

  // Write header
  fprintf(fp, "C %s\n", input_configuration_filename);
  if (input_mesh_filename) fprintf(fp, "M %s\n", input_mesh_filename);
  fprintf(fp, "\n");
  
  // Write overlaps
  for (int i0 = 0; i0 < configuration->NImages(); i0++) {
    RNScalar count0 = overlap_matrix.GridValue(i0, i0);
    for (int i1 = 0; i1 < configuration->NImages(); i1++) {
      RNScalar count1 = overlap_matrix.GridValue(i1, i1);
      RNScalar isect01 = overlap_matrix.GridValue(i0, i1);
      RNScalar isect10 = overlap_matrix.GridValue(i1, i0);
      RNScalar isect = (isect01 < isect10) ? isect01 : isect10;
      RNScalar unio = count0 + count1 - isect;
      RNScalar iou = (unio > 0) ? isect / unio : 0;
      if (iou < 0.001) continue;
      fprintf(fp, "II %d %d   %g %g %g   %g %g\n", i0, i1, iou, isect, unio, count0, count1);
      count++;
    }
  }

  // Close info file
  fclose(fp);
  
  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Overlaps = %d\n", count);
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
WriteImageImageOverlapMatrix(RGBDConfiguration *configuration, const R2Grid& overlap_matrix, const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Writing image->image overlaps to %s ...\n", filename);
    fflush(stdout);
  }

  // Write matrix
  if (!overlap_matrix.WriteFile(filename)) return 0;
  
  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Overlaps = %d\n", overlap_matrix.Cardinality());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
WriteImageImageIOUMatrix(RGBDConfiguration *configuration, const R2Grid& overlap_matrix, const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Compute iou matrix
  R2Grid overlap_iou_matrix(overlap_matrix.XResolution(), overlap_matrix.YResolution());
  for (int i0 = 0; i0 < configuration->NImages(); i0++) {
    RNScalar count0 = overlap_matrix.GridValue(i0, i0);
    for (int i1 = 0; i1 < configuration->NImages(); i1++) {
      RNScalar count1 = overlap_matrix.GridValue(i1, i1);
      RNScalar isect01 = overlap_matrix.GridValue(i0, i1);
      RNScalar isect10 = overlap_matrix.GridValue(i1, i0);
      RNScalar isect = (isect01 < isect10) ? isect01 : isect10;
      RNScalar unio = count0 + count1 - isect;
      RNScalar iou = (unio > 0) ? isect / unio : 0;
      overlap_iou_matrix.SetGridValue(i0, i1, iou);
    }
  }

  // Write iou matrix
  if (strstr(filename, ".png")) overlap_iou_matrix.Multiply(1000);
  if (!overlap_iou_matrix.WriteFile(filename)) return 0;
  
  // Print statistics
  if (print_verbose) {
    printf("Wrote overlap IOU matrix to %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Overlaps = %d\n", overlap_iou_matrix.Cardinality());
    fflush(stdout);
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// PROGRAM ARGUMENT PARSING
////////////////////////////////////////////////////////////////////////

static int 
ParseArgs(int argc, char **argv)
{
  // Check if have an output
  RNBoolean output = FALSE;

  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) print_verbose = 1;
      else if (!strcmp(*argv, "-debug")) print_debug = 1;
      else if (!strcmp(*argv, "-mesh")) { argc--; argv++; input_mesh_filename = *argv; }
      else if (!strcmp(*argv, "-output_image_vertex_overlap_file")) { argc--; argv++; output_image_vertex_overlap_filename = *argv; output = TRUE; }
      else if (!strcmp(*argv, "-output_vertex_image_overlap_file")) { argc--; argv++; output_vertex_image_overlap_filename = *argv; output = TRUE; }
      else if (!strcmp(*argv, "-output_image_image_vertex_overlap_file")) { argc--; argv++; output_image_image_vertex_overlap_filename = *argv; output = TRUE; }
      else if (!strcmp(*argv, "-output_image_image_vertex_overlap_matrix")) { argc--; argv++; output_image_image_vertex_overlap_matrix = *argv; output = TRUE; }
      else if (!strcmp(*argv, "-output_image_image_vertex_iou_matrix")) { argc--; argv++; output_image_image_vertex_iou_matrix = *argv; output = TRUE; }
      else if (!strcmp(*argv, "-output_image_image_pixel_overlap_file")) { argc--; argv++; output_image_image_pixel_overlap_filename = *argv; output = TRUE; }
      else if (!strcmp(*argv, "-output_image_image_pixel_overlap_matrix")) { argc--; argv++; output_image_image_pixel_overlap_matrix = *argv; output = TRUE; }
      else if (!strcmp(*argv, "-output_image_image_pixel_iou_matrix")) { argc--; argv++; output_image_image_pixel_iou_matrix = *argv; output = TRUE; }
      else if (!strcmp(*argv, "-output_image_image_grid_overlap_file")) { argc--; argv++; output_image_image_grid_overlap_filename = *argv; output = TRUE; }
      else if (!strcmp(*argv, "-output_image_image_grid_overlap_matrix")) { argc--; argv++; output_image_image_grid_overlap_matrix = *argv; output = TRUE; }
      else if (!strcmp(*argv, "-output_image_image_grid_iou_matrix")) { argc--; argv++; output_image_image_grid_iou_matrix = *argv; output = TRUE; }
      else if (!strcmp(*argv, "-load_every_kth_image")) { argc--; argv++; load_every_kth_image = atoi(*argv); }
      else if (!strcmp(*argv, "-max_reprojection_distance")) { argc--; argv++; max_reprojection_distance = atof(*argv); }
      else if (!strcmp(*argv, "-max_depth_inconsistency")) { argc--; argv++; max_depth_inconsistency = atof(*argv); }
      else if (!strcmp(*argv, "-max_depth")) { argc--; argv++; max_depth = atof(*argv); }
      else if (!strcmp(*argv, "-pixel_spacing")) { argc--; argv++; pixel_spacing = atoi(*argv); }
      else if (!strcmp(*argv, "-vertex_spacing")) { argc--; argv++; vertex_spacing = atof(*argv); }
      else if (!strcmp(*argv, "-grid_spacing")) { argc--; argv++; grid_spacing = atof(*argv); }
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
    else {
      if (!input_configuration_filename) input_configuration_filename = *argv;
      else if (!output_image_image_pixel_iou_matrix) { output_image_image_pixel_iou_matrix = *argv; output = TRUE; }
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
  }

  // Check filenames
  if (!input_configuration_filename || !output) {
    fprintf(stderr, "Usage: conf2overlap inputconfigurationfile output [-mesh inputmeshfile] [options]\n");
    return 0;
  }

  // Return OK status 
  return 1;
}



////////////////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////////////////

int
main(int argc, char **argv)
{
  // Check number of arguments
  if (!ParseArgs(argc, argv)) exit(1);

  // Read configuration
  RGBDConfiguration *configuration = ReadConfigurationFile(input_configuration_filename);
  if (!configuration) exit(-1);

  // Read mesh
  R3Mesh *mesh = NULL;
  if (input_mesh_filename) {
    mesh = ReadMeshFile(input_mesh_filename);
    if (!mesh) exit(-1);
  }

  // Check if should compute vertex overlaps
  if (mesh && (output_image_image_vertex_overlap_filename || output_image_image_vertex_overlap_matrix || output_image_image_vertex_iou_matrix)) {
    // Compute image-vertex and vertex-image overlaps
    RNArray<R3MeshVertex *> *image_to_vertex_overlaps = new RNArray<R3MeshVertex *> [ configuration->NImages() ];
    RNArray<RGBDImage *> *vertex_to_image_overlaps = new RNArray<RGBDImage *> [ mesh->NVertices() ]; 
    if (!ComputeMeshOverlaps(configuration, mesh, &image_to_vertex_overlaps, &vertex_to_image_overlaps)) return 0;

    // Compute image-image overlap matrix
    R2Grid image_image_vertex_overlap_matrix;
    if (!ComputeImageImageOverlapMatrix(configuration, mesh, image_to_vertex_overlaps, image_image_vertex_overlap_matrix)) exit(-1);
    
    // Write image to vertex overlaps
    if (output_image_vertex_overlap_filename) {
      if (!WriteImageVertexOverlapFile(configuration, mesh, image_to_vertex_overlaps, output_image_vertex_overlap_filename)) exit(-1);
    }
  
    // Write vertex to image overlaps
    if (output_vertex_image_overlap_filename) {
      if (!WriteVertexImageOverlapFile(configuration, mesh, vertex_to_image_overlaps, output_vertex_image_overlap_filename)) exit(-1);
    }
  
    // Write image to image vertex overlaps
    if (output_image_image_vertex_overlap_filename) {
      if (!WriteImageImageOverlapFile(configuration, image_image_vertex_overlap_matrix, output_image_image_vertex_overlap_filename)) exit(-1);
    }
  
    // Write image to image vertex matrix
    if (output_image_image_vertex_overlap_matrix) {
      if (!WriteImageImageOverlapMatrix(configuration, image_image_vertex_overlap_matrix, output_image_image_vertex_overlap_matrix)) exit(-1);
    }

    // Write image to image vertex iou matrix
    if (output_image_image_vertex_iou_matrix) {
      if (!WriteImageImageIOUMatrix(configuration, image_image_vertex_overlap_matrix, output_image_image_vertex_iou_matrix)) exit(-1);
    }
  }

  // Check if should compute pixel overlaps
  if (output_image_image_pixel_overlap_filename || output_image_image_pixel_overlap_matrix || output_image_image_pixel_iou_matrix) {
    // Compute image to image pixel overlaps
    R2Grid image_image_pixel_overlap_matrix;
    if (!ComputePixelOverlaps(configuration, image_image_pixel_overlap_matrix)) exit(-1);
    
    // Write image to image pixel overlap file
    if (output_image_image_pixel_overlap_filename) {
      if (!WriteImageImageOverlapFile(configuration, image_image_pixel_overlap_matrix, output_image_image_pixel_overlap_filename)) exit(-1);
    }
  
    // Write image to image pixel overlap matrix
    if (output_image_image_pixel_overlap_matrix) {
      if (!WriteImageImageOverlapMatrix(configuration, image_image_pixel_overlap_matrix, output_image_image_pixel_overlap_matrix)) exit(-1);
    }

    // Write image to image pixel iou matrix
    if (output_image_image_pixel_iou_matrix) {
      if (!WriteImageImageIOUMatrix(configuration, image_image_pixel_overlap_matrix, output_image_image_pixel_iou_matrix)) exit(-1);
    }
  }
  
  // Check if should compute grid overlaps
  if (output_image_image_grid_overlap_filename || output_image_image_grid_overlap_matrix || output_image_image_grid_iou_matrix) {
    // Compute image to image grid overlaps
    R2Grid image_image_grid_overlap_matrix;
    if (!ComputeGridOverlaps(configuration, image_image_grid_overlap_matrix)) exit(-1);
    
    // Write image to image grid overlap file
    if (output_image_image_grid_overlap_filename) {
      if (!WriteImageImageOverlapFile(configuration, image_image_grid_overlap_matrix, output_image_image_grid_overlap_filename)) exit(-1);
    }
  
    // Write image to image grid overlap matrix
    if (output_image_image_grid_overlap_matrix) {
      if (!WriteImageImageOverlapMatrix(configuration, image_image_grid_overlap_matrix, output_image_image_grid_overlap_matrix)) exit(-1);
    }

    // Write image to image grid iou matrix
    if (output_image_image_grid_iou_matrix) {
      if (!WriteImageImageIOUMatrix(configuration, image_image_grid_overlap_matrix, output_image_image_grid_iou_matrix)) exit(-1);
    }
  }
  
  // Return success 
  return 0;
}



