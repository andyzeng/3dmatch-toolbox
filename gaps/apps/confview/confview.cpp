// Source file for the rgbd viewer program



////////////////////////////////////////////////////////////////////////
// Include files 
////////////////////////////////////////////////////////////////////////

#include "R3Graphics/R3Graphics.h"
#include "RGBD/RGBD.h"
#include "fglut/fglut.h"



////////////////////////////////////////////////////////////////////////
// Program arguments
////////////////////////////////////////////////////////////////////////

static const char *input_configuration_filename = NULL;
static const char *output_configuration_filename = NULL;
static const char *input_mesh_filename = NULL;
static const char *input_overlap_filename = NULL;
static const char *input_overlap_matrix = NULL;
static int load_every_kth_image = 1;
static double texel_spacing = 0;
static int print_verbose = 0;



////////////////////////////////////////////////////////////////////////
// Internal variables
////////////////////////////////////////////////////////////////////////

// GLUT variables 

static int GLUTwindow = 0;
static int GLUTwindow_width = 640;
static int GLUTwindow_height = 480;
static int GLUTmouse[2] = { 0, 0 };
static int GLUTbutton[3] = { 0, 0, 0 };
static int GLUTmouse_drag = 0;
static int GLUTmodifiers = 0;


// Application variables

static RGBDConfiguration configuration;
static R2Grid *image_image_overlaps = NULL;
static RNArray<RGBDImage *> *vertex_image_overlaps = NULL;
static RNArray<R3MeshVertex *> *image_vertex_overlaps = NULL;
static char *screenshot_image_name = NULL;
static int selected_surface_index = -1;
static int selected_image_index = -1;
static int snap_image_index = -1;
static R2Point rubber_box_corners[2] = { R2Point(0,0), R2Point(0,0) };
static RNBoolean rubber_box_active = FALSE;
static int color_scheme = RGBD_PHOTO_COLOR_SCHEME;
static R3Point center(0, 0, 0);
static R3Viewer viewer;



// Display variables

static int show_cameras = 1;
static int show_bboxes = 0;
static int show_images = 0;
static int show_points = 0;
static int show_quads = 0;
static int show_faces = 0;
static int show_edges = 0;
static int show_textures = 0;
static int show_overlaps = 0;
static int show_axes = 0;



////////////////////////////////////////////////////////////////////////
// Read/Write functions
////////////////////////////////////////////////////////////////////////

static int
ReadConfiguration(RGBDConfiguration& configuration, const char *filename) 
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Reading configuration from %s ...\n", filename);
    fflush(stdout);
  }

  // Read file
  if (!configuration.ReadFile(filename, load_every_kth_image)) {
    fprintf(stderr, "Unable to read configuration from %s\n", filename);
    return 0;
  }

#if 0
  // Read all channels ... for now
  if (!configuration.ReadChannels()) {
    fprintf(stderr, "Unable to read channels for %s\n", filename);
    return 0;
  }
#endif
  
  // Set texel spacing if specified on command line                       
  if (texel_spacing > 0) {
    for (int i = 0; i < configuration.NSurfaces(); i++) {
      RGBDSurface *surface = configuration.Surface(i);
      surface->SetWorldTexelSpacing(texel_spacing);
    }
  }

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Images = %d\n", configuration.NImages());
    printf("  # Surfaces = %d\n", configuration.NSurfaces());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
WriteConfiguration(RGBDConfiguration& configuration, const char *filename) 
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Writing configuration to %s ...\n", filename);
    fflush(stdout);
  }

  // Write file
  if (!configuration.WriteFile(filename)) return 0;

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Images = %d\n", configuration.NImages());
    printf("  # Surfaces = %d\n", configuration.NSurfaces());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
ReadMesh(RGBDConfiguration& configuration, const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate mesh
  R3Mesh *mesh = new R3Mesh();
  if (!mesh) {
    fprintf(stderr, "Unable to allocate mesh for %s\n", filename);
    return 0;
  }

  // Read mesh from file
  if (!mesh->ReadFile(filename)) {
    fprintf(stderr, "Unable to read mesh from %s\n", filename);
    return 0;
  }

  // Create surface for mesh
  RGBDSurface *surface = new RGBDSurface(NULL, mesh, texel_spacing);
  if (!surface) {
    fprintf(stderr, "Unable to allocate surface for %s\n", filename);
    return 0;
  }

  // Insert surface
  configuration.InsertSurface(surface);

  // Print statistics
  if (print_verbose) {
    printf("Read mesh from %s ...\n", filename);
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
ReadOverlapMatrix(RGBDConfiguration& configuration, const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate matrix
  image_image_overlaps = new R2Grid();
  if (!image_image_overlaps) {
    fprintf(stderr, "Unable to allocate overlap matrix for %s\n", filename);
    return 0;
  }

  // Read matrix from file
  if (!image_image_overlaps->ReadFile(filename)) {
    fprintf(stderr, "Unable to read overlap matrix from %s\n", filename);
    return 0;
  }

  // Apply scale factor if png
  if (strstr(filename, ".png")) image_image_overlaps->Multiply(0.001);

  // Print statistics
  if (print_verbose) {
    printf("Read overlaps from %s ...\n", filename);
    printf("  # Images = %d\n", image_image_overlaps->XResolution());
    printf("  # Overlaps = %d\n", image_image_overlaps->Cardinality());
    printf("  Mean = %g\n", image_image_overlaps->Mean());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
ReadOverlapFile(RGBDConfiguration& configuration, const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  int count = 0;

  // Check configuration
  if (configuration.NSurfaces() == 0) {
    fprintf(stderr, "Must have mesh to read overlaps\n");
    return 0;
  }

  // Get mesh from configuration
  RGBDSurface *surface = configuration.Surface(0);
  R3Mesh *mesh = surface->mesh;
  if (!mesh || (mesh->NVertices() == 0)) {
    fprintf(stderr, "Must have mesh to read overlaps\n");
    return 0;
  }

  // Allocate arrays of vertex image overlaps
  vertex_image_overlaps = new RNArray<RGBDImage *> [ mesh->NVertices() ];
  if (!vertex_image_overlaps) {
    fprintf(stderr, "Unable to allocate vertex image overlaps for %s\n", filename);
    return 0;
  }

  // Allocate arrays of image vertex overlaps
  image_vertex_overlaps = new RNArray<R3MeshVertex *> [ configuration.NImages() ];
  if (!image_vertex_overlaps) {
    fprintf(stderr, "Unable to allocate image vertex overlaps for %s\n", filename);
    return 0;
  }

  // Allocate matrix of image image overlaps
  if (!image_image_overlaps) {
    image_image_overlaps = new R2Grid(configuration.NImages(), configuration.NImages());
    if (!image_image_overlaps) {
      fprintf(stderr, "Unable to allocate image image overlap matrix\n");
      return 0;
    }
  }

  // Open file
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Unable to open overlap file %s\n", filename);
    return 0;
  }
  
  // Read file
  char cmd[1024];
  while (fscanf(fp, "%s", cmd) == (unsigned int) 1) {
    // Check cmd
    if (!strcmp(cmd, "C")) {
      // Read configuration filename
      char configuration_filename[1024];
      if (fscanf(fp, "%s", configuration_filename) != (unsigned int) 1) {
        fprintf(stderr, "Unable to read configuration filename in %s\n", filename);
        return 0;
      }
    }
    else if (!strcmp(cmd, "M")) {
      // Read mesh filename
      char mesh_filename[1024];
      if (fscanf(fp, "%s", mesh_filename) != (unsigned int) 1) {
        fprintf(stderr, "Unable to read mesh filename in %s\n", filename);
        return 0;
      }
    }
    else if (!strcmp(cmd, "VI")) {
      // Read vertex to image overlaps
      int vertex_index, noverlaps;
      double px, py, pz, nx, ny, nz;
      if (fscanf(fp, "%d%lf%lf%lf%lf%lf%lf%d", &vertex_index,
        &px, &py, &pz, &nx, &ny, &nz, &noverlaps) != (unsigned int) 8) {
        fprintf(stderr, "Unable to read vertex in %s\n", filename);
        return 0;
      }
        
      // Check vertex index
      if ((vertex_index < 0) || (vertex_index >= mesh->NVertices())) {
        fprintf(stderr, "Invalid vertex index %d in %s\n", vertex_index, filename);
        return 0;
      }

      // Find vertex
      R3MeshVertex *vertex = mesh->Vertex(vertex_index);
      if (R3SquaredDistance(mesh->VertexPosition(vertex), R3Point(px, py, pz)) > RN_EPSILON) {
        fprintf(stderr, "Mismatching position for vertex index %d in %s\n", vertex_index, filename);
        return 0;
      }

      // Read overlaps
      for (int i = 0; i < noverlaps; i++) {
        // Read image index
        int image_index;
        if (fscanf(fp, "%d", &image_index) != (unsigned int) 1) {
          fprintf(stderr, "Unable to read image index in %s\n", filename);
          return 0;
        }

        // Check image index
        if ((image_index < 0) || (image_index >= mesh->NVertices())) {
          fprintf(stderr, "Invalid image index %d in %s\n", image_index, filename);
          return 0;
        }
         
        // Find image
        RGBDImage *image = configuration.Image(image_index);
        if (!image) {
          fprintf(stderr, "This should never happen\n");
          return 0;
        }

        // Add overlap
        vertex_image_overlaps[vertex_index].Insert(image);

        // Increment count
        count++;
      }
    }
    else if (!strcmp(cmd, "IV")) {
      // Read image to vertex overlaps
      int image_index, noverlaps;
      if (fscanf(fp, "%d%d", &image_index, &noverlaps) != (unsigned int) 2) {
        fprintf(stderr, "Unable to read image in %s\n", filename);
        return 0;
      }
        
      // Check image index
      if ((image_index < 0) || (image_index >= configuration.NImages())) {
        fprintf(stderr, "Invalid image index %d in %s\n", image_index, filename);
        return 0;
      }

      // Find image
      RGBDImage *image = configuration.Image(image_index);
      if (!image) {
        fprintf(stderr, "This should never happen\n");
        return 0;
      }

      // Read overlaps
      for (int i = 0; i < noverlaps; i++) {
        // Read vertex index
        int vertex_index;
        if (fscanf(fp, "%d", &vertex_index) != (unsigned int) 1) {
          fprintf(stderr, "Unable to read vertex index in %s\n", filename);
          return 0;
        }

        // Check vertex index
        if ((vertex_index < 0) || (vertex_index >= mesh->NVertices())) {
          fprintf(stderr, "Invalid vertex index %d in %s\n", vertex_index, filename);
          return 0;
        }

        // Find vertex
        R3MeshVertex *vertex = mesh->Vertex(vertex_index);
        if (!vertex) {
          fprintf(stderr, "This should never happen\n");
          return 0;
        }

        // Add overlap
        image_vertex_overlaps[image_index].Insert(vertex);

        // Increment count
        count++;
      }
    }
    else if (!strcmp(cmd, "II")) {
      // Read image to image overlap
      RNScalar fraction;
      int image_index1, image_index2, noverlaps;
      if (fscanf(fp, "%d%d%d%lf", &image_index1, &image_index2, &noverlaps, &fraction) != (unsigned int) 4) {
        fprintf(stderr, "Unable to read line in %s\n", filename);
        return 0;
      }

      // Assign entry in image image overlap matrix
      image_image_overlaps->SetGridValue(image_index1, image_index2, fraction);

      // Increment count
      count++;
    }        
  }

  // Close file
  fclose(fp);

  // Print statistics
  if (print_verbose) {
    printf("Read overlaps from %s ...\n", filename);
    printf("  # Images = %d\n", configuration.NImages());
    printf("  # Overlaps = %d\n", count);
    fflush(stdout);
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Draw functions
////////////////////////////////////////////////////////////////////////

static void
DrawCameras(int color_scheme = RGBD_INDEX_COLOR_SCHEME)
{
  // Draw cameras
  for (int i = 0; i < configuration.NImages(); i++) {
    RGBDImage *image = configuration.Image(i);
    if ((color_scheme != RGBD_INDEX_COLOR_SCHEME) && (i == selected_image_index)) image->DrawCamera(RGBD_HIGHLIGHT_COLOR_SCHEME);
    else image->DrawCamera(color_scheme);
  }
}



static void
DrawImages(int color_scheme = RGBD_PHOTO_COLOR_SCHEME)
{
  // Draw images
  for (int i = 0; i < configuration.NImages(); i++) {
    RGBDImage *image = configuration.Image(i);
    if (!image->RedChannel()) continue;
    if ((color_scheme != RGBD_INDEX_COLOR_SCHEME) && (i == selected_image_index)) image->DrawImage(RGBD_HIGHLIGHT_COLOR_SCHEME);
    else image->DrawImage(color_scheme);
  }
}



static void
DrawPoints(int color_scheme = RGBD_PHOTO_COLOR_SCHEME, int skip = 2)
{
  // Draw pixels of all images as points in world space
  for (int i = 0; i < configuration.NImages(); i++) {
    RGBDImage *image = configuration.Image(i);
    if (!image->DepthChannel()) continue;
    if ((color_scheme != RGBD_INDEX_COLOR_SCHEME) && (i == selected_image_index)) image->DrawPoints(RGBD_HIGHLIGHT_COLOR_SCHEME, skip);
    else image->DrawPoints(color_scheme, skip);
  }
}



static void
DrawQuads(int color_scheme = RGBD_PHOTO_COLOR_SCHEME, int skip = 2)
{
  // Draw pixels of all images as quads in world space
  for (int i = 0; i < configuration.NImages(); i++) {
    RGBDImage *image = configuration.Image(i);
    if (!image->DepthChannel()) continue;
    if ((color_scheme != RGBD_INDEX_COLOR_SCHEME) && (i == selected_image_index)) image->DrawQuads(RGBD_HIGHLIGHT_COLOR_SCHEME, skip);
    else image->DrawQuads(color_scheme, skip);
  }
}



static void
DrawBBoxes(int color_scheme = RGBD_INDEX_COLOR_SCHEME)
{
  // Draw pixels of all images as bboxs in world space
  for (int i = 0; i < configuration.NImages(); i++) {
    RGBDImage *image = configuration.Image(i);
    if ((color_scheme != RGBD_INDEX_COLOR_SCHEME) && (i == selected_image_index)) image->DrawBBox(RGBD_HIGHLIGHT_COLOR_SCHEME);
    else image->DrawBBox(color_scheme);
  }
}



static void
DrawFaces(int color_scheme = RGBD_INDEX_COLOR_SCHEME)
{
  // Draw surfaces
  for (int i = 0; i < configuration.NSurfaces(); i++) {
    RGBDSurface *surface = configuration.Surface(i);
    if ((color_scheme != RGBD_INDEX_COLOR_SCHEME) && (i == selected_image_index)) surface->DrawFaces(RGBD_HIGHLIGHT_COLOR_SCHEME);
    else surface->DrawFaces(color_scheme);
  }
}



static void
DrawEdges(int color_scheme = RGBD_INDEX_COLOR_SCHEME)
{
  // Draw surfaces
  for (int i = 0; i < configuration.NSurfaces(); i++) {
    RGBDSurface *surface = configuration.Surface(i);
    if ((color_scheme != RGBD_INDEX_COLOR_SCHEME) && (i == selected_image_index)) surface->DrawEdges(RGBD_HIGHLIGHT_COLOR_SCHEME);
    else surface->DrawEdges(color_scheme);
  }
}



static void
DrawTextures(int color_scheme = RGBD_PHOTO_COLOR_SCHEME)
{
  // Draw surfaces
  for (int i = 0; i < configuration.NSurfaces(); i++) {
    RGBDSurface *surface = configuration.Surface(i);
    if ((color_scheme != RGBD_INDEX_COLOR_SCHEME) && (i == selected_image_index)) surface->DrawTexture(RGBD_HIGHLIGHT_COLOR_SCHEME);
    else surface->DrawTexture(color_scheme);
  }
}



static void
DrawOverlaps(int color_scheme = RGBD_INDEX_COLOR_SCHEME)
{
  // Draw vertex image overlaps
  if (vertex_image_overlaps) {
    if (configuration.NSurfaces() > 0) {
      glLineWidth(3);
      glColor3d(1, 0, 0);
      glBegin(GL_LINES);
      RGBDSurface *surface = configuration.Surface(0);
      R3Mesh *mesh = surface->mesh;
      for (int i = 0; i < mesh->NVertices(); i++) {
        R3MeshVertex *vertex = mesh->Vertex(i);
        const R3Point& position = mesh->VertexPosition(vertex);
        if (R3SquaredDistance(position, center) > 0.05) continue;
        for (int j = 0; j < vertex_image_overlaps[i].NEntries(); j++) {
          RGBDImage *image = vertex_image_overlaps[i][j];
          R3LoadPoint(image->WorldViewpoint());
          R3LoadPoint(position);
        }
        if (vertex_image_overlaps[i].NEntries() > 0) break;
      }
      glEnd();
      glLineWidth(1);
    }
  }

  // Draw vertex image overlaps
  if (image_vertex_overlaps && (selected_image_index >= 0)) {
    if (configuration.NSurfaces() > 0) {
      glPointSize(3);
      glBegin(GL_POINTS);
      RGBDSurface *surface = configuration.Surface(0);
      R3Mesh *mesh = surface->mesh;
      for (int j = 0; j < image_vertex_overlaps[selected_image_index].NEntries(); j++) {
        R3MeshVertex *vertex = image_vertex_overlaps[selected_image_index][j];
        R3LoadPoint(mesh->VertexPosition(vertex));
      }
      glEnd();
      glPointSize(1);
    }
  }

  // Draw image image overlaps
  if (image_image_overlaps && (selected_image_index >= 0)) {
    glLineWidth(3);
    for (int i = 0; i < image_image_overlaps->YResolution(); i++) {
      RNScalar value = image_image_overlaps->GridValue(selected_image_index, i);
      RGBDImage *image = configuration.Image(i);
      RNLoadRgb(1.0 - 5*value, 0, 1.0);
      image->DrawCamera(RGBD_NO_COLOR_SCHEME);
    }
    glLineWidth(1);
 }
}



static void
DrawAxes(void)
{
  // Draw axes 
  RNScalar d = 1;
  glLineWidth(3);
  R3BeginLine();
  glColor3f(1, 0, 0);
  R3LoadPoint(R3zero_point + 0.5 * d * R3negx_vector);
  R3LoadPoint(R3zero_point + d * R3posx_vector);
  R3EndLine();
  R3BeginLine();
  glColor3f(0, 1, 0);
  R3LoadPoint(R3zero_point + 0.5 * d * R3negy_vector);
  R3LoadPoint(R3zero_point + d * R3posy_vector);
  R3EndLine();
  R3BeginLine();
  glColor3f(0, 0, 1);
  R3LoadPoint(R3zero_point + 0.5 * d * R3negz_vector);
  R3LoadPoint(R3zero_point + d * R3posz_vector);
  R3EndLine();
  glLineWidth(1);
}



static void 
DrawRubberBox(void)
{
  // Set rendering modes
  glDrawBuffer(GL_FRONT);
  glDisable(GL_DEPTH_TEST);
  glDepthMask(0);
  glLineStipple(1, 0xFF00);
  glEnable(GL_LINE_STIPPLE);
  glLogicOp(GL_XOR);
  glEnable(GL_COLOR_LOGIC_OP);

  // Set projection matrix
  glMatrixMode(GL_PROJECTION);  
  glPushMatrix();
  glLoadIdentity();
  gluOrtho2D(0, GLUTwindow_width, 0, GLUTwindow_height);

  // Set model view matrix
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  // Draw box
  glBegin(GL_LINE_LOOP);
  glColor3f(1.0, 1.0, 1.0);
  glVertex2f(rubber_box_corners[0][0], rubber_box_corners[0][1]);
  glVertex2f(rubber_box_corners[0][0], rubber_box_corners[1][1]);
  glVertex2f(rubber_box_corners[1][0], rubber_box_corners[1][1]);
  glVertex2f(rubber_box_corners[1][0], rubber_box_corners[0][1]);
  glEnd();

  // Reset projection matrix
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  // Reset model view matrix
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  // Reset rendering modes
  glDisable(GL_COLOR_LOGIC_OP);
  glDisable(GL_LINE_STIPPLE);
  glEnable(GL_DEPTH_TEST);
  glDepthMask(1);
  glDrawBuffer(GL_BACK);
  glFlush();
}



////////////////////////////////////////////////////////////////////////
// Selection with cursor
////////////////////////////////////////////////////////////////////////

static int
Pick(int x, int y,
  RGBDSurface **picked_surface = NULL, RGBDImage **picked_image = NULL,
  R3Point *picked_position = NULL, int pick_tolerance = 10)
{
  // Initialize pick results
  int pick_result = 0;
  if (picked_surface) *picked_surface = NULL;
  if (picked_image) *picked_image = NULL;
  if (picked_position) *picked_position = R3zero_point;

  // Draw everything
  viewer.Load();
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glPointSize(pick_tolerance);
  glLineWidth(pick_tolerance);
  if (picked_image || !picked_surface) {
    if (show_cameras) DrawCameras(RGBD_INDEX_COLOR_SCHEME);
    if (show_bboxes) DrawBBoxes(RGBD_INDEX_COLOR_SCHEME);
    if (show_points) DrawPoints(RGBD_INDEX_COLOR_SCHEME);
    if (show_quads) DrawQuads(RGBD_INDEX_COLOR_SCHEME);
  }
  if (picked_surface || !picked_image) {
    if (show_faces || show_textures) DrawFaces(RGBD_INDEX_COLOR_SCHEME);
  }
  glPointSize(1.0);
  glLineWidth(1.0);
  glFinish();

  // Read color buffer at cursor position
  unsigned char rgba[4];
  glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
  int r = rgba[0] & 0xFF;
  int g = rgba[1] & 0xFF;
  int b = rgba[2] & 0xFF;
  // int a = rgba[3] & 0xFF;

  // Figure out what was picked
  if (r > 0) {
    // Picked surface
    int surface_index = (int) ((configuration.NSurfaces()*r/255.0) + 0.5) - 1;
    if ((surface_index < 0) || (surface_index >= configuration.NSurfaces())) return 0;
    if (picked_surface) *picked_surface = configuration.Surface(surface_index);
    pick_result = RGBD_SURFACE_SELECTION;
  }
  else if ((g > 0) || (b > 0)) {
    // Picked image
    int image_index = (int) ((configuration.NImages()*((g << 8) | b) / 65535.0) + 0.5) - 1;
    if ((image_index < 0) || (image_index >= configuration.NImages())) return 0;
    if (picked_image) *picked_image = configuration.Image(image_index);
    pick_result = RGBD_IMAGE_SELECTION;
  }
  else {
    // Picked background
    return RGBD_NO_SELECTION;
  }

  // Return position
  if (picked_position) {
    // Find hit position
    GLfloat depth;
    GLdouble p[3];
    GLint viewport[4];
    GLdouble modelview_matrix[16];
    GLdouble projection_matrix[16];
    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetDoublev(GL_MODELVIEW_MATRIX, modelview_matrix);
    glGetDoublev(GL_PROJECTION_MATRIX, projection_matrix);
    glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
    gluUnProject(x, y, depth, modelview_matrix, projection_matrix, viewport, &(p[0]), &(p[1]), &(p[2]));
    R3Point position(p[0], p[1], p[2]);
    *picked_position = position;
  }

  // Return success
  return pick_result;
}



static int
ProcessRubberBox(void)
{
  // Create surface ...
  
  // Create rubber box (with sorted corners)
  R2Box rubber_box = R2null_box;
  rubber_box.Union(rubber_box_corners[0]);
  rubber_box.Union(rubber_box_corners[1]);
  if ((rubber_box.XLength() < 10) || (rubber_box.YLength() < 10)) {
    printf("Box is too small -- could not create surface\n");
    fflush(stdout);
    return 0;
  }
   
  // Create random set of points projecting inside rubber box
  int npoints = 0;
  int max_npoints = 1024;
  R3Point *points = new R3Point [ max_npoints ];
  for (int i = 0; i < 10 * max_npoints; i++) {
    int image_index = (int) (RNRandomScalar() * configuration.NImages());
    RGBDImage *image = configuration.Image(image_index);
    double x = RNRandomScalar() * image->NPixels(RN_X);
    double y = RNRandomScalar() * image->NPixels(RN_Y);
    R3Point world_position;
    if (!RGBDTransformImageToWorld(R2Point(x, y), world_position, image)) continue;
    R2Point viewport_position = viewer.ViewportPoint(world_position);
    if (!R2Contains(rubber_box, viewport_position)) continue;
    points[npoints++] = world_position;
    if (npoints >= max_npoints) break;
  }
  
  // Check number of points
  if (npoints < 3) {
    printf("Did not find enough points inside box -- could not create surface\n");
    delete [] points;
    fflush(stdout);
    return 0;
  }
  
  // Compute centroid and principle axes of points
  R3Point centroid = R3Centroid(npoints, points);
  R3Triad axes = R3PrincipleAxes(centroid, npoints, points);
  if (axes[2].Dot(viewer.Camera().Towards()) > 0) {
    axes = R3Triad(axes[0], -axes[1], -axes[2]);
  }
  
  // Compute axis vectors
  R3Vector normal = axes[2]; normal.Normalize();
  R3Vector up = viewer.Camera().Up();
  R3Vector right = normal % up; right.Normalize();
  up = right % normal; up.Normalize();
  
  // Compute radii of points
  RNScalar radius[2] = { 0, 0 };
  for (int i = 0; i < npoints; i++) {
    RNScalar dx = fabs((points[i] - centroid).Dot(right));
    RNScalar dy = fabs((points[i] - centroid).Dot(up));
    if (dx > radius[0]) radius[0] = dx;
    if (dy > radius[1]) radius[1] = dy;
  }

  // Delete points
  delete [] points;

  // Create surface
  R3Rectangle *rectangle = new R3Rectangle(centroid, right, up, radius[0], radius[1]);
  RGBDSurface *surface = new RGBDSurface(NULL, rectangle, texel_spacing);
  if (!surface) {
    printf("Ran out of memory -- could not create surface\n");
    fflush(stdout);
    return 0;
  }

  // Insert surface into configuration
  configuration.InsertSurface(surface);

  // Print message
  printf("Created surface %d\n", configuration.NSurfaces()-1);
  fflush(stdout);

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////////

static void
ResetViewer(void)
{
  // Get bounding box
  R3Box bbox = configuration.WorldBBox();

  // Restrict bounding box only to surfaces, if there are any
  if (configuration.NSurfaces() > 0) {
    bbox = R3null_box;
    for (int i = 0; i < configuration.NSurfaces(); i++) {
      RGBDSurface *surface = configuration.Surface(i);
      bbox.Union(surface->WorldBBox());
    }
  }

  // Initialize viewing center
  center = bbox.Centroid();

  // Initialize viewer
  RNLength r = bbox.DiagonalRadius();
  if (r < 10) r = 10;
  R3Point eye = center - R3negz_vector * (2.5 * r);
  R3Camera camera(eye, R3negz_vector, R3posy_vector, 0.4, 0.4, 0.01 * r, 100.0 * r);
  R2Viewport viewport(0, 0, GLUTwindow_width, GLUTwindow_height);
  viewer.SetViewport(viewport);
  viewer.SetCamera(camera);
}



static void 
AlignWithAxes(void)
{
  // Check images and surfaces
  if (configuration.NImages() == 0) return;
  
  // Compute posz = average image up vector
  R3Vector posz = R3zero_vector;
  for (int i = 0; i < configuration.NImages(); i++) {
    RGBDImage *image = configuration.Image(i);
    posz += image->WorldUp();
  }

  // Compute origin and posy by averaging positions and normals of pixels
  int count = 0;
  R3Point origin = R3zero_point;
  R3Vector posy = R3zero_vector;
  for (int i = 0; i < configuration.NImages(); i++) {
    RGBDImage *image = configuration.Image(i);
    int ix = image->NPixels(RN_X)/2;
    int iy = image->NPixels(RN_Y)/2;
    R3Point world_position;
    if (RGBDTransformImageToWorld(R2Point(ix, iy), world_position, image)) {
      posy += -(image->PixelWorldNormal(ix, iy));
      origin += world_position;
      count++;
    }
  }

  // Compute averages
  if (count > 0) origin /= count;
  else return;

  // Compute orthogonal triad of axes
  posy.Normalize();
  posz.Normalize();
  R3Vector posx = posy % posz; posx.Normalize();
  posz = posx % posy; posz.Normalize();
  R3Triad axes(posx, posy, posz);
  
  // Compute transformation
  R3CoordSystem cs(origin, axes);
  R3Affine transformation(cs.InverseMatrix(), 0);

  // Apply transformation
  configuration.Transform(transformation);
}



static int
DeleteSelectedSurface(void)
{
  // Print message
  if (selected_surface_index >= 0) {
    printf("Deleted surface %d\n", selected_surface_index);
    fflush(stdout);
  }
  else {
    printf("Select a surface first, and then type ctrl-d if you want to delete it\n");
    fflush(stdout);
    return 0;
  }

  // Delete selected surface
  RGBDSurface *selected_surface = configuration.Surface(selected_surface_index);
  delete selected_surface;

  // Reset selected surface index
  selected_surface_index = -1;

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// GLUT user interface functions
////////////////////////////////////////////////////////////////////////

void GLUTStop(void)
{
  // Write configuration
  if (output_configuration_filename) {
    if (!WriteConfiguration(configuration, output_configuration_filename)) exit(-1);
  }

  // Destroy window 
  glutDestroyWindow(GLUTwindow);

  // Exit
  exit(0);
}



void GLUTRedraw(void)
{
  // Set viewing transformation
  viewer.Camera().Load();

  // Clear window 
  // glClearColor(0.0, 0.0, 0.0, 1.0);
  glClearColor(1.0, 1.0, 1.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Draw everything
  if (show_cameras) DrawCameras(color_scheme);
  if (show_bboxes) DrawBBoxes(color_scheme);
  if (show_images) DrawImages(color_scheme);
  if (show_faces) DrawFaces(color_scheme);
  if (show_edges) DrawEdges(color_scheme);
  if (show_points) DrawPoints(color_scheme);
  if (show_quads) DrawQuads(color_scheme);
  if (show_textures) DrawTextures(color_scheme);
  if (show_overlaps) DrawOverlaps();
  if (show_axes) DrawAxes();

  // Capture screenshot image 
  if (screenshot_image_name) {
    if (print_verbose) printf("Creating image %s\n", screenshot_image_name);
    R2Image image(GLUTwindow_width, GLUTwindow_height, 3);
    image.Capture();
    image.Write(screenshot_image_name);
    screenshot_image_name = NULL;
  }

  // Swap buffers 
  glutSwapBuffers();
}    



void GLUTResize(int w, int h)
{
  // Resize window
  glViewport(0, 0, w, h);

  // Resize viewer viewport
  viewer.ResizeViewport(0, 0, w, h);

  // Remember window size 
  GLUTwindow_width = w;
  GLUTwindow_height = h;

  // Redraw
  glutPostRedisplay();
}



void GLUTMotion(int x, int y)
{
  // Invert y coordinate
  y = GLUTwindow_height - y;

  // Compute mouse movement
  int dx = x - GLUTmouse[0];
  int dy = y - GLUTmouse[1];
  
  // Update mouse drag
  GLUTmouse_drag += dx*dx + dy*dy;

  // Check if control key is down
  if (glutGetModifiers() & GLUT_ACTIVE_CTRL) {
    // Update rubber box
    if (rubber_box_active) {
      DrawRubberBox();
      rubber_box_corners[1] = R2Point(x, y);
      DrawRubberBox();
    }
    else {
      rubber_box_active = TRUE;
      rubber_box_corners[0] = R2Point(x, y);
      rubber_box_corners[1] = R2Point(x, y);
      DrawRubberBox();
    }
  }
  else {
    // World in hand navigation 
    if (GLUTbutton[0]) viewer.RotateWorld(1.0, center, x, y, dx, dy);
    else if (GLUTbutton[1]) viewer.ScaleWorld(1.0, center, x, y, dx, dy);
    else if (GLUTbutton[2]) viewer.TranslateWorld(1.0, center, x, y, dx, dy);
    if (GLUTbutton[0] || GLUTbutton[1] || GLUTbutton[2]) glutPostRedisplay();
  }

  // Remember mouse position 
  GLUTmouse[0] = x;
  GLUTmouse[1] = y;
}



void GLUTMouse(int button, int state, int x, int y)
{
  // Invert y coordinate
  y = GLUTwindow_height - y;

  // Mouse is going down
  if (state == GLUT_DOWN) {
    // Reset mouse drag
    GLUTmouse_drag = 0;

    // Process thumbwheel
    if (button == 3) viewer.ScaleWorld(center, 0.9);
    else if (button == 4) viewer.ScaleWorld(center, 1.1);
  }
  else if (button == 0) {
    // Check for double click  
    static RNBoolean double_click = FALSE;
    static RNTime last_mouse_up_time;
    double_click = (!double_click) && (last_mouse_up_time.Elapsed() < 0.4);
    last_mouse_up_time.Read();

    // Check for click (rather than drag)
    if (GLUTmouse_drag < 100) {
      // Reset selected indices
      selected_surface_index = -1;
      selected_image_index = -1;

      // Select image or surface
      R3Point selected_position(0,0,0);
      RGBDImage *selected_image = NULL;
      RGBDSurface *selected_surface = NULL;
      if (Pick(x, y, &selected_surface, &selected_image, &selected_position)) {
        // Remember selected indices
        if (selected_surface) selected_surface_index = selected_surface->ConfigurationIndex();
        if (selected_image) selected_image_index = selected_image->ConfigurationIndex();

        // Print message
        if (selected_surface) {
          printf("Selected surface %d at %g %g %g\n", selected_surface->ConfigurationIndex(),
            selected_position.X(), selected_position.Y(), selected_position.Z());
        }
        if (selected_image) {
          printf("Selected image %d and %g %g %g\n", selected_image->ConfigurationIndex(),
            selected_position.X(), selected_position.Y(), selected_position.Z());
        }
        
        // Set viewing center of rotation and scale if double-click
        // if (double_click) center = selected_position;
        center = selected_position;
      }
    }
    else {
      // Process rubber box 
      if (rubber_box_active) {
        DrawRubberBox();
        ProcessRubberBox();
        rubber_box_corners[0] = R2zero_point;
        rubber_box_corners[1] = R2zero_point;
        rubber_box_active = FALSE;
      }
    }
  }

  // Remember button state 
  int b = (button == GLUT_LEFT_BUTTON) ? 0 : ((button == GLUT_MIDDLE_BUTTON) ? 1 : 2);
  GLUTbutton[b] = (state == GLUT_DOWN) ? 1 : 0;

  // Remember modifiers 
  GLUTmodifiers = glutGetModifiers();

   // Remember mouse position 
  GLUTmouse[0] = x;
  GLUTmouse[1] = y;

  // Redraw
  glutPostRedisplay();
}



void GLUTSpecial(int key, int x, int y)
{
  // Invert y coordinate
  y = GLUTwindow_height - y;

  // Process keyboard button event 
  switch (key) {
  case GLUT_KEY_F1:
    AlignWithAxes();
    ResetViewer();
    break;

  case GLUT_KEY_LEFT:
  case GLUT_KEY_RIGHT:
    // Select next image
    if (configuration.NImages() > 0) {
      snap_image_index += (key == GLUT_KEY_RIGHT) ? 1 : -1;
      if (snap_image_index < 0) snap_image_index = 0;
      if (snap_image_index >= configuration.NImages()) snap_image_index = configuration.NImages()-1;
      RGBDImage *snap_image = configuration.Image(snap_image_index);
      for (int i = 0; i < configuration.NImages(); i++) {
        RGBDImage *image = configuration.Image(i);
        if (image == snap_image) { if (!image->DepthChannel()) image->ReadChannels(); }
        else { if (image->DepthChannel()) image->ReleaseChannels(); }
      }
    }
    break;
  }

  // Remember mouse position 
  GLUTmouse[0] = x;
  GLUTmouse[1] = y;

  // Remember modifiers 
  GLUTmodifiers = glutGetModifiers();

  // Redraw
  glutPostRedisplay();
}



void GLUTKeyboard(unsigned char key, int x, int y)
{
  // Invert y coordinate
  y = GLUTwindow_height - y;

  // Process keyboard button event
  switch (key) {
  case '~': {
    // Dump screen shot to file iX.jpg
    static char buffer[64];
    static int image_count = 1;
    sprintf(buffer, "i%d.jpg", image_count++);
    screenshot_image_name = buffer;
    break; }

  case 1: // ctrl-A
    for (int i = 0; i < configuration.NImages(); i++) {
      RGBDImage *image = configuration.Image(i);
      if (!image->DepthChannel()) image->ReadChannels();
    }
    break;

  case 26: // ctrl-Z
    for (int i = 0; i < configuration.NImages(); i++) {
      RGBDImage *image = configuration.Image(i);
      if (image->DepthChannel()) image->ReleaseChannels();
    }
    break;

  case 4: // ctrl-D
    DeleteSelectedSurface();
    break;

  case 18: // ctrl-R
    ResetViewer();
    break;

  case 'A':
  case 'a':
    show_axes = !show_axes;
    break;

  case 'B':
  case 'b':
    show_bboxes = !show_bboxes;
    break;

  case 'C':
  case 'c':
    show_cameras = !show_cameras;
    break;

  case 'D':
  case 'd':
    if (color_scheme == RGBD_PHOTO_COLOR_SCHEME) color_scheme = RGBD_RENDER_COLOR_SCHEME;
    else if (color_scheme == RGBD_RENDER_COLOR_SCHEME) color_scheme = RGBD_INDEX_COLOR_SCHEME;
    else color_scheme = RGBD_PHOTO_COLOR_SCHEME;
    break;

  case 'E':
  case 'e':
    show_edges = !show_edges;
    break;

  case 'F':
  case 'f':
    show_faces = !show_faces;
    break;

  case 'I':
  case 'i':
    show_images = !show_images;
    break;

  case 'O':
  case 'o':
    show_overlaps = !show_overlaps;
    break;

  case 'P':
  case 'p':
    show_points = !show_points;
    break;

  case 'q':
  case 'Q':
    show_quads = !show_quads;
    break;

  case 'T':
  case 't':
    show_textures = !show_textures;
    break;

  case 'V':
  case 'v':
    if (snap_image_index >= 0) {
      RGBDImage *snap_image = configuration.Image(snap_image_index);
      viewer.RepositionCamera(snap_image->WorldViewpoint());
      viewer.ReorientCamera(snap_image->WorldTowards(), snap_image->WorldUp());
      center = snap_image->WorldViewpoint() + 3 * snap_image->WorldTowards();
    }
    break;
      
  case ' ': {
    RGBDImage *selected_image = NULL;
    if (Pick(x, y, NULL, &selected_image)) {
      if (!selected_image->DepthChannel()) selected_image->ReadChannels();
      else selected_image->ReleaseChannels();
      selected_image_index = selected_image->ConfigurationIndex();
    }
    break; }

  case '!': {
    // Print camera
    const R3Camera& camera = viewer.Camera();
    printf("%g %g %g  %g %g %g  %g %g %g  %g %g  1\n",
           camera.Origin().X(), camera.Origin().Y(), camera.Origin().Z(),
           camera.Towards().X(), camera.Towards().Y(), camera.Towards().Z(),
           camera.Up().X(), camera.Up().Y(), camera.Up().Z(),
           camera.XFOV(), camera.YFOV());
    break; }

  case 27: // ESCAPE
    GLUTStop();
    break;
  }

  // Remember mouse position 
  GLUTmouse[0] = x;
  GLUTmouse[1] = GLUTwindow_height - y;

  // Remember modifiers 
  GLUTmodifiers = glutGetModifiers();

  // Redraw
  glutPostRedisplay();  
}




void GLUTInterface(void)
{
  // Open window
  int argc = 0;
  char *argv[1];
  argv[argc++] = strdup("conf2texture");
  glutInit(&argc, argv);
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(GLUTwindow_width, GLUTwindow_height);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH); // | GLUT_STENCIL
  GLUTwindow = glutCreateWindow("Configuration Viewer");

  // Initialize graphics modes  
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);

  // Initialize lighting
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  static GLfloat lmodel_ambient[] = { 0.2, 0.2, 0.2, 1.0 };
  glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
  glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
  glEnable(GL_NORMALIZE);

  // Define headlight
  static GLfloat light0_diffuse[] = { 0.5, 0.5, 0.5, 1.0 };
  static GLfloat light0_position[] = { 0.0, 0.0, 1.0, 0.0 };
  glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
  glLightfv(GL_LIGHT0, GL_POSITION, light0_position);
  glEnable(GL_LIGHT0);

  // Initialize GLUT callback functions 
  glutDisplayFunc(GLUTRedraw);
  glutReshapeFunc(GLUTResize);
  glutKeyboardFunc(GLUTKeyboard);
  glutSpecialFunc(GLUTSpecial);
  glutMouseFunc(GLUTMouse);
  glutMotionFunc(GLUTMotion);

  // Reset viewer
  ResetViewer();

  // Run main loop -- never returns 
  glutMainLoop();
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
      else if (!strcmp(*argv, "-mesh")) { argc--; argv++; input_mesh_filename = *argv; }
      else if (!strcmp(*argv, "-overlap_file")) { argc--; argv++; input_overlap_filename = *argv; }
      else if (!strcmp(*argv, "-overlap_matrix")) { argc--; argv++; input_overlap_matrix = *argv; }
      else if (!strcmp(*argv, "-load_every_kth_image")) { argc--; argv++; load_every_kth_image = atoi(*argv); }
      else if (!strcmp(*argv, "-texel_spacing")) { argc--; argv++; texel_spacing = atof(*argv); }
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
    else {
      if (!input_configuration_filename) input_configuration_filename = *argv;
      else if (!output_configuration_filename) output_configuration_filename = *argv;
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
  }

  // Check filenames
  if (!input_configuration_filename) {
    fprintf(stderr, "Usage: confview inputconfigurationfile [outputconfigurationfile] [options]\n");
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
  if (!ReadConfiguration(configuration, input_configuration_filename)) exit(-1);

  // Read mesh
  if (input_mesh_filename) {
    if (!ReadMesh(configuration, input_mesh_filename)) exit(-1);
  }

  // Read image-image overlaps
  if (input_overlap_matrix) {
    if (!ReadOverlapMatrix(configuration, input_overlap_matrix)) exit(-1);
  }

  // Read vertex-image overlaps
  if (input_overlap_filename) {
    if (!ReadOverlapFile(configuration, input_overlap_filename)) exit(-1);
  }

  // Begin viewing interface -- never returns
  GLUTInterface();

  // Return success 
  return 0;
}



