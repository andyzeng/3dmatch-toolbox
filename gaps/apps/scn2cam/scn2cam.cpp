// Source file for the scene camera creation program



////////////////////////////////////////////////////////////////////////
// Include files 
////////////////////////////////////////////////////////////////////////

#include "R3Graphics/R3Graphics.h"
#ifdef USE_MESA
#  include "GL/osmesa.h"
#else
#  include "fglut/fglut.h" 
#  define USE_GLUT
#endif



////////////////////////////////////////////////////////////////////////
// Program variables
////////////////////////////////////////////////////////////////////////

// Filename program variables

static char *input_scene_filename = NULL;
static char *input_cameras_filename = NULL;
static char *input_categories_filename = NULL;
static char *output_cameras_filename = NULL;
static char *output_camera_extrinsics_filename = NULL;
static char *output_camera_intrinsics_filename = NULL;
static char *output_camera_names_filename = NULL;
static char *output_nodes_filename = NULL;


// Camera creation variables

static int create_object_cameras = 0;
static int create_room_cameras = 0;
static int create_world_in_hand_cameras = 0;
static int create_path_in_room_cameras = 0;
static int interpolate_camera_trajectory = 0;


// Camera parameter variables

static int width = 640;
static int height = 480;
static double xfov = 0.5; // half-angle in radians
static double eye_height = 1.55;
static double eye_height_radius = 0.05;


// Camera sampling variables

static double position_sampling = 0.25;
static double angle_sampling = RN_PI / 3.0;
static double interpolation_step = 0.1;


// Camera scoring variables

static int scene_scoring_method = 0;
static int object_scoring_method = 0;
static double min_visible_objects = 3;
static double min_visible_fraction = 0.01;
static double min_distance_from_obstacle = 0;
static double min_score = 0;


// Rendering variables

static int glut = 1;
static int mesa = 0;


// Informational program variables

static int print_verbose = 0;
static int print_debug = 0;



////////////////////////////////////////////////////////////////////////
// Internal type definitions
////////////////////////////////////////////////////////////////////////

struct Camera : public R3Camera {
public:
  Camera(void) : R3Camera(), name(NULL) {};
  Camera(const Camera& camera) : R3Camera(camera), name((name) ? strdup(name) : NULL) {};
  Camera(const R3Camera& camera, const char *name) : R3Camera(camera), name((name) ? strdup(name) : NULL) {};
  Camera(const R3Point& origin, const R3Vector& towards, const R3Vector& up, RNAngle xfov, RNAngle yfov, RNLength neardist, RNLength fardist)
    : R3Camera(origin, towards, up, xfov, yfov, neardist, fardist), name(NULL) {};
  ~Camera(void) { if (name) free(name); }
  char *name;
};



////////////////////////////////////////////////////////////////////////
// Internal variables
////////////////////////////////////////////////////////////////////////

// State variables

static R3Scene *scene = NULL;
static RNArray<Camera *> cameras;


// Image types

static const int NODE_INDEX_IMAGE = 0;



////////////////////////////////////////////////////////////////////////
// Input/output functions
////////////////////////////////////////////////////////////////////////

static R3Scene *
ReadScene(char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate scene
  scene = new R3Scene();
  if (!scene) {
    fprintf(stderr, "Unable to allocate scene for %s\n", filename);
    return NULL;
  }

  // Read scene from file
  if (!scene->ReadFile(filename)) {
    delete scene;
    return NULL;
  }

  // Remove references and transformations
  scene->RemoveReferences();
  scene->RemoveTransformations();

  // Print statistics
  if (print_verbose) {
    printf("Read scene from %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Nodes = %d\n", scene->NNodes());
    printf("  # Lights = %d\n", scene->NLights());
    printf("  # Materials = %d\n", scene->NMaterials());
    printf("  # Brdfs = %d\n", scene->NBrdfs());
    printf("  # Textures = %d\n", scene->NTextures());
    printf("  # Referenced models = %d\n", scene->NReferencedScenes());
    fflush(stdout);
  }

  // Return scene
  return scene;
}



static int
ReadCategories(const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Read file
  if (!scene->ReadSUNCGModelFile(filename)) return 0;

  // Print statistics
  if (print_verbose) {
    printf("Read categories from %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    fflush(stdout);
  }

  // Return success
  return 1;
} 



static int
ReadCameras(const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  int camera_count = 0;

  // Get useful variables
  RNScalar neardist = 0.01 * scene->BBox().DiagonalRadius();
  RNScalar fardist = 100 * scene->BBox().DiagonalRadius();
  RNScalar aspect = (RNScalar) height / (RNScalar) width;

  // Open file
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Unable to open cameras file %s\n", filename);
    return 0;
  }

  // Read file
  RNScalar vx, vy, vz, tx, ty, tz, ux, uy, uz, xf, yf, value;
  while (fscanf(fp, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf", &vx, &vy, &vz, &tx, &ty, &tz, &ux, &uy, &uz, &xf, &yf, &value) == (unsigned int) 12) {
    R3Point viewpoint(vx, vy, vz);
    R3Vector towards(tx, ty, tz);
    R3Vector up(ux, uy, uz);
    R3Vector right = towards % up;
    towards.Normalize();
    up = right % towards;
    up.Normalize();
    yf = atan(aspect * tan(xf));
    Camera *camera = new Camera(viewpoint, towards, up, xf, yf, neardist, fardist);
    camera->SetValue(value);
    cameras.Insert(camera);
    camera_count++;
  }

  // Close file
  fclose(fp);

  // Print statistics
  if (print_verbose) {
    printf("Read cameras from %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Cameras = %d\n", camera_count);
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
WriteCameras(const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Open file
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "Unable to open cameras file %s\n", filename);
    return 0;
  }

  // Write file
  for (int i = 0; i < cameras.NEntries(); i++) {
    Camera *camera = cameras.Kth(i);
    R3Point eye = camera->Origin();
    R3Vector towards = camera->Towards();
    R3Vector up = camera->Up();
    fprintf(fp, "%g %g %g  %g %g %g  %g %g %g  %g %g  %g\n",
      eye.X(), eye.Y(), eye.Z(),
      towards.X(), towards.Y(), towards.Z(),
      up.X(), up.Y(), up.Z(),
      camera->XFOV(), camera->YFOV(),
      camera->Value());
  }

  // Close file
  fclose(fp);

  // Print statistics
  if (print_verbose) {
    printf("Wrote cameras to %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Cameras = %d\n", cameras.NEntries());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
WriteCameraExtrinsics(const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Open file
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "Unable to open camera extrinsics file %s\n", filename);
    return 0;
  }

  // Write file
  for (int i = 0; i < cameras.NEntries(); i++) {
    Camera *camera = cameras.Kth(i);
    const R3CoordSystem& cs = camera->CoordSystem();
    R4Matrix matrix = cs.Matrix();
    fprintf(fp, "%g %g %g %g   %g %g %g %g  %g %g %g %g\n",
      matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3], 
      matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3], 
      matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3]);
  }

  // Close file
  fclose(fp);

  // Print statistics
  if (print_verbose) {
    printf("Wrote camera extrinsics to %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Cameras = %d\n", cameras.NEntries());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
WriteCameraIntrinsics(const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Open file
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "Unable to open camera intrinsics file %s\n", filename);
    return 0;
  }

  // Get center of image
  RNScalar cx = 0.5 * width;
  RNScalar cy = 0.5 * height;

  // Write file
  for (int i = 0; i < cameras.NEntries(); i++) {
    Camera *camera = cameras.Kth(i);
    RNScalar fx = 0.5 * width / atan(camera->XFOV());
    RNScalar fy = 0.5 * height / atan(camera->YFOV());
    fprintf(fp, "%g 0 %g   0 %g %g  0 0 1\n", fx, cx, fy, cy);
  }

  // Close file
  fclose(fp);

  // Print statistics
  if (print_verbose) {
    printf("Wrote camera intrinsics to %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
WriteCameraNames(const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Open file
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "Unable to open camera names file %s\n", filename);
    return 0;
  }

  // Write file
  for (int i = 0; i < cameras.NEntries(); i++) {
    Camera *camera = cameras.Kth(i);
    fprintf(fp, "%s\n", (camera->name) ? camera->name : "-");
  }

  // Close file
  fclose(fp);

  // Print statistics
  if (print_verbose) {
    printf("Wrote camera names to %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
WriteNodeNames(const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Open file
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "Unable to open node name file %s\n", filename);
    return 0;
  }

  // Write file
  for (int i = 0; i < scene->NNodes(); i++) {
    R3SceneNode *node = scene->Node(i);
    const char *name = (node->Name()) ? node->Name() : "-";
    fprintf(fp, "%d %s\n", i+1, name);
  }

  // Close file
  fclose(fp);

  // Print statistics
  if (print_verbose) {
    printf("Wrote node names to %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Nodes = %d\n", scene->NNodes());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
WriteCameras(void)
{
  // Write cameras 
  if (output_cameras_filename) {
    if (!WriteCameras(output_cameras_filename)) exit(-1);
  }
  
  // Write camera extrinsics 
  if (output_camera_extrinsics_filename) {
    if (!WriteCameraExtrinsics(output_camera_extrinsics_filename)) exit(-1);
  }
  
  // Write camera intrinsics 
  if (output_camera_intrinsics_filename) {
    if (!WriteCameraIntrinsics(output_camera_intrinsics_filename)) exit(-1);
  }

  // Write camera names 
  if (output_camera_names_filename) {
    if (!WriteCameraNames(output_camera_names_filename)) exit(-1);
  }

  // Write node names
  if (output_nodes_filename) {
    if (!WriteNodeNames(output_nodes_filename)) exit(-1);
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// OpenGL image capture functions
////////////////////////////////////////////////////////////////////////

#if 0
static int
CaptureColor(R2Image& image)
{
  // Capture image 
  image.Capture();

  // Return success
  return 1;
}
#endif



static int
CaptureScalar(R2Grid& scalar_image)
{
  // Capture rgb pixels
  unsigned char *pixels = new unsigned char [ 3 * width * height ];
  glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

  // Fill scalar image
  unsigned char *pixelp = pixels;
  for (int iy = 0; iy < height; iy++) {
    for (int ix = 0; ix < width; ix++) {
      unsigned int value = 0;
      value |= (*pixelp++ << 16) & 0xFF0000;
      value |= (*pixelp++ <<  8) & 0x00FF00;
      value |= (*pixelp++      ) & 0x0000FF;
      scalar_image.SetGridValue(ix, iy, value);
    }
  }

  // Delete rgb pixels
  delete [] pixels;
  
  // Return success
  return 1;
}



#if 0
static int 
CaptureDepth(R2Grid& image)
{
  // Get viewport dimensions
  GLint viewport[4];
  glGetIntegerv(GL_VIEWPORT, viewport);

  // Get modelview  matrix
  static GLdouble modelview_matrix[16];
  // glGetDoublev(GL_MODELVIEW_MATRIX, modelview_matrix);
  for (int i = 0; i < 16; i++) modelview_matrix[i] = 0;
  modelview_matrix[0] = 1.0;
  modelview_matrix[5] = 1.0;
  modelview_matrix[10] = 1.0;
  modelview_matrix[15] = 1.0;
  
  // Get projection matrix
  GLdouble projection_matrix[16];
  glGetDoublev(GL_PROJECTION_MATRIX, projection_matrix);

  // Get viewpoint matrix
  GLint viewport_matrix[16];
  glGetIntegerv(GL_VIEWPORT, viewport_matrix);

  // Allocate pixels
  float *pixels = new float [ image.NEntries() ];

  // Read pixels from frame buffer 
  glReadPixels(0, 0, viewport[2], viewport[3], GL_DEPTH_COMPONENT, GL_FLOAT, pixels); 

  // Resize image
  image.Clear(0.0);
  
  // Convert pixels to depths
  int ix, iy;
  double x, y, z;
  for (int i = 0; i < image.NEntries(); i++) {
    if (RNIsEqual(pixels[i], 1.0)) continue;
    if (RNIsNegativeOrZero(pixels[i])) continue;
    image.IndexToIndices(i, ix, iy);
    gluUnProject(ix, iy, pixels[i], modelview_matrix, projection_matrix, viewport_matrix, &x, &y, &z);
    image.SetGridValue(i, -z);
  }

  // Delete pixels
  delete [] pixels;

  // Return success
  return 1;
}
#endif



static void 
DrawNodeWithOpenGL(R3Scene *scene, R3SceneNode *node, R3SceneNode *selected_node, int image_type)
{
  // Set color based on node index
  RNFlags draw_flags = R3_DEFAULT_DRAW_FLAGS;
  if (image_type == NODE_INDEX_IMAGE) {
    draw_flags = R3_SURFACES_DRAW_FLAG;
    unsigned int node_index = node->SceneIndex() + 1;
    unsigned char color[4];
    color[0] = (node_index >> 16) & 0xFF;
    color[1] = (node_index >>  8) & 0xFF;
    color[2] = (node_index      ) & 0xFF;
    glColor3ubv(color);
  }
  
  // Draw elements
  if (!selected_node || (selected_node == node)) {
    for (int i = 0; i < node->NElements(); i++) {
      R3SceneElement *element = node->Element(i);
      element->Draw(draw_flags);
    }
  }

  // Draw references 
  if (!selected_node || (selected_node == node)) {
    for (int i = 0; i < node->NReferences(); i++) {
      R3SceneReference *reference = node->Reference(i);
      reference->Draw(draw_flags);
    }
  }

  // Draw children
  for (int i = 0; i < node->NChildren(); i++) {
    R3SceneNode *child = node->Child(i);
    DrawNodeWithOpenGL(scene, child, selected_node, image_type);
  }
}




static void 
RenderImageWithOpenGL(R2Grid& image, const R3Camera& camera, R3Scene *scene, R3SceneNode *root_node, R3SceneNode *selected_node, int image_type)
{
  // Clear window
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Initialize transformation
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // Load camera and viewport
  camera.Load();
  glViewport(0, 0, width, height);

  // Initialize graphics modes  
  glEnable(GL_DEPTH_TEST);

  // Draw scene
  R3null_material.Draw();
  DrawNodeWithOpenGL(scene, root_node, selected_node, image_type);
  R3null_material.Draw();

  // Read frame buffer into image
  CaptureScalar(image);

  // Compensate for rendering background black
  image.Substitute(0, R2_GRID_UNKNOWN_VALUE);
  if (image_type == NODE_INDEX_IMAGE) image.Subtract(1.0);
}



////////////////////////////////////////////////////////////////////////
// Raycasting image capture functions
////////////////////////////////////////////////////////////////////////

static void
RenderImageWithRayCasting(R2Grid& image, const R3Camera& camera, R3Scene *scene, R3SceneNode *root_node,  R3SceneNode *selected_node, int image_type)
{
  // Clear image
  image.Clear(R2_GRID_UNKNOWN_VALUE);

  // Setup viewer
  R2Viewport viewport(0, 0, image.XResolution(), image.YResolution());
  R3Viewer viewer(camera, viewport);

  // Render image with ray casting
  for (int iy = 0; iy < image.YResolution(); iy++) {
    for (int ix = 0; ix < image.XResolution(); ix++) {
      R3Ray ray = viewer.WorldRay(ix, iy);
      R3SceneNode *intersection_node = NULL;
      if (root_node->Intersects(ray, &intersection_node)) {
        if (intersection_node) {
          if (!selected_node || (selected_node == intersection_node)) {
            if (image_type == NODE_INDEX_IMAGE) {
              image.SetGridValue(ix, iy, intersection_node->SceneIndex());
            }
          }
        }
      }
    }
  }
}

  

////////////////////////////////////////////////////////////////////////
// Image capture functions
////////////////////////////////////////////////////////////////////////

static void
RenderImage(R2Grid& image, const R3Camera& camera, R3Scene *scene, R3SceneNode *root_node, R3SceneNode *selected_node, int image_type)
{
  // Check rendering method
  if (glut || mesa) RenderImageWithOpenGL(image, camera, scene, root_node, selected_node, image_type);
  else RenderImageWithRayCasting(image, camera, scene, root_node, selected_node, image_type);
}




////////////////////////////////////////////////////////////////////////
// Camera scoring function
////////////////////////////////////////////////////////////////////////

static RNBoolean
IsObject(R3SceneNode *node)
{
  // Check name
  if (!node->Name()) return 0;
  if (strncmp(node->Name(), "Model#", 6)) return 0;

  // Get category identifier
  R3SceneNode *ancestor = node;
  const char *category_identifier = NULL;
  while (!category_identifier && ancestor) {
    category_identifier = node->Info("empty_struct_obj");
    ancestor = ancestor->Parent();
  }

  // Check category identifier
  if (category_identifier) {
    if (strcmp(category_identifier, "2")) return 0;
  }

  // Passed all tests
  return 1;
}



static RNScalar
ObjectCoverageScore(const R3Camera& camera, R3Scene *scene, R3SceneNode *node)
{
  // Allocate points
  const int max_npoints = 1024;
  const int target_npoints = 512;
  static R3Point points[max_npoints];
  static int npoints = 0;

  // Check if same node as last time
  // (NOTE: THIS WILL NOT PARALLELIZE)
  static R3SceneNode *last_node = NULL;
  if (last_node != node) {
    last_node = node;
    npoints = 0;

    // Generate points on surface of node
    RNArea total_area = 0;
    for (int j = 0; j < node->NElements(); j++) {
      R3SceneElement *element = node->Element(j);
      for (int k = 0; k < element->NShapes(); k++) {
        R3Shape *shape = element->Shape(k);
        RNScalar area = shape->Area();
        total_area += area;
      }
    }

    // Check total area
    if (RNIsZero(total_area)) return 0;

    // Generate points 
    for (int i = 0; i < node->NElements(); i++) {
      R3SceneElement *element = node->Element(i);
      for (int j = 0; j < element->NShapes(); j++) {
        R3Shape *shape = element->Shape(j);
        if (shape->ClassID() == R3TriangleArray::CLASS_ID()) {
          R3TriangleArray *triangles = (R3TriangleArray *) shape;
          for (int k = 0; k < triangles->NTriangles(); k++) {
            R3Triangle *triangle = triangles->Triangle(k);
            RNScalar area = triangle->Area();
            RNScalar real_nsamples = target_npoints * area / total_area;
            int nsamples = (int) real_nsamples;
            if (RNRandomScalar() < (real_nsamples - nsamples)) nsamples++;
            for (int m = 0; m < nsamples; m++) {
              if (npoints >= max_npoints) break;
              R3Point point = triangle->RandomPoint();
              points[npoints++] = point;
            }
          }
        }
      }
    }
  }

  // Check number of points
  if (npoints == 0) return 0;

  // Count how many points are visible
  int nvisible = 0;
  for (int i = 0; i < npoints; i++) {
    const R3Point& point = points[i];
    R3Ray ray(camera.Origin(), point);
    const RNScalar tolerance_t = 0.01;
    RNScalar max_t = R3Distance(camera.Origin(), point) + tolerance_t;
    RNScalar hit_t = FLT_MAX;
    R3SceneNode *hit_node = NULL;
    if (scene->Intersects(ray, &hit_node, NULL, NULL, NULL, NULL, &hit_t, 0, max_t)) {
      if ((hit_node == node) && (RNIsEqual(hit_t, max_t, tolerance_t))) nvisible++;
    }
  }

  // Compute score as fraction of points that are visible
  RNScalar score = (RNScalar) nvisible/ (RNScalar) npoints;

  // Return score
  return score;
}



static RNScalar
SceneCoverageScore(const R3Camera& camera, R3Scene *scene, R3SceneNode *room_node = NULL)
{
  // Allocate image for scoring
  R2Grid image(width, height);

  // Compute maximum number of pixels in image
  int max_pixel_count = width * height;
  if (max_pixel_count == 0) return 0;

  // Compute minimum number of pixels per object
  int min_pixel_count_per_object = min_visible_fraction * max_pixel_count;
  if (min_pixel_count_per_object == 0) return 0;

  // Render image
  RenderImage(image, camera, scene, scene->Root(), NULL, NODE_INDEX_IMAGE);

  // Allocate buffer for counting visible pixels of nodes
  int *node_pixel_counts = new int [ scene->NNodes() ];
  for (int i = 0; i < scene->NNodes(); i++) node_pixel_counts[i] = 0;
  
  // Log counts of pixels visible on each node
  for (int i = 0; i < image.NEntries(); i++) {      
    RNScalar value = image.GridValue(i);
    if (value == R2_GRID_UNKNOWN_VALUE) continue;
    int node_index = (int) (value + 0.5);
    if ((node_index < 0) || (node_index >= scene->NNodes())) continue;
    node_pixel_counts[node_index]++;
  }

  // Compute score
  RNScalar sum = 0;
  int node_count = 0;
  for (int i = 0; i < scene->NNodes(); i++) {
    R3SceneNode *node = scene->Node(i);
    if (!IsObject(node)) continue;
    if (room_node && !node->IsDecendent(room_node)) continue;
    if (node_pixel_counts[i] <= min_pixel_count_per_object) continue;
    sum += log(node_pixel_counts[i] / min_pixel_count_per_object);
    node_count++;
  }

  // Compute score (log of product of number of pixels visible in each object)
  RNScalar score = (node_count > min_visible_objects) ? sum : 0;

  // Delete pixel counts
  delete [] node_pixel_counts;

  // Return score
  return score;
}



////////////////////////////////////////////////////////////////////////
// Mask creation functions
////////////////////////////////////////////////////////////////////////

static void
RasterizeIntoZXGrid(R2Grid& grid, R3SceneNode *node,
  const R3Box& world_bbox)
{
  // Check bounding box
  R3Box node_bbox = node->BBox();
  if (!R3Intersects(world_bbox, node_bbox)) return;
  
  // Rasterize elements into grid
  for (int j = 0; j < node->NElements(); j++) {
    R3SceneElement *element = node->Element(j);
    for (int k = 0; k < element->NShapes(); k++) {
      R3Shape *shape = element->Shape(k);
      R3Box shape_bbox = shape->BBox();
      if (!R3Intersects(world_bbox, shape_bbox)) continue;
      if (shape->ClassID() == R3TriangleArray::CLASS_ID()) {
        R3TriangleArray *triangles = (R3TriangleArray *) shape;
        for (int m = 0; m < triangles->NTriangles(); m++) {
          R3Triangle *triangle = triangles->Triangle(m);
          R3Box triangle_bbox = triangle->BBox();
          if (!R3Intersects(world_bbox, triangle_bbox)) continue;
          R3TriangleVertex *v0 = triangle->V0();
          R3Point vp0 = v0->Position();
          R2Point p0(vp0.Z(), vp0.X());
          if (!R2Contains(grid.WorldBox(), p0)) continue;
          R3TriangleVertex *v1 = triangle->V1();
          R3Point vp1 = v1->Position();
          R2Point p1(vp1.Z(), vp1.X());
          if (!R2Contains(grid.WorldBox(), p1)) continue;
          R3TriangleVertex *v2 = triangle->V2();
          R3Point vp2 = v2->Position();
          R2Point p2(vp2.Z(), vp2.X());
          if (!R2Contains(grid.WorldBox(), p2)) continue;
          grid.RasterizeWorldTriangle(p0, p1, p2, 1.0);
        }
      }
    }
  }

  // Rasterize children into grid
  for (int j = 0; j < node->NChildren(); j++) {
    R3SceneNode *child = node->Child(j);
    RasterizeIntoZXGrid(grid, child, world_bbox);
  }
}



static int
ComputeViewpointMask(R3SceneNode *room_node, R2Grid& mask) 
{
  // Get/check room, wall, floor, and ceiling nodes 
  if (!room_node) return 0;
  if (!room_node->Name()) return 0;
  if (strncmp(room_node->Name(), "Room#", 5)) return 0;
  if (room_node->NChildren() < 3) return 0;
  R3SceneNode *floor_node = room_node->Child(0);
  if (!floor_node->Name()) return 0;
  if (strncmp(floor_node->Name(), "Floor#", 6)) return 0;
  R3SceneNode *ceiling_node = room_node->Child(1);
  if (!ceiling_node->Name()) return 0;
  if (strncmp(ceiling_node->Name(), "Ceiling#", 8)) return 0;
  R3SceneNode *wall_node = room_node->Child(2);
  if (!wall_node->Name()) return 0;
  if (strncmp(wall_node->Name(), "Wall#", 5)) return 0;

  // Get bounding boxes in world coordinates
  R3Box room_bbox = room_node->BBox();
  R3Box floor_bbox = floor_node->BBox();
  R3Box ceiling_bbox = ceiling_node->BBox();

  // Get/check grid extent and resolution in world coordinates
  RNScalar grid_sampling_factor = 2;
  RNScalar grid_sample_spacing = min_distance_from_obstacle / grid_sampling_factor;
  if (grid_sample_spacing == 0) grid_sample_spacing = 0.05;
  if (grid_sample_spacing > 0.1) grid_sample_spacing = 0.1;
  R2Box grid_bbox(room_bbox.ZMin(), room_bbox.XMin(), room_bbox.ZMax(), room_bbox.XMax());
  int res1 = (int) (grid_bbox.XLength() / grid_sample_spacing);
  int res2 = (int) (grid_bbox.YLength() / grid_sample_spacing);
  if ((res1 < 3) || (res2 < 3)) return 0;

  // Compute floor mask
  R2Grid floor_mask = R2Grid(res1, res2, grid_bbox);
  RasterizeIntoZXGrid(floor_mask, floor_node, floor_bbox);
  floor_mask.Threshold(0.5, 0, 1);

  // Initialize object mask
  R2Grid object_mask = R2Grid(res1, res2, grid_bbox);
  R3Box object_bbox = room_bbox;
  object_bbox[RN_LO][RN_Y] = floor_bbox[RN_HI][RN_Y] + RN_EPSILON;
  object_bbox[RN_HI][RN_Y] = ceiling_bbox[RN_LO][RN_Y] - RN_EPSILON;

  // Rasterize objects associated with this room into object mask
  for (int i = 0; i < room_node->NChildren(); i++) {
    R3SceneNode *node = room_node->Child(i);
    if ((node == floor_node) || (node == ceiling_node)) continue;
    RasterizeIntoZXGrid(object_mask, node, object_bbox);
  }

  // Rasterize objects associated with no room into object mask
  for (int i = 0; i < room_node->Parent()->NChildren(); i++) {
    R3SceneNode *node = room_node->Parent()->Child(i);
    if (node->NChildren() > 0) continue;
    RasterizeIntoZXGrid(object_mask, node, object_bbox);
  }
  
  // Reverse object mask to by 1 in areas not occupied by objects
  object_mask.Threshold(0.5, 1, 0);

  // Erode object mask to cover viewpoints at least min_distance_from_obstacle
  if (min_distance_from_obstacle > 0) {
    object_mask.Erode(min_distance_from_obstacle / grid_sample_spacing);
  }

  // Combine the two masks
  mask = floor_mask;
  mask.Mask(object_mask);
  
#if 0
  // Debugging
  char buffer[4096];
  sprintf(buffer, "%s_floor_mask.jpg", room_node->Name());
  floor_mask.WriteFile(buffer);
  sprintf(buffer, "%s_object_mask.jpg", room_node->Name());
  object_mask.WriteFile(buffer);
  sprintf(buffer, "%s_mask.jpg", room_node->Name());
  mask.WriteFile(buffer);
#endif
  
  // Return success
  return 1;
}



static int
FindIndexOfRandomPoint(const R2Grid& grid)
{
  // Choose random point in connected component
  int random_point_counter = (int) (RNRandomScalar() * grid.Cardinality());
  for (int i = 0; i < grid.NEntries(); i++) {
    if (grid.GridValue(i) == R2_GRID_UNKNOWN_VALUE) continue;
    if (--random_point_counter == 0) return i;
  }

  // Should not get here
  return -1;
}



static int
FindIndexOfFurthestPointAlongPath(const R2Grid& grid, int start_index,
  int *path = NULL, int *path_size = NULL)
{
  // Initialize Dijksra bookkeeping
  R2Grid parent_grid(grid), distance_grid(grid);
  parent_grid.Clear(R2_GRID_UNKNOWN_VALUE);
  distance_grid.Clear(FLT_MAX);
  const RNScalar *grid_values = distance_grid.GridValues();
  int neighbor_index, ix, iy;
  int end_index = -1;
  
  // Compute shortest path to all nonzero points
  RNHeap<const RNScalar *> heap(0);
  distance_grid.SetGridValue(start_index, 0);
  parent_grid.SetGridValue(start_index, start_index);
  heap.Push(grid_values + start_index);
  while (!heap.IsEmpty()) {
    const RNScalar *grid_entry = heap.Pop();
    int grid_index = grid_entry - grid_values;
    end_index = grid_index;
    grid.IndexToIndices(grid_index, ix, iy);
    for (int dx = -1; dx <= 1; dx++) {
      int nx = ix + dx;
      if ((nx < 0) || (nx > grid.XResolution()-1)) continue;
      for (int dy = -1; dy <= 1; dy++) {
        int ny = iy + dy;
        if ((ny < 0) || (ny > grid.YResolution()-1)) continue;
        grid.IndicesToIndex(nx, ny, neighbor_index);
        if (neighbor_index == grid_index) continue;
        if (grid.GridValue(neighbor_index) != R2_GRID_UNKNOWN_VALUE) {
          RNScalar d = distance_grid.GridValue(grid_index) + sqrt(dx*dx + dy*dy);
          RNScalar old_d = distance_grid.GridValue(neighbor_index);
          if (d < old_d) {
            distance_grid.SetGridValue(neighbor_index, d);
            parent_grid.SetGridValue(neighbor_index, grid_index);
            if (old_d == FLT_MAX) heap.Push(grid_values + neighbor_index);
            else heap.Update(grid_values + neighbor_index);
          }
        }
      }
    }
  }

  // Fill in path
  if (path && path_size) {
    // Construct path backwards
    int count = 0;
    path[count++] = end_index;
    while (path[count-1] != start_index) {
      path[count] = (int) parent_grid.GridValue(path[count-1]);
      count++;
    }

    // Reverse order of path
    for (int i = 0; i < count/2; i++) {
      int swap = path[i];
      path[i] = path[count-1-i];
      path[count-1-i] = swap;
    }

    // Fill in path size
    *path_size = count;
  }

  // Return last visited index
  return end_index;
}



////////////////////////////////////////////////////////////////////////
// Camera creation functions
////////////////////////////////////////////////////////////////////////

static void
CreateObjectCameras(void)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  int camera_count = 0;

  // Get useful variables
  RNScalar neardist = 0.01 * scene->BBox().DiagonalRadius();
  RNScalar fardist = 100 * scene->BBox().DiagonalRadius();
  RNScalar aspect = (RNScalar) height / (RNScalar) width;
  RNAngle yfov = atan(aspect * tan(xfov));
  
  // Create camera with close up view of each object
  for (int i = 0; i < scene->NNodes(); i++) {
    R3SceneNode *node = scene->Node(i);
    if (!node->Name()) continue;
    if (!IsObject(node)) continue;
    if (node->NElements() == 0) continue;
    R3Camera best_camera;

    // Get node's centroid and radius in world coordinate system
    R3Point centroid = node->BBox().Centroid();
    RNScalar radius = node->BBox().DiagonalRadius();

    // Check lots of directions
    int nangles = (int) (RN_TWO_PI / angle_sampling + 0.5);
    RNScalar angle_spacing = (nangles > 1) ? RN_TWO_PI / nangles : RN_TWO_PI;
    for (int j = 0; j < nangles; j++) {
      // Determine view direction
      R3Vector view_direction(-1, 0, 0); 
      view_direction.YRotate((j+RNRandomScalar()) * angle_spacing);
      view_direction.Normalize();

      // Determine camera viewpoint
      RNScalar min_distance = radius;
      RNScalar max_distance = 1.5 * radius/tan(xfov);
      if (min_distance < min_distance_from_obstacle) min_distance = min_distance_from_obstacle;
      if (max_distance < min_distance_from_obstacle) max_distance = min_distance_from_obstacle;
      R3Point viewpoint = centroid - max_distance * view_direction;

      // Project camera viewpoint onto eye height plane (special for planner5d)
      if (node->Parent() && node->Parent()->Parent()) {
        if (node->Parent()->Parent()->Name()) {
          if (strstr(node->Parent()->Parent()->Name(), "Room") || strstr(node->Parent()->Parent()->Name(), "Level")) {
            R3Point floor = node->Parent()->Parent()->Centroid();
            floor[1] = node->Parent()->Parent()->BBox().YMin();
            viewpoint[1] = floor[1] + eye_height;
            viewpoint[1] += 2.0*(RNRandomScalar()-0.5) * eye_height_radius;
          }
        }
      }

      // Ensure centroid is not occluded
      R3Vector back = viewpoint - centroid;
      back.Normalize();
      R3Ray ray(centroid, back);
      RNScalar hit_t = FLT_MAX;
      if (scene->Intersects(ray, NULL, NULL, NULL, NULL, NULL, &hit_t, min_distance, max_distance)) {
        viewpoint = centroid + (hit_t - min_distance_from_obstacle) * back;
      }

      // Compute camera
      R3Vector towards = centroid - viewpoint;
      towards.Normalize();
      R3Vector right = towards % R3posy_vector;
      right.Normalize();
      R3Vector up = right % towards;
      up.Normalize();
      R3Camera camera(viewpoint, towards, up, xfov, yfov, neardist, fardist);

      // Compute score for camera
      camera.SetValue(ObjectCoverageScore(camera, scene, node));
      if (camera.Value() == 0) continue;
      if (camera.Value() < min_score) continue;
                            
      // Remember best camera
      if (camera.Value() > best_camera.Value()) {
        best_camera = camera;
      }
    }

    // Insert best camera
    if (best_camera.Value() > 0) {
      char camera_name[1024];
      const char *node_name = (node->Name()) ? node->Name() : "-";
      const char *parent_name = (node->Parent() && node->Parent()->Name()) ? node->Parent()->Name() : "-";
      sprintf(camera_name, "%s#%s", parent_name, node_name);
      if (print_debug) printf("%s %g\n", camera_name, best_camera.Value());
      Camera *camera = new Camera(best_camera, camera_name);
      cameras.Insert(camera);
      camera_count++;
    }
  }

  // Print statistics
  if (print_verbose) {
    printf("Created object cameras ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Cameras = %d\n", camera_count++);
    fflush(stdout);
  }
}



static void
CreateRoomCameras(void)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  int camera_count = 0;

  // Get useful variables
  RNScalar neardist = 0.01 * scene->BBox().DiagonalRadius();
  RNScalar fardist = 100 * scene->BBox().DiagonalRadius();
  RNScalar aspect = (RNScalar) height / (RNScalar) width;
  RNAngle yfov = atan(aspect * tan(xfov));

  // Create one camera per direction per room 
  for (int i = 0; i < scene->NNodes(); i++) {
    R3SceneNode *room_node = scene->Node(i);
    if (!room_node->Name()) continue;
    if (strncmp(room_node->Name(), "Room#", 5)) continue;

    // Compute viewpoint mask
    R2Grid viewpoint_mask;
    if (!ComputeViewpointMask(room_node, viewpoint_mask)) continue;

    // Sample directions
    int nangles = (int) (RN_TWO_PI / angle_sampling + 0.5);
    RNScalar angle_spacing = (nangles > 1) ? RN_TWO_PI / nangles : RN_TWO_PI;
    for (int j = 0; j < nangles; j++) {
      // Choose one camera for each direction in each room
      R3Camera best_camera;

      // Sample positions 
      R3Box room_bbox = room_node->BBox();
      for (RNScalar z = room_bbox.ZMin(); z < room_bbox.ZMax(); z += position_sampling) {
        for (RNScalar x = room_bbox.XMin(); x < room_bbox.XMax(); x += position_sampling) {
          // Compute position
          R2Point position(x + position_sampling*RNRandomScalar(), z + position_sampling*RNRandomScalar());

          // Check viewpoint mask
          R2Point viewpoint_mask_position(position[1], position[0]); // ZX          
          RNScalar viewpoint_mask_value = viewpoint_mask.WorldValue(viewpoint_mask_position);
          if (viewpoint_mask_value < 0.5) continue;

          // Compute height
          RNScalar y = room_bbox.YMin() + eye_height;
          y += 2.0*(RNRandomScalar()-0.5) * eye_height_radius;
          if (y > room_bbox.YMax()) continue;

          // Compute direction
          RNScalar angle = (j+RNRandomScalar()) * angle_spacing;
          R2Vector direction = R2posx_vector;
          direction.Rotate(angle);
          direction.Normalize();

          // Compute camera
          R3Point viewpoint(position[0], y, position[1]);
          R3Vector towards(direction.X(), -0.2, direction.Y());
          towards.Normalize();
          R3Vector right = towards % R3posy_vector;
          right.Normalize();
          R3Vector up = right % towards;
          up.Normalize();
          R3Camera camera(viewpoint, towards, up, xfov, yfov, neardist, fardist);

          // Compute score for camera
          camera.SetValue(SceneCoverageScore(camera, scene, room_node));
          if (camera.Value() == 0) continue;
          if (camera.Value() < min_score) continue;

          // Remember best camera
          if (camera.Value() > best_camera.Value()) {
            best_camera = camera;
          }
        }
      }

      // Insert best camera for direction in room
      if (best_camera.Value() > 0) {
        if (print_debug) printf("ROOM %s %d : %g\n", room_node->Name(), j, best_camera.Value());
        char name[1024];
        sprintf(name, "%s_%d", room_node->Name(), j);
        Camera *camera = new Camera(best_camera, name);
        cameras.Insert(camera);
        camera_count++;
      }
    }
  }
        
  // Print statistics
  if (print_verbose) {
    printf("Created room cameras ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Cameras = %d\n", camera_count++);
    fflush(stdout);
  }
}



static void
CreatePathInRoomCameras(void)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  int camera_count = 0;

  // Get useful variables
  RNScalar neardist = 0.01 * scene->BBox().DiagonalRadius();
  RNScalar fardist = 100 * scene->BBox().DiagonalRadius();
  RNScalar aspect = (RNScalar) height / (RNScalar) width;
  RNAngle yfov = atan(aspect * tan(xfov));

  // Create one camera trajectory per door
  for (int i = 0; i < scene->NNodes(); i++) {
    R3SceneNode *room_node = scene->Node(i);
    if (!room_node->Name()) continue;
    if (strncmp(room_node->Name(), "Room#", 5)) continue;
    R3Box room_bbox = room_node->BBox();

    // Compute viewpoint mask
    R2Grid viewpoint_mask;
    if (!ComputeViewpointMask(room_node, viewpoint_mask)) continue;

    // Find largest connected component
    R2Grid component_mask = viewpoint_mask;
    component_mask.ConnectedComponentSizeFilter(RN_EPSILON);
    RNScalar component_size = component_mask.Maximum();
    component_mask.Threshold(component_size - 0.5, 0, 1.0);
    component_mask.Substitute(0, R2_GRID_UNKNOWN_VALUE);
    
    // Find path between distant pair of points in largest connected component
    int path_size = 0;
    int random_point_index = FindIndexOfRandomPoint(component_mask);
    if (random_point_index < 0) continue;
    int start_point_index = FindIndexOfFurthestPointAlongPath(component_mask, random_point_index);
    if (start_point_index < 0) continue;
    int *path_indices = new int [ component_mask.NEntries() ];
    int end_point_index = FindIndexOfFurthestPointAlongPath(component_mask, start_point_index, path_indices, &path_size);
    if (end_point_index <= 0) { delete [] path_indices; continue; }

    // Find the lookat point
    RNScalar lookat_weight = 0;
    R3Point lookat_position = R3zero_point;
    for (int j = 0; j < room_node->NChildren(); j++) {
      R3SceneNode *child_node = room_node->Child(j);
      RNScalar weight = child_node->NFacets().Max();
      lookat_position += weight * child_node->Centroid();
      lookat_weight += weight;
    }
    if (lookat_weight > 0) lookat_position /= lookat_weight;
    else lookat_position = room_node->Centroid();
    
    // Sample viewpoints on the path
    int sample_ix, sample_iy;
    int path_step = position_sampling * component_mask.WorldToGridScaleFactor();
    if (path_step == 0) path_step = 1;
    for (int i = 0; i < path_size; i += path_step) {
      // Compute viewpoint
      int sample_index = path_indices[i];
      component_mask.IndexToIndices(sample_index, sample_ix, sample_iy);
      R2Point viewpoint_mask_position = component_mask.WorldPosition(sample_ix+0.5, sample_iy+0.5);  // ZX      

      // Compute height
      RNScalar y = room_bbox.YMin() + eye_height;
      y += 2.0*(RNRandomScalar()-0.5) * eye_height_radius;
      if (y > room_bbox.YMax()) continue;

      // Compute camera
      R3Point viewpoint(viewpoint_mask_position[1], y, viewpoint_mask_position[0]);
      if (R3Contains(viewpoint, lookat_position)) continue;
      R3Vector towards = lookat_position - viewpoint;
      towards.Normalize();
      R3Vector right = towards % R3posy_vector;
      right.Normalize();
      R3Vector up = right % towards;
      up.Normalize();
      R3Camera best_camera(viewpoint, towards, up, xfov, yfov, neardist, fardist);

      // Insert camera
      if (print_debug) printf("PATH %s : %d / %d\n", room_node->Name(), i, path_size);
      char name[1024];
      sprintf(name, "%s_%d", room_node->Name(), i);
      Camera *camera = new Camera(best_camera, name);
      cameras.Insert(camera);
      camera_count++;
    }

    // Delete temporary memory
    delete [] path_indices;
  }
        
  // Print statistics
  if (print_verbose) {
    printf("Created room cameras ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Cameras = %d\n", camera_count++);
    fflush(stdout);
  }
}



static void
CreateWorldInHandCameras(void)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  int camera_count = 0;

  // Get useful variables
  R3Point centroid = scene->Centroid();
  RNScalar radius = scene->BBox().DiagonalRadius();
  RNScalar neardist = 0.01 * radius;
  RNScalar fardist = 100 * radius;
  RNScalar aspect = (RNScalar) height / (RNScalar) width;
  RNAngle yfov = atan(aspect * tan(xfov));
  RNLength distance = 2.5 * radius;

  // Determine number of cameras
  int ncameras = 0;
  if (position_sampling > 0) {
    RNArea area_of_viewpoint_sphere = 4.0 * RN_PI * distance * distance;
    RNScalar area_per_camera = 4.0 * position_sampling * position_sampling;
    int position_ncameras = (int) (area_of_viewpoint_sphere / area_per_camera + 0.5);
    if (position_ncameras > ncameras) ncameras = position_ncameras;
  }
  if (angle_sampling > 0) {
    RNArea area_of_viewpoint_sphere = 4.0 * RN_PI * distance * distance;
    RNScalar arc_length = (distance * angle_sampling);
    RNScalar area_per_camera = arc_length * arc_length;
    int angle_ncameras = (int) (area_of_viewpoint_sphere / area_per_camera + 0.5);
    if (angle_ncameras > ncameras) ncameras = angle_ncameras;
  }
  if (ncameras == 0) ncameras = 1024;
  
  // Create cameras at random directions from viewpoint sphere looking at centroid
  for (int i = 0; i < ncameras; i++) {
    // Compute view directions
    R3Vector towards = R3RandomDirection();
    towards.Normalize();
    R3Vector right = towards % R3posy_vector;
    if (RNIsZero(right.Length())) continue;
    right.Normalize();
    R3Vector up = right % towards;
    if (RNIsZero(up.Length())) continue;
    up.Normalize();

    // Compute eyepoint
    RNScalar d = distance + (2.0*RNRandomScalar()-1.0) * position_sampling;
    R3Point viewpoint = centroid - d * towards;

    // Compute name
    char name[1024];
    sprintf(name, "WORLDINHAND#%d", i);

    // Create camera
    R3Camera c(viewpoint, towards, up, xfov, yfov, neardist, fardist);
    Camera *camera = new Camera(c, name);
    cameras.Insert(camera);
    camera_count++;
  }

  // Print statistics
  if (print_verbose) {
    printf("Created world in hand cameras ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Cameras = %d\n", camera_count++);
    fflush(stdout);
  }
}



////////////////////////////////////////////////////////////////////////
// Camera interpolation functions
////////////////////////////////////////////////////////////////////////

static int
InterpolateCameraTrajectory(RNLength trajectory_step = 0.1)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Set some camera parameters based on first camera
  RNLength xf = cameras.Head()->XFOV();
  RNLength yf = cameras.Head()->YFOV();
  RNLength neardist = cameras.Head()->Near();
  RNLength fardist = cameras.Head()->Far();
  
  // Create spline data
  int nkeypoints = cameras.NEntries();
  R3Point *viewpoint_keypoints =  new R3Point [ nkeypoints ];
  R3Point *towards_keypoints =  new R3Point [ nkeypoints ];
  R3Point *up_keypoints =  new R3Point [ nkeypoints ];
  RNScalar *parameters = new RNScalar [ nkeypoints ];
  for (int i = 0; i < cameras.NEntries(); i++) {
    R3Camera *camera = cameras.Kth(i);
    viewpoint_keypoints[i] = camera->Origin();
    towards_keypoints[i] = camera->Towards().Point();
    up_keypoints[i] = camera->Up().Point();
    if (i == 0) parameters[i] = 0;
    else parameters[i] = parameters[i-1] + R3Distance(viewpoint_keypoints[i], viewpoint_keypoints[i-1]) + R3InteriorAngle(towards_keypoints[i].Vector(), towards_keypoints[i-1].Vector());
  }

  // Create splines
  R3CatmullRomSpline viewpoint_spline(viewpoint_keypoints, parameters, nkeypoints);
  R3CatmullRomSpline towards_spline(towards_keypoints, parameters, nkeypoints);
  R3CatmullRomSpline up_spline(up_keypoints, parameters, nkeypoints);

  // Delete cameras
  for (int i = 0; i < cameras.NEntries(); i++) delete cameras[i];
  cameras.Empty();
  
  // Resample splines
  for (RNScalar u = viewpoint_spline.StartParameter(); u <= viewpoint_spline.EndParameter(); u += trajectory_step) {
    R3Point viewpoint = viewpoint_spline.PointPosition(u);
    R3Point towards = towards_spline.PointPosition(u);
    R3Point up = up_spline.PointPosition(u);
    Camera *camera = new Camera(viewpoint, towards.Vector(), up.Vector(), xf, yf, neardist, fardist);
    char name[1024];
    sprintf(name, "T%f", u);
    camera->name = strdup(name);
    cameras.Insert(camera);
  }

  // Delete spline data
  delete [] viewpoint_keypoints;
  delete [] towards_keypoints;
  delete [] up_keypoints;
  delete [] parameters;

  // Print statistics
  if (print_verbose) {
    printf("Interpolated camera trajectory ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Cameras = %d\n", cameras.NEntries());
    fflush(stdout);
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Camera processing functions
////////////////////////////////////////////////////////////////////////

static int
SortCameras(void)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Sort the cameras
  cameras.Sort(R3CompareCameras);

  // Print statistics
  if (print_verbose) {
    printf("Sorted cameras ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Cameras = %d\n", cameras.NEntries());
    fflush(stdout);
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Create and write functions
////////////////////////////////////////////////////////////////////////

static void
CreateAndWriteCameras(void)
{
  // Create cameras
  if (create_object_cameras) CreateObjectCameras();
  if (create_room_cameras) CreateRoomCameras();
  if (create_path_in_room_cameras) CreatePathInRoomCameras();
  if (create_world_in_hand_cameras) CreateWorldInHandCameras();

  // Create trajectory from cameras
  if (interpolate_camera_trajectory) {
    if (!InterpolateCameraTrajectory(interpolation_step)) exit(-1);
  }
  else {
    SortCameras();
  }

  // Write cameras
  WriteCameras();

  // Exit program
  exit(0);
}



static int
CreateAndWriteCamerasWithGlut(void)
{
#ifdef USE_GLUT
  // Open window
  int argc = 1;
  char *argv[1];
  argv[0] = strdup("scn2cam");
  glutInit(&argc, argv);
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(width, height);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH); 
  glutCreateWindow("Scene Camera Creation");

  // Initialize GLUT callback functions 
  glutDisplayFunc(CreateAndWriteCameras);

  // Run main loop  -- never returns 
  glutMainLoop();

  // Return success -- actually never gets here
  return 1;
#else
  // Not supported
  RNAbort("Program was not compiled with glut.  Recompile with make.\n");
  return 0;
#endif
}



static int
CreateAndWriteCamerasWithMesa(void)
{
#ifdef USE_MESA
  // Create mesa context
  OSMesaContext ctx = OSMesaCreateContextExt(OSMESA_RGBA, 32, 0, 0, NULL);
  if (!ctx) {
    fprintf(stderr, "Unable to create mesa context\n");
    return 0;
  }

  // Create frame buffer
  void *frame_buffer = malloc(width * height * 4 * sizeof(GLubyte) );
  if (!frame_buffer) {
    fprintf(stderr, "Unable to allocate mesa frame buffer\n");
    return 0;
  }

  // Assign mesa context
  if (!OSMesaMakeCurrent(ctx, frame_buffer, GL_UNSIGNED_BYTE, width, height)) {
    fprintf(stderr, "Unable to make mesa context current\n");
    return 0;
  }

  // Create cameras
  CreateAndWriteCameras();

  // Delete mesa context
  OSMesaDestroyContext(ctx);

  // Delete frame buffer
  free(frame_buffer);

  // Return success
  return 1;
#else
  // Not supported
  RNAbort("Program was not compiled with mesa.  Recompile with make mesa.\n");
  return 0;
#endif
}



////////////////////////////////////////////////////////////////////////
// Program argument parsing
////////////////////////////////////////////////////////////////////////

static int 
ParseArgs(int argc, char **argv)
{
  // Initialize variables to track whether to assign defaults
  int create_cameras = 0;
  int output = 0;
  
  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) print_verbose = 1;
      else if (!strcmp(*argv, "-debug")) print_debug = 1;
      else if (!strcmp(*argv, "-glut")) { mesa = 0; glut = 1; }
      else if (!strcmp(*argv, "-mesa")) { mesa = 1; glut = 0; }
      else if (!strcmp(*argv, "-raycast")) { mesa = 0; glut = 0; }
      else if (!strcmp(*argv, "-categories")) { argc--; argv++; input_categories_filename = *argv; }
      else if (!strcmp(*argv, "-input_cameras")) { argc--; argv++; input_cameras_filename = *argv; }
      else if (!strcmp(*argv, "-output_camera_extrinsics")) { argc--; argv++; output_camera_extrinsics_filename = *argv; output = 1; }
      else if (!strcmp(*argv, "-output_camera_intrinsics")) { argc--; argv++; output_camera_intrinsics_filename = *argv; output = 1; }
      else if (!strcmp(*argv, "-output_camera_names")) { argc--; argv++; output_camera_names_filename = *argv; output = 1; }
      else if (!strcmp(*argv, "-output_nodes")) { argc--; argv++; output_nodes_filename = *argv; output = 1; }
      else if (!strcmp(*argv, "-interpolate_camera_trajectory")) { interpolate_camera_trajectory = 1; }
      else if (!strcmp(*argv, "-width")) { argc--; argv++; width = atoi(*argv); }
      else if (!strcmp(*argv, "-height")) { argc--; argv++; height = atoi(*argv); }
      else if (!strcmp(*argv, "-xfov")) { argc--; argv++; xfov = atof(*argv); }
      else if (!strcmp(*argv, "-eye_height")) { argc--; argv++; eye_height = atof(*argv); }
      else if (!strcmp(*argv, "-eye_height_radius")) { argc--; argv++; eye_height_radius = atof(*argv); }
      else if (!strcmp(*argv, "-min_distance_from_obstacle")) { argc--; argv++; min_distance_from_obstacle = atof(*argv); }
      else if (!strcmp(*argv, "-min_visible_objects")) { argc--; argv++; min_visible_objects = atoi(*argv); }
      else if (!strcmp(*argv, "-min_score")) { argc--; argv++; min_score = atof(*argv); }
      else if (!strcmp(*argv, "-scene_scoring_method")) { argc--; argv++; scene_scoring_method = atoi(*argv); }
      else if (!strcmp(*argv, "-object_scoring_method")) { argc--; argv++; object_scoring_method = atoi(*argv); }
      else if (!strcmp(*argv, "-position_sampling")) { argc--; argv++; position_sampling = atof(*argv); }
      else if (!strcmp(*argv, "-angle_sampling")) { argc--; argv++; angle_sampling = atof(*argv); }
      else if (!strcmp(*argv, "-interpolation_step")) { argc--; argv++; interpolation_step = atof(*argv); }
      else if (!strcmp(*argv, "-create_object_cameras") || !strcmp(*argv, "-create_leaf_node_cameras")) {
        create_cameras = create_object_cameras = 1;
        angle_sampling = RN_PI / 6.0;
      }
      else if (!strcmp(*argv, "-create_room_cameras")) {
        create_cameras = create_room_cameras = 1;
        angle_sampling = RN_PI / 2.0;
      }
      else if (!strcmp(*argv, "-create_path_in_room_cameras")) {
        create_cameras = create_path_in_room_cameras = 1;
      }
      else if (!strcmp(*argv, "-create_world_in_hand_cameras")) {
        create_cameras = create_world_in_hand_cameras = 1;
      }
      else {
        fprintf(stderr, "Invalid program argument: %s", *argv);
        exit(1);
      }
      argv++; argc--;
    }
    else {
      if (!input_scene_filename) input_scene_filename = *argv;
      else if (!output_cameras_filename) { output_cameras_filename = *argv; output = 1; }
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
  }

  // Set default camera options
  if (!input_cameras_filename && !create_cameras) {
    create_room_cameras = 1;
  }

  // Check filenames
  if (!input_scene_filename || !output) {
    fprintf(stderr, "Usage: scn2cam inputscenefile outputcamerafile\n");
    return 0;
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

  // Read scene
  if (!ReadScene(input_scene_filename)) exit(-1);

  // Read cameras
  if (input_cameras_filename) {
    if (!ReadCameras(input_cameras_filename)) exit(-1);
  }

  // Read categories
  if (input_categories_filename) {
    if (!ReadCategories(input_categories_filename)) exit(-1);
  }

  // Create and write new cameras 
  if (mesa) CreateAndWriteCamerasWithMesa();
  else if (glut) CreateAndWriteCamerasWithGlut();
  else CreateAndWriteCameras();

  // Return success 
  return 0;
}




