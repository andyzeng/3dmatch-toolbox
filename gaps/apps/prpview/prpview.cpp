// Source file for the mesh viewer program



// Include files 

#include "R3Graphics/R3Graphics.h"
#include "fglut/fglut.h"



// Program variables

static char *mesh_name = NULL;
static char *properties_name = NULL;
static char *points_name = NULL;
static char *image_name = NULL;
static R3Vector initial_camera_towards(-0.57735, -0.57735, -0.57735);
static R3Vector initial_camera_up(-0.57735, 0.57735, 0.5773);
static R3Point initial_camera_origin(0,0,0);
static RNBoolean initial_camera = FALSE;
static int color_with_redness = 0;
static int color_with_labels = 0;
static int print_verbose = 0;



// Type definitions

struct Point {
  R3Point position;
  R3Vector normal;
};



// Application variables

static R3Viewer *viewer = NULL;
static R3Mesh *mesh = NULL;
static RNArray<Point *> *points = NULL;
static R3MeshPropertySet *properties = NULL;
static R3MeshProperty *current_property = NULL;
static int current_property_index = 0;
static RNInterval percentile_range(0,0);
static RNInterval value_range(0,0);



// GLUT variables 

static int GLUTwindow = 0;
static int GLUTwindow_height = 800;
static int GLUTwindow_width = 800;
static int GLUTmouse[2] = { 0, 0 };
static int GLUTbutton[3] = { 0, 0, 0 };
static int GLUTmodifiers = 0;



// Display variables

static int show_faces = 0;
static int show_points = 0;
static int show_values = 1;
static int show_isocontours = 0;
static int show_global_minima = 0;
static int show_global_maxima = 0;
static int show_local_minima = 0;
static int show_local_maxima = 0;
static int show_statistics = 1;
static int show_text = 0;
static int show_backfacing = 0;
static int nisosteps = 50;



////////////////////////////////////////////////////////////////////////
// Utility property value functions
////////////////////////////////////////////////////////////////////////

static RNScalar
Value(R3MeshVertex *vertex)
{
  // Return value of current property at vertex
  return current_property->VertexValue(vertex);
}



static RNScalar
NormalizedValue(R3MeshVertex *vertex)
{
  // Return value between 0 and 1
  RNScalar value = current_property->VertexValue(vertex);
  RNScalar diameter = value_range.Diameter();
  if (diameter > 0) value = (value - value_range.Min()) / diameter;
  if (value < 0) value = 0;
  else if (value > 1) value = 1;
  return value;
}



static RNRgb 
NormalizedColor(R3MeshVertex *vertex)
{
  // Check color scheme
  RNRgb c(0, 0, 0);
  if (color_with_labels) {
    // Color with deterministic value based on int label
    int ivalue = (current_property->VertexValue(vertex) + 0.5);
    c[0] = (ivalue % 7) / 6.0;
    c[1] = (ivalue % 5) / 4.0;
    c[2] = (ivalue % 2) / 1.0;
  }
  else {
    // Compute normalized value
    RNScalar value = NormalizedValue(vertex);

    // Compute color
    if (color_with_redness) {
      c[0] = 0.8;
      c[1] = 0.8 * (1 - value);
      c[2] = 0.8 * (1 - value);
    }
    else {
      if (value < 0.5) {
        c[0] = 1 - 2 * value;
        c[1] = 2 * value;
      }
      else {
        c[1] = 1 - 2 * (value - 0.5);
        c[2] = 2 * (value - 0.5);
      }
    }
  }
  
  // Return color
  return c;
}


////////////////////////////////////////////////////////////////////////
// Glut user interface functions
////////////////////////////////////////////////////////////////////////

void GLUTDrawText(const R3Point& p, const char *s)
{
  // Draw text string s and position p
  glRasterPos3d(p[0], p[1], p[2]);
  while (*s) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *(s++));
}
  


void GLUTDrawText(const R2Point& p, const char *s)
{
  // Draw text string s and position p
  R3Ray ray = viewer->WorldRay((int) p[0], (int) p[1]);
  R3Point position = ray.Point(2 * viewer->Camera().Near());
  glRasterPos3d(position[0], position[1], position[2]);
  while (*s) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *(s++));
}



void GLUTDrawText(const R2Point& p, const char *str, double value)
{
  // Draw text string s and position p
  char buffer[1024];
  char *s = buffer;
  sprintf(buffer, "%s:%g", str, value);
  R3Ray ray = viewer->WorldRay((int) p[0], (int) p[1]);
  R3Point position = ray.Point(2 * viewer->Camera().Near());
  glRasterPos3d(position[0], position[1], position[2]);
  while (*s) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *(s++));
}



void GLUTStop(void)
{
  // Destroy window 
  glutDestroyWindow(GLUTwindow);

  // Exit
  exit(0);
}



void GLUTRedraw(void)
{
  // Set viewing transformation
  viewer->Camera().Load();

  // Clear window 
  glClearColor(1, 1, 1, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Set backface culling
  if (show_backfacing) glDisable(GL_CULL_FACE);
  else glEnable(GL_CULL_FACE);

  // Set overhead light
  static GLfloat light2_position[] = { 3.0, 4.0, 5.0, 0.0 };
  glLightfv(GL_LIGHT2, GL_POSITION, light2_position);
  // glEnable(GL_LIGHT2);

  // Draw faces
  if (show_faces) {
    glEnable(GL_LIGHTING);
    static GLfloat ambient_material[] = { 0.1, 0.1, 0.1, 1.0 };
    static GLfloat diffuse_material[] = { 0.8, 0.8, 0.8, 1.0 };
    // static GLfloat specular_material[] = { 0.2, 0.2, 0.2, 1.0};
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient_material); 
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse_material); 
    // glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular_material); 
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 1);
    mesh->DrawFaces();
  }

  // Draw points
  if (points && show_points) {
    // Draw points
    glEnable(GL_LIGHTING);
    static GLfloat regular_material[] = { 1, 1, 0, 1 };
    static GLfloat redness_material[] = { 0, 0, 1, 1 };
    GLfloat *material = (color_with_redness) ? redness_material : regular_material;
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, material);
    RNLength radius = 0.008 * mesh->BBox().LongestAxisLength();
    for (int i = 0; i < points->NEntries(); i++) {
      Point *point = points->Kth(i);
      R3Sphere(point->position, radius).Draw();
    }
  }

  // Draw values
  if (show_values) {
    glEnable(GL_LIGHTING);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(2, 1);
    for (int i = 0; i < mesh->NFaces(); i++) {
      R3MeshFace *face = mesh->Face(i);
      glBegin(GL_POLYGON);
      for (int j = 0; j < 3; j++) {
        R3MeshVertex *vertex = mesh->VertexOnFace(face, j);
        static GLfloat material[4] = { 0.8, 0.8, 0.8, 1 };
        RNRgb color = NormalizedColor(vertex);
        material[0] = color[0];
        material[1] = color[1];
        material[2] = color[2];
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, material);
        R3LoadNormal(mesh->VertexNormal(vertex));
        R3LoadPoint(mesh->VertexPosition(vertex));
      }
      glEnd();
    }
    glDisable(GL_POLYGON_OFFSET_FILL);
  }

  // Draw isocontours
  if (show_isocontours) {
    glDisable(GL_LIGHTING);
    glColor3d(1, 1, 1);
    glLineWidth(1);
    glBegin(GL_LINES);
    for (int isostep = 1; isostep < nisosteps; isostep++) {
      double isolevel = (double) isostep / (double) nisosteps;
      for (int i = 0; i < mesh->NFaces(); i++) {
        R3MeshFace *face = mesh->Face(i);
        for (int j = 0; j < 3; j++) {
          R3MeshVertex *vertex = mesh->VertexOnFace(face, j);
          RNScalar value = NormalizedValue(vertex);
          R3MeshVertex *vertex1 = mesh->VertexOnFace(face, (j+1)%3);
          RNScalar value1 = NormalizedValue(vertex1);
          R3MeshVertex *vertex2 = mesh->VertexOnFace(face, (j+2)%3);
          RNScalar value2 = NormalizedValue(vertex2);
          RNScalar t1, t2;
          if (value < isolevel) {
            if ((value1 <= isolevel) || (value2 <= isolevel)) continue;
            t1 = (isolevel - value) / (value1 - value);
            t2 = (isolevel - value) / (value2 - value);
          }
          else if (value > isolevel) {
            if ((value1 >= isolevel) || (value2 >= isolevel)) continue;
            t1 = (value - isolevel) / (value - value1);
            t2 = (value - isolevel) / (value - value2);
          }
          else continue;
          R3Point position = mesh->VertexPosition(vertex);
          R3Point position1 = mesh->VertexPosition(vertex1);
          R3Point position2 = mesh->VertexPosition(vertex2);
          R3Vector vector1 = t1 * (position1 - position);
          R3Vector vector2 = t2 * (position2 - position);
          R3Point point1 = position + vector1;
          R3Point point2 = position + vector2;
          R3LoadPoint(point1);
          R3LoadPoint(point2);
        }
      }
    }
    glEnd();
    glLineWidth(1);
  }

  // Draw vertices
  if (show_global_maxima || show_global_minima || show_local_maxima || show_local_minima) {
    glEnable(GL_LIGHTING);
    for (int i = 0; i < mesh->NVertices(); i++) {
      R3MeshVertex *vertex = mesh->Vertex(i);
      R3Point position = mesh->VertexPosition(vertex);
      RNScalar value = Value(vertex);
      if (show_global_minima && (value == current_property->Minimum())) {
        static GLfloat material[] = { 1, 1, 1, 1 };
        material[0] = 1; material[1] = 0; material[2] = 0;
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, material); 
        RNScalar radius = 0.01 * mesh->BBox().DiagonalLength();
        R3Sphere(position, radius).Draw();
      }
      if (show_global_maxima && (value == current_property->Maximum())) {
        static GLfloat material[] = { 1, 1, 1, 1 };
        material[0] = 0; material[1] = 1; material[2] = 0;
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, material); 
        RNScalar radius = 0.01 * mesh->BBox().DiagonalLength();
        R3Sphere(position, radius).Draw();
      }
      if (show_local_minima && (current_property->IsLocalMinimum(vertex))) {
        static GLfloat material[] = { 1, 1, 1, 1 };
        material[0] = 0.5; material[1] = 0; material[2] = 0;
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, material); 
        RNScalar radius = 0.005 * mesh->BBox().DiagonalLength();
        R3Sphere(position, radius).Draw();
      }
      if (show_local_maxima && (current_property->IsLocalMaximum(vertex))) {
        static GLfloat material[] = { 1, 1, 1, 1 };
        material[0] = 0; material[1] = 0.5; material[2] = 0;
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, material); 
        RNScalar radius = 0.005 * mesh->BBox().DiagonalLength();
        R3Sphere(position, radius).Draw();
      }
    }
  }

  // Draw values
  if (show_values) {
    glDisable(GL_LIGHTING);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(2, 1);
    for (int i = 0; i < mesh->NFaces(); i++) {
      R3MeshFace *face = mesh->Face(i);
      glBegin(GL_POLYGON);
      for (int j = 0; j < 3; j++) {
        R3MeshVertex *vertex = mesh->VertexOnFace(face, j);
        RNLoadRgb(NormalizedColor(vertex));
        R3LoadPoint(mesh->VertexPosition(vertex));
      }
      glEnd();
    }
    glDisable(GL_POLYGON_OFFSET_FILL);
  }

  // Draw text
  if (show_text) {
    char buffer[1024];
    glDisable(GL_LIGHTING);
    RNLength d = 0.004 * mesh->BBox().DiagonalRadius();
    for (int i = 0; i < mesh->NVertices(); i++) {
      R3MeshVertex *vertex = mesh->Vertex(i);
      R3Point position = mesh->VertexPosition(vertex);
      const R3Vector normal = mesh->VertexNormal(vertex);
      sprintf(buffer, "%g", Value(vertex));
      GLUTDrawText(position + d * normal, buffer);
    }
  }

  // Draw statistics
  if (show_statistics) {
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    double y = GLUTwindow_height - 20;
    GLUTDrawText(R2Point(10, y), current_property->Name()); y-= 20;
    GLUTDrawText(R2Point(10, y), "Mean", current_property->Mean()); y-= 20;
    GLUTDrawText(R2Point(10, y), "Min", current_property->Minimum()); y-= 20;
    GLUTDrawText(R2Point(10, y), "Max", current_property->Maximum()); y-= 20;
    GLUTDrawText(R2Point(10, y), "Stddev", current_property->StandardDeviation()); y-= 20;
    GLUTDrawText(R2Point(10, y), "Entropy", current_property->Entropy()); y-= 20;
    GLUTDrawText(R2Point(10, y), "Min Display Value", value_range.Min()); y-= 20;
    GLUTDrawText(R2Point(10, y), "Max Display Value", value_range.Max()); y-= 20;
    GLUTDrawText(R2Point(10, y), "Min Display Percentile", percentile_range.Min()); y-= 20;
    GLUTDrawText(R2Point(10, y), "Max Display Percentile", percentile_range.Max()); y-= 20;
  }

  // Capture image and exit
  if (image_name) {
    R2Image image(GLUTwindow_width, GLUTwindow_height, 3);
    image.Capture();
    image.Write(image_name);
    GLUTStop();
  }

  // Swap buffers 
  glutSwapBuffers();
}    



void GLUTResize(int w, int h)
{
  // Resize window
  glViewport(0, 0, w, h);

  // Resize viewer viewport
  viewer->ResizeViewport(0, 0, w, h);

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
  
  // World in hand navigation 
  R3Point origin = mesh->BBox().Centroid();
  if (GLUTbutton[0]) viewer->RotateWorld(1.0, origin, x, y, dx, dy);
  else if (GLUTbutton[1]) viewer->ScaleWorld(1.0, origin, x, y, dx, dy);
  else if (GLUTbutton[2]) viewer->TranslateWorld(1.0, origin, x, y, dx, dy);
  if (GLUTbutton[0] || GLUTbutton[1] || GLUTbutton[2]) glutPostRedisplay();

  // Remember mouse position 
  GLUTmouse[0] = x;
  GLUTmouse[1] = y;
}



void GLUTMouse(int button, int state, int x, int y)
{
  // Invert y coordinate
  y = GLUTwindow_height - y;
  
  // Process mouse button event

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
  case GLUT_KEY_HOME:
  case GLUT_KEY_END:
  case GLUT_KEY_PAGE_DOWN:
  case GLUT_KEY_PAGE_UP: {
    if (key == GLUT_KEY_PAGE_DOWN) current_property_index--;
    else if (key == GLUT_KEY_PAGE_UP) current_property_index++;
    else if (key == GLUT_KEY_HOME) current_property_index = 0;
    else if (key == GLUT_KEY_END) current_property_index = properties->NProperties() - 1;
    if (current_property_index < 0) current_property_index = 0;
    if (current_property_index >= properties->NProperties()) current_property_index = properties->NProperties() - 1;
    current_property = properties->Property(current_property_index);
    percentile_range.Reset(10, 90);
    RNScalar min_value = current_property->Percentile(percentile_range.Min());
    RNScalar max_value = current_property->Percentile(percentile_range.Max());
    value_range.Reset(min_value, max_value);
    break; }

  case GLUT_KEY_RIGHT:
  case GLUT_KEY_LEFT:
  case GLUT_KEY_DOWN:
  case GLUT_KEY_UP: {
    RNScalar mid = percentile_range.Mid();
    RNScalar radius = percentile_range.Radius();
    if (key == GLUT_KEY_LEFT) { if (radius >= 1) radius -= 1; }
    else if (key == GLUT_KEY_RIGHT) { if ((mid - radius >= 1) && (mid + radius <= 99)) radius += 1; }
    else if (key == GLUT_KEY_DOWN) { if (mid - radius >= 1) mid -= 1; }
    else if (key == GLUT_KEY_UP) { if (mid + radius <= 99) mid += 1; }
    percentile_range.Reset(mid - radius, mid + radius);
    RNScalar min_value = current_property->Percentile(percentile_range.Min());
    RNScalar max_value = current_property->Percentile(percentile_range.Max());
    value_range.Reset(min_value, max_value);
    break; }
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
  // Process keyboard button event 
  switch (key) {
  case 'B':
  case 'b':
    show_backfacing = !show_backfacing;
    break;

  case 'F':
  case 'f':
    show_faces = !show_faces;
    break;

  case 'G':
    show_global_maxima = !show_global_maxima;
    break;

  case 'g':
    show_global_minima = !show_global_minima;
    break;

  case 'I':
  case 'i':
    show_isocontours = !show_isocontours;
    break;

  case 'L':
    show_local_maxima = !show_local_maxima;
    break;

  case 'l':
    show_local_minima = !show_local_minima;
    break;

  case 'P':
  case 'p':
    show_points = !show_points;
    break;

  case 'S':
  case 's':
    show_statistics = !show_statistics;
    break;

  case 'T':
  case 't':
    show_text = !show_text;
    break;

  case 'V':
  case 'v':
    show_values = !show_values;
    break;

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




#if 0

void GLUTIdle(void)
{
  // Set current window
  if ( glutGetWindow() != GLUTwindow ) 
    glutSetWindow(GLUTwindow);  

  // Redraw
  glutPostRedisplay();
}

#endif



void GLUTInit(int *argc, char **argv)
{
  // Open window 
  glutInit(argc, argv);
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(GLUTwindow_width, GLUTwindow_height);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH); // | GLUT_STENCIL
  GLUTwindow = glutCreateWindow("Property Viewer");

  // Initialize background color 
  glClearColor(200.0/255.0, 200.0/255.0, 200.0/255.0, 1.0);

  // Initialize lighting
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  static GLfloat lmodel_ambient[] = { 0.2, 0.2, 0.2, 1.0 };
  glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
  glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
  glEnable(GL_NORMALIZE);
  glEnable(GL_LIGHTING); 

  // Initialize headlight
  static GLfloat light0_diffuse[] = { 1, 1, 1, 1 };
  static GLfloat light0_position[] = { 0.0, 0.0, 1.0, 0.0 };
  glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
  glLightfv(GL_LIGHT0, GL_POSITION, light0_position);
  glEnable(GL_LIGHT0);

  // Initialize backlight
  static GLfloat light1_diffuse[] = { 1, 1, 1, 1 };
  static GLfloat light1_position[] = { 0.0, 0.0, -1.0, 0.0 };
  glLightfv(GL_LIGHT1, GL_DIFFUSE, light1_diffuse);
  glLightfv(GL_LIGHT1, GL_POSITION, light1_position);
  glEnable(GL_LIGHT1);

  // Initialize graphics modes  
  glEnable(GL_DEPTH_TEST);
  glShadeModel(GL_SMOOTH);

  // Initialize GLUT callback functions 
  glutDisplayFunc(GLUTRedraw);
  glutReshapeFunc(GLUTResize);
  glutKeyboardFunc(GLUTKeyboard);
  glutSpecialFunc(GLUTSpecial);
  glutMouseFunc(GLUTMouse);
  glutMotionFunc(GLUTMotion);
}



void GLUTMainLoop(void)
{
  // Initialize current property
  current_property_index = 0;
  current_property = properties->Property(current_property_index);
  percentile_range.Reset(10, 90);
  if (value_range.Diameter() <= 0) {
    RNScalar min_value = current_property->Percentile(percentile_range.Min());
    RNScalar max_value = current_property->Percentile(percentile_range.Max());
    value_range.Reset(min_value, max_value);
  }

  // Run main loop -- never returns 
  glutMainLoop();
}


 
static R3Viewer *
CreateBirdsEyeViewer(R3Mesh *mesh)
{
  // Setup camera view looking down the Z axis
  R3Box bbox = mesh->BBox();
  assert(!bbox.IsEmpty());
  RNLength r = bbox.DiagonalRadius();
  assert((r > 0.0) && RNIsFinite(r));
  if (!initial_camera) initial_camera_origin = mesh->Centroid() - initial_camera_towards * (2.5 * r);
  R3Camera camera(initial_camera_origin, initial_camera_towards, initial_camera_up, 0.4, 0.4, 0.01 * r, 100.0 * r);
  R2Viewport viewport(0, 0, GLUTwindow_width, GLUTwindow_height);
  return new R3Viewer(camera, viewport);
}



static R3Mesh *
ReadMesh(char *filename)
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
    delete mesh;
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("Read mesh from %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Faces = %d\n", mesh->NFaces());
    printf("  # Edges = %d\n", mesh->NEdges());
    printf("  # Vertices = %d\n", mesh->NVertices());
    fflush(stdout);
  }

  // Return mesh
  return mesh;
}



static RNArray<Point *> *
ReadPoints(R3Mesh *mesh, const char *filename)
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

  // Parse input filename extension
  const char *extension;
  if (!(extension = strrchr(filename, '.'))) {
    printf("Filename %s has no extension (e.g., .ply)\n", filename);
    return 0;
  }

  // Check input filename extension
  if (!strcmp(extension, ".pts")) {
    // Open points file
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
      fprintf(stderr, "Unable to open points file: %s\n", filename);
      return NULL;
    }

    // Read points
    float coordinates[6];
    while (fread(coordinates, sizeof(float), 6, fp) == (unsigned int) 6) {
      Point *point = new Point();
      point->position.Reset(coordinates[0], coordinates[1], coordinates[2]);
      point->normal.Reset(coordinates[3], coordinates[4], coordinates[5]);
      points->Insert(point);
    }

    // Close points file
    fclose(fp);
  }
  else if (!strcmp(extension, ".xyz")) {
    // Open points file
    FILE *fp = fopen(filename, "r");
    if (!fp) {
      fprintf(stderr, "Unable to open points file: %s\n", filename);
      return NULL;
    }

    // Read points
    double x, y, z;
    while (fscanf(fp, "%lf%lf%lf", &x, &y, &z) == (unsigned int) 3) {
      Point *point = new Point();
      point->position.Reset(x, y, z);
      point->normal.Reset(0, 0, 0);
      points->Insert(point);
    }

    // Close points file
    fclose(fp);
  }
  else if (!strcmp(extension, ".vts")) {
    // Open points file
    FILE *fp = fopen(filename, "r");
    if (!fp) {
      fprintf(stderr, "Unable to open points file: %s\n", filename);
      return NULL;
    }

    // Read points
    int vertex_index;
    double x, y, z;
    while (fscanf(fp, "%d%lf%lf%lf", &vertex_index, &x, &y, &z) == (unsigned int) 4) {
      Point *point = new Point();
      point->position.Reset(x, y, z);
      point->normal.Reset(0, 0, 0);
      points->Insert(point);
    }

    // Close points file
    fclose(fp);
  }
  else if (!strcmp(extension, ".pid")) {
    // Open points file
    FILE *fp = fopen(filename, "r");
    if (!fp) {
      fprintf(stderr, "Unable to open points file: %s\n", filename);
      return NULL;
    }

    // Read points
    int vertex_index;
    while (fscanf(fp, "%d", &vertex_index) == (unsigned int) 1) {
      if ((vertex_index >= 0) && (vertex_index < mesh->NVertices())) {
        R3MeshVertex *vertex = mesh->Vertex(vertex_index);
        Point *point = new Point();
        point->position = mesh->VertexPosition(vertex);
        point->normal = mesh->VertexNormal(vertex);
        points->Insert(point);
      }
    }

    // Close points file
    fclose(fp);
  }
  else {
    fprintf(stderr, "Unrecognized point file extension: %s\n", extension);
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("Read point file: %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Points = %d\n", points->NEntries());
    fflush(stdout);
  }

  // Return points
  return points;
}



static R3MeshPropertySet *
ReadProperties(R3Mesh *mesh, const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate properties
  R3MeshPropertySet *properties = new R3MeshPropertySet(mesh);
  if (!properties) {
    fprintf(stderr, "Unable to allocate properties for %s\n", properties_name);
    return NULL;
  }

  // Read properties from file
  if (!properties->Read(filename)) {
    delete properties;
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("Read properties from %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Properties = %d\n", properties->NProperties());
    printf("  # Vertices = %d\n", properties->Mesh()->NVertices());
    fflush(stdout);
  }

  // Return property set
  return properties;
}



int ParseArgs(int argc, char **argv)
{
  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) { print_verbose = 1; }
      else if (!strcmp(*argv, "-points")) { argc--; argv++; points_name = *argv; show_points = 1; }
      else if (!strcmp(*argv, "-image")) { argc--; argv++; image_name = *argv; }
      else if (!strcmp(*argv, "-no_statistics")) { show_statistics = 0; }
      else if (!strcmp(*argv, "-faces")) { show_faces = 1; show_values = 0; }
      else if (!strcmp(*argv, "-redness")) { color_with_redness = 1; }
      else if (!strcmp(*argv, "-labels")) { color_with_labels = 1; }
      else if (!strcmp(*argv, "-back")) { show_backfacing = 1; }
      else if (!strcmp(*argv, "-value_range")) { 
        RNScalar min_value, max_value;
        argv++; argc--; min_value = atof(*argv);
        argv++; argc--; max_value = atof(*argv);
        value_range.Reset(min_value, max_value);
      }
      else if (!strcmp(*argv, "-camera")) {
        RNCoord x, y, z, tx, ty, tz, ux, uy, uz;
        argv++; argc--; x = atof(*argv);
        argv++; argc--; y = atof(*argv);
        argv++; argc--; z = atof(*argv);
        argv++; argc--; tx = atof(*argv);
        argv++; argc--; ty = atof(*argv);
        argv++; argc--; tz = atof(*argv);
        argv++; argc--; ux = atof(*argv);
        argv++; argc--; uy = atof(*argv);
        argv++; argc--; uz = atof(*argv);
        initial_camera_origin = R3Point(x, y, z);
        initial_camera_towards.Reset(tx, ty, tz);
        initial_camera_up.Reset(ux, uy, uz);
        initial_camera = TRUE;
      }
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
    else {
      if (!mesh_name) mesh_name = *argv;
      else if (!properties_name) properties_name = *argv;
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
  }

  // Check mesh filename
  if (!mesh_name || !properties_name) {
    fprintf(stderr, "Usage: prpview meshfile propertiesfile.\n");
    return 0;
  }

  // Return OK status 
  return 1;
}



int main(int argc, char **argv)
{
  // Parse program arguments
  if (!ParseArgs(argc, argv)) exit(-1);

  // Initialize GLUT
  GLUTInit(&argc, argv);

  // Read mesh
  mesh = ReadMesh(mesh_name);
  if (!mesh) exit(-1);

  // Read points
  if (points_name) {
    points = ReadPoints(mesh, points_name);
    if (!points) exit(-1);
  }

  // Read properties
  properties = ReadProperties(mesh, properties_name);
  if (!properties) exit(-1);

  // Create viewer
  viewer = CreateBirdsEyeViewer(mesh);
  if (!viewer) exit(-1);

  // Run GLUT interface
  GLUTMainLoop();

  // Return success 
  return 0;
}

















