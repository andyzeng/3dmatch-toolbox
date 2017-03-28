// Source file for the mesh viewer program



// Include files 

#include "R3Graphics/R3Graphics.h"
#include "fglut/fglut.h"



// Program variables

static RNArray<char *> mesh_names;
static RNArray<char *> point_names;
static char *image_name = NULL;
static R3Vector initial_camera_towards(-0.57735, -0.57735, -0.57735);
static R3Vector initial_camera_up(-0.57735, 0.57735, 0.5773);
static R3Point initial_camera_origin(0,0,0);
static RNBoolean initial_camera = FALSE;
static RNScalar world_radius_scale = 1.0;
static int print_verbose = 0;
static int print_debug = 0;
static int nrows = 32;



// Type definitions

struct Point {
  R3Point position;
  R3Vector normal;
};

struct Model {
  Model(void) : mesh(NULL), points(NULL), 
    bbox(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX) { name[0] = '\0'; };
  R3Mesh *mesh;
  RNArray<Point *> *points;
  R3Box bbox;
  char name[256];
};



// GLUT variables 

static int GLUTwindow = 0;
static int GLUTwindow_height = 768;
static int GLUTwindow_width = 1024;
static int GLUTmouse[2] = { 0, 0 };
static int GLUTbutton[3] = { 0, 0, 0 };
static int GLUTmodifiers = 0;



// Application variables

static R3Viewer *viewer = NULL;
static RNArray<Model *> models;
static int selected_point_index = -1;



// Display variables

static const int max_models = 4;
static int show_model[max_models] = { 1, 1, 1, 1 };
static int show_faces = 1;
static int show_edges = 0;
static int show_vertices = 0;
static int show_points = 0;
static int show_normals = 0;
static int show_matches = 0;
static int show_backfacing = 0;
static int show_point_order = 0;
static int show_point_names = 0;



// Color definitions

static const int max_mesh_colors = 4;
static float mesh_colors[max_mesh_colors][4] = {
  { 0.5, 0.5, 0.5, 1 },
  { 1, 0.2, 0.2, 1 },
  { 0.2, 1, 0.2, 1 },
  { 0.2, 0.2, 1, 1 }
};

static const int max_point_colors = 36;
static float point_colors[max_point_colors][4] = {
  { 0.0, 1.0, 0.0, 1.0 },
  { 0.0, 0.0, 1.0, 1.0 },
  { 1.0, 0.0, 0.0, 1.0 },
  { 1.0, 1.0, 0.0, 1.0 },
  { 1.0, 0.0, 1.0, 1.0 },
  { 0.0, 1.0, 1.0, 1.0 },
  { 1.0, 0.5, 0.0, 1.0 },
  { 1.0, 0.0, 0.5, 1.0 },
  { 0.5, 1.0, 0.0, 1.0 },
  { 0.5, 0.0, 1.0, 1.0 },
  { 0.0, 1.0, 0.5, 1.0 },
  { 0.0, 0.5, 1.0, 1.0 },
  { 0.5, 0.5, 1.0, 1.0 },
  { 1.0, 0.5, 0.5, 1.0 },
  { 0.5, 1.0, 0.5, 1.0 },
  { 0.5, 0.5, 1.0, 1.0 },
  { 1.0, 0.2, 0.0, 1.0 },
  { 1.0, 0.0, 0.2, 1.0 },
  { 0.2, 1.0, 0.0, 1.0 },
  { 0.2, 0.0, 1.0, 1.0 },
  { 0.0, 1.0, 0.2, 1.0 },
  { 0.0, 0.2, 1.0, 1.0 },
  { 0.2, 0.2, 1.0, 1.0 },
  { 1.0, 0.2, 0.2, 1.0 },
  { 0.2, 1.0, 0.2, 1.0 },
  { 0.2, 0.2, 1.0, 1.0 },
  { 1.0, 0.8, 0.0, 1.0 },
  { 1.0, 0.0, 0.8, 1.0 },
  { 0.8, 1.0, 0.0, 1.0 },
  { 0.8, 0.0, 1.0, 1.0 },
  { 0.0, 1.0, 0.8, 1.0 },
  { 0.0, 0.8, 1.0, 1.0 },
  { 0.8, 0.8, 1.0, 1.0 },
  { 1.0, 0.8, 0.8, 1.0 },
  { 0.8, 1.0, 0.8, 1.0 },
  { 0.8, 0.8, 1.0, 1.0 },
};



void GLUTDrawText(const R3Point& p, const char *s)
{
  // Draw text string s and position p
  glRasterPos3d(p[0], p[1], p[2]);
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
  glClearColor(1.0, 1.0, 1.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Set backface culling
  if (show_backfacing) glDisable(GL_CULL_FACE);
  else glEnable(GL_CULL_FACE);

  // Set lights
  static GLfloat light0_position[] = { 3.0, 4.0, 5.0, 0.0 };
  glLightfv(GL_LIGHT0, GL_POSITION, light0_position);
  static GLfloat light1_position[] = { -3.0, -2.0, -3.0, 0.0 };
  glLightfv(GL_LIGHT1, GL_POSITION, light1_position);

  // Draw all models
  for (int m = 0; m < models.NEntries(); m++) {
    if (!show_model[m]) continue;
    Model *model = models.Kth(m);
    R3Mesh *mesh = model->mesh;
    RNArray<Point *> *points = model->points;

    // Draw mesh
    if (mesh) { 
      // Draw faces
      if (show_faces) {
        glEnable(GL_LIGHTING);
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, mesh_colors[m%max_mesh_colors]); 
        mesh->DrawFaces();
      }
   
      // Draw edges
      if (show_edges) {
        glDisable(GL_LIGHTING);
        glColor3f(0.2, 0.2, 0.2);
        mesh->DrawEdges();
      }

      // Draw vertices
      if (show_vertices) {
        glEnable(GL_LIGHTING);
        static GLfloat material[] = { 0.8, 0.4, 0.2, 1.0 };
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, material); 
        mesh->DrawVertices();
      }
    }

#if 1
    // Draw points
    if (points && show_points) {
      glEnable(GL_LIGHTING);
      glPointSize(3);
      glBegin(GL_POINTS);
      for (int i = 0; i < points->NEntries(); i++) {
        Point *point = points->Kth(i);
        int point_color_index = (show_point_order) ? i%max_point_colors : m%max_point_colors;
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, point_colors[point_color_index]);
        R3LoadNormal(point->normal);
        R3LoadPoint(point->position);
      }
      glEnd();
      glPointSize(1);
    }
#else
    // Draw points
    if (points && show_points) {
      // Draw points
      glEnable(GL_LIGHTING);
      RNLength radius = 0.008 * model->bbox.LongestAxisLength();
      for (int i = 0; i < points->NEntries(); i++) {
        Point *point = points->Kth(i);
        int point_color_index = (show_point_order) ? i%max_point_colors : m%max_point_colors;
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, point_colors[point_color_index]);
        R3Sphere(point->position, radius).Draw();
        glDisable(GL_LIGHTING);
        glLineWidth(3.0);
        glColor4fv(point_colors[point_color_index]);
        R3Span(point->position, point->position + (4 * radius) * point->normal).Draw();
        glLineWidth(1.0);
        glEnable(GL_LIGHTING);
        if (i == selected_point_index) { 
          glEnable(GL_LIGHT2);
          R3Sphere(point->position, 2 * radius).Draw();
          glDisable(GL_LIGHT2);
        }
      }
    }
#endif

    // Draw point normals
    if (points && show_normals) {
      glDisable(GL_LIGHTING);
      glBegin(GL_LINES);
      RNLength radius = 0.008 * model->bbox.LongestAxisLength();
      for (int i = 0; i < points->NEntries(); i++) {
        Point *point = points->Kth(i);
        int point_color_index = (show_point_order) ? i%max_point_colors : m%max_point_colors;
        glColor4fv(point_colors[point_color_index]);
        R3LoadPoint(point->position);
        R3LoadPoint(point->position + radius*point->normal);
      }
      glEnd();
    }

    // Draw point names
    if (points && show_point_names) {
      char buffer[256];
      glDisable(GL_LIGHTING);
      glColor3f(0, 0, 0);
      RNLength radius = 0.02 * model->bbox.LongestAxisLength();
      for (int i = 0; i < points->NEntries(); i++) {
        Point *point = points->Kth(i);
        R3Point position = point->position + radius * point->normal;
        int point_color_index = (show_point_order) ? i%max_point_colors : 0;
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, point_colors[point_color_index]);
        sprintf(buffer, "%d", i);
        GLUTDrawText(position, buffer);
      }
    }

    // Draw matches
    if ((m > 0) && show_matches) {
      // Draw matches between points based on indices
      glDisable(GL_LIGHTING);
      Model *model0 = models.Kth(0);
      RNArray<Point *> *points0 = model0->points;
      for (int i = 0; i < points->NEntries(); i++) {
        if (i >= points0->NEntries()) break;
        Point *point = points->Kth(i);
        Point *point0 = points0->Kth(i);
        glLineWidth(2.0);
        glColor4fv(point_colors[i%max_point_colors]);
        R3Span(point->position, point0->position).Draw();
        glLineWidth(1.0);
      }
    }
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
  R3Point origin = models.Head()->bbox.Centroid();
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
  if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_DOWN)) {
    // Check for double click
    static RNTime last_mouse_down_time;
    static RNBoolean double_click = FALSE;
    static const RNScalar double_click_interval = 0.5;
    double_click = (!double_click) && (last_mouse_down_time.Elapsed() < double_click_interval);
    last_mouse_down_time.Read();

    // Select point
    if (double_click) {
      selected_point_index = -1;
      RNLength selected_distance = FLT_MAX;
      R2Point mouse_position(x, y);
      for (int i = 0; i < models.NEntries(); i++) {
        Model *model = models.Kth(i);
        if (!show_model[i]) continue;
        if (!model->points) continue;
        for (int j = 0; j < model->points->NEntries(); j++) {
          Point *point = model->points->Kth(j);
          R2Point point_projection = viewer->ViewportPoint(point->position);
          RNLength distance = R2Distance(mouse_position, point_projection);
          if ((distance < 10) && (distance < selected_distance)) {
            selected_point_index = j;
            selected_distance = distance;
          }
        }
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
  case '1':
  case '2':
  case '3':
  case '4':
    show_model[key - '1'] = !show_model[key - '1'];
    break;

  case 'B':
  case 'b':
    show_backfacing = !show_backfacing;
    break;

  case 'C': 
  case 'c': {
    // Print camera
    const R3Camera& camera = viewer->Camera();
    printf("#camera  %g %g %g  %g %g %g  %g %g %g  %g \n",
           camera.Origin().X(), camera.Origin().Y(), camera.Origin().Z(),
           camera.Towards().X(), camera.Towards().Y(), camera.Towards().Z(),
           camera.Up().X(), camera.Up().Y(), camera.Up().Z(),
           camera.YFOV());
    }
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
    show_point_names = !show_point_names;
    break;

  case 'M':
  case 'm':
    show_matches = !show_matches;
    break;

  case 'N':
  case 'n':
    show_normals = !show_normals;
    break;

  case 'O':
  case 'o':
    show_point_order = !show_point_order;
    break;

 case 'P':
  case 'p':
    show_points = !show_points;
    break;

  case 'V':
  case 'v':
    show_vertices = !show_vertices;
    break;

  case 27: // ESCAPE
    exit(0);
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




void GLUTInit(int *argc, char **argv)
{
  // Open window
  glutInit(argc, argv);
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(GLUTwindow_width, GLUTwindow_height);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH); // | GLUT_STENCIL
  GLUTwindow = glutCreateWindow("OpenGL Viewer");

  // Initialize background color
  glClearColor(200.0/255.0, 200.0/255.0, 200.0/255.0, 1.0);

  // Initialize lights
  static GLfloat lmodel_ambient[] = { 0.2, 0.2, 0.2, 1.0 };
  glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
  glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
  static GLfloat light0_diffuse[] = { 1.0, 1.0, 1.0, 1.0 };
  glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
  glEnable(GL_LIGHT0);
  static GLfloat light1_diffuse[] = { 0.5, 0.5, 0.5, 1.0 };
  glLightfv(GL_LIGHT1, GL_DIFFUSE, light1_diffuse);
  glEnable(GL_LIGHT1);
  glEnable(GL_NORMALIZE);
  glEnable(GL_LIGHTING);

  // Initialize graphics modes
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);

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
  // Run main loop -- never returns
  glutMainLoop();
}


////////////////////////////////////////////////////////////////////////
// I/O Functions
////////////////////////////////////////////////////////////////////////


static R3Mesh *
ReadMesh(const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate mesh
  R3Mesh *mesh = new R3Mesh();
  assert(mesh);

  // Read mesh from file
  if (!mesh->ReadFile(filename)) {
    delete mesh;
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("Read mesh file: %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Faces = %d\n", mesh->NFaces());
    printf("  # Edges = %d\n", mesh->NEdges());
    printf("  # Vertices = %d\n", mesh->NVertices());
    fflush(stdout);
  }

  // Return success
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
  else if (!strcmp(extension, ".xyzn")) {
    // Open points file
    FILE *fp = fopen(filename, "r");
    if (!fp) {
      fprintf(stderr, "Unable to open points file: %s\n", filename);
      return NULL;
    }

    // Read points
    char buffer[4096];
    while (fgets(buffer, 4096, fp)) {
      double px, py, pz, nx, ny, nz;
      if (sscanf(buffer, "%lf%lf%lf%lf%lf%lf", &px, &py, &pz, &nx, &ny, &nz) == (unsigned int) 6) {
        Point *point = new Point();
        point->position.Reset(px, py, pz);
        point->normal.Reset(nx, ny, nz);
        points->Insert(point);
      }
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



static Model *
ReadModel(const char *mesh_name, const char *point_name)
{
  // Create model
  Model *model = new Model();
  assert(model);

  // Fill in model name
  char *namep = strrchr(point_name, '/');
  namep = (namep) ? namep + 1 : (char *) point_name;
  strcpy(model->name, namep);
  namep = strrchr(model->name, '.');
  if (namep) *namep = '\0';

  // Read mesh
  model->mesh = NULL;
  if (mesh_name) {
    model->mesh = ReadMesh(mesh_name);
    if (!model->mesh) return NULL;
    model->bbox = model->mesh->BBox();
  }

  // Read points
  model->points = NULL;
  if (point_name) {
    model->points = ReadPoints(model->mesh, point_name);
    if (!model->points) return NULL;
    for (int i = 0; i < model->points->NEntries(); i++) {
      model->bbox.Union(model->points->Kth(i)->position);
    }
  }

  // Return model
  return model;
}



static R3Viewer *
CreateBirdsEyeViewer(Model *model)
{
    // Setup camera view looking down the Z axis
    R3Box bbox = model->bbox;
    assert(!bbox.IsEmpty());
    RNLength r = bbox.DiagonalRadius();
    assert((r > 0.0) && RNIsFinite(r));
    if (!initial_camera) initial_camera_origin = bbox.Centroid() - initial_camera_towards * (2 * r);
    R3Camera camera(initial_camera_origin, initial_camera_towards, initial_camera_up, 0.4, 0.4, 0.01 * r, 100.0 * r);
    R2Viewport viewport(0, 0, GLUTwindow_width, GLUTwindow_height);
    return new R3Viewer(camera, viewport);
}



int ParseArgs(int argc, char **argv)
{
  // Check number of arguments
  if ((argc == 2) && (*argv[1] == '-')) {
    printf("Usage: ptsview filename [options]\n");
    exit(0);
  }

  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) { print_verbose = 1; }
      else if (!strcmp(*argv, "-debug")) { print_debug = 1; }
      else if (!strcmp(*argv, "-image")) { argc--; argv++; image_name = *argv; }
      else if (!strcmp(*argv, "-scale")) { argc--; argv++; world_radius_scale = atof(*argv); }
      else if (!strcmp(*argv, "-back")) { show_backfacing = TRUE; }
      else if (!strcmp(*argv, "-nrows")) { argc--; argv++; nrows = atoi(*argv); }
      else if (!strcmp(*argv, "-window")) { 
        argv++; argc--; GLUTwindow_width = atoi(*argv); 
        argv++; argc--; GLUTwindow_height = atoi(*argv); 
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
      else { 
        fprintf(stderr, "Invalid program argument: %s", *argv); 
        exit(1); 
      }
      argv++; argc--;
    }
    else {
      char *ext = strrchr(*argv, '.');
      if (ext && (!strcmp(ext, ".pts") || !strcmp(ext, ".xyz") || !strcmp(ext, ".xyzn") || !strcmp(ext, ".vts") || !strcmp(ext, ".pid"))) point_names.Insert(*argv);
      else if (ext && (!strcmp(ext, ".off") || !strcmp(ext, ".ply") || !strcmp(ext, ".obj"))) mesh_names.Insert(*argv);
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
  }

  // Check points name
  if (point_names.IsEmpty()) {
    fprintf(stderr, "You did not specify a points filename\n");
    return FALSE;
  }

  // Update display settings
  if (!point_names.IsEmpty()) show_points = TRUE;

  // Return OK status 
  return 1;
}



int main(int argc, char **argv)
{
  // Parse program arguments
  if (!ParseArgs(argc, argv)) exit(-1);

  // Initialize GLUT
  GLUTInit(&argc, argv);

  // Read files
  for (int i = 0; i < point_names.NEntries(); i++) {
    // Get points name
    char *point_name = point_names.Kth(i);

    // Get mesh name
    char *mesh_name = NULL;
    if (mesh_names.NEntries() > i) {
      mesh_name = mesh_names.Kth(i);
    }

    // Read model
    Model *model = ReadModel(mesh_name, point_name);
    if (!model) exit(-1);

    // Add model
    models.Insert(model);
  }

  // Create viewer
  viewer = CreateBirdsEyeViewer(models.Head());
  if (!viewer) exit(-1);

  // Run GLUT interface
  GLUTMainLoop();

  // Return success 
  return 0;
}

















