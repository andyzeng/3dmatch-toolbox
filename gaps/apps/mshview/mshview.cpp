// Source file for the mesh viewer program



// Include files 

#include "R3Graphics/R3Graphics.h"
#include <fglut/fglut.h>



// Program variables

static RNArray<char *> mesh_names;
static const char *image_name = NULL;
static RNRgb background(200.0/255.0, 200.0/255.0, 200.0/255.0);
static RNBoolean initial_camera = FALSE;
static R3Point initial_camera_origin(0,0,0);
static R3Vector initial_camera_towards(0, 0, -1);
static R3Vector initial_camera_up(0, 1, 0);
// static R3Vector initial_camera_towards(0.115655, 0.447639, -0.886704);
// static R3Vector initial_camera_up(0.00610775, 0.892357, 0.451289);
// static R3Vector initial_camera_towards(-0.57735, -0.57735, -0.57735);
// static R3Vector initial_camera_up(-0.57735, 0.57735, 0.5773);
static int print_verbose = 0;



// GLUT variables 

static int GLUTwindow = 0;
static int GLUTwindow_height = 480;
static int GLUTwindow_width = 640;
static int GLUTmouse[2] = { 0, 0 };
static int GLUTbutton[3] = { 0, 0, 0 };
static int GLUTmodifiers = 0;



// Application variables

static R3Viewer *viewer = NULL;
static RNArray<R3Mesh *> meshes;



// Display variables

static int show_faces = 1;
static int show_edges = 0;
static int show_vertices = 0;
static int show_materials = 0;
static int show_segments = 0;
static int show_boundaries = 0;
static int show_axes = 0;
static int show_face_normals = 0;
static int show_face_names = 0;
static int show_edge_names = 0;
static int show_vertex_colors = 0;
static int show_vertex_normals = 0;
static int show_vertex_texcoords = 0;
static int show_vertex_names = 0;
static int show_material_names = 0;
static int show_segment_names = 0;
static int show_backfacing = 0;
static int current_mesh = -1;
static R3Point world_origin(0, 0, 0);



// Colors

const int ncolors = 72;
const RNRgb colors[ncolors] = {
  RNRgb(0.5, 0.5, 0.5), RNRgb(1, 0, 0), RNRgb(0, 0, 1), 
  RNRgb(0, 1, 0), RNRgb(0, 1, 1), RNRgb(1, 0, 1), 
  RNRgb(1, 0.5, 0), RNRgb(0, 1, 0.5), RNRgb(0.5, 0, 1), 
  RNRgb(0.5, 1, 0), RNRgb(0, 0.5, 1), RNRgb(1, 0, 0.5), 
  RNRgb(0.5, 0, 0), RNRgb(0, 0.5, 0), RNRgb(0, 0, 0.5), 
  RNRgb(0.5, 0.5, 0), RNRgb(0, 0.5, 0.5), RNRgb(0.5, 0, 0.5),
  RNRgb(0.7, 0, 0), RNRgb(0, 0.7, 0), RNRgb(0, 0, 0.7), 
  RNRgb(0.7, 0.7, 0), RNRgb(0, 0.7, 0.7), RNRgb(0.7, 0, 0.7), 
  RNRgb(0.7, 0.3, 0), RNRgb(0, 0.7, 0.3), RNRgb(0.3, 0, 0.7), 
  RNRgb(0.3, 0.7, 0), RNRgb(0, 0.3, 0.7), RNRgb(0.7, 0, 0.3), 
  RNRgb(0.3, 0, 0), RNRgb(0, 0.3, 0), RNRgb(0, 0, 0.3), 
  RNRgb(0.3, 0.3, 0), RNRgb(0, 0.3, 0.3), RNRgb(0.3, 0, 0.3),
  RNRgb(1, 0.3, 0.3), RNRgb(0.3, 1, 0.3), RNRgb(0.3, 0.3, 1), 
  RNRgb(1, 1, 0.3), RNRgb(0.3, 1, 1), RNRgb(1, 0.3, 1), 
  RNRgb(1, 0.5, 0.3), RNRgb(0.3, 1, 0.5), RNRgb(0.5, 0.3, 1), 
  RNRgb(0.5, 1, 0.3), RNRgb(0.3, 0.5, 1), RNRgb(1, 0.3, 0.5), 
  RNRgb(0.5, 0.3, 0.3), RNRgb(0.3, 0.5, 0.3), RNRgb(0.3, 0.3, 0.5), 
  RNRgb(0.5, 0.5, 0.3), RNRgb(0.3, 0.5, 0.5), RNRgb(0.5, 0.3, 0.5),
  RNRgb(0.3, 0.5, 0.5), RNRgb(0.5, 0.3, 0.5), RNRgb(0.5, 0.5, 0.3), 
  RNRgb(0.3, 0.3, 0.5), RNRgb(0.5, 0.3, 0.3), RNRgb(0.3, 0.5, 0.3), 
  RNRgb(0.3, 0.8, 0.5), RNRgb(0.5, 0.3, 0.8), RNRgb(0.8, 0.5, 0.3), 
  RNRgb(0.8, 0.3, 0.5), RNRgb(0.5, 0.8, 0.3), RNRgb(0.3, 0.5, 0.8), 
  RNRgb(0.8, 0.5, 0.5), RNRgb(0.5, 0.8, 0.5), RNRgb(0.5, 0.5, 0.8), 
  RNRgb(0.8, 0.8, 0.5), RNRgb(0.5, 0.8, 0.8), RNRgb(0.8, 0.5, 0.8)
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

  // Draw every mesh
  for (int m = 0; m < meshes.NEntries(); m++) {
    R3Mesh *mesh = meshes[m];
    if ((current_mesh != -1) && (m != current_mesh)) continue;

    // Draw faces
    if (show_faces) {
      if (show_vertex_colors) glDisable(GL_LIGHTING);
      else if (show_vertex_texcoords) glDisable(GL_LIGHTING);
      else { glEnable(GL_LIGHTING); glColor3d(0.8, 0.8, 0.8); }
      glBegin(GL_TRIANGLES);
      for (int i = 0; i < mesh->NFaces(); i++) {
        R3MeshFace *face = mesh->Face(i);
        if (show_materials) RNLoadRgb(colors[1 + mesh->FaceMaterial(face)%(ncolors-1)]); 
        else if (show_segments) RNLoadRgb(colors[1 + mesh->FaceSegment(face)%(ncolors-1)]); 
        else if (meshes.NEntries() > 1) RNLoadRgb(colors[1 + m%(ncolors-1)]); 
        else if (!show_vertex_colors) R3LoadNormal(mesh->FaceNormal(face));
        for (int j = 0; j < 3; j++) {
          R3MeshVertex *vertex = mesh->VertexOnFace(face, j);
          if (show_vertex_colors) R3LoadRgb(mesh->VertexColor(vertex));
          else if (show_vertex_texcoords) R3LoadRgb(mesh->VertexTextureCoords(vertex).X(), mesh->VertexTextureCoords(vertex).Y(), 0.0);
          R3LoadPoint(mesh->VertexPosition(vertex));
        }
      }
      glEnd();
    }

    // Draw edges
    if (show_edges) {
      glDisable(GL_LIGHTING);
      glColor3f(1.0, 0.0, 0.0);
      mesh->DrawEdges();
    }

    // Draw boundary edges
    if (show_boundaries) {
      glLineWidth(3);
      glDisable(GL_LIGHTING);
      glColor3f(0.0, 0.0, 0.0);
      for (int i = 0; i < mesh->NEdges(); i++) {
        R3MeshEdge *edge = mesh->Edge(i);
        if (mesh->FaceOnEdge(edge, 0) && mesh->FaceOnEdge(edge, 1)) continue;
        mesh->DrawEdge(edge);
      }
      glLineWidth(1);
    }

    // Draw vertices
    if (show_vertices) {
      glDisable(GL_LIGHTING);
      glColor3d(0.0, 0.0, 1.0);
      glBegin(GL_POINTS);
      for (int i = 0; i < mesh->NVertices(); i++) {
        R3MeshVertex *vertex = mesh->Vertex(i);
        if (show_vertex_colors) R3LoadRgb(mesh->VertexColor(vertex));
        R3LoadPoint(mesh->VertexPosition(vertex));
      }
      glEnd();
    }

    // Draw face normals
    if (show_face_normals) {
      glDisable(GL_LIGHTING);
      glColor3f(0.2, 0.2, 0.2);
      glBegin(GL_LINES);
      RNScalar radius = 0.02 * mesh->BBox().DiagonalLength();
      for (int i = 0; i < mesh->NFaces(); i++) {
        R3MeshFace *face = mesh->Face(i);
        R3Point position = mesh->FaceCentroid(face);
        R3Vector normal = mesh->FaceNormal(face);
        R3LoadPoint(position);
        R3LoadPoint(position + radius * normal);
      }
      glEnd();
    }

    // Draw vertex normals
    if (show_vertex_normals) {
      glDisable(GL_LIGHTING);
      glColor3d(0.5, 0.5, 0.5);
      glBegin(GL_LINES);
      RNScalar radius = 0.02 * mesh->BBox().DiagonalLength();
      for (int i = 0; i < mesh->NVertices(); i++) {
        R3MeshVertex *vertex = mesh->Vertex(i);
        R3Point position = mesh->VertexPosition(vertex);
        R3Vector normal = mesh->VertexNormal(vertex);
        if (show_vertex_colors) R3LoadRgb(mesh->VertexColor(vertex));
        R3LoadPoint(position);
        R3LoadPoint(position + radius * normal);
      }
      glEnd();
    }

    // Draw face names
    if (show_face_names) {
      glDisable(GL_LIGHTING);
      glColor3f(0, 0, 0);
      for (int i = 0; i < mesh->NFaces(); i++) {
        R3MeshFace *face = mesh->Face(i);
        char buffer[256];
        sprintf(buffer, "%d", mesh->FaceID(face));
        GLUTDrawText(mesh->FaceCentroid(face), buffer);
      }
    }

    // Draw edge names   
    if (show_edge_names) {
      glDisable(GL_LIGHTING);
      glColor3f(0.8, 0.0, 0.0);
      for (int i = 0; i < mesh->NEdges(); i++) {
        R3MeshEdge *edge = mesh->Edge(i);
        char buffer[256];
        sprintf(buffer, "%d", mesh->EdgeID(edge));
        GLUTDrawText(mesh->EdgeMidpoint(edge), buffer);
      }
    }

    // Draw vertex names
    if (show_vertex_names) {
      glDisable(GL_LIGHTING);
      glColor3f(0.5, 0.3, 0.1);
      for (int i = 0; i < mesh->NVertices(); i++) {
        R3MeshVertex *vertex = mesh->Vertex(i);
        char buffer[256];
        sprintf(buffer, "%d", mesh->VertexID(vertex));
        if (show_vertex_colors) R3LoadRgb(mesh->VertexColor(vertex));
        GLUTDrawText(mesh->VertexPosition(vertex), buffer);
      }
    }

    // Draw material names
    if (show_material_names) {
      glDisable(GL_LIGHTING);
      glColor3f(0, 0, 1);
      for (int i = 0; i < mesh->NFaces(); i++) {
        R3MeshFace *face = mesh->Face(i);
        char buffer[256];
        sprintf(buffer, "%d", mesh->FaceMaterial(face));
        GLUTDrawText(mesh->FaceCentroid(face), buffer);
      }
    }

    // Draw segment names
    if (show_segment_names) {
      glDisable(GL_LIGHTING);
      glColor3f(0, 0, 1);
      for (int i = 0; i < mesh->NFaces(); i++) {
        R3MeshFace *face = mesh->Face(i);
        char buffer[256];
        sprintf(buffer, "%d", mesh->FaceSegment(face));
        GLUTDrawText(mesh->FaceCentroid(face), buffer);
      }
    }
  }

  // Draw axes
  if (show_axes) {
    RNScalar d = meshes[0]->BBox().DiagonalRadius();
    glDisable(GL_LIGHTING);
    glLineWidth(3);
    R3BeginLine();
    glColor3f(1, 0, 0);
    R3LoadPoint(R3zero_point + d * R3negx_vector);
    R3LoadPoint(R3zero_point + d * R3posx_vector);
    R3EndLine();
    R3BeginLine();
    glColor3f(0, 1, 0);
    R3LoadPoint(R3zero_point + d * R3negy_vector);
    R3LoadPoint(R3zero_point + d * R3posy_vector);
    R3EndLine();
    R3BeginLine();
    glColor3f(0, 0, 1);
    R3LoadPoint(R3zero_point + d * R3negz_vector);
    R3LoadPoint(R3zero_point + d * R3posz_vector);
    R3EndLine();
    glLineWidth(1);
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
  if (GLUTbutton[0]) viewer->RotateWorld(1.0, world_origin, x, y, dx, dy);
  else if (GLUTbutton[1]) viewer->ScaleWorld(1.0, world_origin, x, y, dx, dy);
  else if (GLUTbutton[2]) viewer->TranslateWorld(1.0, world_origin, x, y, dx, dy);
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

  // Process mouse button event
  if ((button == GLUT_LEFT) && (state == GLUT_UP)) {
    // Check for double click  
    static RNBoolean double_click = FALSE;
    static RNTime last_mouse_down_time;
    double_click = (!double_click) && (last_mouse_down_time.Elapsed() < 0.4);
    last_mouse_down_time.Read();

    // Set world origin
    if (double_click) {
      R3Ray ray = viewer->WorldRay(x, y);
      RNScalar best_t = FLT_MAX;
      for (int m = 0; m < meshes.NEntries(); m++) {
        R3Mesh *mesh = meshes[m];
        if ((current_mesh != -1) && (m != current_mesh)) continue;
        R3MeshIntersection intersection;
        if (mesh->Intersection(ray, &intersection)) {
          if (intersection.t < best_t) {
            world_origin = intersection.point;
            best_t = intersection.t;
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
  // Invert y coordinate
  y = GLUTwindow_height - y;

  // Process keyboard button event 
  switch (key) {
  case '0':
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
    if (key == '0') current_mesh = -1;
    else if (key - '1' < meshes.NEntries()) current_mesh = key - '1';
    else printf("Unable to select mesh %d\n", key - '1');
    break;

  case 'A':
  case 'a':
    show_axes = !show_axes;
    break;

  case 'B':
  case 'b':
    show_backfacing = !show_backfacing;
    break;

  case 'C': 
  case 'c':
    show_vertex_colors = !show_vertex_colors;
    break;
    
  case 'E':
    show_edge_names = !show_edge_names;
    break;

  case 'e':
    show_edges = !show_edges;
    break;

  case 'F':
    show_face_names = !show_face_names;
    break;

  case 'f':
    show_faces = !show_faces;
    break;

  case 'M':
    show_material_names = !show_material_names;
    break;

  case 'm':
    show_materials = !show_materials;
    break;

  case 'N':
    show_vertex_normals = !show_vertex_normals;
    break;

  case 'n':
    show_face_normals = !show_face_normals;
    break;

  case 'S':
    show_segment_names = !show_segment_names;
    break;

  case 's':
    show_segments = !show_segments;
    break;

  case 'T':
  case 't':
    show_vertex_texcoords = !show_vertex_texcoords;
    break;

  case 'V':
    show_vertex_names = !show_vertex_names;
    break;

  case 'v':
    show_vertices = !show_vertices;
    break;

  case 'X':
  case 'x':
    show_boundaries = !show_boundaries;
    break;

  case ' ': {
    // Print camera
    const R3Camera& camera = viewer->Camera();
    fprintf(stderr, "%g %g %g  %g %g %g  %g %g %g  %g %g  1\n",
           camera.Origin().X(), camera.Origin().Y(), camera.Origin().Z(),
           camera.Towards().X(), camera.Towards().Y(), camera.Towards().Z(),
           camera.Up().X(), camera.Up().Y(), camera.Up().Z(),
           camera.XFOV(), camera.YFOV());
    printf("%g %g %g  %g %g %g  %g %g %g  %g %g  1\n",
           camera.Origin().X(), camera.Origin().Y(), camera.Origin().Z(),
           camera.Towards().X(), camera.Towards().Y(), camera.Towards().Z(),
           camera.Up().X(), camera.Up().Y(), camera.Up().Z(),
           camera.XFOV(), camera.YFOV());
    break; }
      
  case 16:  // ctrl-P
    image_name = "mshview.png";
    break; 

  case 27: // ESCAPE
    GLUTStop();
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
  GLUTwindow = glutCreateWindow("OpenGL Viewer");

  // Initialize background color 
  glClearColor(background[0], background[1], background[2], 1.0);

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

  // Initialize color settings
  glEnable(GL_COLOR_MATERIAL);
  glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);

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
  // Set world origin
  if (meshes.NEntries() > 0) {
    world_origin = meshes[0]->BBox().Centroid();
  }

  // Run main loop -- never returns 
  glutMainLoop();
}


 
int
ReadMeshes(const RNArray<char *>& mesh_names)
{
  // Read each mesh
  for (int i = 0; i < mesh_names.NEntries(); i++) {
    char *mesh_name = mesh_names[i];

    // Allocate mesh
    R3Mesh *mesh = new R3Mesh();
    assert(mesh);

    // Read mesh from file
    if (!mesh->ReadFile(mesh_name)) {
      delete mesh;
      return 0;
    }

    // Add mesh to list
    meshes.Insert(mesh);

    // Print statistics
    if (print_verbose) {
      printf("Read mesh from %s ...\n", mesh_name);
      printf("  # Faces = %d\n", mesh->NFaces());
      printf("  # Edges = %d\n", mesh->NEdges());
      printf("  # Vertices = %d\n", mesh->NVertices());
      fflush(stdout);
    }
  }

  // Return number of meshes
  return meshes.NEntries();
}



static R3Viewer *
CreateBirdsEyeViewer(R3Mesh *mesh)
{
    // Setup camera view looking down the Z axis
    R3Box bbox = mesh->BBox();
    assert(!bbox.IsEmpty());
    RNLength r = bbox.DiagonalRadius();
    assert((r > 0.0) && RNIsFinite(r));
    if (!initial_camera) initial_camera_origin = bbox.Centroid() - initial_camera_towards * (2.5 * r);
    R3Camera camera(initial_camera_origin, initial_camera_towards, initial_camera_up, 0.4, 0.4, 0.01 * r, 100.0 * r);
    R2Viewport viewport(0, 0, GLUTwindow_width, GLUTwindow_height);
    return new R3Viewer(camera, viewport);
}



int ParseArgs(int argc, char **argv)
{
  // Check number of arguments
  if ((argc == 2) && (*argv[1] == '-')) {
    printf("Usage: meshviewer filename [options]\n");
    exit(0);
  }

  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) { print_verbose = 1; }
      else if (!strcmp(*argv, "-image")) { argc--; argv++; image_name = *argv; }
      else if (!strcmp(*argv, "-vertices")) { show_vertices = TRUE; }
      else if (!strcmp(*argv, "-edges")) { show_edges = TRUE; }
      else if (!strcmp(*argv, "-back")) { show_backfacing = TRUE; }
      else if (!strcmp(*argv, "-boundaries")) { show_boundaries = TRUE; }
      else if (!strcmp(*argv, "-axes")) { show_axes = TRUE; }
      else if (!strcmp(*argv, "-segments")) { show_segments = TRUE; }
      else if (!strcmp(*argv, "-materials")) { show_materials = TRUE; }
      else if (!strcmp(*argv, "-vertex_colors")) { show_vertex_colors = TRUE; }
      else if (!strcmp(*argv, "-window")) { 
        argv++; argc--; GLUTwindow_width = atoi(*argv); 
        argv++; argc--; GLUTwindow_height = atoi(*argv); 
      }
      else if (!strcmp(*argv, "-background")) { 
        argv++; argc--; background[0] = atof(*argv); 
        argv++; argc--; background[1] = atof(*argv); 
        argv++; argc--; background[2] = atof(*argv);
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
      mesh_names.Insert(*argv);
      argv++; argc--;
    }
  }

  // Check mesh filename
  if (mesh_names.IsEmpty()) {
    fprintf(stderr, "You did not specify a mesh file name.\n");
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
  if (!ReadMeshes(mesh_names)) exit(-1);

  // Create viewer
  viewer = CreateBirdsEyeViewer(meshes[0]);
  if (!viewer) exit(-1);

  // Run GLUT interface
  GLUTMainLoop();

  // Return success 
  return 0;
}

















