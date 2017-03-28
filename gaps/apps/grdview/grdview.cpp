// Source file for the pdb viewer program



// Include files 

#include "R3Graphics/R3Graphics.h"
#include "fglut/fglut.h"



// Program variables

static char *grid_name = NULL;
static char *image_name = NULL;
static RNScalar background_color[3] = { 1, 1, 1 };
static int print_verbose = 0;



// GLUT variables 

static int GLUTwindow = 0;
static int GLUTwindow_height = 1024;
static int GLUTwindow_width = 1024;
static int GLUTmouse[2] = { 0, 0 };
static int GLUTbutton[3] = { 0, 0, 0 };
static int GLUTmodifiers = 0;



// Application variables

static R3Viewer *viewer = NULL;
static R3Grid *grid = NULL;
static R3Point initial_camera_origin = R3Point(0.0, 0.0, 0.0);
static R3Vector initial_camera_towards = R3Vector(0.0, 0.0, -1.0);
static R3Vector initial_camera_up = R3Vector(0.0, 1.0, 0.0);
static RNBoolean initial_camera = FALSE;



// Display variables

static int show_points = 0;
static int show_text = 0;
static int show_isosurface = 1;
static int show_slices[3] = { 0, 0, 0 };
static int show_box = 1;
static int show_axes = 0;
static int show_principle_axes = 0;
static float show_threshold = -98765;
static int show_slice_coords[3] = { 0, 0, 0 };



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
  glClearColor(background_color[0], background_color[1], background_color[2], 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Set lights
  static GLfloat light0_position[] = { 3.0, 4.0, 5.0, 0.0 };
  glLightfv(GL_LIGHT0, GL_POSITION, light0_position);
  static GLfloat light1_position[] = { -3.0, -2.0, -3.0, 0.0 };
  glLightfv(GL_LIGHT1, GL_POSITION, light1_position);

  // Draw grid
  if (show_points) {
    glPointSize(5.0);
    glDisable(GL_LIGHTING);
    glDepthMask(FALSE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glBegin(GL_POINTS);
    RNScalar mean = grid->Mean();
    RNScalar scale = (RNIsZero(mean)) ? 1.0 : 0.25 / grid->Mean();
    for (int i = 0; i < grid->XResolution(); i++) {
      for (int j = 0; j < grid->YResolution(); j++) {
        for (int k = 0; k < grid->ZResolution(); k++) {
          RNScalar value = grid->GridValue(i, j, k);
          if (value <= show_threshold) continue;
          RNScalar intensity = 1 - scale * value;
          glColor4f(intensity, intensity, intensity, 0.5*(1-intensity));
          double x = i + 1.0 - (7*i%5)/6.0 - (7*j%5)/6.0 - (7*k%5)/6.0;
          double y = j + 1.0 - (7*i%5)/6.0 - (7*j%5)/6.0 - (7*k%5)/6.0;
          double z = k + 1.0 - (7*i%5)/6.0 - (7*j%5)/6.0 - (7*k%5)/6.0;
          glVertex3d(x, y, z);
        }
      }
    }
    glEnd();
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ZERO);
    glDepthMask(TRUE);
    glPointSize(1.0);
  }

  // Draw grid text
  if (show_text) {
    char buffer[64];
    glDisable(GL_LIGHTING);
    RNLoadRgb(0.0, 0.0, 1.0);
    for (int i = 0; i < grid->XResolution(); i++) {
      for (int j = 0; j < grid->YResolution(); j++) {
        for (int k = 0; k < grid->ZResolution(); k++) {
          RNScalar value = grid->GridValue(i, j, k);
          if (value <= show_threshold) continue;
          sprintf(buffer, "%.2g", value);
          GLUTDrawText(R3Point((RNScalar) i, (RNScalar) j, (RNScalar) k), buffer);
        }
      }
    }
  }

  // Draw grid isosurface
  if (show_isosurface) {
    glDisable(GL_LIGHTING);
    RNLoadRgb(1.0, 0.0, 0.0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    grid->DrawIsoSurface(show_threshold);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  }

  // Draw grid X slices
  if (show_slices[RN_X] || show_slices[RN_Y] || show_slices[RN_Z]) {
    glDisable(GL_LIGHTING);
    RNLoadRgb(1.0, 1.0, 1.0);
    if (show_slices[RN_X]) grid->DrawSlice(RN_X, show_slice_coords[RN_X]);
    if (show_slices[RN_Y]) grid->DrawSlice(RN_Y, show_slice_coords[RN_Y]);
    if (show_slices[RN_Z]) grid->DrawSlice(RN_Z, show_slice_coords[RN_Z]);
  }

  // Draw grid bounding box
  if (show_box) {
    glDisable(GL_LIGHTING);
    glColor3f(0, 0, 0);
    R3Box(0, 0, 0, grid->XResolution()-1, grid->YResolution()-1, grid->ZResolution()-1).Outline();
#if 0
    if (show_slices[0]) R3Box(show_slice_coords[0], 0, 0, show_slice_coords[0], grid->YResolution()-1, grid->ZResolution()-1).Outline();
    if (show_slices[1]) R3Box(0, show_slice_coords[1], 0, grid->XResolution()-1, show_slice_coords[1], grid->ZResolution()-1).Outline();
    if (show_slices[2]) R3Box(0, 0, show_slice_coords[2], grid->XResolution()-1, grid->YResolution()-1, show_slice_coords[2]).Outline();
#endif
  }

  // Draw axes
  if (show_axes) {
    glDisable(GL_LIGHTING);
    glLineWidth(3);
    glColor3f(1, 0, 0);
    R3BeginLine();
    R3LoadPoint(grid->XResolution()/2.0, grid->YResolution()/2.0, grid->ZResolution()/2.0);
    R3LoadPoint(grid->XResolution()/1.0, grid->YResolution()/2.0, grid->ZResolution()/2.0);
    R3EndLine();
    glColor3f(0, 1, 0);
    R3BeginLine();
    R3LoadPoint(grid->XResolution()/2.0, grid->YResolution()/2.0, grid->ZResolution()/2.0);
    R3LoadPoint(grid->XResolution()/2.0, grid->YResolution()/1.0, grid->ZResolution()/2.0);
    R3EndLine();
    glColor3f(0, 0, 1);
    R3BeginLine();
    R3LoadPoint(grid->XResolution()/2.0, grid->YResolution()/2.0, grid->ZResolution()/2.0);
    R3LoadPoint(grid->XResolution()/2.0, grid->YResolution()/2.0, grid->ZResolution()/1.0);
    R3EndLine();
    glLineWidth(1);
  }

  // Draw principle axes
  if (show_principle_axes) {
    static R3Point *center = NULL;
    static R3Vector axes[3];
    if (!center) {
      RNScalar variances[3];
      center = new R3Point(grid->GridCentroid());
      R3Triad triad = grid->GridPrincipleAxes(center, variances);
      axes[0] = sqrt(variances[0]) * triad[0];
      axes[1] = sqrt(variances[1]) * triad[1];
      axes[2] = sqrt(variances[2]) * triad[2];
    }
    glDisable(GL_LIGHTING);
    glLineWidth(3);
    glColor3f(1, 0, 0);
    R3BeginLine();
    R3LoadPoint(*center - axes[0]);
    R3LoadPoint(*center + axes[0]);
    R3EndLine();
    glColor3f(0, 1, 0);
    R3BeginLine();
    R3LoadPoint(*center - axes[1]);
    R3LoadPoint(*center + axes[1]);
    R3EndLine();
    glColor3f(0, 0, 1);
    R3BeginLine();
    R3LoadPoint(*center - axes[2]);
    R3LoadPoint(*center + axes[2]);
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
  R3Point origin = grid->GridBox().Centroid();
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

  // Set useful variables
  static RNBoolean first = TRUE;
  static RNScalar threshold_step = 1;
  static RNInterval threshold_range(0,0);
  if (first) {
    RNInterval grid_range = grid->Range();
    threshold_step = grid->StandardDeviation() / 10;
    threshold_range.Reset(grid_range.Min() + 1.0E-20, grid_range.Max() - 1.0E-20);
    first = FALSE;
  }

  // Process keyboard button event 
  switch (key) {
  case GLUT_KEY_DOWN:
    show_threshold -= threshold_step;
    if (show_threshold < threshold_range.Min()) show_threshold = threshold_range.Min();
    break;

  case GLUT_KEY_UP:
    show_threshold += threshold_step;
    if (show_threshold > threshold_range.Max()) show_threshold = threshold_range.Max();
    break;

  case GLUT_KEY_LEFT:
    threshold_step /= 2;
    break;

  case GLUT_KEY_RIGHT:
    threshold_step *= 2;
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
  // Process keyboard button event 
  switch (key) {
  case 'A':
  case 'a':
    show_axes = !show_axes;
    break;

  case 'B':
  case 'b':
    show_box = !show_box;
    break;

  case 'G':
  case 'g':
    show_points = !show_points;
    break;

  case 'I':
  case 'i':
    show_isosurface = !show_isosurface;
    break;

  case 'P':
  case 'p':
    show_principle_axes = !show_principle_axes;
    break;

  case 'T':
  case 't':
    show_text = !show_text;
    break;

  case 'W':
  case 'w':
    show_slices[0] = !show_slices[0];
    show_slices[1] = show_slices[0];
    show_slices[2] = show_slices[0];
    break;

  case 'X':
    show_slice_coords[RN_X]++;
    if (show_slice_coords[RN_X] >= grid->XResolution()) 
      show_slice_coords[RN_X] = grid->XResolution() - 1;
    break;

  case 'x':
    show_slice_coords[RN_X]--;
    if (show_slice_coords[RN_X] < 0)
      show_slice_coords[RN_X] = 0;
    break;

  case 'Y':
    show_slice_coords[RN_Y]++;
    if (show_slice_coords[RN_Y] >= grid->YResolution()) 
      show_slice_coords[RN_Y] = grid->YResolution() - 1;
    break;

  case 'y':
    show_slice_coords[RN_Y]--;
    if (show_slice_coords[RN_Y] < 0)
      show_slice_coords[RN_Y] = 0;
    break;

  case 'Z':
    show_slice_coords[RN_Z]++;
    if (show_slice_coords[RN_Z] >= grid->ZResolution()) 
      show_slice_coords[RN_Z] = grid->ZResolution() - 1;
    break;

  case 'z':
    show_slice_coords[RN_Z]--;
    if (show_slice_coords[RN_Z] < 0)
      show_slice_coords[RN_Z] = 0;
    break;

  case ' ': {
    // Print camera
    const R3Camera& camera = viewer->Camera();
    printf("#camera  %g %g %g  %g %g %g  %g %g %g  %g \n",
           camera.Origin().X(), camera.Origin().Y(), camera.Origin().Z(),
           camera.Towards().X(), camera.Towards().Y(), camera.Towards().Z(),
           camera.Up().X(), camera.Up().Y(), camera.Up().Z(),
           camera.YFOV());
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

  // Initialize GLUT callback functions 
  glutDisplayFunc(GLUTRedraw);
  glutReshapeFunc(GLUTResize);
  glutKeyboardFunc(GLUTKeyboard);
  glutSpecialFunc(GLUTSpecial);
  glutMouseFunc(GLUTMouse);
  glutMotionFunc(GLUTMotion);

  // Initialize font
#if (RN_OS == RN_WINDOWSNT)
  int font = glGenLists(256);
  wglUseFontBitmaps(wglGetCurrentDC(), 0, 256, font); 
  glListBase(font);
#endif
}



void GLUTMainLoop(void)
{
  // Set default values
  if (show_threshold == -98765) {
    RNScalar max_threshold = grid->Maximum() - 1.0E-20;
    show_threshold = grid->Mean() + 3 * grid->StandardDeviation();
    if (show_threshold > max_threshold) show_threshold = max_threshold;
  }

  // Set slice coords to middle of grid
  show_slice_coords[RN_X] = grid->XResolution()/2;
  show_slice_coords[RN_Y] = grid->YResolution()/2;
  show_slice_coords[RN_Z] = grid->ZResolution()/2;

  // Run main loop -- never returns 
  glutMainLoop();
}



static R3Grid *
ReadGrid(char *grid_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate grid
  R3Grid *grid = new R3Grid();
  if (!grid) {
    RNFail("Unable to allocated grid");
    return NULL;
  }

  // Read grid 
  if (!grid->ReadFile(grid_name)) {
    RNFail("Unable to read grid file %s", grid_name);
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("Read grid ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Resolution = %d %d %d\n", grid->XResolution(), grid->YResolution(), grid->ZResolution());
    const R3Box bbox = grid->WorldBox();
    printf("  World Box = ( %g %g %g ) ( %g %g %g )\n", bbox[0][0], bbox[0][1], bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]);
    printf("  Spacing = %g\n", grid->GridToWorldScaleFactor());
    printf("  Cardinality = %d\n", grid->Cardinality());
    printf("  Volume = %g\n", grid->Volume());
    RNInterval grid_range = grid->Range();
    printf("  Minimum = %g\n", grid_range.Min());
    printf("  Maximum = %g\n", grid_range.Max());
    printf("  L1Norm = %g\n", grid->L1Norm());
    fflush(stdout);
  }

  // Return success
  return grid;
}



static R3Viewer *
CreateViewer(R3Grid *grid)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Get pdb bounding box
  R3Box bbox = grid->GridBox();
  assert(!bbox.IsEmpty());
  RNLength r = bbox.DiagonalRadius();
  assert((r > 0.0) && RNIsFinite(r));

  // Setup camera view looking down the Z axis
  if (!initial_camera) initial_camera_origin = bbox.Centroid() - initial_camera_towards * (2.5 * r);;
  R3Camera camera(initial_camera_origin, initial_camera_towards, initial_camera_up, 0.4, 0.4, 0.1 * r, 1000.0 * r);
  R2Viewport viewport(0, 0, GLUTwindow_width, GLUTwindow_height);
  R3Viewer *viewer = new R3Viewer(camera, viewport);

  // Print statistics
  if (print_verbose) {
    printf("Created viewer ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Origin = %g %g %g\n", camera.Origin().X(), camera.Origin().Y(), camera.Origin().Z());
    printf("  Towards = %g %g %g\n", camera.Towards().X(), camera.Towards().Y(), camera.Towards().Z());
    printf("  Up = %g %g %g\n", camera.Up().X(), camera.Up().Y(), camera.Up().Z());
    printf("  Fov = %g %g\n", camera.XFOV(), camera.YFOV());
    printf("  Near = %g\n", camera.Near());
    printf("  Far = %g\n", camera.Far());
    fflush(stdout);
  }

  // Return viewer
  return viewer;
}



static int 
ParseArgs(int argc, char **argv)
{
  // Check number of arguments
  if ((argc == 2) && (*argv[1] == '-')) {
    printf("Usage: symsam [options]\n");
    exit(0);
  }

  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) print_verbose = 1; 
      else if (!strcmp(*argv, "-show_points")) show_points = 1; 
      else if (!strcmp(*argv, "-show_text")) show_text = 1; 
      else if (!strcmp(*argv, "-show_isosurface")) show_isosurface = 1; 
      else if (!strcmp(*argv, "-show_axes")) show_axes = 1; 
      else if (!strcmp(*argv, "-dont_show_isosurface")) show_isosurface = 0; 
      else if (!strcmp(*argv, "-show_threshold")) {
        argc--; argv++; show_threshold = atof(*argv); 
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
      else if (!strcmp(*argv, "-background")) { 
        argc--; argv++; background_color[0] = atof(*argv); 
        argc--; argv++; background_color[1] = atof(*argv); 
        argc--; argv++; background_color[2] = atof(*argv); 
      }
      else if (!strcmp(*argv, "-image")) { 
        argc--; argv++; image_name = *argv; 
      }
      else { 
        fprintf(stderr, "Invalid program argument: %s", *argv); 
        exit(1); 
      }
      argv++; argc--;
    }
    else {
      if (!grid_name) grid_name = *argv;
      else { 
        fprintf(stderr, "Invalid program argument: %s", *argv); 
        exit(1); 
      }
      argv++; argc--;
    }
  }

  // Check pdb filename
  if (!grid_name) {
    fprintf(stderr, "You did not specify a pdb file.\n");
    return 0;
  }

  // Return OK status 
  return 1;
}



int 
main(int argc, char **argv)
{
  // Initialize GLUT
  GLUTInit(&argc, argv);

  // Parse program arguments
  if (!ParseArgs(argc, argv)) exit(-1);

  // Read grid
  grid = ReadGrid(grid_name);
  if (!grid) exit(-1);

  // Create viewer
  viewer = CreateViewer(grid);
  if (!viewer) exit(-1);

  // Run GLUT interface
  GLUTMainLoop();

  // Return success 
  return 0;
}



