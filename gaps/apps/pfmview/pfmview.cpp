// Source file for the mesh viewer program



// Include files 

#include "R2Shapes/R2Shapes.h"
#include <fglut/fglut.h>



// Program variables

static RNArray<char *> grid_names;
static int print_verbose = 0;



// GLUT variables 

static int GLUTwindow = 0;
static int GLUTwindow_height = 480;
static int GLUTwindow_width = 640;
static int GLUTmouse[2] = { 0, 0 };
static int GLUTbutton[3] = { 0, 0, 0 };
static int GLUTmodifiers = 0;



// Grid viewing variables

static RNArray<R2Grid *> grids;
static R2Grid *selected_grid = NULL;
static int selected_grid_index = -1;
static R2Box selected_grid_window = R2null_box;
static R2Point selected_grid_position(RN_UNKNOWN, RN_UNKNOWN);
static RNInterval selected_grid_range(0,0);
static int color_type = 0; // 0=gray, 1=red-green-blue, 2=labels



static void
SelectGrid(int index)
{
  // Check index
  if (index < 0) index = 0;
  if (index > grids.NEntries()-1) index = grids.NEntries()-1;
  if (index == selected_grid_index) return;
  R2Grid *grid = grids.Kth(index);

  // Set window title
  glutSetWindowTitle(grid_names.Kth(index));

  // Update display variables
  if (!selected_grid ||
      (selected_grid->XResolution() != grid->XResolution()) ||
      (selected_grid->YResolution() != grid->YResolution()) ||
      (0 && (selected_grid->WorldToGridTransformation() != grid->WorldToGridTransformation()))) {
    RNScalar window_aspect = (double) GLUTwindow_width / (double) GLUTwindow_height;
    RNScalar grid_aspect = (double) grid->XResolution() / (double) grid->YResolution();
    R2Point origin = grid->GridBox().Centroid();
    R2Vector diagonal = grid->GridBox().Max() - origin;
    diagonal[0] *= window_aspect / grid_aspect;
    selected_grid_window = R2Box(origin - diagonal, origin + diagonal);
    selected_grid_position.Reset(RN_UNKNOWN,RN_UNKNOWN);
  }

  // Update min and max values
  selected_grid_range = grid->Range();

  // Update selected grid 
  selected_grid_index = index;
  selected_grid = grid;
}



static RNRgb 
Color(RNScalar value)
{
  // Check for unknown value
  if (value == R2_GRID_UNKNOWN_VALUE) {
    if (color_type == 0) return RNRgb(1, 0.5, 0);
    else return RNblack_rgb;
  }

  // Compute color
  RNRgb c(0, 0, 0);
  if (color_type == 0) {
    // Draw gray value
    RNScalar value_min = selected_grid_range.Min();
    RNScalar value_width = selected_grid_range.Diameter();
    RNScalar value_scale = (value_width > 0) ? 1.0 / value_width : 1.0;
    RNScalar normalized_value = value_scale * (value - value_min);
    c[0] = normalized_value;
    c[1] = normalized_value;
    c[2] = normalized_value;
  }
  else if (color_type == 1) {
    // Draw heatmap value
    RNScalar value_min = selected_grid_range.Min();
    RNScalar value_width = selected_grid_range.Diameter();
    RNScalar value_scale = (value_width > 0) ? 1.0 / value_width : 1.0;
    RNScalar normalized_value = value_scale * (value - value_min);
    if (normalized_value < 0.5) {
      c[0] = 1 - 2 * normalized_value;
      c[1] = 2 * normalized_value;
    }
    else {
      c[1] = 1 - 2 * (normalized_value - 0.5);
      c[2] = 2 * (normalized_value - 0.5);
    }
  }
  else if (color_type == 2) {
    // Draw deterministic color based on int label
    int ivalue = value + 0.5;
    c[0] = 0.1 + (ivalue % 7) / 6.0;
    c[1] = 0.1 + (ivalue % 5) / 4.0;
    c[2] = 0.1 + (ivalue % 2) / 1.0;
  }

  // Return color
  return c;
}


void GLUTDrawText(const R2Point& p, const char *s)
{
  // Draw text string s and position p
  glRasterPos2d(p[0], p[1]);
  while (*s) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *(s++));
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
  // Check grid
  if (!selected_grid) return;

  // Clear window 
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Set projection matrix
  glMatrixMode(GL_PROJECTION);  
  glPushMatrix();
  glLoadIdentity();
  gluOrtho2D(selected_grid_window.XMin(), selected_grid_window.XMax(), selected_grid_window.YMin(), selected_grid_window.YMax()); 

  // Set model view matrix
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  // Draw grid values
  int xmin = (selected_grid_window.XMin() > 1) ? selected_grid_window.XMin() : 1;
  int ymin = (selected_grid_window.YMin() > 1) ? selected_grid_window.YMin() : 1;
  int xmax = (selected_grid_window.XMax()+1 < selected_grid->XResolution()-1) ? selected_grid_window.XMax()+1 : selected_grid->XResolution()-1;
  int ymax = (selected_grid_window.YMax()+1 < selected_grid->YResolution()-1) ? selected_grid_window.YMax()+1 : selected_grid->YResolution()-1;
  for (int j = ymin; j <= ymax; j++) {
    glBegin(GL_TRIANGLE_STRIP);
    for (int i = xmin; i <= xmax; i++) {
      for (int k = -1; k <= 0; k++) { 
        RNScalar value = selected_grid->GridValue(i, j+k);
        RNRgb color = Color(value);
        RNLoadRgb(color);
        glVertex2i(i, j+k);
      }
    }
    glEnd();
  }

  // Draw value at selected grid position
  if ((selected_grid_position.X() != RN_UNKNOWN) && (selected_grid_position.Y() != RN_UNKNOWN)) {
    int ix = (int) (selected_grid_position.X() + 0.5);
    if ((ix >= 0) && (ix < selected_grid->XResolution())) {
      int iy = (int) (selected_grid_position.Y() + 0.5);
      if ((iy >= 0) && (iy < selected_grid->YResolution())) {
        RNScalar value = selected_grid->GridValue(ix, iy);
        char buffer[1024];
        if (value != R2_GRID_UNKNOWN_VALUE) sprintf(buffer, "%d %d : %g", ix, iy, value);
        else sprintf(buffer, "%d %d : %s", ix, iy, "Unknown");
        // RNRgb color = Color(value);
        // RNRgb complement = RNwhite_rgb - color;
        RNLoadRgb(RNmagenta_rgb);
        R2Box(selected_grid_position - 0.5 * R2ones_vector, selected_grid_position + 0.5 * R2ones_vector);
        GLUTDrawText(selected_grid_position + 2 * R2ones_vector, buffer);
      }
    }
  }

  // Reset projection matrix
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  // Reset model view matrix
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  // Swap buffers 
  glutSwapBuffers();
}    



void GLUTResize(int w, int h)
{
  // Resize window
  glViewport(0, 0, w, h);

  // Remember window size 
  GLUTwindow_width = w;
  GLUTwindow_height = h;

  // Update selected grid window
  if (selected_grid) {
    RNScalar window_aspect = (double) GLUTwindow_width / (double) GLUTwindow_height;
    RNScalar grid_aspect = (double) selected_grid->XResolution() / (double) selected_grid->YResolution();
    R2Point origin = selected_grid->GridBox().Centroid();
    R2Vector diagonal = selected_grid->GridBox().Max() - origin;
    diagonal[0] *= window_aspect / grid_aspect;
    selected_grid_window = R2Box(origin - diagonal, origin + diagonal);
  }

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
  
  // View manipulation
  if (selected_grid) {
    if (GLUTbutton[0]) {
      // Query
      RNScalar px = x * selected_grid_window.XLength() / (double) GLUTwindow_width + selected_grid_window.XMin();
      RNScalar py = y * selected_grid_window.YLength() / (double) GLUTwindow_height + selected_grid_window.YMin();
      selected_grid_position.Reset(px, py);
      glutPostRedisplay();
    }
    else if (GLUTbutton[1]) {
      // Zoom
      RNScalar scale_factor = 1;
      scale_factor *= 1.0-(double)dx/(double)GLUTwindow_width;
      scale_factor *= 1.0-(double)dy/(double)GLUTwindow_height;
      scale_factor *= scale_factor;
      selected_grid_window.Inflate(scale_factor);
      glutPostRedisplay();
    }
    else if (GLUTbutton[2]) {
      // Pan
      RNScalar tx = -dx * selected_grid_window.XLength() / (double) GLUTwindow_width;
      RNScalar ty = -dy * selected_grid_window.YLength() / (double) GLUTwindow_height;
      selected_grid_window.Translate(R2Vector(tx, ty));
      glutPostRedisplay();
    }
  }

  // Remember mouse position 
  GLUTmouse[0] = x;
  GLUTmouse[1] = y;
}



void GLUTMouse(int button, int state, int x, int y)
{
  // Invert y coordinate
  y = GLUTwindow_height - y;
  
  // Process mouse button event
  if (button == 0) {
    if (state == GLUT_DOWN) {
      // Query
      RNScalar px = x * selected_grid_window.XLength() / (double) GLUTwindow_width + selected_grid_window.XMin();
      RNScalar py = y * selected_grid_window.YLength() / (double) GLUTwindow_height + selected_grid_window.YMin();
      selected_grid_position.Reset(px, py);
      glutPostRedisplay();
    }
    else {
      selected_grid_position.Reset(RN_UNKNOWN, RN_UNKNOWN);
      glutPostRedisplay();
    }
  }
  else if ((button == 3) || (button == 4)) {
    if (state == GLUT_DOWN) {
      // Zoom with wheel
      RNScalar scale_factor = (button == 3) ? 0.9 : 1.1;
      selected_grid_window.Inflate(scale_factor);
      glutPostRedisplay();
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
  case GLUT_KEY_PAGE_UP:
  case GLUT_KEY_PAGE_DOWN:
    if (selected_grid) {
      int shift = 0;
      if (key == GLUT_KEY_PAGE_UP) shift = -1;
      else if (key == GLUT_KEY_PAGE_DOWN) shift = 1;
      SelectGrid(selected_grid_index + shift);
      glutPostRedisplay();
    }
    break;

  case GLUT_KEY_LEFT: 
  case GLUT_KEY_RIGHT: 
  case GLUT_KEY_UP: 
  case GLUT_KEY_DOWN: 
    if (selected_grid) {
      RNScalar selected_grid_minimum = selected_grid->Minimum();
      RNScalar selected_grid_maximum = selected_grid->Maximum();
      RNScalar selected_grid_radius = 0.5 * (selected_grid_maximum - selected_grid_minimum);
      RNScalar center = selected_grid_range.Mid();
      RNScalar radius = selected_grid_range.Radius();
      if (key == GLUT_KEY_LEFT) radius *= 0.95;
      else if (key == GLUT_KEY_RIGHT) radius *= 1.05;
      else if (key == GLUT_KEY_UP) center += 0.01 * selected_grid_radius;
      else if (key == GLUT_KEY_DOWN) center -= 0.01 * selected_grid_radius;
      if (radius > selected_grid_radius) radius = selected_grid_radius;
      if (center - radius < selected_grid_minimum) center = selected_grid_minimum + radius;
      if (center + radius > selected_grid_maximum) center = selected_grid_maximum - radius;
      RNScalar min_value = center - radius;
      RNScalar max_value = center + radius;
      selected_grid_range.Reset(min_value, max_value);
      glutPostRedisplay();
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
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
    SelectGrid(key - '1');
    break;

  case 'C': 
  case 'c': 
    color_type = ((color_type + 1) % 3);
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
  // Set window dimensions
  GLUTwindow_width = 1024;
  GLUTwindow_height = 768;
  if (!grids.IsEmpty()) {
    R2Grid *grid = grids.Head();
    RNScalar aspect = (RNScalar) grid->YResolution() / (RNScalar) grid->XResolution();
    GLUTwindow_height = aspect * GLUTwindow_width;
  }

  // Open window 
  glutInit(argc, argv);
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(GLUTwindow_width, GLUTwindow_height);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH); // | GLUT_STENCIL
  GLUTwindow = glutCreateWindow("pfmview");

  // Initialize background color 
  glClearColor(00, 0.0, 0.0, 1.0);

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
  // Select first grid
  SelectGrid(0);

  // Run main loop -- never returns 
  glutMainLoop();
}


 
int
ReadGrids(const RNArray<char *>& grid_names)
{
  // Read each grid
  for (int i = 0; i < grid_names.NEntries(); i++) {
    char *grid_name = grid_names[i];

    // Allocate grid
    R2Grid *grid = new R2Grid();
    assert(grid);

    // Read grid from file
    if (!grid->Read(grid_name)) {
      delete grid;
      return 0;
    }

    // Add grid to list
    grids.Insert(grid);

    // Print statistics
    if (print_verbose) {
      const R2Box& box = grid->WorldBox();
      RNInterval grid_range = grid->Range();
      printf("Read grid from %s ...\n", grid_name);
      printf("  Resolution = %d %d\n", grid->XResolution(), grid->YResolution());
      printf("  World Box = ( %g %g ) ( %g %g )\n", box[0][0], box[0][1], box[1][0], box[1][1]);
      printf("  Spacing = %g\n", grid->GridToWorldScaleFactor());
      printf("  Cardinality = %d\n", grid->Cardinality());
      printf("  Minimum = %g\n", grid_range.Min());
      printf("  Maximum = %g\n", grid_range.Max());
      printf("  L1Norm = %g\n", grid->L1Norm());
      printf("  L2Norm = %g\n", grid->L2Norm());
      fflush(stdout);
    }
  }

  // Return number of grids
  return grids.NEntries();
}



static int 
ParseArgs(int argc, char **argv)
{
  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) { print_verbose = 1; }
      else if (!strcmp(*argv, "-gray_colors")) color_type = 0;
      else if (!strcmp(*argv, "-heatmap_colors")) color_type = 1;
      else if (!strcmp(*argv, "-label_colors")) color_type = 2;
      else if (!strcmp(*argv, "-selected_grid_range")) {
        argc--; argv++; RNScalar min_value = atof(*argv);
        argc--; argv++; RNScalar max_value = atof(*argv);
        selected_grid_range.Reset(min_value, max_value);
      }
      else {
        fprintf(stderr, "Invalid program argument: %s", *argv);
        exit(1);
      }
      argv++; argc--;
    }
    else {
      grid_names.Insert(*argv);
      argv++; argc--;
    }
  }

  // Check grid filename
  if (grid_names.IsEmpty()) {
    fprintf(stderr, "Usage: pfmview grid_name [options].\n");
    return 0;
  }

  // Return OK status 
  return 1;
}



int main(int argc, char **argv)
{
  // Parse program arguments
  if (!ParseArgs(argc, argv)) exit(-1);

  // Read grid
  if (!ReadGrids(grid_names)) exit(-1);

  // Initialize GLUT
  GLUTInit(&argc, argv);

  // Run GLUT interface
  GLUTMainLoop();

  // Return success 
  return 0;
}








