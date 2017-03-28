// Source file for the surfel tree viewer program



////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "R3Graphics/R3Graphics.h"
#include "R3Surfels/R3Surfels.h"
#include "R3SurfelViewer.h"
#include "fglut/fglut.h" 
#include "align.h"
#include "debug.h"



////////////////////////////////////////////////////////////////////////
// Global variables
////////////////////////////////////////////////////////////////////////

// Program arguments

static char *scene_name = NULL;
static char *database_name = NULL;
static char *model_name = NULL;
static char *output_image_name = NULL;
static int print_verbose = 0;


// Application variables

R3SurfelScene *scene = NULL;
R3SurfelViewer *viewer = NULL;
R3Scene *model = NULL;


// Display variables

static int color_scheme = R3_SURFEL_VIEWER_COLOR_BY_RGB;
static int cull_backfacing = 0;
static int show_model = 1;
static int show_model_names = 0;
static RNRgb background_color(0,0,0);
static R3Point initial_camera_eye(0,0,0);
static R3Vector initial_camera_towards(0,0,0);
static R3Vector initial_camera_up(0,0,0);
static RNBoolean initial_camera = FALSE;



// Glut variables

static int GLUTwindow = 0;
static int GLUTwindow_width = 1024;
static int GLUTwindow_height = 768;



////////////////////////////////////////////////////////////////////////
// Surfel I/O Functions
////////////////////////////////////////////////////////////////////////

static R3SurfelScene *
OpenScene(const char *scene_name, const char *database_name)
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
  if (!scene->OpenFile(scene_name, database_name, "r+", "r+")) {
    delete scene;
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("Opened scene ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Objects = %d\n", scene->NObjects());
    printf("  # Labels = %d\n", scene->NLabels());
    printf("  # Object Properties = %d\n", scene->NObjectProperties());
    printf("  # Label Properties = %d\n", scene->NLabelProperties());
    printf("  # Object Relationships = %d\n", scene->NObjectRelationships());
    printf("  # Label Relationships = %d\n", scene->NLabelRelationships());
    printf("  # Assignments = %d\n", scene->NLabelAssignments());
    printf("  # Features = %d\n", scene->NFeatures());
    printf("  # Scans = %d\n", scene->NScans());
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
    printf("  # Scans = %d\n", scene->NScans());
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



static R3Scene *
ReadModel(char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate model
  R3Scene *model = new R3Scene();
  if (!model) {
    fprintf(stderr, "Unable to allocate model for %s\n", filename);
    return NULL;
  }

  // Read model from file
  if (!model->ReadFile(filename)) {
    delete model;
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("Read model from %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Nodes = %d\n", model->NNodes());
    fflush(stdout);
  }

  // Return model
  return model;
}



////////////////////////////////////////////////////////////////////////
// GLUT callback functions
////////////////////////////////////////////////////////////////////////

static void 
DrawText(const R3Point& p, const char *s, void *font = GLUT_BITMAP_HELVETICA_12)
{
  // Draw text string s and position p
  glRasterPos3d(p[0], p[1], p[2]);
  while (*s) glutBitmapCharacter(font, *(s++));
}
 


void GLUTStop(void)
{
  // Terminate viewer
  if (viewer) viewer->Terminate();

  // Close scene
  if (scene) CloseScene(scene);

  // Destroy window 
  glutDestroyWindow(GLUTwindow);

  // Exit
  exit(0);
}



void GLUTRedraw(void)
{
  // Redraw with viewer
  if (viewer->Redraw()) {
    glutPostRedisplay();
  }

  // Draw model
  if (model) {
    // Draw model faces
    if (show_model) {
      glEnable(GL_LIGHTING);
      static GLfloat material[] = { 0.0, 0.8, 0.0, 1.0 };
      glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, material); 
      model->Draw(R3_DEFAULT_DRAW_FLAGS);
      glDisable(GL_LIGHTING);
    }

    // Draw model node names
    if (show_model_names) {
      glDisable(GL_LIGHTING);
      glColor3d(1, 1, 1);
      for (int i = 0; i < model->NNodes(); i++) {
        R3SceneNode *node = model->Node(i);
        if (!node->NChildren() == 0) continue;
        R3Point p = node->Centroid() + 1.5 * node->BBox().ZRadius() * R3posz_vector;
        DrawText(p, node->Name());
      }
    }
  }

  // Draw debug info
  if (DebugRedraw(viewer)) {
    glutPostRedisplay();
  }    

  // Capture image and exit
  if (output_image_name) {
    R2Image image(GLUTwindow_width, GLUTwindow_height, 3);
    image.Capture();
    image.Write(output_image_name);
    GLUTStop();
  }

  // Swap buffers 
  glutSwapBuffers();
}    



void GLUTResize(int w, int h)
{
  // Remember window dimensions
  GLUTwindow_width = w;
  GLUTwindow_height = h;

  // Resize viewer
  if (viewer->Resize(w, h)) {
    glutPostRedisplay();
  }

  // Send resize to debug function
  if (DebugResize(viewer, w, h)) {
    glutPostRedisplay();
  }    
}



void GLUTMouseMotion(int x, int y)
{
  // Invert y coordinate
  y = GLUTwindow_height - y;
  
  // Send mouse motion to viewer
  if (viewer->MouseMotion(x, y)) {
    glutPostRedisplay();
  }

  // Send mouse motion to debug function
  if (DebugMouseMotion(viewer, x, y)) {
    glutPostRedisplay();
  }    
}



void GLUTMouseButton(int button, int state, int x, int y)
{
  // Invert y coordinate
  y = GLUTwindow_height - y;
  
  // Determine button status
  int b = (button == GLUT_LEFT_BUTTON) ? 0 : ((button == GLUT_MIDDLE_BUTTON) ? 1 : 2);
  int s = (state == GLUT_DOWN) ? 1 : 0;

  // Determine modifiers
  int modifiers = glutGetModifiers();
  int shift = (modifiers & GLUT_ACTIVE_SHIFT);
  int ctrl = (modifiers & GLUT_ACTIVE_CTRL);
  int alt = (modifiers & GLUT_ACTIVE_ALT);

  // Send mouse event to viewer
  if (viewer->MouseButton(x, y, b, s, shift, ctrl, alt)) {
    glutPostRedisplay();
  }

  // Send mouse event to debug function
  if (DebugMouseButton(viewer, x, y, b, s, shift, ctrl, alt)) {
    glutPostRedisplay();
  }    
}



void GLUTSpecial(int key, int x, int y)
{
  // Invert y coordinate
  y = GLUTwindow_height - y;

  // Determine modifiers
  int modifiers = glutGetModifiers();
  int shift = (modifiers & GLUT_ACTIVE_SHIFT);
  int ctrl = (modifiers & GLUT_ACTIVE_CTRL);
  int alt = (modifiers & GLUT_ACTIVE_ALT);

  // Translate key
  int translated_key = key;
  switch (key) {
    case GLUT_KEY_DOWN: translated_key = R3_SURFEL_VIEWER_DOWN_KEY; break;
    case GLUT_KEY_UP: translated_key = R3_SURFEL_VIEWER_UP_KEY; break;
    case GLUT_KEY_LEFT: translated_key = R3_SURFEL_VIEWER_LEFT_KEY; break;
    case GLUT_KEY_RIGHT: translated_key = R3_SURFEL_VIEWER_RIGHT_KEY; break;
    case GLUT_KEY_HOME: translated_key = R3_SURFEL_VIEWER_HOME_KEY; break;
    case GLUT_KEY_END: translated_key = R3_SURFEL_VIEWER_END_KEY; break;
    case GLUT_KEY_INSERT: translated_key = R3_SURFEL_VIEWER_INSERT_KEY; break;
    case GLUT_KEY_F1: translated_key = R3_SURFEL_VIEWER_F1_KEY; break;
    case GLUT_KEY_F2: translated_key = R3_SURFEL_VIEWER_F2_KEY; break;
    case GLUT_KEY_F3: translated_key = R3_SURFEL_VIEWER_F3_KEY; break;
    case GLUT_KEY_F4: translated_key = R3_SURFEL_VIEWER_F4_KEY; break;
    case GLUT_KEY_F5: translated_key = R3_SURFEL_VIEWER_F5_KEY; break;
    case GLUT_KEY_F6: translated_key = R3_SURFEL_VIEWER_F6_KEY; break;
    case GLUT_KEY_F7: translated_key = R3_SURFEL_VIEWER_F7_KEY; break;
    case GLUT_KEY_F8: translated_key = R3_SURFEL_VIEWER_F8_KEY; break;
    case GLUT_KEY_F9: translated_key = R3_SURFEL_VIEWER_F9_KEY; break;
    case GLUT_KEY_F10: translated_key = R3_SURFEL_VIEWER_F10_KEY; break;
    case GLUT_KEY_F11: translated_key = R3_SURFEL_VIEWER_F11_KEY; break;
    case GLUT_KEY_F12: translated_key = R3_SURFEL_VIEWER_F12_KEY; break;
  }

  // Send keyboard event to viewer
  if (viewer->Keyboard(x, y, translated_key, shift, ctrl, alt)) {
    glutPostRedisplay();
  }

  // Send keyboard event to debug function
  if (DebugKeyboard(viewer, x, y, translated_key, shift, ctrl, alt)) {
    glutPostRedisplay();
  }    
}



void GLUTKeyboard(unsigned char key, int x, int y)
{
  // Invert y coordinate
  y = GLUTwindow_height - y;

  // Determine modifiers
  int modifiers = glutGetModifiers();
  int shift = (modifiers & GLUT_ACTIVE_SHIFT);
  int ctrl = (modifiers & GLUT_ACTIVE_CTRL);
  int alt = (modifiers & GLUT_ACTIVE_ALT);

  // Convert control keys 
  int translated_key = key;
  if ((key >= 1) && (key <= 26)) {
    if (shift) translated_key += 64;
    else translated_key += 96;
    ctrl = GLUT_ACTIVE_CTRL;
  }

  // Process keyboard event
  if (alt) {
    if (translated_key == 'M') {
      show_model_names = !show_model_names;
      glutPostRedisplay();
    }
    else if (translated_key == 'm') {
      show_model = !show_model;
      glutPostRedisplay();
    }
  }
  else {
    if (translated_key == 27) { // ESC
      exit(0);
    }
    else if (translated_key == 32) { // SPACE
      const R3Camera& camera = viewer->Camera();
      printf("#camera  %g %g %g  %g %g %g  %g %g %g\n",
        camera.Origin().X(), camera.Origin().Y(), camera.Origin().Z(),
        camera.Towards().X(), camera.Towards().Y(), camera.Towards().Z(),
        camera.Up().X(), camera.Up().Y(), camera.Up().Z());
    }
  }

  // Send keyboard event to viewer
  if (viewer->Keyboard(x, y, translated_key, shift, ctrl, alt)) {
    glutPostRedisplay();
  }

  // Send keyboard event to debug function
  if (DebugKeyboard(viewer, x, y, translated_key, shift, ctrl, alt)) {
    glutPostRedisplay();
  }    
}



void GLUTInit(int *argc, char **argv)
{
  // Open window
  glutInit(argc, argv);
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(GLUTwindow_width, GLUTwindow_height);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_ALPHA);
  GLUTwindow = glutCreateWindow("Surfel Viewer");

  // Initialize GLUT callback functions
  glutDisplayFunc(GLUTRedraw);
  glutReshapeFunc(GLUTResize);
  glutKeyboardFunc(GLUTKeyboard);
  glutSpecialFunc(GLUTSpecial);
  glutMouseFunc(GLUTMouseButton);
  glutMotionFunc(GLUTMouseMotion);
  atexit(GLUTStop);

  // Initialize viewer
  viewer->Initialize();
}



void GLUTMainLoop(void)
{
  // Set viewer properties
  viewer->SetSurfelColorScheme(color_scheme);
  viewer->SetBackgroundColor(background_color);
  viewer->SetBackfacingVisibility(1 - cull_backfacing);

  // Set camera 
  if (initial_camera) {
    R3Camera camera(initial_camera_eye, initial_camera_towards, initial_camera_up, 0.4, 0.4, 0.01, 100000.0);
    viewer->SetCamera(camera);
  }

  // Run main loop -- never returns
  glutMainLoop();
}



////////////////////////////////////////////////////////////////////////
// Argument Parsing Functions
////////////////////////////////////////////////////////////////////////

static int 
ParseArgs(int argc, char **argv)
{
  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) print_verbose = 1;
      else if (!strcmp(*argv, "-image")) { 
        argc--; argv++; output_image_name = *argv; 
      }
      else if (!strcmp(*argv, "-color")) { 
        argc--; argv++; color_scheme = atoi(*argv); 
      }
      else if (!strcmp(*argv, "-cull_backfacing")) { 
        cull_backfacing = 1; 
      }
      else if (!strcmp(*argv, "-window")) { 
        argv++; argc--; GLUTwindow_width = atoi(*argv); 
        argv++; argc--; GLUTwindow_height = atoi(*argv); 
      }
      else if (!strcmp(*argv, "-background")) { 
        argv++; argc--; background_color[0] = atof(*argv); 
        argv++; argc--; background_color[1] = atof(*argv); 
        argv++; argc--; background_color[2] = atof(*argv); 
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
        initial_camera_eye = R3Point(x, y, z);
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
      if (!scene_name) scene_name = *argv;
      else if (!database_name) database_name = *argv;
      else if (!model_name) model_name = *argv;
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
  }

  // Check surfels name
  if (!scene_name || !database_name) {
    fprintf(stderr, "Usage: sflview scenefile databasefile [options]\n");
    return FALSE;
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

  // Open scene
  scene = OpenScene(scene_name, database_name);
  if (!scene) exit(-1);

  // Read model
  if (model_name) {
    model = ReadModel(model_name);
    if (!model) exit(-1);
  }

  // Create viewer
  viewer = new R3SurfelViewer(scene);
  if (!viewer) exit(-1);

  // Initialize GLUT
  GLUTInit(&argc, argv);

  // Run GLUT interface
  GLUTMainLoop();

  // Return success 
  return 0;
}
