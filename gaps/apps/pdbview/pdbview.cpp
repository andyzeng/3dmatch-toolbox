// Source file for the pdb viewer program



// Include files 

#include "R3Graphics/R3Graphics.h"
#include "R3Shapes/R3Shapes.h"
#include "PDB/PDB.h"
#include "fglut/fglut.h"



// Display types

typedef enum {
  NULL_DISPLAY_SCHEME,
  POINT_DISPLAY_SCHEME,
  BALL_DISPLAY_SCHEME,
  // STICK_DISPLAY_SCHEME,
  NUM_DISPLAY_SCHEMES
} DisplayScheme;

typedef enum {
  FILE_COLOR_SCHEME,
  MODEL_COLOR_SCHEME,
  CHAIN_COLOR_SCHEME,
  RESIDUE_COLOR_SCHEME,
  ATOM_COLOR_SCHEME,
  NUM_COLOR_SCHEMES
} ColorScheme;



// Program variables

static char *pdb_name = NULL;
static char *grid_name = NULL;
static char *image_name = NULL;
static char *ligand_name = NULL;
static char *asa_name = NULL;
static char *consurf_name = NULL;
static char *hssp_name = NULL;
static char *jsd_name = NULL;
static RNScalar background_color[3] = { 1, 1, 1 };
static R3Point initial_camera_origin = R3Point(0.0, 0.0, 0.0);
static R3Vector initial_camera_towards = R3Vector(0.0, 0.0, -1.0);
static R3Vector initial_camera_up = R3Vector(0.0, 1.0, 0.0);
static RNBoolean initial_camera = FALSE;
static RNBoolean biomolecule = FALSE;
static RNScalar world_radius = 0;
static int print_verbose = 0;



// GLUT variables 

static int GLUTwindow = 0;
static int GLUTwindow_height = 1024;
static int GLUTwindow_width = 1024;
static int GLUTmouse[2] = { 0, 0 };
static int GLUTbutton[3] = { 0, 0, 0 };
static int GLUTmodifiers = 0;



// Application variables

static PDBFile *file = NULL;
static PDBResidue *ligand = NULL;
static R3Grid *grid = NULL;
static R3Viewer *viewer = NULL;



// Display variables

static int protein_display_scheme = BALL_DISPLAY_SCHEME;
static int hetatom_display_scheme = BALL_DISPLAY_SCHEME;
static int ligand_display_scheme = BALL_DISPLAY_SCHEME;
static int protein_color_scheme = RESIDUE_COLOR_SCHEME;
static int hetatom_color_scheme = RESIDUE_COLOR_SCHEME;
static int ligand_color_scheme = RESIDUE_COLOR_SCHEME;
static int show_backbone_only = 0;
static int show_charge = 0;
static int show_conservation = 0;
static int show_hydrophobicity = 0;
static int show_accessible_surface_area = 0;
static int show_occupancy = 0;
static int show_temperature_factor = 0;
static int show_grid_isosurface = 0;
static int show_names = 0;
static RNScalar grid_threshold = 1.0E-6;



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



void GLUTIdle(void)
{
  // Redraw
  glutPostRedisplay();
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

  // Draw PDB file
  if (file) {
    // Show protein
    if (protein_display_scheme != NULL_DISPLAY_SCHEME) {
      glEnable(GL_LIGHTING);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      RNRgb color(0.5, 0.25, 0.5);
      for (int m = 0; m < file->NModels(); m++) {
        PDBModel *model = file->Model(m);
        if (protein_color_scheme == MODEL_COLOR_SCHEME) color = model->Color();
        for (int i = 0; i < model->NChains(); i++) {
          PDBChain *chain = model->Chain(i);
          if (protein_color_scheme == CHAIN_COLOR_SCHEME) color = chain->Color();
          for (int j = 0; j < chain->NResidues(); j++) {
            PDBResidue *residue = chain->Residue(j);
            if (residue == ligand) continue;
            if (protein_color_scheme == RESIDUE_COLOR_SCHEME) color = residue->Color();
            for (int k = 0; k < residue->NAtoms(); k++) {
              PDBAtom *atom = residue->Atom(k);
              if (show_backbone_only && !atom->IsBackbone()) continue;
              if (atom->IsHetAtom()) continue;
              if (show_charge) {
                if (atom->Charge() == PDB_UNKNOWN) color = RNblue_rgb;
                else { RNScalar c = 0.5 + atom->Charge(); color[0] = c; color[1] = c; color[2] = c; }
              }
              else if (show_conservation) {
                if (residue->conservation == PDB_UNKNOWN) color = RNblue_rgb;
                else { color[0] = 1 - residue->conservation; color[1] = 1 - residue->conservation; color[2] = 1 - residue->conservation; }
              }
              else if (show_hydrophobicity) {
                if (atom->IsHetAtom()) color = RNblue_rgb;
                else { RNScalar h = 1 - (atom->Hydrophobicity() + 1)/3; color[0] = h; color[1] = h; color[2] = h; }
              }
              else if (show_temperature_factor) {
                if (atom->IsHetAtom()) color = RNblue_rgb;
                else { RNScalar t = 10 * (atom->tempFactor - 1.2); color[0] = t; color[1] = t; color[2] = t; }
              }
              else if (show_accessible_surface_area) {
                if (atom->accessible_surface_area == PDB_UNKNOWN) color = RNblue_rgb;
                else { RNScalar asa = atom->accessible_surface_area / 50; color[0] = asa; color[1] = asa; color[2] = asa; }
              }
              else if (protein_color_scheme == ATOM_COLOR_SCHEME) color = atom->Color();
              if (show_occupancy && (atom->Occupancy() < 1.0)) glEnable(GL_BLEND);
              static GLfloat material[4];
              material[0] = color[0]; material[1] = color[1]; material[2] = color[2]; material[3] = atom->Occupancy();
              glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, material); 
              RNLength radius = (protein_display_scheme == BALL_DISPLAY_SCHEME) ? atom->Radius() : 0.25;
              R3Sphere(atom->Position(), radius).Draw();
              if (show_occupancy && (atom->Occupancy() < 1.0)) glDisable(GL_BLEND);
            }
          }
        }
      }
    }

    // Show hetatoms
    if (hetatom_display_scheme != NULL_DISPLAY_SCHEME) {
      glEnable(GL_LIGHTING);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      RNRgb color(0.5, 0.25, 0.5);
      for (int m = 0; m < file->NModels(); m++) {
        PDBModel *model = file->Model(m);
        if (hetatom_color_scheme == MODEL_COLOR_SCHEME) color = model->Color();
        for (int i = 0; i < model->NChains(); i++) {
          PDBChain *chain = model->Chain(i);
          if (hetatom_color_scheme == CHAIN_COLOR_SCHEME) color = chain->Color();
          for (int j = 0; j < chain->NResidues(); j++) {
            PDBResidue *residue = chain->Residue(j);
            if (residue == ligand) continue;
            if (hetatom_color_scheme == RESIDUE_COLOR_SCHEME) color = residue->Color();
            for (int k = 0; k < residue->NAtoms(); k++) {
              PDBAtom *atom = residue->Atom(k);
              if (!atom->IsHetAtom()) continue;
              if (show_charge) {
                if (atom->Charge() == PDB_UNKNOWN) color = RNblue_rgb;
                else { RNScalar c = 0.5 + atom->Charge(); color[0] = c; color[1] = c; color[2] = c; }
              }
              else if (show_conservation) {
                if (residue->conservation == PDB_UNKNOWN) color = RNblue_rgb;
                else { color[0] = 1 - residue->conservation; color[1] = 1 - residue->conservation; color[2] = 1 - residue->conservation; }
              }
              else if (show_hydrophobicity) {
                if (atom->IsHetAtom()) color = RNblue_rgb;
                else { RNScalar h = 1 - (atom->Hydrophobicity() + 1)/3; color[0] = h; color[1] = h; color[2] = h; }
              }
              else if (show_accessible_surface_area) {
                if (atom->accessible_surface_area == PDB_UNKNOWN) color = RNblue_rgb;
                else { RNScalar asa = atom->accessible_surface_area / 50; color[0] = asa; color[1] = asa; color[2] = asa; }
              }
              else if (hetatom_color_scheme == ATOM_COLOR_SCHEME) color = atom->Color();
              if (show_occupancy && (atom->Occupancy() < 1.0)) glEnable(GL_BLEND);
              static GLfloat material[4];
              material[0] = color[0]; material[1] = color[1]; material[2] = color[2]; material[3] = atom->Occupancy();
              glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, material); 
              RNLength radius = (hetatom_display_scheme == BALL_DISPLAY_SCHEME) ? atom->Radius() : 0.25;
              R3Sphere(atom->Position(), radius).Draw();
              if (show_occupancy && (atom->Occupancy() < 1.0)) glDisable(GL_BLEND);
            }
          }
        }
      }
    }

    // Show ligand
    if (ligand && (ligand_display_scheme != NULL_DISPLAY_SCHEME)) {
      // Show atoms
      if (ligand_display_scheme != NULL_DISPLAY_SCHEME) {
        PDBChain *chain = ligand->Chain();
        glEnable(GL_LIGHTING);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        static RNRgb default_color(0, 1, 0);
        RNRgb color = default_color;
        if (ligand_color_scheme == CHAIN_COLOR_SCHEME) color = chain->Color();
        else if (ligand_color_scheme == RESIDUE_COLOR_SCHEME) color = ligand->Color();
        for (int i = 0; i < ligand->NAtoms(); i++) {
          PDBAtom *atom = ligand->Atom(i);
          if (show_occupancy && (atom->Occupancy() < 1.0)) glEnable(GL_BLEND);
          if (ligand_color_scheme == ATOM_COLOR_SCHEME) color = atom->Color();
          static GLfloat material[4];
          material[0] = color[0]; material[1] = color[1]; material[2] = color[2]; material[3] = atom->Occupancy();
          glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, material); 
          RNLength radius = (ligand_display_scheme == BALL_DISPLAY_SCHEME) ? atom->Radius() : 0.5;
          R3Sphere(atom->Position(), radius).Draw();
          if (show_occupancy && (atom->Occupancy() < 1.0)) glDisable(GL_BLEND);
        }
      }
    }

    // Show names
    if (show_names) {
      glDisable(GL_LIGHTING);
      for (int m = 0; m < file->NModels(); m++) {
        PDBModel *model = file->Model(m);
        if ((protein_display_scheme != NULL_DISPLAY_SCHEME) && (protein_color_scheme == MODEL_COLOR_SCHEME)) {
          RNLoadRgb(model->Color());
          GLUTDrawText(model->BBox().Centroid(), model->Name());
        }
        for (int i = 0; i < model->NChains(); i++) {
          PDBChain *chain = model->Chain(i);
          if ((protein_display_scheme != NULL_DISPLAY_SCHEME) && (protein_color_scheme == CHAIN_COLOR_SCHEME)) {
            RNLoadRgb(chain->Color());
            GLUTDrawText(chain->BBox().Centroid(), chain->Name());
          }
          for (int j = 0; j < chain->NResidues(); j++) {
            PDBResidue *residue = chain->Residue(j);
            if (((residue == ligand) && (ligand_display_scheme != NULL_DISPLAY_SCHEME) && (ligand_color_scheme == RESIDUE_COLOR_SCHEME)) ||
                ((!residue->AminoAcid()) && (hetatom_display_scheme != NULL_DISPLAY_SCHEME) && (hetatom_color_scheme == RESIDUE_COLOR_SCHEME)) ||
                ((residue->AminoAcid()) && ((protein_display_scheme != NULL_DISPLAY_SCHEME)) && (protein_color_scheme == RESIDUE_COLOR_SCHEME))) {
              RNLoadRgb(residue->Color());
              GLUTDrawText(residue->BBox().Centroid(), residue->Name());
            }
            for (int k = 0; k < residue->NAtoms(); k++) {
              PDBAtom *atom = residue->Atom(k);
              if (((residue == ligand) && (ligand_display_scheme != NULL_DISPLAY_SCHEME) && (ligand_color_scheme == ATOM_COLOR_SCHEME)) ||
                  ((!residue->AminoAcid()) && (hetatom_display_scheme != NULL_DISPLAY_SCHEME) && (hetatom_color_scheme == ATOM_COLOR_SCHEME)) ||
                  ((residue->AminoAcid()) && (protein_display_scheme != NULL_DISPLAY_SCHEME) && (protein_color_scheme == ATOM_COLOR_SCHEME))) {
                RNLoadRgb(atom->Color());
                GLUTDrawText(atom->Position(), atom->Name());
              }
            }
          }
        }
      }
    }
  }
    
  // Show grid
  if (grid && show_grid_isosurface) {
    // Push grid transformation
    const R3Affine& grid_to_world = grid->GridToWorldTransformation();
    grid_to_world.Push();

    // Draw grid isosurface
    glDisable(GL_LIGHTING);
    RNLoadRgb(1.0, 0.0, 0.0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    grid->DrawIsoSurface(grid_threshold);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // Pop grid transformation
    grid_to_world.Pop();
  }
    
  // Capture image 
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
  R3Point origin = file->BBox().Centroid();
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
  case GLUT_KEY_DOWN:
    grid_threshold *= 0.9;
    if (grid_threshold < 1.0E-6) grid_threshold = 0.0;
    break;

  case GLUT_KEY_UP:
    grid_threshold *= 1.1;
    if (grid_threshold < 1.0E-6) grid_threshold = 1.0E-6;
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
  case '~': {
    glReadBuffer(GL_FRONT);
    R2Image image(GLUTwindow_width, GLUTwindow_height, 3);
    image.Capture();
    char imagename[256];
    static int image_count = 1;
    sprintf(imagename, "i%d.jpg", image_count++);
    image.Write(imagename);
    glReadBuffer(GL_BACK);
    break; }

  case 'A':
  case 'a':
    show_accessible_surface_area = !show_accessible_surface_area;
    break;

  case 'B':
  case 'b':
    show_backbone_only = !show_backbone_only;
    break;

  case 'C':
  case 'c':
    show_conservation = !show_conservation;
    break;

  case 'H':
    hetatom_color_scheme = (hetatom_color_scheme + 1) % NUM_COLOR_SCHEMES;
    break;

  case 'h':
    hetatom_display_scheme = (hetatom_display_scheme + 1) % NUM_DISPLAY_SCHEMES;
    break;

  case 'I':
  case 'i':
    show_grid_isosurface = !show_grid_isosurface;
    break;

  case 'L':
    ligand_color_scheme = (ligand_color_scheme + 1) % NUM_COLOR_SCHEMES;
    break;

  case 'l':
    ligand_display_scheme = (ligand_display_scheme + 1) % NUM_DISPLAY_SCHEMES;
    break;

  case 'N':
  case 'n':
    show_names = !show_names;
    break;

  case 'O':
  case 'o':
    show_occupancy = !show_occupancy;
    break;

  case 'P':
    protein_color_scheme = (protein_color_scheme + 1) % NUM_COLOR_SCHEMES;
    break;

  case 'p':
    protein_display_scheme = (protein_display_scheme + 1) % NUM_DISPLAY_SCHEMES;
    break;

  case 'R':
  case 'r':
    show_charge = !show_charge;
    break;

  case 'T':
  case 't':
    show_temperature_factor = !show_temperature_factor;
    break;

  case 'Y':
  case 'y':
    show_hydrophobicity = !show_hydrophobicity;
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



void GLUTInit(int *argc, char **argv)
{
  // Open window 
  glutInit(argc, argv);
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(GLUTwindow_width, GLUTwindow_height);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH); // | GLUT_STENCIL
  GLUTwindow = glutCreateWindow("PDB Viewer");

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
  glPointSize(3); 

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
  // Set the window's title
  glutSetWindowTitle(pdb_name);

  // Run main loop -- never returns 
  glutMainLoop();
}

 

static PDBFile *
ReadPDB(char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate PDBFile
  PDBFile *file = new PDBFile(filename);
  if (!file) {
    RNFail("Unable to allocate PDB file for %s", filename);
    return NULL;
  }

  // Read DB file
  if (!file->ReadFile(filename)) {
    RNFail("Unable to read PDB file: %s", filename);
    return NULL;
  }

  // Get biomolecule
  if (biomolecule && (!file->IsBiomolecule())) {
    PDBFile *file2 = file->CopyBiomolecule();
    delete file;
    file = file2;
  }

  // Print statistics
  if (print_verbose) {
    printf("Read PDB file ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    for (int i =0; i < file->NModels(); i++) {
      PDBModel *model = file->Model(i);
      printf("  Model %s ...\n", model->Name());
      printf("  # Chains = %d\n", model->NChains());
      printf("  # Residues = %d\n", model->NResidues());
      printf("  # Atoms = %d\n", model->NAtoms());
    }
    fflush(stdout);
  }

  // Return success
  return file;
}



static PDBResidue *
FindLigand(PDBFile *file, const char *ligand_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Find ligand
  PDBResidue *ligand = file->FindResidue(ligand_name);
  if (!ligand) {
    fprintf(stderr, "Unable to find ligand %s in %s\n", ligand_name, pdb_name); 
    return 0;
  }

  // Print statistics
  if (print_verbose) {
    printf("Found ligand ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Atoms = %d\n", ligand->NAtoms());
    fflush(stdout);
  }

  // Return ligand
  return ligand;
}



static int
ReadConsurfFiles(PDBFile *file, char *consurf_basename)
{
  // Check number of models
  if (file->NModels() < 1) {
    fprintf(stderr, "File must have at least one model to read conservation scores: %s.\n", consurf_basename);
    return 0;
  }

  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Read conservation files
  int nresidues = file->ReadConsurfFiles(consurf_basename);

  // Print statistics
  if (print_verbose && (nresidues > 0)) {
    printf("Read consurf files ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Residues = %d\n", nresidues);
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
ReadHsspFiles(PDBFile *file, char *hssp_basename)
{
  // Check number of models
  if (file->NModels() < 1) {
    fprintf(stderr, "File must have at least one model to read conservation scores: %s.\n", hssp_basename);
    return 0;
  }

  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Read conservation files
  int nresidues = file->ReadHsspFiles(hssp_basename);

  // Print statistics
  if (print_verbose && (nresidues > 0)) {
    printf("Read hssp files ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf(" # Residues = %d\n", nresidues);
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
ReadJsdFiles(PDBFile *file, char *jsd_basename)
{
  // Check number of models
  if (file->NModels() < 1) {
    fprintf(stderr, "File must have at least one model to read conservation scores: %s.\n", jsd_basename);
    return 0;
  }

  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Read conservation files
  int nresidues = file->ReadJsdFiles(jsd_basename);

  // Print statistics
  if (print_verbose && (nresidues > 0)) {
    printf("Read jsd files ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf(" # Residues = %d\n", nresidues);
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
ReadASAFile(PDBFile *file, char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Read ASA file
  int natoms = file->ReadASAFile(filename);
  if (natoms == 0) return 0;

  // Print statistics
  if (print_verbose) {
    printf("Read accessible surface area file ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Surface Atoms = %d\n", natoms);
    fflush(stdout);
  }

  // Return success
  return 1;
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

  // Set the grid threshold
  grid_threshold = grid->Mean();

  // Print statistics
  if (print_verbose) {
    printf("Read grid ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Dimensions = %d %d %d\n", grid->XResolution(), grid->YResolution(), grid->ZResolution());
    printf("  Maximum = %g\n", grid->Maximum());
    printf("  L2Norm = %g\n", grid->L2Norm());
    fflush(stdout);
  }

  // Return success
  return grid;
}



static R3Viewer *
CreateViewer(PDBFile *file, PDBResidue *ligand)
{
  // Get file bounding box
  R3Box bbox = file->BBox();
  assert(!bbox.IsEmpty());
  RNLength r = bbox.DiagonalRadius();
  assert((r > 0.0) && RNIsFinite(r));
  if (world_radius == 0.0) world_radius = r;

  // Setup default camera view looking down the Z axis
  if (!initial_camera) initial_camera_origin = bbox.Centroid() - initial_camera_towards * (2 * world_radius);;
  R3Camera camera(initial_camera_origin, initial_camera_towards, initial_camera_up, 0.4, 0.4, 0.1 * r, 1000.0 * r);
  R2Viewport viewport(0, 0, GLUTwindow_width, GLUTwindow_height);
  R3Viewer *viewer = new R3Viewer(camera, viewport);

  // Zoom in on ligand, if there is one
  if (!initial_camera && ligand) {
    // Reset camera
    R3Box ligand_bbox = ligand->BBox();
    R3Point ligand_center = ligand_bbox.Centroid();
    R3Point file_center = file->BBox().Centroid();
    R3Vector towards = file_center - ligand_center;
    RNLength towards_length = towards.Length();
    if (RNIsZero(towards_length)) towards = R3negz_vector;
    else towards /= towards_length;
    R3Vector up = R3posy_vector;
    RNScalar dot = up.Dot(towards);
    if (RNIsEqual(fabs(dot), 1.0)) up = R3posx_vector;
    R3Point eye = ligand_center - towards * (2 * world_radius);;
    viewer->ResetCamera(eye, towards, up);
  }

  // Return viewer
  return viewer;
}



static int 
ParseArgs(int argc, char **argv)
{
  // Check number of arguments
  if ((argc < 2) || (!strcmp(argv[1], "-help"))) {
    printf("Usage: pdbview filename [options]\n");
    exit(0);
  }

  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) print_verbose = 1; 
      else if (!strcmp(*argv, "-biomolecule")) biomolecule = 1; 
      else if (!strcmp(*argv, "-ligand")) { argc--; argv++; ligand_name = *argv; }
      else if (!strcmp(*argv, "-grid")) { argc--; argv++; grid_name = *argv; }
      else if (!strcmp(*argv, "-image")) { argc--; argv++; image_name = *argv; }
      else if (!strcmp(*argv, "-consurf")) { argc--; argv++; consurf_name = *argv; }
      else if (!strcmp(*argv, "-hssp")) { argc--; argv++; hssp_name = *argv; }
      else if (!strcmp(*argv, "-jsd")) { argc--; argv++; jsd_name = *argv; }
      else if (!strcmp(*argv, "-asa")) { argc--; argv++; asa_name = *argv; }
      else if (!strcmp(*argv, "-radius")) { argc--; argv++; world_radius = atof(*argv); }
      else if (!strcmp(*argv, "-colors")) {
        argv++; argc--; protein_color_scheme = atoi(*argv) % NUM_COLOR_SCHEMES; 
        argv++; argc--; hetatom_color_scheme = atoi(*argv) % NUM_COLOR_SCHEMES; 
        argv++; argc--; ligand_color_scheme = atoi(*argv) % NUM_COLOR_SCHEMES; 
      }
      else if (!strcmp(*argv, "-displays")) {
        argv++; argc--; protein_display_scheme = atoi(*argv) % NUM_DISPLAY_SCHEMES; 
        argv++; argc--; hetatom_display_scheme = atoi(*argv) % NUM_DISPLAY_SCHEMES; 
        argv++; argc--; ligand_display_scheme = atoi(*argv) % NUM_DISPLAY_SCHEMES; 
      }
      else if (!strcmp(*argv, "-background")) { 
        argc--; argv++; background_color[0] = atof(*argv); 
        argc--; argv++; background_color[1] = atof(*argv); 
        argc--; argv++; background_color[2] = atof(*argv); 
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
      if (!pdb_name) pdb_name = *argv;
      else { 
        fprintf(stderr, "Invalid program argument: %s", *argv); 
        exit(1); 
      }
      argv++; argc--;
    }
  }

  // Check filename
  if (!pdb_name) {
    fprintf(stderr, "You did not specify a file.\n");
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

  // Read pdb
  file = ReadPDB(pdb_name);
  if (!file) exit(-1);

  // Find ligand
  if (ligand_name) {
    ligand = FindLigand(file, ligand_name);
    if (!ligand) exit(-1);
  }

  // Read conservation files
  if (jsd_name) {
    int status = ReadJsdFiles(file, jsd_name);
    if (!status) exit(-1);
  }
  else if (hssp_name) {
    int status = ReadHsspFiles(file, hssp_name);
    if (!status) exit(-1);
  }
  else if (consurf_name) {
    int status = ReadConsurfFiles(file, consurf_name);
    if (!status) exit(-1);
  }

  // Read accessible surface area file
  if (asa_name) {
    int status = ReadASAFile(file, asa_name);
    if (!status) exit(-1);
  }

  // Read grid
  if (grid_name) {
    grid = ReadGrid(grid_name);
    if (!grid) exit(-1);
  }

  // Create viewer
  viewer = CreateViewer(file, ligand);
  if (!viewer) exit(-1);

  // Run GLUT interface
  GLUTMainLoop();

  // Return success 
  return 0;
}



