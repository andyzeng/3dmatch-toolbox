/* Source file for the surfel scene viewer class */



////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "R3Graphics/R3Graphics.h"
#include "R3Surfels/R3Surfels.h"
#include "R3SurfelViewer.h"



////////////////////////////////////////////////////////////////////////
// Surfel viewer constructor/destructor
////////////////////////////////////////////////////////////////////////

R3SurfelViewer::
R3SurfelViewer(R3SurfelScene *scene)
  : scene(NULL),
    resident_nodes(),
    viewer(),
    viewing_extent(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX),
    center_point(0,0,0),
    current_scan_index(-1),
    surfel_size(2),
    surfel_visibility(1),
    normal_visibility(0),
    backfacing_visibility(1),
    background_visibility(1),
    object_property_visibility(0),
    object_label_visibility(0),
    object_name_visibility(0),
    node_bbox_visibility(0),
    block_bbox_visibility(0),
    scan_viewpoint_visibility(0),
    center_point_visibility(0),
    axes_visibility(0),
    surfel_color_scheme(R3_SURFEL_VIEWER_COLOR_BY_RGB),
    normal_color(0,1,0),
    background_color(0,0,0),
    object_property_color(0,1,1),
    object_name_color(1,1,0),
    node_bbox_color(0,0,1),
    block_bbox_color(0,1,0),
    scan_viewpoint_color(0,1,1),
    center_point_color(1,0,0),
    adapt_working_set_automatically(0),
    target_resolution(30),
    focus_radius(0),
    window_height(0),
    window_width(0),
    shift_down(0),
    ctrl_down(0),
    alt_down(0),
    start_timer(),
    frame_timer(),
    frame_time(-1),
    image_name(NULL)
{
  // Initialize mouse button state
  mouse_button[0] = 0;
  mouse_button[1] = 0;
  mouse_button[2] = 0;

  // Initialize mouse positions
  mouse_position[0] = 0;
  mouse_position[1] = 0;

  // Initialize mouse positions
  mouse_down_position[0] = 0;
  mouse_down_position[1] = 0;

  // Initialize mouse drag distance
  mouse_drag_distance_squared = 0;

  // Initialize timers
  start_timer.Read();
  frame_timer.Read();

  // Set the scene
  if (scene) SetScene(scene);
}



R3SurfelViewer::
~R3SurfelViewer(void)
{
}



////////////////////////////////////////////////////////////////////////
// Text drawing utility functions
////////////////////////////////////////////////////////////////////////

#include "fglut/fglut.h"

static void 
DrawText(const R2Point& p, const char *s, void *font = GLUT_BITMAP_HELVETICA_12)
{
  // Draw text string s and position p
  glRasterPos2d(p[0], p[1]);
  while (*s) glutBitmapCharacter(font, *(s++));
}
  


static void 
DrawText(const R3Point& p, const char *s, void *font = GLUT_BITMAP_HELVETICA_12)
{
  // Draw text string s and position p
  glRasterPos3d(p[0], p[1], p[2]);
  while (*s) glutBitmapCharacter(font, *(s++));
}
  

////////////////////////////////////////////////////////////////////////
// Coloring utility functions
////////////////////////////////////////////////////////////////////////

void
LoadColor(int k)
{
  // Make array of colors
  const int ncolors = 72;
  const RNRgb colors[ncolors] = {
    RNRgb(0.5, 0.2, 0.2), RNRgb(0, 1, 0), RNRgb(0, 0, 1), 
    RNRgb(0.3, 0.6, 0), RNRgb(0, 1, 1), RNRgb(1, 0, 1), 
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

  // Return color
  if (k == 0) RNLoadRgb(colors[0]);
  else RNLoadRgb(colors[1 + (k % (ncolors-1))]);
}



void
LoadColor(double value)
{
  // Compute rgb
  GLdouble r, g, b;
  if (value < 0) {
    r = 0;
    g = 0;
    b = 1;
  }
  else if (value < 0.1) {
    value *= 10;
    r = 0;
    g = value;
    b = 1;
  }
  else if (value < 0.5) {
    value = (value - 0.1) * 2.5;
    r = 0;
    g = 1;
    b = 1 - value;
  }
  else if (value < 0.9) {
    value = (value - 0.5) * 2.5;
    r = value;
    g = 1;
    b = 0;
  }
  else if (value < 1) {
    value = (value - 0.9) * 10;
    r = 1;
    g = 1 - value;
    b = 0;
  }
  else {
    r = 1;
    g = 0;
    b = 0;
  }

  // Load rgb
  glColor3d(r, g, b);
}



static void
LoadViewingExtent(const R3SurfelViewer *viewer)
{
  // Disable all clip planes
  for (int i = 0; i < 6; i++) {
    glDisable(GL_CLIP_PLANE0 + i);
  }

  // Check viewer
  if (viewer) {
    // Check extent
    const R3Box& extent = viewer->ViewingExtent();
    if (extent.IsEmpty()) return;
    const R3SurfelScene *scene = viewer->Scene();
    const R3Box& bbox = scene->BBox();  
    if (R3Contains(extent, bbox)) return;

    // Load lo clip planes
    for (int dim = RN_X; dim <= RN_Z; dim++) {
      if (extent[RN_LO][dim] > bbox[RN_LO][dim]) {
        GLdouble plane_equation[4] = { 0, 0, 0, 0 };
        plane_equation[dim] = 1.0;
        plane_equation[3] = -extent[RN_LO][dim];
        glClipPlane(GL_CLIP_PLANE0 + dim, plane_equation);
        glEnable(GL_CLIP_PLANE0 + dim);
      }
    }

    // Load hi clip planes
    for (int dim = RN_X; dim <= RN_Z; dim++) {
      if (extent[RN_HI][dim] < bbox[RN_HI][dim]) {
        GLdouble plane_equation[4] = { 0, 0, 0, 0 };
        plane_equation[dim] = -1.0;
        plane_equation[3] = extent[RN_HI][dim];
        glClipPlane(GL_CLIP_PLANE0 + 3 + dim, plane_equation);
        glEnable(GL_CLIP_PLANE0 + 3 + dim);
      }
    }
  }  
}



#if 0
static void
DrawSurfelQuad(const R3SurfelViewer *viewer, const R3SurfelBlock *block, const R3Surfel *surfel, unsigned char alpha)
{
  // Get color
  unsigned char color[4];
  color[0] = surfel->R();
  color[1] = surfel->G();
  color[2] = surfel->B();
  color[3] = alpha;
  
  // Get position
  const R3Point& block_origin = block->Origin();
  double px = surfel->X() + block_origin.X();
  double py = surfel->Y() + block_origin.Y();
  double pz = surfel->Z() + block_origin.Z();
  R3Point p(px, py, pz);

  // Get radius
  RNScalar r = 0.5 / block->Resolution();

  // Get axis vectors
  R3Vector normal(surfel->NX(), surfel->NY(), surfel->NZ());
  RNDimension dim = normal.MinDimension();
  R3Vector axis1 = normal % R3xyz_triad[dim];  axis1.Normalize();
  R3Vector axis2 = normal % axis1; axis2.Normalize();

  // Draw square
  glColor4ubv(color);
  // R3LoadNormal(normal);
  R3LoadPoint(p - r*axis1 - r*axis2);
  R3LoadPoint(p + r*axis1 - r*axis2);
  R3LoadPoint(p + r*axis1 + r*axis2);
  R3LoadPoint(p - r*axis1 + r*axis2);
}
#endif



////////////////////////////////////////////////////////////////////////
// UI event handler functions
////////////////////////////////////////////////////////////////////////

int R3SurfelViewer::
Redraw(void)
{
  // Check scene
  if (!scene) return 0;

  // Set viewing transformation
  viewer.Camera().Load();

  // Clear window 
  glClearColor(background_color[0], background_color[1], background_color[2], 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Set lights
  static GLfloat light0_position[] = { 3.0, 4.0, 5.0, 0.0 };
  glLightfv(GL_LIGHT0, GL_POSITION, light0_position);
  static GLfloat light1_position[] = { -3.0, -2.0, -3.0, 0.0 };
  glLightfv(GL_LIGHT1, GL_POSITION, light1_position);

  // Set viewing extent
  LoadViewingExtent(this);

  // Set draw modes
  glDisable(GL_LIGHTING);
  glPointSize(surfel_size);
  glLineWidth(1);
  if (backfacing_visibility) glDisable(GL_CULL_FACE);
  else glEnable(GL_CULL_FACE);
  RNLength distance_to_closest_scan = FLT_MAX;
  if (surfel_color_scheme == R3_SURFEL_VIEWER_COLOR_BY_VIEWPOINT) {
    for (int i = 0; i < resident_nodes.NNodes(); i++) {
      R3SurfelNode *node = resident_nodes.Node(i);
      R3SurfelScan *scan = node->Scan();
      if (!scan) continue;
      RNLength d = R3Distance(viewer.Camera().Origin(), scan->Viewpoint());
      if (d < distance_to_closest_scan) distance_to_closest_scan = d;
    }
    // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // glEnable(GL_BLEND);
    // glDepthMask(FALSE);
  }

  // Draw surfels
  if (surfel_visibility) {
    // Draw resident nodes
    if (surfel_color_scheme == R3_SURFEL_VIEWER_COLOR_BY_CURRENT_LABEL) {
      for (int i = 0; i < resident_nodes.NNodes(); i++) {
        R3SurfelNode *node = resident_nodes.Node(i);
        R3SurfelObject *object = node->Object();
        if (!object) continue;
        R3SurfelLabel *label = object->CurrentLabel();
        int label_index = (label) ? label->SceneIndex() : 0;
        LoadColor(label_index);
        node->Draw(0); 
      }
    }
    else if (surfel_color_scheme == R3_SURFEL_VIEWER_COLOR_BY_GROUND_TRUTH_LABEL) {
      for (int i = 0; i < resident_nodes.NNodes(); i++) {
        R3SurfelNode *node = resident_nodes.Node(i);
        R3SurfelObject *object = node->Object();
        if (!object) continue;
        R3SurfelLabel *label = object->GroundTruthLabel();
        int label_index = (label) ? label->SceneIndex() : 0;
        LoadColor(label_index);
        node->Draw(0); 
      }
    }
    else if (surfel_color_scheme == R3_SURFEL_VIEWER_COLOR_BY_OBJECT) {
      // Draw with colors based on nodes
      for (int i = 0; i < resident_nodes.NNodes(); i++) {
        R3SurfelNode *node = resident_nodes.Node(i);
        R3SurfelObject *object = node->Object();
        int object_index = (object) ? object->SceneIndex() : 0;
        LoadColor(object_index);
        node->Draw(0); 
      }
    }
    else if (surfel_color_scheme == R3_SURFEL_VIEWER_COLOR_BY_NODE) {
      // Draw with colors based on nodes
      for (int i = 0; i < resident_nodes.NNodes(); i++) {
        R3SurfelNode *node = resident_nodes.Node(i);
        LoadColor(i);
        node->Draw(0); 
      }
    }
    else if (surfel_color_scheme == R3_SURFEL_VIEWER_COLOR_BY_BLOCK) {
      // Draw with colors based on blocks
      int count = 0;
      for (int i = 0; i < resident_nodes.NNodes(); i++) {
        R3SurfelNode *node = resident_nodes.Node(i);
        for (int j = 0; j < node->NBlocks(); j++) {
          R3SurfelBlock *block = node->Block(j);
          LoadColor(count++);
          block->Draw(0);
        }
      }
    }
    else if (surfel_color_scheme == R3_SURFEL_VIEWER_COLOR_BY_HEIGHT) {
      // Draw with colors based on heights
      for (int i = 0; i < resident_nodes.NNodes(); i++) {
        R3SurfelNode *node = resident_nodes.Node(i);
        glBegin(GL_POINTS);
        for (int j = 0; j < node->NBlocks(); j++) {
          R3SurfelBlock *block = node->Block(j);
          const R3Point& block_origin = block->Origin();
          for (int k = 0; k < block->NSurfels(); k++) {
            const R3Surfel *surfel = block->Surfel(k);
            double x = surfel->X() + block_origin.X();
            double y = surfel->Y() + block_origin.Y();
            double z = surfel->Z() + block_origin.Z();
            double value = 0.1 * (y - center_point.Y());
            LoadColor(value);
            glVertex3d(x, y, z);
          }
        }
        glEnd();
      }
    }
    else if (surfel_color_scheme == R3_SURFEL_VIEWER_COLOR_BY_NORMAL) {
      // Draw with colors based on normals
      for (int i = 0; i < resident_nodes.NNodes(); i++) {
        R3SurfelNode *node = resident_nodes.Node(i);
        glBegin(GL_POINTS);
        for (int j = 0; j < node->NBlocks(); j++) {
          R3SurfelBlock *block = node->Block(j);
          const R3Point& block_origin = block->Origin();
          for (int k = 0; k < block->NSurfels(); k++) {
            const R3Surfel *surfel = block->Surfel(k);
            double x = surfel->X() + block_origin.X();
            double y = surfel->Y() + block_origin.Y();
            double z = surfel->Z() + block_origin.Z();
            if (!surfel->HasNormal()) glColor3d(0.5, 0.5, 0.5);
            else glColor3f(surfel->NX(), surfel->NY(), surfel->NZ());
            glVertex3d(x, y, z);
          }
        }
        glEnd();
      }
    }
    else if (surfel_color_scheme == R3_SURFEL_VIEWER_COLOR_BY_SHADING) {
      // Draw with colors based on shading
      glEnable(GL_LIGHTING);
      glBegin(GL_POINTS);
      for (int i = 0; i < resident_nodes.NNodes(); i++) {
        R3SurfelNode *node = resident_nodes.Node(i);
        for (int j = 0; j < node->NBlocks(); j++) {
          R3SurfelBlock *block = node->Block(j);
          const R3Point& block_origin = block->Origin();
          for (int k = 0; k < block->NSurfels(); k++) {
            const R3Surfel *surfel = block->Surfel(k);
            double x = surfel->X() + block_origin.X();
            double y = surfel->Y() + block_origin.Y();
            double z = surfel->Z() + block_origin.Z();
            glNormal3f(surfel->NX(), surfel->NY(), surfel->NZ());
            glVertex3d(x, y, z);
          }
        }
      }
      glEnd();
      glDisable(GL_LIGHTING);
    }
    else if (surfel_color_scheme == R3_SURFEL_VIEWER_COLOR_BY_VIEWPOINT) {
      // Draw with RGB surfel colors, with alpha based on distance to viewpoint
      glBegin(GL_POINTS);
      for (int i = 0; i < resident_nodes.NNodes(); i++) {
        R3SurfelNode *node = resident_nodes.Node(i);
        R3SurfelScan *scan = node->Scan();
        if (!scan) continue;
        
        // Check distance to closest scan
        if (distance_to_closest_scan < FLT_MAX) {
          RNLength d = R3Distance(viewer.Camera().Origin(), scan->Viewpoint());
          if (d > distance_to_closest_scan + 0.25) continue;
        }

        // Compute alpha for node
        // RNScalar dot = viewer.Camera().Towards().Dot(scan->Towards());
        unsigned char alpha = 255;
       
        // Draw surfels
        for (int j = 0; j < node->NBlocks(); j++) {
          R3SurfelBlock *block = node->Block(j);
          const R3Point& block_origin = block->Origin();
          for (int k = 0; k < block->NSurfels(); k++) {
            const R3Surfel *surfel = block->Surfel(k);

            // Check surfel
            if (!backfacing_visibility) {
              R3Vector normal(surfel->NX(), surfel->NY(), surfel->NZ());
              if (normal.Dot(viewer.Camera().Towards()) >= 0) continue;
            }

            // Set color
            glColor4ub(surfel->R(), surfel->G(), surfel->B(), alpha);

            // Draw surfel
            double x = surfel->X() + block_origin.X();
            double y = surfel->Y() + block_origin.Y();
            double z = surfel->Z() + block_origin.Z();
            glVertex3d(x, y, z);
          }
        }
      }
      glEnd();
    }
    else {
      // Draw with RGB surfel colors
      glBegin(GL_POINTS);
      for (int i = 0; i < resident_nodes.NNodes(); i++) {
        R3SurfelNode *node = resident_nodes.Node(i);
        for (int j = 0; j < node->NBlocks(); j++) {
          R3SurfelBlock *block = node->Block(j);
          const R3Point& block_origin = block->Origin();
          for (int k = 0; k < block->NSurfels(); k++) {
            const R3Surfel *surfel = block->Surfel(k);

            // Check surfel
            if (!backfacing_visibility) {
              R3Vector normal(surfel->NX(), surfel->NY(), surfel->NZ());
              if (normal.Dot(viewer.Camera().Towards()) >= 0) continue;
            }

            // Set color

            // Draw surfel
            double x = surfel->X() + block_origin.X();
            double y = surfel->Y() + block_origin.Y();
            double z = surfel->Z() + block_origin.Z();
            glColor3ubv(surfel->Color());
            glVertex3d(x, y, z);
          }
        }
      }
      glEnd();
    }
  }

  // Reset the point size
  glDisable(GL_BLEND);
  glDepthMask(TRUE);
  glPointSize(1);

  // Draw normals
  if (normal_visibility) {
    RNLoadRgb(normal_color);
    glBegin(GL_LINES);
    RNLength r = 0.025;
    for (int i = 0; i < resident_nodes.NNodes(); i++) {
      R3SurfelNode *node = resident_nodes.Node(i);
      for (int j = 0; j < node->NBlocks(); j++) {
        R3SurfelBlock *block = node->Block(j);
        const R3Point& block_origin = block->Origin();
        for (int k = 0; k < block->NSurfels(); k++) {
          const R3Surfel *surfel = block->Surfel(k);
          double px = surfel->X() + block_origin.X();
          double py = surfel->Y() + block_origin.Y();
          double pz = surfel->Z() + block_origin.Z();
          double nx = surfel->NX();
          double ny = surfel->NY();
          double nz = surfel->NZ();
          R3LoadPoint(px, py, pz);
          R3LoadPoint(px + r * nx, py + r * ny, pz + r * nz);
        }
      }
    }
    glEnd();
  }

  // Draw object properties
  if (object_property_visibility) {
    RNLoadRgb(object_property_color);
    for (int i = 0; i < scene->NObjectProperties(); i++) {
      R3SurfelObjectProperty *property = scene->ObjectProperty(i);
      property->Draw(0);
    }
  }

  // Draw object labels
  if (object_label_visibility) {
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, viewer.Viewport().Width(), 0, viewer.Viewport().Height());
    glDisable(GL_DEPTH_TEST);
    glDepthMask(FALSE);
    for (int i = 0; i < scene->NObjects(); i++) {
      R3SurfelObject *object = scene->Object(i);
      for (int i = 0; i < object->NLabelAssignments(); i++) {
        R3SurfelLabelAssignment *assignment = object->LabelAssignment(i);
        if (assignment->Originator() == R3_SURFEL_LABEL_ASSIGNMENT_GROUND_TRUTH_ORIGINATOR) continue;
        RNBoolean confirmed = (assignment->Originator() == R3_SURFEL_LABEL_ASSIGNMENT_HUMAN_ORIGINATOR) ? 1 : 0;
        R3SurfelLabel *label = assignment->Label();
        R3Point position = object->Centroid();
        position[2] = object->BBox().ZMax() + 1;
        R2Point p = viewer.ViewportPoint(position);
        void *font = (confirmed) ? GLUT_BITMAP_HELVETICA_18 : GLUT_BITMAP_HELVETICA_12;
        int width = glutBitmapLength(font, (const unsigned char *) label->Name());
        p[0] -= width / 2;
        RNLoadRgb(label->Color());
        DrawText(p, label->Name(), font);
        break;
      }
    }
    glDepthMask(TRUE);
    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
  }

  // Draw object names
  if (object_name_visibility) {
    RNLoadRgb(object_name_color);
    for (int i = 0; i < scene->NObjects(); i++) {
      R3SurfelObject *object = scene->Object(i);
      if (object->NParts() > 0) continue;
      if (!object->Name()) continue;
      R3Point position = object->Centroid();
      position[2] = object->BBox().ZMax() + 1;
      DrawText(position, object->Name(), GLUT_BITMAP_HELVETICA_12);
    }
  }

  // Draw node bounding boxes
  if (node_bbox_visibility) {
    RNLoadRgb(node_bbox_color);
    for (int i = 0; i < resident_nodes.NNodes(); i++) {
      R3SurfelNode *node = resident_nodes.Node(i);
      node->BBox().Outline();
      if (node->NParts() > 0) glColor3d(0, 1, 0);
      else glColor3d(1, 0, 0);
    }
  }

  // Draw block bounding boxes
  if (block_bbox_visibility) {
    RNLoadRgb(block_bbox_color);
    for (int i = 0; i < resident_nodes.NNodes(); i++) {
      R3SurfelNode *node = resident_nodes.Node(i);
      for (int j = 0; j < node->NBlocks(); j++) {
        R3SurfelBlock *block = node->Block(j);
        block->BBox().Outline();
      }
    }
  }

  // Draw scan viewpoints
  if (scan_viewpoint_visibility) {
    glDisable(GL_LIGHTING);
    RNLoadRgb(scan_viewpoint_color);
    glBegin(GL_LINES);
    for (int i = 0; i < scene->NScans(); i++) {
      R3SurfelScan *scan = scene->Scan(i);
      const R3Point& viewpoint = scan->Viewpoint();
      const R3Vector towards = scan->Towards();
      const R3Vector up = scan->Up();
      R3LoadPoint(viewpoint);
      R3LoadPoint(viewpoint + towards);
      R3LoadPoint(viewpoint);
      R3LoadPoint(viewpoint + 0.5 * up);
    }
    glEnd();
  }

  // Draw center point
  if (center_point_visibility) {
    glEnable(GL_LIGHTING);
    GLfloat color[4];
    color[0] = center_point_color[0];
    color[1] = center_point_color[1];
    color[2] = center_point_color[2];
    color[3] = 1;
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color);
    R3Sphere(center_point, 1).Draw();
    glDisable(GL_LIGHTING);
  }

  // Reset viewing modes
  LoadViewingExtent(NULL);

  // Draw axes
  if (axes_visibility) {
    RNScalar d = 1.0;
    glDisable(GL_LIGHTING);
    glLineWidth(3);
    R3BeginLine();
    glColor3f(1, 0, 0);
    R3LoadPoint(R3zero_point);
    R3LoadPoint(R3zero_point + d * R3posx_vector);
    R3EndLine();
    R3BeginLine();
    glColor3f(0, 1, 0);
    R3LoadPoint(R3zero_point);
    R3LoadPoint(R3zero_point + d * R3posy_vector);
    R3EndLine();
    R3BeginLine();
    glColor3f(0, 0, 1);
    R3LoadPoint(R3zero_point);
    R3LoadPoint(R3zero_point + d * R3posz_vector);
    R3EndLine();
    glLineWidth(1);
  }

  // Capture image and exit
  if (image_name) {
    R2Image image(viewer.Viewport().Width(), viewer.Viewport().Height(), 3);
    image.Capture();
    image.Write(image_name);
    free(image_name);
    image_name = NULL;
  }

  // Update the frame time
  if (frame_time < 0) frame_time = 0.05;
  else frame_time = frame_timer.Elapsed();
  frame_timer.Read();

  // Adapt working set
  if (adapt_working_set_automatically) {
    // Adjust target resolution based on frame time
    if (frame_time > 0.05) {
      SetTargetResolution(0.9 * TargetResolution());
    }
    else if (frame_time < 0.025) {
      SetTargetResolution(1.1 * TargetResolution());
    }

    // Make gross estimate of visible radius
    RNLength camera_height = viewer.Camera().Origin().Z();
    RNLength visible_radius = camera_height;

    // Adjust focus radius 
    SetFocusRadius(visible_radius);

    // Adjust surfel size based on visible radius
    if ((TargetResolution() > 0) && (visible_radius > 0)) {
      RNLength window_width = viewer.Viewport().Width();
      RNLength npixels = window_width / (visible_radius * TargetResolution());
      SetSurfelSize(npixels);
    }
  }

  // Return whether need redraw
  return 0;
}    



int R3SurfelViewer::
Resize(int w, int h)
{
  // Resize window
  glViewport(0, 0, w, h);

  // Resize viewer viewport
  viewer.ResizeViewport(0, 0, w, h);

  // Remember window size
  window_width = w;
  window_height = h;

  // Return whether need redraw
  return 1;
}



// TEMPORARY HACK
static R3SurfelViewer *flying_viewer = NULL;
static void StaticIdle(void) { if (flying_viewer) flying_viewer->Idle(); } 


int R3SurfelViewer::
Idle(void)
{
  // Update viewer
  RNScalar step_factor = 0;
  if (mouse_button[1]) step_factor = 0.1;
  else if (mouse_button[2]) step_factor = -0.1;
  RNScalar pitch_factor = 1;
  int ix = flying_viewer->mouse_position[0];
  int iy = flying_viewer->mouse_position[1];
  flying_viewer->viewer.FlyCamera(pitch_factor, 0.0, 0.0, step_factor, TRUE, TRUE, ix, iy, 0, 0);

  // Post redisplay
  glutPostRedisplay();

  // Return success
  return TRUE;
}



int R3SurfelViewer::
MouseMotion(int x, int y)
{
  // Initialize
  int redraw = 0;

  // Compute mouse movement
  int dx = x - mouse_position[0];
  int dy = y - mouse_position[1];

  // Set viewing center point
  R3Point viewing_center_point = center_point;
  const R3Camera& camera = viewer.Camera();
  R3Plane camera_plane(camera.Origin(), camera.Towards());
  RNScalar signed_distance = R3SignedDistance(camera_plane, viewing_center_point);
  if (signed_distance < 0) viewing_center_point -= (signed_distance - 1) * camera.Towards();
  
  // World in hand navigation 
  if (!flying_viewer) {
    if (shift_down && mouse_button[0]) viewer.ScaleWorld(2.0, viewing_center_point, x, y, dx, dy);
    else if (ctrl_down && mouse_button[0]) viewer.TranslateWorld(2.0, viewing_center_point, x, y, dx, dy);
    else if (mouse_button[0]) RotateWorld(1.0, viewing_center_point, x, y, dx, dy);
    else if (mouse_button[1]) viewer.ScaleWorld(2.0, viewing_center_point, x, y, dx, dy);
    else if (mouse_button[2]) viewer.TranslateWorld(2.0, viewing_center_point, x, y, dx, dy);
    if (mouse_button[0] || mouse_button[1] || mouse_button[2]) redraw = 1;
  }

  // Remember mouse position 
  mouse_position[0] = x;
  mouse_position[1] = y;

  // Update mouse drag movement
  mouse_drag_distance_squared += dx*dx + dy*dy;

  // Return whether need redraw
  return redraw;
}



int R3SurfelViewer::
MouseButton(int x, int y, int button, int state, int shift, int ctrl, int alt)
{
  // Initialize
  int redraw = 0;

  // Process mouse button event
  if (state == 1) {
    // Button is going down
    mouse_drag_distance_squared = 0;

    // Remember mouse down position 
    mouse_down_position[0] = x;
    mouse_down_position[1] = y;

    // Start walking when middle or right button goes down with ctrl
    if (ctrl && ((button == 1) || (button == 2))) { 
      flying_viewer = this; 
      glutIdleFunc(StaticIdle); 
    }
  }
  else {
    // Check for drag
    RNBoolean drag = (mouse_drag_distance_squared > 10 * 10);

    // Check for double click
    static RNBoolean double_click = FALSE;
    static RNTime last_mouse_down_time;
    double_click = !drag && !double_click && (last_mouse_down_time.Elapsed() < 0.4);
    last_mouse_down_time.Read();

    // Set center point on left double-click 
    if ((button == 0) && double_click) {
      R3Point pick_position;
      R3SurfelNode *node = PickNode(x, y, &pick_position);
      if (node) SetCenterPoint(pick_position);
    }

    // Stop walking when button goes up
    flying_viewer = NULL;
    glutIdleFunc(NULL);
  }

  // Remember mouse position 
  mouse_position[0] = x;
  mouse_position[1] = y;

  // Remember button state 
  mouse_button[button] = state;

  // Remember modifiers 
  shift_down = shift;
  ctrl_down = ctrl;
  alt_down = alt;

  // Return whether need redraw
  return redraw;
}



int R3SurfelViewer::
Keyboard(int x, int y, int key, int shift, int ctrl, int alt)
{
  // Initialize redraw status
  int redraw = 1;

  // Process debugging commands
  if (alt) {
    // Process debugging commands
    switch (key) {
    case 'A':
    case 'a':
      SetAxesVisibility(-1);
      break;
      
    case 'B':
      SetBlockBBoxVisibility(-1);
      break;

    case 'b':
      SetNodeBBoxVisibility(-1);
      break;

    case 'C':
    case 'c':
      surfel_color_scheme = (surfel_color_scheme + 1) % R3_SURFEL_VIEWER_NUM_COLOR_SCHEMES;
      break;
      
    case 'F':
    case 'f':
      SetBackfacingVisibility(-1);
      break;
      
    case 'I':
    case 'i':
      SetObjectNameVisibility(-1);
      break;
      
    case 'L':
    case 'l':
      SetObjectLabelVisibility(-1);
      break;
      
    case 'N':
    case 'n':
      SetNormalVisibility(-1);
      break;
      
    case 'O':
    case 'o':
      SetCenterPointVisibility(-1);
      break;
      
    case 'P':
    case 'p':
      SetObjectPropertyVisibility(-1);
      break;

    case 'R':
    case 'r':
      SetCenterPoint(scene->Centroid());
      break;

    case 'S':
    case 's':
      SetSurfelVisibility(-1);
      break;

    case 'v':
      JumpToNextScanViewpoint(1);
      break;
      
    case 'V':
      JumpToNextScanViewpoint(-1);
      break;
      
    case 'W':
      UpdateWorkingSet(viewer);
      break;

    case 'w': {
      R3Point pick_position;
      R3SurfelNode *node = PickNode(x, y, &pick_position);
      if (node) SetCenterPoint(pick_position);
      break; }

    case 'X':
    case 'x':
      SetBackgroundVisibility(-1);
      break;
      
    case 'Y':
    case 'y':
      SetScanViewpointVisibility(-1);
      break;
      
    case 'Q': 
    case 'q': {
      R3Point pick_position(0,0,0);
      R3SurfelNode *node = PickNode(x, y, &pick_position);
      if (node) {
        SetCenterPoint(pick_position);
        printf("%g %g %g\n", pick_position[0], pick_position[1], pick_position[2]);
        while (node) {
          const char *node_name = (node->Name()) ? node->Name() : "-";
          R3SurfelObject *object = node->Object();
          const char *object_name = (object && (object->Name())) ? object->Name() : "-";
          char object_index[128];
          if (object) sprintf(object_index, "%d", object->SceneIndex());
          else sprintf(object_index, "%s", "-");
          R3SurfelLabel *ground_truth_label = (object) ? object->GroundTruthLabel() : NULL;
          const char *ground_truth_label_name = (ground_truth_label && (ground_truth_label->Name())) ? ground_truth_label->Name() : "-";
          R3SurfelLabel *current_label = (object) ? object->CurrentLabel() : NULL;
          const char *current_label_name = (current_label && (current_label->Name())) ? current_label->Name() : "-";
          printf("  %4d %4d %-30s  :  %-6s %-30s  :  %-30s %-30s\n",  
                 node->TreeLevel(), node->NParts(), node_name, 
                 object_index, object_name, 
                 current_label_name, ground_truth_label_name);
          node = node->Parent();        
        }
      }
      break; }

    default:
      redraw = 0;
      break;
    }
  }
  else if (ctrl) {
    switch(key) {
    default:
      redraw = 0;
      break;
    }
  }
  else {
    // Process other keyboard events
    switch (key) {
    case R3_SURFEL_VIEWER_F1_KEY:
    case R3_SURFEL_VIEWER_F2_KEY:
    case R3_SURFEL_VIEWER_F3_KEY:
    case R3_SURFEL_VIEWER_F4_KEY: 
      ZoomCamera(key - R3_SURFEL_VIEWER_F1_KEY + 1);
      break; 

    case R3_SURFEL_VIEWER_UP_KEY:
      SetTargetResolution(1.5 * TargetResolution());
      break;

    case R3_SURFEL_VIEWER_DOWN_KEY:
      SetTargetResolution(0.67 * TargetResolution());
      break;

    case R3_SURFEL_VIEWER_RIGHT_KEY:
      SetFocusRadius(1.1 * FocusRadius());
      break;

    case R3_SURFEL_VIEWER_LEFT_KEY:
      SetFocusRadius(0.9 * FocusRadius());
      break;

    case R3_SURFEL_VIEWER_PAGE_UP_KEY: 
      if (viewing_extent.IsEmpty()) viewing_extent = scene->BBox();
      if (shift) viewing_extent[RN_LO][RN_Z] += 0.1;
      else viewing_extent[RN_HI][RN_Z] += 0.1;
      if (R3Contains(viewing_extent, scene->BBox())) viewing_extent = R3null_box;
      break;

    case R3_SURFEL_VIEWER_PAGE_DOWN_KEY: 
      if (viewing_extent.IsEmpty()) viewing_extent = scene->BBox();
      if (shift) viewing_extent[RN_LO][RN_Z] -= 0.1;
      else viewing_extent[RN_HI][RN_Z] -= 0.1;
      if (R3Contains(viewing_extent, scene->BBox())) viewing_extent = R3null_box;
      break;

    case '-': 
      SetSurfelSize(0.9 * SurfelSize());
      break;

    case '+': 
      SetSurfelSize(1.1 * SurfelSize());
      break;

    default:
      redraw = 0;
      break;
    }
  }

  // Remember mouse position 
  mouse_position[0] = x;
  mouse_position[1] = y;

  // Remember modifiers 
  shift_down = shift;
  ctrl_down = ctrl;
  alt_down = alt;

  // Return whether need redraw
  return redraw;
}



////////////////////////////////////////////////////////////////////////
// COMMANDS
////////////////////////////////////////////////////////////////////////

void R3SurfelViewer::
Initialize(void)
{
  // Initialize lights
  static GLfloat lmodel_ambient[] = { 0.2, 0.2, 0.2, 1.0 };
  glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
  glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
  static GLfloat light0_diffuse[] = { 1.0, 1.0, 1.0, 1.0 };
  static GLfloat light0_position[] = { 0.0, 0.0, -1.0, 0.0 };
  glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
  glLightfv(GL_LIGHT0, GL_POSITION, light0_position);
  glEnable(GL_LIGHT0);
  static GLfloat light1_diffuse[] = { 0.5, 0.5, 0.5, 1.0 };
  static GLfloat light1_position[] = { 0.0, -1.0, 0.0, 0.0 };
  glLightfv(GL_LIGHT1, GL_DIFFUSE, light1_diffuse);
  glLightfv(GL_LIGHT1, GL_POSITION, light1_position);
  glEnable(GL_LIGHT1);
  glEnable(GL_NORMALIZE);

  // Initialize graphics modes
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
}



void R3SurfelViewer::
Terminate(void)
{
}



void R3SurfelViewer::
ZoomCamera(RNScalar scale)
{
  // Zoom into center point so that an area of radius scale is visible
  R3Point eye = center_point - 2 * scale * viewer.Camera().Towards();
  viewer.RepositionCamera(eye);
}



void R3SurfelViewer::
JumpToNextScanViewpoint(int delta)
{
  // Check number of scans
  if (scene->NScans() == 0) return;

  // Update current scan index
  if (current_scan_index < 0) current_scan_index = 0;
  else current_scan_index += delta;
  if (current_scan_index < 0) current_scan_index = 0;
  if (current_scan_index >= scene->NScans()) current_scan_index = scene->NScans() - 1;

  // Set camera
  R3SurfelScan *scan = scene->Scan(current_scan_index);
  viewer.RepositionCamera(scan->Viewpoint());
  viewer.ReorientCamera(scan->Towards(), scan->Up());

  // Update working set
  R3SurfelNode *node = scan->Node();
  if (!node) {
    SetCenterPoint(scan->Viewpoint() + 2.0 * scan->Towards());
  }
  else {
    EmptyWorkingSet();
    center_point = node->Centroid();
    InsertIntoWorkingSet(node, TRUE);
  }
}


int R3SurfelViewer::
WriteImage(const char *filename)
{
  // Check if can write file
  FILE *fp = fopen(filename, "w");
  if (!fp) return 0;
  else fclose(fp);

  // Remember image name -- capture image next redraw
  if (image_name) free(image_name);
  if (!filename) image_name = NULL;
  else image_name = strdup(filename);

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Scene manipulation utility functions
////////////////////////////////////////////////////////////////////////

void R3SurfelViewer::
SetScene(R3SurfelScene *scene)
{
  // Remember scene
  this->scene = scene;

  // Set center point
  center_point = scene->Centroid();
  // center_point[1] = scene->BBox().YMin();

  // Set focus radius
  focus_radius = 400;
  if (focus_radius > 0.5 * scene->BBox().DiagonalRadius()) {
    focus_radius = 0.5 * scene->BBox().DiagonalRadius();
  }

  // Set camera and viewport
  R3Box bbox = scene->BBox();
  RNLength r = bbox.DiagonalRadius();
  static const R3Vector up(0, 1, 0);
  static const R3Vector towards(0, 0, -1);
  R3Point eye = scene->Centroid() - towards * (2 * r); 
  R3Camera camera(eye, towards, up, 0.4, 0.4, 0.01, 100000.0);
  R2Viewport viewport(0, 0, window_width, window_height);
  viewer.SetViewport(viewport);
  viewer.SetCamera(camera);

  // Lock coarsest blocks in memory (~500MB)
  // ReadCoarsestBlocks(32 * 1024 * 1024);

  // Update working set
  UpdateWorkingSet();
}



////////////////////////////////////////////////////////////////////////
// Working set utility functions
////////////////////////////////////////////////////////////////////////

void R3SurfelViewer::
ReadCoarsestBlocks(RNScalar max_complexity)
{
  // Just checking
  if (!scene) return;

  // Get convenient variables
  R3SurfelTree *tree = scene->Tree();
  if (!tree) return;

  // Seed breadth first search with root nodes
  RNQueue<R3SurfelNode *> queue;
  for (int i = 0; i < tree->NNodes(); i++) {
    R3SurfelNode *node = tree->Node(i);
    queue.Insert(node);
  }

  // Visit nodes in breadth first search reading blocks
  RNScalar total_complexity = 0;
  while (!queue.IsEmpty()) {
    R3SurfelNode *node = queue.Pop();
    if (total_complexity + node->Complexity() > max_complexity) break;
    total_complexity += node->Complexity();
    node->ReadBlocks();
    for (int i = 0; i < node->NParts(); i++) {
      R3SurfelNode *part = node->Part(i);
      queue.Push(part);
    }
  }
}



void R3SurfelViewer::
ReleaseCoarsestBlocks(RNScalar max_complexity)
{
  // Just checking
  if (!scene) return;

  // Get convenient variables
  R3SurfelTree *tree = scene->Tree();
  if (!tree) return;

  // Seed breadth first search with root nodes
  RNQueue<R3SurfelNode *> queue;
  for (int i = 0; i < tree->NNodes(); i++) {
    R3SurfelNode *node = tree->Node(i);
    queue.Insert(node);
  }

  // Visit nodes in breadth first search reading blocks
  RNScalar total_complexity = 0;
  while (!queue.IsEmpty()) {
    R3SurfelNode *node = queue.Pop();
    if (total_complexity + node->Complexity() > max_complexity) break;
    total_complexity += node->Complexity();
    node->ReleaseBlocks();
    for (int i = 0; i < node->NParts(); i++) {
      R3SurfelNode *part = node->Part(i);
      queue.Push(part);
    }
  }
}



void R3SurfelViewer::
EmptyWorkingSet(void)
{
  // Just checking
  if (!scene) return;

  // Release blocks from resident nodes
  resident_nodes.ReleaseBlocks();

  // Empty resident nodes
  resident_nodes.Empty();
}



void R3SurfelViewer::
UpdateWorkingSet(const R3Point& center, RNScalar resolution, RNScalar radius)
{
  // Just checking
  if (!scene) return;

  // Get convenient variables
  R3SurfelTree *tree = scene->Tree();
  if (!tree) return;

  // Find new set of nodes
  R3SurfelNodeSet new_resident_nodes;
  new_resident_nodes.InsertNodes(tree, center, radius, -FLT_MAX, FLT_MAX, resolution, RN_EPSILON);

  // Read new working set
  new_resident_nodes.ReadBlocks();

  // Release old working set
  resident_nodes.ReleaseBlocks();

  // Now use newnodes 
  resident_nodes = new_resident_nodes;
}



void R3SurfelViewer::
UpdateWorkingSet(const R3Viewer& view)
{
  // Just checking
  if (!scene) return;
  R3SurfelTree *tree = scene->Tree();
  if (!tree) return;
 
  // Get convenient variables
  R3Point eye = view.Camera().Origin();
  R3Vector towards = view.Camera().Towards();
  R2Box viewport_box = view.Viewport().BBox();
  if (viewport_box.Area() == 0) return;

  // Allocate temporary memory
  int xres = viewport_box.XLength();
  int yres = int (xres * (double) viewport_box.YLength() / (double) viewport_box.XLength());
  int *visible_marks = new int [ tree->NNodes() ];
  int *resident_marks = new int [ tree->NNodes() ];
  int *item_buffer = new int [ xres * yres ];
  RNScalar *depth_buffer = new RNScalar [ xres * yres ];
  RNScalar *viewpoint_distance_buffer = new RNScalar [ xres * yres ];
  for (int i = 0; i < tree->NNodes(); i++) { resident_marks[i] = visible_marks[i] = 0; }
  for (int i = 0; i < resident_nodes.NNodes(); i++) resident_marks[resident_nodes.Node(i)->TreeIndex()] = 1;

  // Create viewer
  R3Viewer tmp_viewer(view);
  tmp_viewer.ResizeViewport(0, 0, xres, yres);

  RNBoolean done = FALSE;
  const int max_iterations = 8;
  for (int iteration = 0; iteration < max_iterations; iteration++) {
    // Check if done
    if (done) break;
    done = TRUE;

    // Initialize buffers
    for (int i = 0; i < xres * yres; i++) {
      item_buffer[i] = -1;
      depth_buffer[i] = FLT_MAX;
      viewpoint_distance_buffer[i] = FLT_MAX;
    }
    for (int i = 0; i < tree->NNodes(); i++) {
      visible_marks[i] = 0;
    }

    // Render surfels into item buffer
    for (int i = 0; i < resident_nodes.NNodes(); i++) {
      R3SurfelNode *node = resident_nodes.Node(i);
      for (int j = 0; j < node->NBlocks(); j++) {
        R3SurfelBlock *block = node->Block(j);
        const R3Point& block_origin = block->Origin();
        for (int k = 0; k < block->NSurfels(); k++) {
          const R3Surfel *surfel = block->Surfel(k);
          double wx = surfel->X() + block_origin.X();
          double wy = surfel->Y() + block_origin.Y();
          double wz = surfel->Z() + block_origin.Z();
          R3Point wp(wx, wy, wz);
          RNScalar depth = towards.Dot(wp - eye);
          if (depth <= 0) continue; 
          R2Point vp = tmp_viewer.ViewportPoint(wp);
          if (vp == R2infinite_point) continue; 
          int vx = (int) (vp.X() + 0.5);
          if ((vx < 0) || (vx >= xres)) continue; 
          int vy = (int) (vp.Y() + 0.5);
          if ((vy < 0) || (vy >= yres)) continue; 
          int index = xres*vy + vx;
          if (RNIsGreater(depth, depth_buffer[index])) continue;
          RNLength viewpoint_distance = (node->Scan()) ? R3Distance(tmp_viewer.Camera().Origin(), node->Scan()->Viewpoint()) : RN_INFINITY;
          if (RNIsEqual(depth, depth_buffer[index], 1E-3) && RNIsGreater(viewpoint_distance, viewpoint_distance_buffer[index])) continue;
          item_buffer[index] = node->TreeIndex();
          depth_buffer[index] = depth;
          viewpoint_distance_buffer[index] = viewpoint_distance;
        }
      }
    }

    // Mark visible nodes
    for (int i = 0; i < xres * yres; i++) {
      int node_index = item_buffer[i];
      if (node_index < 0) continue;
      if (visible_marks[node_index] == 0) {
        visible_marks[node_index] = 1;
      }
    }

    // Remove invisible nodes from working set
    for (int i = 0; i < resident_nodes.NNodes(); i++) {
      R3SurfelNode *node = resident_nodes.Node(i);
      assert(resident_marks[node->TreeIndex()]);
      if (visible_marks[node->TreeIndex()] == 1) continue;
      resident_marks[node->TreeIndex()] = 0;
      RemoveFromWorkingSet(node);
    }
    
    // Add parts of visible nodes to working set
    for (int i = 0; i < xres * yres; i++) {
      int node_index = item_buffer[i];
      if (node_index < 0) continue;
      assert(depth_buffer[i] < FLT_MAX);
      assert(visible_marks[node_index] == 1);
      if (resident_marks[node_index] == 0) continue;
      R3SurfelNode *node = tree->Node(node_index);
      if (node->NParts() == 0) continue;
      resident_marks[node_index] = 0;
      RemoveFromWorkingSet(node);
      for (int j = 0; j < node->NParts(); j++) {
        R3SurfelNode *part = node->Part(j);
        resident_marks[part->TreeIndex()] = 1;
        InsertIntoWorkingSet(part);
        done = FALSE;
      }
    }

    if (done) break;
  }

  // Delete temporary memory
  delete [] visible_marks;
  delete [] resident_marks;
  delete [] item_buffer;
  delete [] depth_buffer;
  delete [] viewpoint_distance_buffer;
}



void R3SurfelViewer::
InsertIntoWorkingSet(R3SurfelNode *node, RNBoolean full_resolution)
{
  // Just checking
  if (!scene) return;

  // Recurse to children
  if (full_resolution) {
    if (node->NParts() == 0) {
      InsertIntoWorkingSet(node, FALSE);
    }
    else {
      for (int i = 0; i < node->NParts(); i++) {
        R3SurfelNode *part = node->Part(i);
        InsertIntoWorkingSet(part, full_resolution);
      }
    }
  }
  else {
    // Read blocks 
    node->ReadBlocks();

    // Insert into resident nodes
    resident_nodes.InsertNode(node);
  }
}



void R3SurfelViewer::
RemoveFromWorkingSet(R3SurfelNode *node, RNBoolean full_resolution)
{
  // Just checking
  if (!scene) return;

  // Recurse to children
  if (full_resolution) {
    if (node->NParts() == 0) {
      RemoveFromWorkingSet(node, FALSE);
    }
    else {
      for (int i = 0; i < node->NParts(); i++) {
        R3SurfelNode *part = node->Part(i);
        RemoveFromWorkingSet(part, full_resolution);
      }
    }
  }
  else {
    // Release blocks 
    node->ReleaseBlocks();

    // Remove from resident nodes
    resident_nodes.RemoveNode(node);
  }
}



void R3SurfelViewer::
RotateWorld(RNScalar factor, const R3Point& origin, int, int, int dx, int dy)
{
  // Rotate world based on mouse (dx)
  if ((dx == 0) && (dy == 0)) return;
  RNLength vx = (RNLength) dx / (RNLength) viewer.Viewport().Width();
  RNLength vy = (RNLength) dy / (RNLength) viewer.Viewport().Height();
  RNAngle theta = -1 * factor * 4.0 * vx;
  viewer.RotateWorld(origin, viewer.Camera().Up(), theta);
  RNAngle phi = factor * 4.0 * vy;
  // RNAngle max_phi = R3InteriorAngle(viewer.Camera().Towards(), R3posy_vector) - RN_PI_OVER_TWO;
  // RNAngle min_phi = -1.0 * R3InteriorAngle(viewer.Camera().Towards(), R3negz_vector);
  // if (phi < min_phi) phi = min_phi;
  // if (phi > max_phi) phi = max_phi;
  viewer.RotateWorld(origin, viewer.Camera().Right(), phi);
}



#if 1

R3SurfelNode *R3SurfelViewer::
PickNode(int x, int y, R3Point *picked_position, 
  R3SurfelBlock **picked_block, const R3Surfel **picked_surfel,
  RNBoolean exclude_nonobjects, RNBoolean exclude_aerial) 
{
  // XXX TEMPORARY XXX
  exclude_aerial = TRUE;

  // How close the cursor has to be to a point (in pixels)
  int pick_tolerance = 10;

  // Clear window 
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Set viewing transformation
  viewer.Load();

  // Set OpenGL stuff
  glPointSize(pick_tolerance);    

  // Draw everything
  for (int i = 0; i < resident_nodes.NNodes(); i++) {
    R3SurfelNode *node = resident_nodes.Node(i);

    // Set color
    unsigned char rgba[4];
    int node_index = i + 1;
    rgba[0] = (node_index >> 16) & 0xFF;
    rgba[1] = (node_index >> 8) & 0xFF;
    rgba[2] = node_index & 0xFF;
    rgba[3] = 0xFE;
    glColor4ubv(rgba);

    // Draw blocks
    for (int j = 0; j < node->NBlocks(); j++) {
      R3SurfelBlock *block = node->Block(j);
      glPushMatrix();
      const R3Point& origin = block->Origin();
      glTranslated(origin[0], origin[1], origin[2]);
      glBegin(GL_POINTS);
      for (int k = 0; k < block->NSurfels(); k++) {
        const R3Surfel *surfel = block->Surfel(k);
        if (exclude_aerial && surfel->IsAerial()) continue;
        glVertex3fv(surfel->Coords());
      }
      glEnd();
      glPopMatrix();
    }
  }

  // Reset OpenGL stuff
  glPointSize(1);
  glFinish();

  // Read color buffer at cursor position
  unsigned char rgba[4];
  glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
  if (rgba[3] == 0) return NULL;

  // Determine node index
  int r = rgba[0] & 0xFF;
  int g = rgba[1] & 0xFF;
  int b = rgba[2] & 0xFF;
  int a = rgba[3] & 0xFF;
  if (a != 0xFE) return NULL;
  int node_index = (r << 16) | (g << 8) | b;
  node_index--;

  // Determine node
  if (node_index < 0) return NULL;
  if (node_index >= resident_nodes.NNodes()) return NULL;
  R3SurfelNode *hit_node = resident_nodes.Node(node_index);

  // Find node part of an object
  R3SurfelNode *picked_node = hit_node;
  if (exclude_nonobjects) {
    // Find node associated with object
    picked_node = NULL;

    // Check if hit node is part of an object
    if (hit_node->Object()) {
      picked_node = hit_node;
    }

    // Check if hit node has ancestor that is part of an object
    if (picked_node == NULL) {
      R3SurfelNode *ancestor = hit_node->Parent();
      while (ancestor) {
        if (ancestor->Object()) { picked_node = ancestor; break; }
        ancestor = ancestor->Parent();
      }
    }
    
    // Check if hit node has descendent that is part of an object
    if (picked_node == NULL) {
      R3Ray ray = viewer.WorldRay(x, y);
      RNScalar t, picked_t = FLT_MAX;
      RNArray<R3SurfelNode *> stack;
      stack.Insert(hit_node);
      while (!stack.IsEmpty()) {
        R3SurfelNode *node = stack.Tail();
        stack.RemoveTail();
        for (int i = 0; i < node->NParts(); i++) {
          stack.Insert(node->Part(i));
        }
        if (node->Object()) {
          if (R3Intersects(ray, node->BBox(), NULL, NULL, &t)) {
            if (t < picked_t) {
              picked_node = node;
              picked_t = t;
            }
          }
        }
      }
    }
  }
    
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
  if (picked_position) *picked_position = position;

  // Find hit surfel
  if (picked_block || picked_surfel) {
    // Create pointset in vicinity of picked position
    R3Point position(p[0], p[1], p[2]);
    R3SurfelSphereConstraint sphere_constraint(R3Sphere(position, 0.1));
    R3SurfelPointSet *pointset = CreatePointSet(scene, NULL, &sphere_constraint);
    if (pointset) {
      // Find surfel point closest to picked position
      R3SurfelPoint *closest_point = NULL;
      RNLength closest_distance = FLT_MAX;
      for (int i = 0; i < pointset->NPoints(); i++) {
        R3SurfelPoint *point = pointset->Point(i);
        RNLength distance = R3SquaredDistance(point->Position(), position);
        if (distance < closest_distance) {
          closest_distance = distance;
          closest_point = point;
        }
      }

      // Return closest point
      if (closest_point) {
        if (picked_position) *picked_position = closest_point->Position();
        if (picked_block) *picked_block = closest_point->Block();
        if (picked_surfel) *picked_surfel = closest_point->Surfel();
      }

      // Delete point set
      delete pointset;
    }
  }

  // Return picked node
  return picked_node;
}

#else

R3SurfelNode *R3SurfelViewer::
PickNode(int x, int y, R3Point *picked_position, 
  R3SurfelBlock **picked_block, const R3Surfel **picked_surfel,
  RNBoolean exclude_nonobjects, RNBoolean exclude_aerial) 
{
  // XXX TEMPORARY XXX
  exclude_aerial = TRUE;

  // Initialize result
  if (picked_position) *picked_position = R3zero_point;
  if (picked_block) picked_block = NULL;
  if (picked_surfel) picked_surfel = NULL;

  // Check cursor position
  R2Point cursor_position(x,y);
  if (!R2Contains(viewer.Viewport().BBox(), cursor_position)) {
    return NULL;
  }

  // Allocate select buffer
  const int SELECT_BUFFER_SIZE = 1024;
  GLuint select_buffer[SELECT_BUFFER_SIZE];
  GLint select_buffer_hits;

  // Initialize select buffer
  glSelectBuffer(SELECT_BUFFER_SIZE, select_buffer);
  glRenderMode(GL_SELECT);
  glInitNames();
  glPushName(0);

  // Draw surfels with pick names into selection buffer
  GLint viewport[4];
  glViewport(0, 0, Viewport().Width(), Viewport().Height());
  glGetIntegerv(GL_VIEWPORT, viewport);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPickMatrix((GLdouble) x, (GLdouble) y, 16, 16, viewport);
  viewer.Camera().Load(TRUE);
  glMatrixMode(GL_MODELVIEW);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  for (int i = 0; i < resident_nodes.NNodes(); i++) {
    R3SurfelNode *node = resident_nodes.Node(i);
    glLoadName(i + 1);
    for (int j = 0; j < node->NBlocks(); j++) {
      R3SurfelBlock *block = node->Block(j);
      glPushMatrix();
      const R3Point& origin = block->Origin();
      glTranslated(origin[0], origin[1], origin[2]);
      glBegin(GL_POINTS);
      for (int k = 0; k < block->NSurfels(); k++) {
        const R3Surfel *surfel = block->Surfel(k);
        if (exclude_aerial && surfel->IsAerial()) continue;
        glVertex3fv(surfel->Coords());
      }
      glEnd();
      glPopMatrix();
    }
  }
  glFlush();
  select_buffer_hits = glRenderMode(GL_RENDER);

  // Process select buffer to find front-most hit
  GLuint hit = 0;
  GLuint hit_z = 0xFFFFFFFF;
  GLuint *bufp = select_buffer;
  GLuint numnames, z1, z2;
  for (int i = 0; i < select_buffer_hits; i++) {
    numnames = *bufp++;
    z1 = *bufp++;
    z2 = *bufp++;
    while (numnames--) {
      if (z1 < hit_z) {
        hit = *bufp;
        hit_z = z1/2 + z2/2;
      }
      bufp++;
    }
  }

  // Check if hit anything
  if (hit <= 0) return NULL;

  // Find hit node
  hit--; // subtract the one added to avoid zero
  if (hit < 0) return NULL;
  if (hit >= (GLuint) resident_nodes.NNodes()) return NULL;
  R3SurfelNode *hit_node = resident_nodes.Node(hit);

  // Find node part of an object
  R3SurfelNode *picked_node = hit_node;
  if (exclude_nonobjects) {
    // Find node associated with object
    picked_node = NULL;

    // Check if hit node is part of an object
    if (hit_node->Object()) {
      picked_node = hit_node;
    }

    // Check if hit node has ancestor that is part of an object
    if (picked_node == NULL) {
      R3SurfelNode *ancestor = hit_node->Parent();
      while (ancestor) {
        if (ancestor->Object()) { picked_node = ancestor; break; }
        ancestor = ancestor->Parent();
      }
    }
    
    // Check if hit node has descendent that is part of an object
    if (picked_node == NULL) {
      R3Ray ray = viewer.WorldRay(x, y);
      RNScalar t, picked_t = FLT_MAX;
      RNArray<R3SurfelNode *> stack;
      stack.Insert(hit_node);
      while (!stack.IsEmpty()) {
        R3SurfelNode *node = stack.Tail();
        stack.RemoveTail();
        for (int i = 0; i < node->NParts(); i++) {
          stack.Insert(node->Part(i));
        }
        if (node->Object()) {
          if (R3Intersects(ray, node->BBox(), NULL, NULL, &t)) {
            if (t < picked_t) {
              picked_node = node;
              picked_t = t;
            }
          }
        }
      }
    }
  }
    
  // Find hit position
  GLdouble p[3];
  GLdouble modelview_matrix[16];
  GLdouble projection_matrix[16];
  glGetDoublev(GL_MODELVIEW_MATRIX, modelview_matrix);
  glGetDoublev(GL_PROJECTION_MATRIX, projection_matrix);
  GLdouble z = (GLdouble) hit_z / (GLdouble) 0xFFFFFFFF;
  gluUnProject(x, y, z, modelview_matrix, projection_matrix, viewport, &(p[0]), &(p[1]), &(p[2]));
  if (picked_position) picked_position->Reset(p[0], p[1], p[2]);

  // Find hit surfel
  if (picked_block || picked_surfel) {
    // Create pointset in vicinity of picked position
    R3Point position(p[0], p[1], p[2]);
    R3SurfelSphereConstraint sphere_constraint(R3Sphere(position, 0.1));
    R3SurfelPointSet *pointset = CreatePointSet(scene, NULL, &sphere_constraint);
    if (pointset) {
      // Find surfel point closest to picked position
      R3SurfelPoint *closest_point = NULL;
      RNLength closest_distance = FLT_MAX;
      for (int i = 0; i < pointset->NPoints(); i++) {
        R3SurfelPoint *point = pointset->Point(i);
        RNLength distance = R3SquaredDistance(point->Position(), position);
        if (distance < closest_distance) {
          closest_distance = distance;
          closest_point = point;
        }
      }

      // Return closest point
      if (closest_point) {
        if (picked_position) *picked_position = closest_point->Position();
        if (picked_block) *picked_block = closest_point->Block();
        if (picked_surfel) *picked_surfel = closest_point->Surfel();
      }

      // Delete point set
      delete pointset;
    }
  }

  // Return picked node
  return picked_node;
}

#endif



void R3SurfelViewer::
DrawObject(R3SurfelObject *object, RNFlags draw_flags) const
{
  // Draw nodes
  for (int i = 0; i < object->NNodes(); i++) {
    R3SurfelNode *node = object->Node(i);
    if (!node->DrawResidentDescendents(draw_flags)) {
      if (!node->DrawResidentAncestor(draw_flags)) {
        // const char *object_name = (object->Name()) ? object->Name() : "None";
        // fprintf(stderr, "Did not draw object %s\n", object_name);
      }
    }
  }

  // Draw parts
  for (int i = 0; i < object->NParts(); i++) {
    R3SurfelObject *part = object->Part(i);
    DrawObject(part, draw_flags);
  }
}
  
  
  
////////////////////////////////////////////////////////////////////////
// OBJECT EDITING 
////////////////////////////////////////////////////////////////////////

int R3SurfelViewer::
SplitLeafNodes(R3SurfelNode *source_node, const R3SurfelConstraint& constraint, 
  RNArray<R3SurfelNode *> *nodesA, RNArray<R3SurfelNode *> *nodesB)
{
  // Get convenient variables
  R3SurfelTree *tree = scene->Tree();
  if (!tree) return 0;
  if (!source_node) source_node = tree->RootNode();
  if (!source_node) return 0;

  // Split leaf nodes, WHILE UPDATING NODES IN VIEWER'S RESIDENT SET
  int countA = 0;
  int countB = 0;
  RNArray<R3SurfelNode *> stack;
  stack.Insert(source_node);
  while (!stack.IsEmpty()) {
    R3SurfelNode *node = stack.Tail();
    stack.RemoveTail();
    if (node->NParts() == 0) {
      // Check if node is resident in working set
      int resident_index = resident_nodes.NodeIndex(node);
    
      // Split leaf node
      RNArray<R3SurfelNode *> partsA, partsB;
      if (tree->SplitLeafNodes(node, constraint, &partsA, &partsB)) {
        // Update resident set
        if (resident_index > 0) {
          node->ReleaseBlocks(); // ???
          resident_nodes.RemoveNode(resident_index);
          for (int j = 0; j < partsA.NEntries(); j++) {
            R3SurfelNode *partA = partsA.Kth(j);
            resident_nodes.InsertNode(partA);
          }
          for (int j = 0; j < partsB.NEntries(); j++) {
            R3SurfelNode *partB = partsB.Kth(j);
            resident_nodes.InsertNode(partB);
          }
        }
      }

      // Insert parts into result
      if (nodesA) nodesA->Append(partsA);
      if (nodesB) nodesB->Append(partsB);
      countA += partsA.NEntries();
      countB += partsB.NEntries();
    }
    else {
      for (int i = 0; i < node->NParts(); i++) {
        R3SurfelNode *part = node->Part(i);
        stack.Insert(part);
      }
    } 
  }

  // Return status
  if (countA == 0) return 0;
  if (countB == 0) return 0;
  return 1;
}



