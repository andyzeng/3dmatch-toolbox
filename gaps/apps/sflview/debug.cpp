/* Source file for the surfel tree debugging visualization */



////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "R3Graphics/R3Graphics.h"
#include "R3Surfels/R3Surfels.h"
#include "R3SurfelViewer.h"
#include "map.h"
#include "align.h"
#include "debug.h"



////////////////////////////////////////////////////////////////////////
// Global variables
////////////////////////////////////////////////////////////////////////

// Drawing toggles

int debug0 = 0;
int debug1 = 0;
int debug2 = 0;
int debug3 = 0;
int debug4 = 0;
int debug5 = 0;
int debug6 = 0;
int debug7 = 0;
int debug8 = 0;
int debug9 = 0;



// Colors

static GLfloat colors[24][4] = {
  {1,0,0,1}, {0,1,0,1}, {0,0,1,1}, {1,0,1,1}, {0,1,1,1}, {1,1,0,1}, 
  {1,.3,.7,1}, {1,.7,.3,1}, {.7,1,.3}, {.3,1,.7,1}, {.7,.3,1,1}, {.3,.7,1,1}, 
  {1,.5,.5,1}, {.5,1,.5,1}, {.5,.5,1,1}, {1,.5,1,1}, {.5,1,1,1}, {1,1,.5,1}, 
  {.5,0,0,1}, {0,.5,0,1}, {0,0,.5,1}, {.5,0,.5,1}, {0,.5,.5,1}, {.5,.5,0,1} 
};



////////////////////////////////////////////////////////////////////////
// OBJECT STUFF
////////////////////////////////////////////////////////////////////////

int
CreateObject(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();
  int status = 0;

#if 0
  // Create object with above ground grid
  static R3SurfelPointSet *pointset = NULL;
  static R3Grid *grid = NULL;
  static R3SurfelObject *object = NULL;
  static R3Point last_center_point(0,0,0);
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    if (pointset) delete pointset; pointset = NULL;
    if (grid) delete grid; grid = NULL;
    if (object) { object->ReleaseBlocks(); object = NULL; }
    R3SurfelCylinderConstraint cylinder_constraint(center_point, 4);
    R3SurfelPointSet *pointset1 = CreatePointSet(scene, NULL, &cylinder_constraint);
    R3Plane plane = FitSupportPlane(pointset1);
    R3SurfelPlaneConstraint plane_constraint(plane, FALSE, FALSE, TRUE, 0.25);
    R3SurfelPointSet *pointset2 = CreatePointSet(pointset1, &plane_constraint);
    pointset = CreateConnectedPointSet(pointset2, center_point);
    delete pointset2;
    delete pointset1;
    if (!pointset) return 0;
    if (pointset->NPoints() == 0) { delete pointset; return 0; }
    if (pointset->BBox().Volume() == 0) { delete pointset; return 0; }
    grid = CreateGrid(pointset);
    if (!grid) return 0;
    if (grid->Cardinality() == 0) { delete grid; return 0; }
    object = CreateObject(scene, pointset);
    if (object) object->ReadBlocks();
    status = 1;
  }
#elif 0
  // Create object with pointset
  static R3SurfelPointSet *pointset = NULL;
  static R3SurfelObject *object = NULL;
  static R3Point last_center_point(0,0,0);
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    if (pointset) delete pointset; pointset = NULL;
    if (object) { object->ReleaseBlocks(); object = NULL; }
    R3SurfelCylinderConstraint cylinder_constraint(center_point, 4);
    R3SurfelPointSet *pointset1 = CreatePointSet(scene, NULL, &cylinder_constraint);
    R3Plane plane = FitSupportPlane(pointset1);
    R3SurfelPlaneConstraint plane_constraint(plane, FALSE, FALSE, TRUE, 0.25);
    R3SurfelPointSet *pointset2 = CreatePointSet(pointset1, &plane_constraint);
    pointset = CreateConnectedPointSet(pointset2, center_point);
    delete pointset2;
    delete pointset1;
    if (!pointset) return 0;
    if (pointset->NPoints() == 0) { delete pointset; return 0; }
    if (pointset->BBox().Volume() == 0) { delete pointset; return 0; }
    object = CreateObject(scene, pointset);
    if (object) object->ReadBlocks();
    viewer->UpdateWorkingSet();
    status = 1;
  }
#else
  // Create object with copy of pointset
  static R3SurfelPointSet *pointset = NULL;
  static R3SurfelObject *object = NULL;
  static R3Point last_center_point(0,0,0);
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    if (pointset) delete pointset; pointset = NULL;
    if (object) { object->ReleaseBlocks(); object = NULL; }
    R3SurfelCylinderConstraint cylinder_constraint(center_point, 4);
    R3SurfelPointSet *pointset1 = CreatePointSet(scene, NULL, &cylinder_constraint);
    R3Plane plane = FitSupportPlane(pointset1);
    R3SurfelPlaneConstraint plane_constraint(plane, FALSE, FALSE, TRUE, 0.25);
    R3SurfelPointSet *pointset2 = CreatePointSet(pointset1, &plane_constraint);
    pointset = CreateConnectedPointSet(pointset2, center_point);
    delete pointset2;
    delete pointset1;
    if (!pointset) return 0;
    if (pointset->NPoints() == 0) { delete pointset; return 0; }
    if (pointset->BBox().Volume() == 0) { delete pointset; return 0; }
    char object_name[1024];
    static int object_name_counter = 1;
    sprintf(object_name, "O%d", object_name_counter++);
    object = CreateObject(scene, pointset, NULL, object_name, NULL, object_name, TRUE);
    if (object) object->ReadBlocks();
    viewer->UpdateWorkingSet();
    status = 1;
  }
#endif

  // Draw object
  if (object) {
    glDisable(GL_LIGHTING);
    RNLoadRgb(0.0, 1.0, 0.0);
    glPointSize(5);
    object->Draw(0);
    object->BBox().Outline();
    glPointSize(1);
  }

#if 0
  // Draw pointset
  if (pointset) {
    glDisable(GL_LIGHTING);
    RNLoadRgb(0.0, 1.0, 1.0);
    glPointSize(5);
    pointset->Draw(0);
    glPointSize(1);
  }

  // Draw grid
  if (grid) {
    glDisable(GL_LIGHTING);
    RNLoadRgb(1.0, 0.0, 0.0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    const R3Affine& grid_to_world = grid->GridToWorldTransformation();
    grid_to_world.Push();
    grid->DrawIsoSurface(0.5);
    grid->GridBox().Outline();
    grid_to_world.Pop();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  }
#endif

  // Return whether object was created
  return status;
}



int
RemoveObject(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();
  int status = 0;

  // Create object with above ground grid
  static R3Point last_center_point(0,0,0);
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    
    // Find closest object
    R3SurfelObject *object = NULL;
    RNLength distance = FLT_MAX;
    for (int i = 0; i < scene->NObjects(); i++) {
      R3SurfelObject *o = scene->Object(i);
      if (o->NNodes() == 0) continue;
      RNLength d = R3Distance(o->BBox(), center_point);
      if (d < distance) {
        object = o;
        distance = d;
      }
    }
    
    if (object) {
      scene->RemoveObject(object);
      status = 1;
    }
  }

  // Return whether removed object
  return status;
}



int
MergeObjects(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();
  int status = 0;

  // Create object with above ground grid
  static R3Point last_center_point(0,0,0);
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    
    // Find two closest objects
    R3SurfelObject *object1 = NULL;
    R3SurfelObject *object2 = NULL;
    RNLength d1 = FLT_MAX;
    RNLength d2 = FLT_MAX;
    for (int i = 0; i < scene->NObjects(); i++) {
      R3SurfelObject *o = scene->Object(i);
      if (o->NNodes() == 0) continue;
      RNLength d = R3Distance(o->BBox(), center_point);
      if (d < d1) {
        object2 = object1;
        d2 = d1;
        object1 = o;
        d1 = d;
      }
      else if (d < d2) {
        object2 = o;
        d2 = d;
      }
    }
    
    if (object1 && object2) {
      scene->MergeObject(object1, object2);
      status = 1;
    }
  }

  // Return whether merged objects
  return status;
}



int
SplitObject(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();
  int status = 0;

  // Split object 
  static R3Point last_center_point(0,0,0);
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    
    // Find closest object
    R3SurfelObject *object = NULL;
    RNLength distance = FLT_MAX;
    for (int i = 0; i < scene->NObjects(); i++) {
      R3SurfelObject *o = scene->Object(i);
      if (o->NNodes() == 0) continue;
      RNLength d = R3Distance(o->BBox(), center_point);
      if (d < distance) {
        object = o;
        distance = d;
      }
    }

    // Split object
    if (object) {
      R3SurfelCylinderConstraint constraint(center_point, 4);
      SplitObject(object, &constraint);
      status = 1;
    }
  }

  // Return whether split objects
  return status;
}



int
ReadObjects(R3SurfelViewer *viewer)
{
  // Check if first time
  static int first = 1;
  if (!first) return 0;
  first = 0;

  // Empty working set
  // viewer->EmptyWorkingSet();

  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();

  // Open file
  FILE *fp = fopen("objects.txt", "r");
  if (!fp) {
    fprintf(stderr, "Unable to open objects.txt\n");
    return 0;
  }

  // Read file
  double x, y, z;
  char label_name[1024];
  while (fscanf(fp, "%s%lf%lf%lf", label_name, &x, &y, &z) == (unsigned int) 4) {
    // Get label
    R3SurfelLabel *label = scene->FindLabelByName(label_name);
    if (!label) {
      fprintf(stderr, "Unable to find label %s\n", label_name);
      continue;
    }

    printf("HERE1 %s %g %g %g\n", label_name, x, y, z);

    // Create pointset
    R3Point center_point(x, y, z);
    R3SurfelCylinderConstraint cylinder_constraint(center_point, 4);
    R3SurfelPointSet *pointset1 = CreatePointSet(scene, NULL, &cylinder_constraint);
    if (!pointset1) { fprintf(stderr, "Unable to create pointset1\n"); continue; }
    if (pointset1->NPoints() == 0) { fprintf(stderr, "Empty pointset1\n"); delete pointset1; continue; }
    R3Plane plane = FitSupportPlane(pointset1);
    R3SurfelPlaneConstraint plane_constraint(plane, FALSE, FALSE, TRUE, 0.25);
    R3SurfelPointSet *pointset2 = CreatePointSet(pointset1, &plane_constraint);
    delete pointset1; 
    if (!pointset2) { fprintf(stderr, "Unable to create pointset2\n"); continue; }
    if (pointset2->NPoints() == 0) { fprintf(stderr, "Empty pointset2\n"); delete pointset2; continue; }
    R3SurfelPointGraph *graph = new R3SurfelPointGraph(*pointset2);
    delete pointset2;
    if (!graph) { fprintf(stderr, "Unable to create graph\n"); continue; }
    if (graph->NPoints() == 0) { fprintf(stderr, "Empty graph\n"); delete graph; continue; }
    R3SurfelPointSet *pointset = CreateConnectedPointSet(graph, center_point);
    delete graph;
    if (!pointset) { fprintf(stderr, "Unable to create pointset\n"); continue; }
    if (pointset->NPoints() == 0) { fprintf(stderr, "Empty pointset\n"); delete pointset; continue; }

    printf("HERE2 %s %g %g %g %d\n", label_name, x, y, z, pointset->NPoints());

    // Create object
    char object_name[1024];
    sprintf(object_name, "%s_%.3f_%.3f_%.3f", label_name, x, y, z);
    R3SurfelObject *object = CreateObject(scene, pointset, NULL, object_name, NULL, object_name, TRUE);
    if (!object) { fprintf(stderr, "Unable to create object %s %g %g %g\n", label_name, x, y, z); delete pointset; continue; }

    printf("HERE3 %s %g %g %g\n", label_name, x, y, z);

    // Assign label
    R3SurfelLabelAssignment *assignment = new R3SurfelLabelAssignment(object, label, R3_SURFEL_LABEL_ASSIGNMENT_GROUND_TRUTH_ORIGINATOR);
    if (!assignment) { fprintf(stderr, "Unable to create assignment %s %g %g %g\n", label_name, x, y, z); delete pointset; continue; }
    scene->InsertLabelAssignment(assignment);

    printf("HERE4 %s %g %g %g\n", label_name, x, y, z);

    // Delete pointset
    delete pointset;
  }

  // Close file
  fclose(fp);

  // Update working set
  viewer->UpdateWorkingSet();

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// MEAN SHIFT STUFF
////////////////////////////////////////////////////////////////////////

static R3Point
MeanShift(R3SurfelScene *scene,
  const R3Point& start_position, RNLength radius)
{
  // This only works if radius < 2

  // Create point set
  R3Point current_position = start_position;
  for (int iter = 0; iter < 10; iter++) {
    // Create point set
    R3SurfelCylinderConstraint neighborhood_constraint(current_position, 2);
    R3SurfelPointSet *pointset1 = CreatePointSet(scene, NULL, &neighborhood_constraint);
    R3Plane bottom_plane = FitSupportPlane(pointset1);
    R3Plane top_plane(bottom_plane[0], bottom_plane[1], bottom_plane[2], bottom_plane[3] - 2.5);
    R3SurfelCylinderConstraint radius_constraint(current_position, radius);
    R3SurfelPlaneConstraint bottom_plane_constraint(bottom_plane, FALSE, FALSE, TRUE, 0.25);
    R3SurfelPlaneConstraint top_plane_constraint(top_plane, TRUE, FALSE, FALSE, 0.25);
    R3SurfelMultiConstraint multiconstraint;
    multiconstraint.InsertConstraint(&radius_constraint);
    multiconstraint.InsertConstraint(&bottom_plane_constraint);
    multiconstraint.InsertConstraint(&top_plane_constraint);
    R3SurfelPointSet *pointset = CreatePointSet(pointset1, &multiconstraint);
    delete pointset1;
    
    // Sum positions
    RNScalar total_weight = 0;
    R3Point total_position = R3zero_point;
    for (int i = 0; i < pointset->NPoints(); i++) {
      const R3SurfelPoint *point = pointset->Point(i);
      R3Point position = point->Position();
      total_position += position;
      total_weight += 1;
    }

    // Delete point set
    delete pointset;

    // Find centroid
    if (total_weight == 0) break;
    R3Point centroid = total_position / total_weight;
    if (R3SquaredDistance(centroid, current_position) < 0.0001) break;

    // printf("%12.6f %12.6f %12.6f : %9.1f %9.6f \n", 
    //   centroid.X(), centroid.Y(), centroid.Z(),
    //   total_weight, R3Distance(current_position, centroid));

    // Set current position and iterate
    current_position = centroid;
  }

  // Return mean shift centroid
  return current_position;
}



void
DrawMeanShift(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();

  // Update mean shift point
  static R3Point mean_shift_point(0,0,0);
  static R3Point last_center_point(0,0,0);
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    mean_shift_point = MeanShift(scene, center_point, 0.5);
  }

  // Draw mean shift point
  glEnable(GL_LIGHTING);
  GLfloat color[4] = { 0, 1, 1, 1 }; 
  glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color);
  R3Sphere(mean_shift_point, 0.5).Draw();
  glDisable(GL_LIGHTING);
}


#if 0

void
MeanShift(R3Grid *grid)
{
  printf("Begin ...\n");
  printf("  Cardinality = %d\n", grid->Cardinality());
  printf("  Maximum = %g\n", grid->Maximum());
  printf("  Mean = %g\n", grid->Mean());
  printf("  L1Norm = %g\n", grid->L1Norm());
  printf("  L2Norm = %g\n", grid->L2Norm());

  // Create gaussian kernel
  static RNScalar kernel[5][5][5];
  const double sigma = 1;
  const double denom = -2 * sigma * sigma;
  for (int jz = -2; jz <=2; jz++) {
    for (int jy = -2; jy <= 2; jy++) {
      for (int jx = -2; jx <= 2; jx++) {
        double dd = jx*jx + jy*jy + jz*jz;
        kernel[jx+2][jy+2][jz+2] = exp(dd/denom);
      }
    }
  }

  // Iteratively shift towards highest density
  for (int iter = 0; iter < 128; iter++) {
    // Innocent until proven guilty
    RNScalar done = TRUE;

    // Initialize next grid
    R3Grid next_grid(grid->XResolution(), grid->YResolution(), grid->ZResolution());

    // Consider every grid cell
    for (int iz = 0; iz < grid->ZResolution(); iz++) {
      for (int iy = 0; iy < grid->YResolution(); iy++) {
        for (int ix = 0; ix < grid->XResolution(); ix++) {
          RNScalar value = grid->GridValue(ix, iy, iz);
          if (value == 0) continue;

          // Compute centroid of neighbors
          RNScalar total_weight = 0;
          R3Vector total_displacement(0,0,0);
          for (int jz = -2; jz <=2; jz++) {
            for (int jy = -2; jy <= 2; jy++) {
              for (int jx = -2; jx <= 2; jx++) {
                RNScalar neighbor_value = grid->GridValue(ix+jx,iy+jy,iz+jz);
                if (neighbor_value <= 0) continue;
                RNScalar weight = kernel[jx+2][jy+2][jz+2] * neighbor_value;
                total_displacement += weight * R3Vector(jx, jy, jz);
                total_weight += weight;
              }
            }
          }

          // Check total weight
          if (total_weight == 0) continue;

#if 1
          // Apply displacement
          R3Vector displacement = total_displacement / total_weight;
          if (!R3Contains(displacement, R3zero_vector)) done = FALSE;
          RNScalar x = ix + displacement[0];
          RNScalar y = iy + displacement[1];
          RNScalar z = iz + displacement[2];
          next_grid.RasterizeGridPoint(x, y, z, value);
#else
          // Apply discrete displacmeent
          R3Vector displacement = total_displacement / total_weight;
          int dx = (int) (displacement.X() + 0.5);
          int dy = (int) (displacement.Y() + 0.5);
          int dz = (int) (displacement.Z() + 0.5);
          if ((dx != 0) || (dy != 0) || (dz != 0)) done = FALSE;
          next_grid.AddGridValue(ix+dx, iy+dy, iz+dz, value);
#endif
        }
      }
    }

    // Check if done
    if (done) { printf("Done %d\n", iter); break; }

    // Copy next grid
    grid->Copy(next_grid);
  }

  printf("End ...\n");
  printf("  Cardinality = %d\n", grid->Cardinality());
  printf("  Maximum = %g\n", grid->Maximum());
  printf("  Mean = %g\n", grid->Mean());
  printf("  L1Norm = %g\n", grid->L1Norm());
  printf("  L2Norm = %g\n", grid->L2Norm());
}



static void
DrawMeanShift(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  R3SurfelTree *tree = scene->Tree();
  const R3Point& center_point = viewer->CenterPoint();

  // Create segment pointset
  static R3Grid *grid = NULL;
  static R3SurfelPointSet *segment = NULL;
  static R3Point last_center_point(0,0,0);
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    if (grid) delete grid;
    if (segment) delete segment;
    R3CylinderConstraint constraint(center_point);
    R3SurfelPointSet *pointset = CreatePointSet(scener, NULL, &cylinder_constraint);
    segment = CreateAboveGroundPointSet(pointset);
    grid = CreateGrid(segment, 0.1);
    grid->Dilate(2);
    grid->Erode(2);
    grid->Threshold(0.5, 0, 1);
    // MeanShift(grid);
    delete pointset;
  }

  // Draw grid
  if (grid) {
    glDisable(GL_LIGHTING);
    RNLoadRgb(0.0, 1.0, 0.0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    const R3Affine& grid_to_world = grid->GridToWorldTransformation();
    grid_to_world.Push();
    grid->DrawIsoSurface(0.5);
    grid->GridBox().Outline();
    grid_to_world.Pop();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  }

  // Draw segment
  if (segment && viewer->ObjectNameVisibility()) {
    glDisable(GL_LIGHTING);
    RNLoadRgb(1.0, 1.0, 0.0);
    glPointSize(5);
    segment->Draw(0);
    glPointSize(1);
  }
}

#endif



////////////////////////////////////////////////////////////////////////
// GRAPH STUFF
////////////////////////////////////////////////////////////////////////

void
DrawGraph(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();

  // Create graph
  static R3SurfelPointGraph *graph = NULL;
  static R3Point last_center_point(0,0,0);
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    if (graph) delete graph;
    R3SurfelCylinderConstraint cylinder_constraint(center_point, 4);
    R3SurfelPointSet *pointset1 = CreatePointSet(scene, NULL, &cylinder_constraint);
    R3Plane plane = FitSupportPlane(pointset1);
    R3SurfelPlaneConstraint plane_constraint(plane, FALSE, FALSE, TRUE, 0.25);
    R3SurfelPointSet *pointset2 = CreatePointSet(pointset1, &plane_constraint);
    // graph = new R3SurfelPointGraph(*pointset2);
    graph = new R3SurfelPointGraph(*pointset1);
    delete pointset2;
    delete pointset1;
  }

  // Draw graph
  if (graph) {
    glDisable(GL_LIGHTING);
    glColor3d(0, 1, 0);
    graph->Draw(0);
  }
}



void
DrawPrunedGraph(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();

  // Create graph
  static R3SurfelPointGraph *graph = NULL;
  static R3Point last_center_point(0,0,0);
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    if (graph) delete graph;
    R3SurfelCylinderConstraint cylinder_constraint(center_point, 4);
    R3SurfelPointSet *pointset1 = CreatePointSet(scene, NULL, &cylinder_constraint);
    R3Plane plane = FitSupportPlane(pointset1);
    R3SurfelPlaneConstraint plane_constraint(plane, FALSE, FALSE, TRUE, 0.25);
    R3SurfelPointSet *pointset2 = CreatePointSet(pointset1, &plane_constraint);
    graph = new R3SurfelPointGraph(*pointset2);
    graph->RemoveOutlierEdges(1);
    delete pointset2;
    delete pointset1;
  }

  // Draw graph
  if (graph) {
    glDisable(GL_LIGHTING);
    glColor3d(1, 1, 0);
    graph->Draw(0);
  }
}



////////////////////////////////////////////////////////////////////////
// PLANE STUFF
////////////////////////////////////////////////////////////////////////

void
DrawEstimateSupportPlane(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();

  // Update
  static R3Plane plane(0,0,0,0);
  static R3Point last_center_point(0,0,0);
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    plane = EstimateSupportPlane(scene, center_point);
  }

  // Draw the best plane
  if ((plane[0] != 0) || (plane[1] != 0) || (plane[2] != 0)) {
    glEnable(GL_LIGHTING);
    GLfloat color[4] = { 1, 1, 0, 1 }; 
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color);
    R3Point p = center_point;
    p.Project(plane);
    R3Sphere(p, 1).Draw();
    glDisable(GL_LIGHTING);
    glColor3d(1, 1, 0);
    glLineWidth(3);
    R3Span(p, p + 5 * plane.Normal()).Draw();
    R3Span(p, p - 5 * plane.Normal()).Draw();
    glLineWidth(1);
  }
}



void
DrawFitSupportPlane(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();

  // Update
  static R3Plane plane(0,0,0,0);
  static R3Point last_center_point(0,0,0);
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    plane = FitSupportPlane(scene, center_point);
  }

  // Draw the best plane
  if ((plane[0] != 0) || (plane[1] != 0) || (plane[2] != 0)) {
    glEnable(GL_LIGHTING);
    GLfloat color[4] = { 0, 1, 1, 1 }; 
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color);
    R3Point p = center_point;
    p.Project(plane);
    R3Sphere(p, 1).Draw();
    glDisable(GL_LIGHTING);
    glColor3d(0, 1, 1);
    glLineWidth(3);
    R3Span(p, p + 5 * plane.Normal()).Draw();
    R3Span(p, p - 5 * plane.Normal()).Draw();
    glLineWidth(1);
  }
}



////////////////////////////////////////////////////////////////////////
// SEGMENTATION STUFF
////////////////////////////////////////////////////////////////////////

void
DrawPointSet(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();

  // Create segment pointset
  static R3SurfelPointSet *segment = NULL;
  static R3Point last_center_point(0,0,0);
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    if (segment) delete segment;
    R3SurfelCylinderConstraint constraint(center_point, 4);
    segment = CreatePointSet(scene, NULL, &constraint);
  }

  // Draw segment
  if (segment) {
    glDisable(GL_LIGHTING);
    glColor3d(1, 0, 1);
    glDisable(GL_DEPTH_TEST);
    glPointSize(5);
    segment->Draw(0);
    glPointSize(1);
    glEnable(GL_DEPTH_TEST);
  }
}



void
DrawAbovePointSet(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();

  // Create segment pointset
  static R3SurfelPointSet *segment = NULL;
  static R3Point last_center_point(0,0,0);
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    if (segment) delete segment;
    R3SurfelCylinderConstraint cylinder_constraint(center_point, 4);
    R3SurfelPointSet *pointset = CreatePointSet(scene, NULL, &cylinder_constraint);
    R3Plane plane = FitSupportPlane(pointset);
    R3SurfelPlaneConstraint plane_constraint(plane, FALSE, FALSE, TRUE, 0.25);
    segment = CreatePointSet(pointset, &plane_constraint);
    delete pointset;
  }

  // Draw segment
  if (segment) {
    glDisable(GL_LIGHTING);
    glColor3d(0, 1, 0);
    glDisable(GL_DEPTH_TEST);
    glPointSize(5);
    segment->Draw(0);
    glPointSize(1);
    glEnable(GL_DEPTH_TEST);
  }
}



void
DrawConnectedPointSet(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();

  // Create segment pointset
  static R3SurfelPointSet *segment = NULL;
  static R3Point last_center_point(0,0,0);
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    if (segment) delete segment;
    R3SurfelCylinderConstraint cylinder_constraint(center_point, 4);
    R3SurfelPointSet *pointset1 = CreatePointSet(scene, NULL, &cylinder_constraint);
    segment = CreateConnectedPointSet(pointset1, center_point);
    delete pointset1;
  }

  // Draw segment
  if (segment) {
    glDisable(GL_LIGHTING);
    glColor3d(1, 1, 0);
    glDisable(GL_DEPTH_TEST);
    glPointSize(5);
    segment->Draw(0);
    glPointSize(1);
    glEnable(GL_DEPTH_TEST);
  }
}



void
DrawAboveConnectedPointSet(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();

  // Create segment pointset
  static R3SurfelPointSet *segment = NULL;
  static R3Point last_center_point(0,0,0);
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    if (segment) delete segment;
    R3SurfelCylinderConstraint cylinder_constraint(center_point, 4);
    R3SurfelPointSet *pointset1 = CreatePointSet(scene, NULL, &cylinder_constraint);
    R3Plane plane = FitSupportPlane(pointset1);
    R3SurfelPlaneConstraint plane_constraint(plane, FALSE, FALSE, TRUE, 0.25);
    R3SurfelPointSet *pointset2 = CreatePointSet(pointset1, &plane_constraint);
    segment = CreateConnectedPointSet(pointset2, center_point);
    delete pointset2;
    delete pointset1;
  }

  // Draw segment
  if (segment) {
    glDisable(GL_LIGHTING);
    glColor3d(1, 1, 0);
    glDisable(GL_DEPTH_TEST);
    glPointSize(5);
    segment->Draw(0);
    glPointSize(1);
    glEnable(GL_DEPTH_TEST);
  }
}




void
DrawAboveConnectedPointGraph(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();

  // Create segment pointset
  static R3SurfelPointSet *segment = NULL;
  static R3Point last_center_point(0,0,0);
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    if (segment) delete segment;
    R3SurfelCylinderConstraint cylinder_constraint(center_point, 4);
    R3SurfelPointSet *pointset1 = CreatePointSet(scene, NULL, &cylinder_constraint);
    R3Plane plane = FitSupportPlane(pointset1);
    R3SurfelPlaneConstraint plane_constraint(plane, FALSE, FALSE, TRUE, 0.25);
    R3SurfelPointSet *pointset2 = CreatePointSet(pointset1, &plane_constraint);
    R3SurfelPointGraph *graph = new R3SurfelPointGraph(*pointset2);
    segment = CreateConnectedPointSet(graph, center_point);
    delete graph;
    delete pointset2;
    delete pointset1;
  }

  // Draw segment
  if (segment) {
    glDisable(GL_LIGHTING);
    glColor3d(1, 0, 1);
    glDisable(GL_DEPTH_TEST);
    glPointSize(5);
    segment->Draw(0);
    glPointSize(1);
    glEnable(GL_DEPTH_TEST);
  }
}


R3Vector *
CreateDirections(R3SurfelPointGraph *graph)
{
  // Check graph
  if (graph->NPoints() == 0) return NULL;
  if (graph->MaxNeighbors() < 2) return NULL;

  // Allocate directions
  R3Vector *directions = new R3Vector [ graph->NPoints() ];
  if (!directions) {
    fprintf(stderr, "Unable to allocate directions\n");
    return NULL;
  }

  // Allocate temporary memory
  R3Point *positions = new R3Point [graph->MaxNeighbors() + 1];

  // Compute directions
  for (int i = 0; i < graph->NPoints(); i++) {
    const R3SurfelPoint *point = graph->Point(i);
#if 0
    if (graph->NNeighbors(i) < 2) { 
      // Punt
      directions[i] = R3zero_vector; 
    }
    else {
      // Compute direction with PCA of neighborhood
      positions[0] = point->Position();
      for (int j = 0; j < graph->NNeighbors(i); j++) {
        const R3SurfelPoint *neighbor = graph->Neighbor(i, j);
        positions[j+1] = neighbor->Position();
      }
      int npositions = graph->NNeighbors(i) + 1;
      RNScalar variances[3];
      R3Triad triad = R3PrincipleAxes(positions[0], npositions, positions, NULL, variances);
      directions[i] = sqrt(variances[2]) * triad[2];
      if ((i % 100) == 0) printf("%g : %g %g %g\n", variances[2], triad[2][0], triad[2][1], triad[2][2]);
    }
#else
    directions[i] = R3zero_vector;
    if (graph->NNeighbors(i) > 0) {
      R3Point sum(0,0,0);
      for (int j = 0; j < graph->NNeighbors(i); j++) {
        const R3SurfelPoint *neighbor = graph->Neighbor(i, j);
        sum += neighbor->Position().Vector();
      }
      R3Point avg = sum / graph->NNeighbors(i);
      directions[i] = avg - point->Position();
    }
#endif
  }

  // Delete positions
  delete [] positions;

  // Return directions
  return directions;
}



static void
DrawDirections(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();

  // Create directions
  static R3Vector *directions = NULL;
  static R3SurfelPointGraph *graph = NULL;
  static R3Point last_center_point(0,0,0);
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    R3SurfelCylinderConstraint cylinder_constraint(center_point, 4);
    R3SurfelPointSet *pointset = CreatePointSet(scene, NULL, &cylinder_constraint);
    graph = new R3SurfelPointGraph(*pointset);
    delete pointset;
    directions = CreateDirections(graph);
  }

  // Draw directions
  if (graph && directions) {
    glDisable(GL_LIGHTING);
    glColor3d(0, 0, 1);
    glBegin(GL_LINES);
    for (int i = 0; i < graph->NPoints(); i++) {
      const R3SurfelPoint *point = graph->Point(i);
      R3Point position = point->Position();
      R3LoadPoint(position);
      R3LoadPoint(position + directions[i]);
    }
    glEnd();
  }
}



static void
DrawGrid(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();

  // Create segment pointset
  static R3Grid *grid = NULL;
  static R3Point last_center_point(0,0,0);
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    R3SurfelCylinderConstraint cylinder_constraint(center_point, 4);
    R3SurfelPointSet *pointset = CreatePointSet(scene, NULL, &cylinder_constraint);
    grid = CreateGrid(pointset);
    delete pointset;
  }

  // Draw grid
  if (grid) {
    glDisable(GL_LIGHTING);
    RNLoadRgb(0.0, 1.0, 0.0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    const R3Affine& grid_to_world = grid->GridToWorldTransformation();
    grid_to_world.Push();
    grid->DrawIsoSurface(0.5);
    grid->GridBox().Outline();
    grid_to_world.Pop();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  }
}



static void
DrawOverheadImage(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();

  // Read overhead image
  static R2Image *overhead_image = NULL;
  static R2Texture *overhead_texture = NULL;
  if (!overhead_image) {
    overhead_image = new R2Image();
    if (!overhead_image->Read("overhead.png")) return;
    overhead_texture = new R2Texture(overhead_image, 
      R2_CLAMP_TEXTURE_WRAP, R2_CLAMP_TEXTURE_WRAP, 
      R2_LINEAR_MIPMAP_LINEAR_TEXTURE_FILTER, R2_LINEAR_TEXTURE_FILTER, 
      R2_MODULATE_TEXTURE_BLEND);
  }

  // Draw overhead image
  if (overhead_texture) {
    glDisable(GL_LIGHTING);
    RNLoadRgb(1.0, 1.0, 1.0);
    overhead_texture->Draw();
    R3BeginPolygon();
    const R3Box& bbox = scene->BBox();
    R3LoadTextureCoords(0.0, 0.0);  R3LoadPoint(bbox[0][0], bbox[0][1], center_point[2]);
    R3LoadTextureCoords(1.0, 0.0);  R3LoadPoint(bbox[1][0], bbox[0][1], center_point[2]);
    R3LoadTextureCoords(1.0, 1.0);  R3LoadPoint(bbox[1][0], bbox[1][1], center_point[2]);
    R3LoadTextureCoords(0.0, 1.0);  R3LoadPoint(bbox[0][0], bbox[1][1], center_point[2]);
    R3EndPolygon();
    R2null_texture.Draw();
  }
}



static void
DrawPlanarGrids(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();

  // Create grids
  static RNArray<R3PlanarGrid *> *grids = NULL;
  static R3Point last_center_point(0,0,0);
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;

    // Delete previous grids
    if (grids) {
      for (int i = 0; i < grids->NEntries(); i++) delete grids->Kth(i);
      delete grids;
    }

    // Create new grids in area around center point
    R3Box box(center_point - 5 * R3ones_vector, center_point + 5 * R3ones_vector);
    R3SurfelBoxConstraint constraint(box);
    grids = CreatePlanarGrids(scene, NULL, &constraint);
  }

  // Draw grids
  if (grids) {
    glDisable(GL_LIGHTING);
    for (int i = 0; i < grids->NEntries(); i++) {
      R3PlanarGrid *proxy = grids->Kth(i);
      glColor4fv(colors[i%24]); 
      proxy->Draw();
    }
  }
}



static void
CreatePlanarObjects(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();

  // Create objects
  static R3Point last_center_point(0,0,0);
  static RNArray<R3SurfelObject *> * objects = NULL;
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    if (objects) {
      for (int i = 0; i < objects->NEntries(); i++) {
        R3SurfelObject *object = objects->Kth(i);
        object->ReleaseBlocks();
      }
      delete objects;
      objects = NULL;
    }
    if (1) {
      R3Box box(center_point - 5 * R3ones_vector, center_point + 5 * R3ones_vector);
      R3SurfelBoxConstraint constraint(box);
      objects = CreatePlanarObjects(scene, NULL, &constraint);
    }
    if (objects) {
      printf("%d\n", objects->NEntries());
      for (int i = 0; i < objects->NEntries(); i++) {
        R3SurfelObject *object = objects->Kth(i);
        object->ReadBlocks();
      }
    }
  }

  // Draw objects
  if (objects) {
    glDisable(GL_LIGHTING);
    for (int i = 0; i < objects->NEntries(); i++) {
      R3SurfelObject *object = objects->Kth(i);
      glColor4fv(colors[i%24]); 
      object->Draw(0); 
    }      
  }
}



static void
CreateClusterObjects(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();

  // Create objects
  static R3Point last_center_point(0,0,0);
  static RNArray<R3SurfelObject *> * objects = NULL;
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    if (objects) {
      for (int i = 0; i < objects->NEntries(); i++) {
        R3SurfelObject *object = objects->Kth(i);
        for (int j = 0; j < object->NNodes(); j++) {
          R3SurfelNode *node = object->Node(j);
          node->ReadBlocks();
        }
      }
      delete objects;
      objects = NULL;
    }
    if (1) {
      R3Box box(center_point - 5 * R3ones_vector, center_point + 5 * R3ones_vector);
      R3SurfelBoxConstraint constraint(box);
      objects = CreateClusterObjects(scene, NULL, &constraint);
    }
    if (objects) {
      printf("%d\n", objects->NEntries());
      for (int i = 0; i < objects->NEntries(); i++) {
        R3SurfelObject *object = objects->Kth(i);
        for (int j = 0; j < object->NNodes(); j++) {
          R3SurfelNode *node = object->Node(j);
          node->ReadBlocks();
        }
      }
    }
  }

  // Draw objects
  if (objects) {
    glDisable(GL_LIGHTING);
    for (int i = 0; i < objects->NEntries(); i++) {
      R3SurfelObject *object = objects->Kth(i);
      glColor4fv(colors[i%24]); 
      for (int j = 0; j < object->NNodes(); j++) {
        R3SurfelNode *node = object->Node(j);
        node->Draw(0); 
      }
    }      
  }
}



////////////////////////////////////////////////////////////////////////
// ENTRY POINTS
////////////////////////////////////////////////////////////////////////

int
DebugRedraw(R3SurfelViewer *viewer)
{
  int status = 0;
  if (debug1) DrawPointSet(viewer);
  if (debug2) DrawGraph(viewer);
  if (debug4) DrawMeanShift(viewer);
  if (debug5) DrawConnectedPointSet(viewer);
  if (debug6) DrawAbovePointSet(viewer);
  if (debug7) DrawAboveConnectedPointGraph(viewer);
  if (debug8) DrawAlign(viewer);
  if (debug9) status |= ExtractObjectWithMesh(viewer);
  // if (debug8) CreateObject(viewer);
  // if (debug9) ReadObjects(viewer);
  // if (debug1) DrawEstimateSupportPlane(viewer);
  // if (debug7) status |= RemoveObject(viewer);
  // if (debug8) status |= SplitObject(viewer);
  // if (debug9) status |= MergeObjects(viewer);
  // if (debug3) DrawGrid(viewer);
  // if (debug7) CreateClusterObjects(viewer);
  // if (debug8) DrawPlanarGrids(viewer);
  // if (debug9) CreatePlanarObjects(viewer);
  // if (debug1) DrawEstimateSupportPlane(viewer);
  // if (debug8) DrawOverheadImage(viewer);
  // if (debug2) DrawFitSupportPlane(viewer);
  // if (debug8) DrawPrunedGraph(viewer);
  // if (debug8) DrawDirections(viewer);
  // if (debug2) DrawGrid(viewer);
  // DrawMap(viewer);
  return status;
}



int 
DebugResize(R3SurfelViewer *viewer, int w, int h)
{
  // Do nothing
  return 0;
}



int DebugMouseMotion(R3SurfelViewer *viewer, int x, int y)
{
  // Do nothing
  return 0;
}



int DebugMouseButton(R3SurfelViewer *viewer, int x, int y, int button, int state, int shift, int ctrl, int alt)
{
  // Do nothing
  return 0;
}



int
DebugKeyboard(R3SurfelViewer *viewer, int x, int y, int key, int shift, int ctrl, int alt)
{
  // Initialize redraw
  int redraw = 0;

  // Toggle debugging
  if (alt) {
    redraw= 1;
    switch (key) {
    case '0': debug0 = !debug0; break;
    case '1': debug1 = !debug1; break;
    case '2': debug2 = !debug2; break;
    case '3': debug3 = !debug3; break;
    case '4': debug4 = !debug4; break;
    case '5': debug5 = !debug5; break;
    case '6': debug6 = !debug6; break;
    case '7': debug7 = !debug7; break;
    case '8': debug8 = !debug8; break;
    case '9': debug9 = !debug9; break;
    default: redraw = 0; break;
    }
  }

  /// Return whether should redraw
  return redraw;
}
