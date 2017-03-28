/* Source file for the surfel alignment utilities */



////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "R3Graphics/R3Graphics.h"
#include "R3Surfels/R3Surfels.h"
#include "R3SurfelViewer.h"
#include "model.cpp"



int
DrawAlign(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();

  // Fit mesh to point cloud
  static Model *mesh_model = NULL;
  static Model *surfel_model = NULL;
  static R3Affine mesh_to_surfels = R3identity_affine;
  static R3Point last_center_point(0,0,0);
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;

    // Reset everything
    mesh_model = NULL;
    if (surfel_model) delete surfel_model; 
    surfel_model = NULL;

    // Create mesh models
    static RNArray<Model *> *mesh_models = NULL;
    if (!mesh_models) {
      mesh_models = CreateMeshModels("models/list.txt");
      if (!mesh_models) abort();
    }
    
    // Create surfel model
    surfel_model = CreateSurfelModel(scene, R2Point(center_point.X(), center_point.Y()));
    if (!surfel_model) return 0;

    // Find best fitting mesh model
    int ncorrespondences;
    RNScalar object_coverage, mesh_coverage, rmsd;
    mesh_model = BestFitModel(surfel_model, *mesh_models, &mesh_to_surfels, 
      &object_coverage, &mesh_coverage, &rmsd, &ncorrespondences);
    if (!mesh_model) return 0;

    printf("HEREA %g %g : %g %g %g : %g %g : %g : %d / %d %d : %s\n", 
     center_point.X(), center_point.Y(), 
     surfel_model->origin.X(), surfel_model->origin.Y(), surfel_model->origin.Z(), 
     mesh_coverage, object_coverage, rmsd, ncorrespondences, 
     (surfel_model->points) ? surfel_model->points->NEntries() : -1, 
     (mesh_model->points) ? mesh_model->points->NEntries() : -1, 
     (mesh_model) ? mesh_model->name : "None");
  }
 
  // Draw surfel points
  if (surfel_model) {
    glDisable(GL_LIGHTING);
    glPointSize(5);
    glColor3d(1, 1, 1);
    glBegin(GL_POINTS);
    for (int i = 0; i < surfel_model->points->NEntries(); i++) 
      R3LoadPoint(surfel_model->points->Kth(i)->position);
    glEnd();
    glPointSize(1);
  }

  // Draw mesh points
  if (mesh_model) {
    // Draw points
    glDisable(GL_LIGHTING);
    glColor3d(1, 0, 0);
    mesh_to_surfels.Push();
    glPointSize(5);
    glBegin(GL_POINTS);
    for (int i = 0; i < mesh_model->points->NEntries(); i++) 
      R3LoadPoint(mesh_model->points->Kth(i)->position);
    glEnd();
    glPointSize(1);
    mesh_to_surfels.Pop();
 
    // // Draw mesh
    // glEnable(GL_LIGHTING);
    // GLfloat color[4] = { 0.5, 0.5, 0.5, 1 }; 
    // glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color);
    // glColor4fv(color);
    // mesh_to_surfels.Push();
    // mesh_model->mesh->Draw();
    // mesh_to_surfels.Pop();
    // glDisable(GL_LIGHTING);
  }

  // Return whether need another redraw
  return 0;
}



int
ExtractObjectWithMesh(R3SurfelViewer *viewer)
{
  // Get convenient variables
  R3SurfelScene *scene = viewer->Scene();
  const R3Point& center_point = viewer->CenterPoint();
  R3SurfelTree *tree = scene->Tree();
  if (!tree) return 0;
  R3SurfelObject *root_object = scene->RootObject();
  if (!root_object) return 0;
  R3SurfelNode *root_node = tree->RootNode();
  if (!root_node) return 0;
  int status = 0;

  // Fit mesh to object
  static R3Point last_center_point(0,0,0);
  static R3Affine mesh_to_surfels = R3identity_affine;
  static R3SurfelObject *object = NULL;
  static R3Mesh *mesh = NULL;
  if (!R3Contains(center_point, last_center_point)) {
    last_center_point = center_point;
    if (object) object->ReleaseBlocks();
    object = NULL;
    mesh = NULL;

    // Create mesh models
    static RNArray<Model *> *mesh_models = NULL;
    if (!mesh_models) {
      mesh_models = CreateMeshModels("models");
      if (!mesh_models) abort();
    }

    // Create surfel model
    Model *surfel_model = CreateSurfelModel(scene,  R2Point(center_point.X(), center_point.Y()), 4.0, 4.0);
    if (!surfel_model) return 0;
      
    // Find best fitting mesh model
    RNScalar surfel_coverage, mesh_coverage, rmsd;
    Model *mesh_model = BestFitModel(surfel_model, *mesh_models, &mesh_to_surfels, &surfel_coverage, &mesh_coverage, &rmsd);
    if (!mesh_model) { delete surfel_model; return 0; }
    mesh = mesh_model->mesh;

    // Create constraint
    R3SurfelMultiConstraint constraint;
    // R3Plane plane = EstimateSupportPlane(scene, center_point);
    // R3SurfelPlaneConstraint plane_constraint(plane, FALSE, FALSE, TRUE, 0.25);
    R3SurfelMeshConstraint mesh_constraint(mesh, mesh_to_surfels.Inverse(), 0.25);
    R3SurfelObjectConstraint object_constraint(NULL, TRUE);
    // constraint.InsertConstraint(&plane_constraint);
    constraint.InsertConstraint(&mesh_constraint);
    constraint.InsertConstraint(&object_constraint);

    // Split leaf nodes
    RNArray<R3SurfelNode *> nodes;
    viewer->SplitLeafNodes(root_node, constraint, &nodes);
    if (nodes.IsEmpty()) return 0;
    
    // Create object
    object = new R3SurfelObject(NULL);

    // Insert nodes satisfying constraint into object
    for (int i = 0; i < nodes.NEntries(); i++) {
      R3SurfelNode *node = nodes.Kth(i);
      R3SurfelObject *old_object = node->Object();
      if (old_object) old_object->RemoveNode(node);
      object->InsertNode(node);
    }

    // Update properties
    object->UpdateProperties();
    
    // Insert object into scene
    scene->InsertObject(object, root_object);

    // Read object
    if (object) object->ReadBlocks();

    printf("HEREB %s : %g : %p %p\n", 
     (mesh_model) ? mesh_model->name : "None",  
     (object) ? object->Complexity() : -1,
     mesh, object);

    // Delete surfel model
    delete surfel_model;

    // Redraw
    status = 1;
  }
 
  // Draw object
  if (object) {
    glDisable(GL_LIGHTING);
    glColor3d(0, 0, 1);
    object->Draw();
  }

  // Draw mesh
  if (mesh) {
    glEnable(GL_LIGHTING);
    GLfloat color[4] = { 0, 1, 0, 1 }; 
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color);
    glColor4fv(color);
    mesh_to_surfels.Push();
    mesh->Draw();
    mesh_to_surfels.Pop();
    glDisable(GL_LIGHTING);
  }

  // Return whether need another redraw
  return status;
}



