////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "FET.h"



////////////////////////////////////////////////////////////////////////
// Constructors
////////////////////////////////////////////////////////////////////////

FETShape::
FETShape(FETReconstruction *reconstruction)
  : reconstruction(NULL),
    reconstruction_index(-1),
    parents(),
    children(),
    features(),
    matches(),
    initial_transformation(R3identity_affine),
    current_transformation(R3identity_affine),
    ground_truth_transformation(R3identity_affine),
    kdtree(NULL),
    viewpoint(0, 0, 0),
    towards(0, 0, 0),
    up(0, 0, 0),
    bbox(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX),
    origin(RN_UNKNOWN, RN_UNKNOWN, RN_UNKNOWN),
    name(NULL)
{
  // Initialize variable index
  for (int i = 0; i < max_variables; i++) variable_index[i] = -1;

  // Initialize variable inertias
  for (int i = 0; i < max_variables; i++) variable_inertias[i] = 1;
  variable_inertias[FET_SX] = RN_INFINITY;
  variable_inertias[FET_SY] = RN_INFINITY;
  variable_inertias[FET_SZ] = RN_INFINITY;

  // Insert into reconstruction
  if (reconstruction) reconstruction->InsertShape(this);
}



FETShape::
FETShape(const FETShape& shape)
  : reconstruction(NULL),
    reconstruction_index(-1),
    parents(),
    children(),
    features(),
    matches(),
    initial_transformation(shape.initial_transformation),
    current_transformation(shape.current_transformation),
    ground_truth_transformation(shape.ground_truth_transformation),
    kdtree(NULL),
    viewpoint(shape.viewpoint),
    towards(shape.towards),
    up(shape.up),
    bbox(shape.bbox),
    origin(shape.origin),
    name((shape.name) ? strdup(shape.name) : NULL)
{
  // Initialize variable index
  for (int i = 0; i < max_variables; i++) this->variable_index[i] = -1;

  // Copy variable inertias
  for (int i = 0; i < max_variables; i++) this->variable_inertias[i] = shape.variable_inertias[i];

#if 0
  // Insert into reconstruction
  if (shape.reconstruction) shape.reconstruction->InsertShape(this);

  // Copy matches
  for (int i = 0; i < shape.NMatches(); i++) {
    FETMatch *match = shape.Match(i);
    InsertMatch(new Match(*match), match->FETShapeIndex(this));
  }

  // Copy features
  for (int i = 0; i < shape.NFeatures(); i++) {
    FETFeature *feature = shape.Feature(i);
    InsertFeature(new Feature(*feature));
  }

  // Copy children
  for (int i = 0; i < shape.NChildren(); i++) {
    FETShape *child = shape.Child(i);
    InsertChild(new FETShape(*child));
  }

  // Insert into parents
  for (int i = 0; i < shape.NParents(); i++) {
    FETShape *parent = shape.Parent(i);
    parent->InsertChild(this);
  }
#endif
}



FETShape::
~FETShape(void) 
{
  // Delete name
  if (name) free(name);
  
  // Delete kdtree
  if (kdtree) delete kdtree;
  
  // Remove from parents
  while (NParents() > 0) {
    FETShape *parent = Parent(NParents()-1);
    parent->RemoveChild(this);
  }

  // Remove children
  while (NChildren() > 0) {
    FETShape *child = Child(NChildren()-1);
    RemoveChild(child);
  }

  // Delete matches 
  while (NMatches() > 0) {
    FETMatch *match = Match(NMatches()-1);
    delete match;
  }

  // Delete features 
  while (NFeatures() > 0) {
    FETFeature *feature = Feature(NFeatures()-1);
    delete feature;
  }

  // Remove from reconstruction
  if (reconstruction) reconstruction->RemoveShape(this);
}



////////////////////////////////////////////////////////////////////////
// Properties
////////////////////////////////////////////////////////////////////////

R3Box FETShape::
BBox(void) const
{
  // Return bounding box
  if (bbox.IsEmpty()) ((FETShape *) this)->UpdateBBox();
  return bbox;
}



RNLength FETShape::
AverageFeatureRadius(void) const
{
  // Check features
  if (NFeatures() == 0) return 0.0;
  
  // Return average radius
  RNLength total_radius = 0;
  for (int i = 0; i < NFeatures(); i++) {
    FETFeature *feature = Feature(i);
    total_radius += feature->Radius();
  }

  // Return average
  return total_radius / NFeatures();
}



////////////////////////////////////////////////////////////////////////
// Hierarchy manipulation
////////////////////////////////////////////////////////////////////////

void FETShape::
InsertChild(FETShape *child)
{
  // Just checking
  assert(child->reconstruction == this->reconstruction);
  assert(!children.FindEntry(child));

  // Insert parent/child
  this->children.Insert(child);
  child->parents.Insert(this);
}



void FETShape::
RemoveChild(FETShape *child)
{
  // Just checking
  assert(child->reconstruction == this->reconstruction);
  assert(children.FindEntry(child));

  // Remove parent/child
  this->children.Remove(child);
  child->parents.Remove(this);
}



////////////////////////////////////////////////////////////////////////
// Feature manipulation
////////////////////////////////////////////////////////////////////////

void FETShape::
InsertFeature(FETFeature *feature)
{
  // Just checking
  assert(feature->reconstruction == this->reconstruction);
  assert(feature->shape_index == -1);
  assert(feature->shape == NULL);
  
  // Insert feature
  feature->shape = this;
  feature->shape_index = features.NEntries();
  features.Insert(feature);

  // Update bounding box
  if (!bbox.IsEmpty()) {
    bbox.Union(feature->Position(TRUE));
  }
}



void FETShape::
RemoveFeature(FETFeature *feature)
{
  // Just checking
  assert(feature->shape_index >= 0);
  assert(feature->shape_index < features.NEntries());
  assert(feature->shape == this);

  // Remove feature
  RNArrayEntry *entry = features.KthEntry(feature->shape_index);
  FETFeature *tail = features.Tail();
  tail->shape_index = feature->shape_index;
  features.EntryContents(entry) = tail;
  features.RemoveTail();
  feature->shape_index = -1;
  feature->shape = NULL;

  // Reset bounding box
  InvalidateBBox();
}



////////////////////////////////////////////////////////////////////////
// Match manipulation
////////////////////////////////////////////////////////////////////////

void FETShape::
InsertMatch(FETMatch *match, int k)
{
  // Just checking
  assert(match->reconstruction == this->reconstruction);
  assert(match->shape_indices[k] == -1);
  assert(match->shapes[k] == NULL);

  // Insert match
  match->shapes[k] = this;
  match->shape_indices[k] = matches.NEntries();
  matches.Insert(match);
}



void FETShape::
RemoveMatch(FETMatch *match, int k)
{
  // Just checking
  assert(match->shape_indices[k] >= 0);
  assert(match->shape_indices[k] < matches.NEntries());
  assert(match->shapes[k] == this);

  // Remove match
  RNArrayEntry *entry = matches.KthEntry(match->shape_indices[k]);
  FETMatch *tail = matches.Tail();
  tail->shape_indices[tail->ShapeIndex(this)] = match->shape_indices[k];
  matches.EntryContents(entry) = tail;
  matches.RemoveTail();
  match->shape_indices[k] = -1;
  match->shapes[k] = NULL;
}



////////////////////////////////////////////////////////////////////////
// Transformation manipulation
////////////////////////////////////////////////////////////////////////

void FETShape::
ResetTransformation(void)
{
  // Reset transformation
  current_transformation = initial_transformation;

  // Need to update bounding box
  InvalidateBBox();
}



void FETShape::
SetTransformation(const R3Affine& transformation)
{
  // Set transformation
  this->current_transformation = transformation;

  // Need to update bounding box
  InvalidateBBox();
}



void FETShape::
PerturbTransformation(RNLength translation_magnitude, RNAngle rotation_magnitude)
{
  // Perturb transformation
  R3Point c = Origin();
  Transform(c);
  R3Affine old_transformation = current_transformation;
  current_transformation = R3identity_affine;
  current_transformation.Translate(translation_magnitude * R3RandomDirection());
  current_transformation.Translate(c.Vector());
  current_transformation.Rotate(R3RandomDirection(), rotation_magnitude);
  current_transformation.Translate(-c.Vector());
  current_transformation.Transform(old_transformation);

  // Need to update bounding box
  InvalidateBBox();
}



////////////////////////////////////////////////////////////////////////
// Display
////////////////////////////////////////////////////////////////////////

void FETShape::
Draw(void) const
{
  // Push transformation
  current_transformation.Push();

  // Draw all features
  for (int i = 0; i < NFeatures(); i++) {
    FETFeature *feature = Feature(i);
    feature->Draw();
  }

  // Pop transformation
  current_transformation.Pop();
}



////////////////////////////////////////////////////////////////////////
// Input / output functions
////////////////////////////////////////////////////////////////////////

int FETShape::
ReadAscii(FILE *fp)
{
  // Read shape
  int dummy = 0;
  int r, nparents, nfeatures;
  char name_buffer[256];
  RNScalar c[3], current_matrix[16], ground_truth_matrix[16];
  fscanf(fp, "%d", &r);
  fscanf(fp, "%s", name_buffer);
  fscanf(fp, "%d", &nparents);
  fscanf(fp, "%d", &nfeatures);
  fscanf(fp, "%lf%lf%lf", &c[0], &c[1], &c[2]);
  fscanf(fp, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf",
    &current_matrix[0], &current_matrix[1], &current_matrix[2], &current_matrix[3],
    &current_matrix[4], &current_matrix[5], &current_matrix[6], &current_matrix[7],
    &current_matrix[8], &current_matrix[9], &current_matrix[10], &current_matrix[11],
    &current_matrix[12], &current_matrix[13], &current_matrix[14], &current_matrix[15]);
  fscanf(fp, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf",
    &ground_truth_matrix[0], &ground_truth_matrix[1], &ground_truth_matrix[2], &ground_truth_matrix[3],
    &ground_truth_matrix[4], &ground_truth_matrix[5], &ground_truth_matrix[6], &ground_truth_matrix[7],
    &ground_truth_matrix[8], &ground_truth_matrix[9], &ground_truth_matrix[10], &ground_truth_matrix[11],
    &ground_truth_matrix[12], &ground_truth_matrix[13], &ground_truth_matrix[14], &ground_truth_matrix[15]);
  for (int k = 0; k < max_variables; k++) fscanf(fp, "%lf", &variable_inertias[k]);
  for (int k = 0; k < 4; k++) fscanf(fp, "%d", &dummy);
  
  // Check reconstruction index
  assert(((r == -1) && !reconstruction) || (r == reconstruction_index));
  
  // Read parents
  for (int i = 0; i < nparents; i++) {
    int p;
    fscanf(fp, "%d", &p);
    if (reconstruction) {
      FETShape *parent = reconstruction->Shape(p);
      parent->InsertChild(this);
    }
  }

  // Assign transformation
  current_transformation.Reset(R4Matrix(current_matrix), 0);
  initial_transformation.Reset(R4Matrix(current_matrix), 0);
  ground_truth_transformation.Reset(R4Matrix(ground_truth_matrix), 0);

  // Assign name
  if (strcmp(name_buffer, "None") && (strcmp(name_buffer, "none"))) name = strdup(name_buffer);

  // Assign origin
  if (RNIsNotEqual(c[0], RN_UNKNOWN) && RNIsNotEqual(c[0], RN_UNKNOWN) && RNIsNotEqual(c[0], RN_UNKNOWN)) {
    origin.Reset(c[0], c[1], c[2]);
  }
  
  // Return success
  return 1;
}



int FETShape::
WriteAscii(FILE *fp) const
{
  // Write shape
  int dummy = 0;
  int r = reconstruction_index;
  int nparents = NParents();
  int nfeatures = NFeatures();
  char name_buffer[256] = { '\0' };
  if (name) strncpy(name_buffer, name, 255);
  else strncpy(name_buffer, "None", 255);
  R4Matrix current_matrix = current_transformation.Matrix();
  R4Matrix ground_truth_matrix = ground_truth_transformation.Matrix();
  fprintf(fp, "%d ", r);
  fprintf(fp, "%s ", name_buffer);
  fprintf(fp, "%d ", nparents);
  fprintf(fp, "%d ", nfeatures);
  fprintf(fp, "%g %g %g ", origin[0], origin[1], origin[2]);
  fprintf(fp, "%g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g ",
    current_matrix[0][0], current_matrix[0][1], current_matrix[0][2], current_matrix[0][3],
    current_matrix[1][0], current_matrix[1][1], current_matrix[1][2], current_matrix[1][3],
    current_matrix[2][0], current_matrix[2][1], current_matrix[2][2], current_matrix[2][3],
    current_matrix[3][0], current_matrix[3][1], current_matrix[3][2], current_matrix[3][3]);
  fprintf(fp, "%g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g ",
    ground_truth_matrix[0][0], ground_truth_matrix[0][1], ground_truth_matrix[0][2], ground_truth_matrix[0][3],
    ground_truth_matrix[1][0], ground_truth_matrix[1][1], ground_truth_matrix[1][2], ground_truth_matrix[1][3],
    ground_truth_matrix[2][0], ground_truth_matrix[2][1], ground_truth_matrix[2][2], ground_truth_matrix[2][3],
    ground_truth_matrix[3][0], ground_truth_matrix[3][1], ground_truth_matrix[3][2], ground_truth_matrix[3][3]);
  for (int k = 0; k < max_variables; k++) fprintf(fp, "%g ", variable_inertias[k]);
  for (int k = 0; k < 4; k++) fprintf(fp, "%d ", dummy);

  // Write parents
  for (int i = 0; i < nparents; i++) {
    FETShape *parent = Parent(i);
    fprintf(fp, "%d ", parent->reconstruction_index);
  }
    
  // Return success
  return 1;
}

  

int FETShape::
ReadBinary(FILE *fp)
{
  // Read shape
  int dummy = 0;
  int r, nparents, nfeatures;
  char name_buffer[256];
  float v[3], t[3], u[3];
  RNScalar c[3], current_matrix[16], ground_truth_matrix[16];
  fread(&r, sizeof(int), 1, fp);
  fread(name_buffer, sizeof(char), 256, fp);
  fread(&nparents, sizeof(int), 1, fp);
  fread(&nfeatures, sizeof(int), 1, fp);
  fread(&c[0], sizeof(RNScalar), 3, fp);
  fread(&current_matrix[0], sizeof(RNScalar), 16, fp);
  fread(&ground_truth_matrix[0], sizeof(RNScalar), 16, fp);
  fread(variable_inertias, sizeof(RNScalar), max_variables, fp);
  fread(&v[0], sizeof(float), 3, fp);
  fread(&t[0], sizeof(float), 3, fp);
  fread(&u[0], sizeof(float), 3, fp);
  for (int k = 0; k < 7; k++) fread(&dummy, sizeof(int), 1, fp);
  if (name_buffer[0] != '\0') name = strdup(name_buffer);
  current_transformation.Reset(R4Matrix(current_matrix), 0);
  initial_transformation.Reset(R4Matrix(current_matrix), 0);
  ground_truth_transformation.Reset(R4Matrix(ground_truth_matrix), 0);
  viewpoint.Reset(v[0], v[1], v[2]);
  towards.Reset(t[0], t[1], t[2]);
  up.Reset(u[0], u[1], u[2]);
  origin.Reset(c[0], c[1], c[2]);
  
  // Check reconstruction index
  assert(((r == -1) && !reconstruction) || (r == reconstruction_index));
  
  // Read parents
  for (int i = 0; i < nparents; i++) {
    int p;
    fread(&p, sizeof(int), 1, fp);
    if (reconstruction) {
      FETShape *parent = reconstruction->Shape(p);
      parent->InsertChild(this);
    }
  }

  // Return success
  return 1;
}



int FETShape::
WriteBinary(FILE *fp) const
{
  // Write shape
  int dummy = 0;
  int nparents = NParents();
  int nfeatures = NFeatures();
  char name_buffer[256] = { '\0' };
  float v[3]; v[0] = (float) viewpoint[0]; v[1] = (float) viewpoint[1]; v[2] = (float) viewpoint[2];
  float t[3]; t[0] = (float) towards[0]; t[1] = (float) towards[1]; t[2] = (float) towards[2];
  float u[3]; u[0] = (float) up[0]; u[1] = (float) up[1]; u[2] = (float) up[2];
  if (name) strncpy(name_buffer, name, 255);
  R4Matrix current_matrix = current_transformation.Matrix();
  R4Matrix ground_truth_matrix = ground_truth_transformation.Matrix();
  fwrite(&reconstruction_index, sizeof(int), 1, fp);
  fwrite(name_buffer, sizeof(char), 256, fp);
  fwrite(&nparents, sizeof(int), 1, fp);
  fwrite(&nfeatures, sizeof(int), 1, fp);
  fwrite(origin.Coords(), sizeof(RNScalar), 3, fp);
  fwrite(&current_matrix[0][0], sizeof(RNScalar), 16, fp);
  fwrite(&ground_truth_matrix[0][0], sizeof(RNScalar), 16, fp);
  fwrite(variable_inertias, sizeof(RNScalar), max_variables, fp);
  fwrite(v, sizeof(float), 3, fp);
  fwrite(t, sizeof(float), 3, fp);
  fwrite(u, sizeof(float), 3, fp);
  for (int k = 0; k < 7; k++) fwrite(&dummy, sizeof(int), 1, fp);

  // Write parents
  for (int i = 0; i < nparents; i++) {
    FETShape *parent = Parent(i);
    fwrite(&parent->reconstruction_index, sizeof(int), 1, fp);
  }
    
  // Return success
  return 1;
}

  

////////////////////////////////////////////////////////////////////////
// Update functions
////////////////////////////////////////////////////////////////////////

void FETShape::
UpdateVariableIndex(int& nvariables)
{
  // XXXX THIS IS A TEMPORARY HACK XXXX
  // OPTIMIZATION SEEMS UNSTABLE OTHERWISE
  // HERE BECAUSE INERTIAS ARE READ FROM FILE
  // variable_inertias[FET_RX] = RN_INFINITY;
  // variable_inertias[FET_RY] = RN_INFINITY;

  // Update variable index
  for (int i = 0; i < max_variables; i++) {
    if (variable_inertias[i] >= RN_INFINITY) variable_index[i] = -1;
    else variable_index[i] = nvariables++;
  }
}

 

    
void FETShape::
UpdateVariableValues(const RNScalar *x)
{
#if 0
  for (int i = 0; i < max_variables; i++) {
    if (variable_index[i] < 0) continue;
    if (RNIsZero(x[variable_index[i]], 0.001)) continue;
    printf("%6d %2d %9.6f\n", reconstruction_index, i, x[variable_index[i]]);
  }
#endif
  
  // Extract variable values
  R3Vector translation(0,0,0);
  R3Vector rotation(0,0,0);
  R3Vector scale(0,0,0);
  translation[0] = (variable_index[FET_TX] >= 0) ? x[variable_index[FET_TX]] : 0.0;
  translation[1] = (variable_index[FET_TY] >= 0) ? x[variable_index[FET_TY]] : 0.0;
  translation[2] = (variable_index[FET_TZ] >= 0) ? x[variable_index[FET_TZ]] : 0.0;
  rotation[0] = (variable_index[FET_RX] >= 0) ? x[variable_index[FET_RX]] : 0.0;
  rotation[1] = (variable_index[FET_RY] >= 0) ? x[variable_index[FET_RY]] : 0.0;
  rotation[2] = (variable_index[FET_RZ] >= 0) ? x[variable_index[FET_RZ]] : 0.0;
  scale[0] = (variable_index[FET_SX] >= 0) ? x[variable_index[FET_SX]] : 0.0;
  scale[1] = (variable_index[FET_SY] >= 0) ? x[variable_index[FET_SY]] : 0.0;
  scale[2] = (variable_index[FET_SZ] >= 0) ? x[variable_index[FET_SZ]] : 0.0;
  
  // Extract center of rotation and scale
  R3Point c = Origin();
  Transform(c);

  // Compute transformation
  R3Affine transformation = R3identity_affine;
  transformation.Translate(translation);
  transformation.Translate(c.Vector());
  transformation.Rotate(rotation);
  transformation.Scale(R3ones_vector + scale);
  transformation.Translate(-c.Vector());
  transformation.Transform(current_transformation);
  current_transformation = transformation;

  // Need to update bounding box
  InvalidateBBox();
}



void FETShape::
UpdateFeatureProperties(void)
{
  // Get some useful variables
  const int max_neighbor_points = 16;
  RNLength max_neighbor_distance = RN_INFINITY;

  // Build shape's feature kdtree 
  if (!kdtree) {
    FETFeature tmp; int position_offset = (unsigned char *) &(tmp.position) - (unsigned char *) &tmp;
    kdtree = new R3Kdtree<FETFeature *>(features, position_offset);
    if (!kdtree) RNAbort("Cannot build kdtree");
  }

  // Update normal and radius for every feature (if none already)
  for (int i = 0; i < NFeatures(); i++) {
    FETFeature *feature = Feature(i);

    // Check if already up-to-date
    if ((feature->Radius() > 0) && (feature->Normal() != R3zero_vector))  continue;

    // Get features in neighborhood
    RNArray<FETFeature *> neighbor_features;
    if (!kdtree->FindClosest(feature, RN_EPSILON, max_neighbor_distance, max_neighbor_points, neighbor_features)) continue;
    assert(neighbor_features.NEntries() <= max_neighbor_points);
    if (neighbor_features.NEntries() < 3) continue;

    // Get array of points for passing to centroid and principle axes functions
    int num_neighbor_points = neighbor_features.NEntries();
    R3Point neighbor_points[max_neighbor_points];
    for (int j = 0; j < num_neighbor_points; j++) {
      FETFeature *neighbor_feature = neighbor_features.Kth(j);
      neighbor_points[j] = neighbor_feature->Position();
    }

    // Compute neighborhood properties
    RNScalar neighborhood_variances[3];
    R3Point neighborhood_centroid = R3Centroid(num_neighbor_points, neighbor_points);
    R3Triad neighborhood_axes = R3PrincipleAxes(neighborhood_centroid, num_neighbor_points, neighbor_points, NULL, neighborhood_variances);
    if (neighborhood_variances[0] < RN_EPSILON) continue;
    RNLength radius = sqrt(neighborhood_variances[0]);
     R3Vector direction = neighborhood_axes[0];
    R3Vector normal = neighborhood_axes[2];
    R3Plane tangent_plane(feature->Position(), normal);
    if (R3SignedDistance(tangent_plane, this->Centroid()) < 0) normal.Flip();
      
    // Update feature properties
    feature->SetDirection(direction);
    feature->SetNormal(normal);
    feature->SetRadius(radius);

    // Update shape type ???
    // if (variances[1] / variances[0] < 0.25) feature->SetShapeType(LINE_FEATURE_SHAPE);
    // else if (variances[2] / variances[0] < 0.25) feature->SetShapeType(PLANE_FEATURE_SHAPE);
    // else feature->SetShapeType(POINT_FEATURE_SHAPE);
  }
}



void FETShape::
UpdateBBox(void)
{
  // Update bounding box
  bbox = R3null_box;
  for (int i = 0; i < NFeatures(); i++) {
    FETFeature *feature = Feature(i);
    bbox.Union(feature->Position(TRUE));
  }
}



void FETShape::
InvalidateBBox(void)
{
  // Invalidate bounding box
  bbox.Reset(R3Point(FLT_MAX, FLT_MAX, FLT_MAX), R3Point(-FLT_MAX, -FLT_MAX, -FLT_MAX));

  // Invalidate parent bounding boxes
  for (int i = 0; i < NParents(); i++) {
    FETShape *parent = Parent(i);
    parent->InvalidateBBox();
  }

  // Invalidate reconstruction bounding box
  if (reconstruction) reconstruction->InvalidateBBox();
}



////////////////////////////////////////////////////////////////////////
// Search functions
////////////////////////////////////////////////////////////////////////

FETFeature *FETShape::
FindClosestFeature(const R3Point& query_position, 
  RNLength min_euclidean_distance, RNLength max_euclidean_distance) 
{
  // Build shape's feature kdtree 
  if (!kdtree) {
    FETFeature tmp; int position_offset = (unsigned char *) &(tmp.position) - (unsigned char *) &tmp;
    kdtree = new R3Kdtree<FETFeature *>(features, position_offset);
    if (!kdtree) RNAbort("Cannot build kdtree");
  }

  // Find closest feature
  return kdtree->FindClosest(query_position, min_euclidean_distance, max_euclidean_distance);
}



FETFeature *FETShape::
FindClosestFeature(FETFeature *query_feature, const R3Affine& query_transformation,
  RNLength min_euclidean_distance, RNLength max_euclidean_distance, 
  RNLength *max_descriptor_distances, RNAngle max_normal_angle,
  RNScalar min_distinction, RNScalar min_salience, RNBoolean discard_boundaries)
{
  // Load compatibility parameters
  FETCompatibilityParameters compatibility(max_euclidean_distance,
   max_descriptor_distances, max_normal_angle,
   min_distinction, min_salience, discard_boundaries);

  // Build kdtree 
  if (!kdtree) {
    FETFeature tmp; int position_offset = (unsigned char *) &(tmp.position) - (unsigned char *) &tmp;
    kdtree = new R3Kdtree<FETFeature *>(features, position_offset);
    if (!kdtree) RNAbort("Cannot build kdtree");
  }

  // Temporarily transform query feature
  R3Point saved_query_position = query_feature->position;
  R3Vector saved_query_direction = query_feature->direction;
  R3Vector saved_query_normal = query_feature->normal;
  query_feature->position.Transform(query_transformation);
  query_feature->direction.Transform(query_transformation);
  query_feature->normal.Transform(query_transformation);
  query_feature->position.InverseTransform(current_transformation);
  query_feature->direction.InverseTransform(current_transformation);
  query_feature->normal.InverseTransform(current_transformation);

  // Find closest feature
  FETFeature *closest_feature = kdtree->FindClosest(query_feature, 
    min_euclidean_distance, max_euclidean_distance, 
    AreFeaturesCompatible, &compatibility);

  // Restore query feature
  query_feature->position = saved_query_position;
  query_feature->direction = saved_query_direction;
  query_feature->normal = saved_query_normal;

  // Return closest feature
  return closest_feature;
}



int FETShape::
FindAllFeatures(FETFeature *query_feature, const R3Affine& query_transformation, RNArray<FETFeature *>& result,
  RNLength min_euclidean_distance, RNLength max_euclidean_distance, 
  RNLength *max_descriptor_distances, RNAngle max_normal_angle,
  RNScalar min_distinction, RNScalar min_salience,
  RNBoolean discard_boundaries) 
{
  // Load compatibility parameters
  FETCompatibilityParameters compatibility(max_euclidean_distance,
   max_descriptor_distances, max_normal_angle,
   min_distinction, min_salience, discard_boundaries);

  // Build shape's feature kdtree 
  if (!kdtree) {
    FETFeature tmp; int position_offset = (unsigned char *) &(tmp.position) - (unsigned char *) &tmp;
    kdtree = new R3Kdtree<FETFeature *>(features, position_offset);
    if (!kdtree) RNAbort("Cannot build kdtree");
  }

  // Temporarily transform query feature
  R3Point saved_query_position = query_feature->position;
  R3Vector saved_query_direction = query_feature->direction;
  R3Vector saved_query_normal = query_feature->normal;
  query_feature->position.Transform(query_transformation);
  query_feature->direction.Transform(query_transformation);
  query_feature->normal.Transform(query_transformation);
  query_feature->position.InverseTransform(current_transformation);
  query_feature->direction.InverseTransform(current_transformation);
  query_feature->normal.InverseTransform(current_transformation);

  // Find all features
  kdtree->FindAll(query_feature, 
    min_euclidean_distance, max_euclidean_distance, 
    AreFeaturesCompatible, &compatibility, 
    result);

  // Restore query feature
  query_feature->position = saved_query_position;
  query_feature->direction = saved_query_direction;
  query_feature->normal = saved_query_normal;

  // Return success
  return result.NEntries();
}



int FETShape::
ComputeTransformedPointCoordinates(const R3Point& position, 
  RNAlgebraic *& px, RNAlgebraic *& py, RNAlgebraic *& pz) const
{
  // Get transformation variables
  RNPolynomial tx, ty, tz, rx, ry, rz, sx(1, 0), sy (1, 0), sz(1, 0);
  if (variable_index[FET_TX] >= 0) tx.AddTerm(1.0, variable_index[FET_TX], 1.0, TRUE); 
  if (variable_index[FET_TY] >= 0) ty.AddTerm(1.0, variable_index[FET_TY], 1.0, TRUE); 
  if (variable_index[FET_TZ] >= 0) tz.AddTerm(1.0, variable_index[FET_TZ], 1.0, TRUE);
  if (variable_index[FET_RX] >= 0) rx.AddTerm(1.0, variable_index[FET_RX], 1.0, TRUE); 
  if (variable_index[FET_RY] >= 0) ry.AddTerm(1.0, variable_index[FET_RY], 1.0, TRUE); 
  if (variable_index[FET_RZ] >= 0) rz.AddTerm(1.0, variable_index[FET_RZ], 1.0, TRUE);
  if (variable_index[FET_SX] >= 0) sx.AddTerm(1.0, variable_index[FET_SX], 1.0, TRUE); 
  if (variable_index[FET_SY] >= 0) sy.AddTerm(1.0, variable_index[FET_SY], 1.0, TRUE); 
  if (variable_index[FET_SZ] >= 0) sz.AddTerm(1.0, variable_index[FET_SZ], 1.0, TRUE);

  // Get origin and point after initial transformation
  R3Point c = Origin();
  R3Point p = position;
  c.Transform(Transformation());
  p.Transform(Transformation());
  R3Vector d = p - c;

  // Get rotated vector from transformed origin to transformed point
  // dx =     d.X() - rz*d.Y() + ry*d.Z();
  // dy =  rz*d.X()      d.Y() - rx*d.Z();
  // dz = -ry*d.X() + rx*d.Y()      d.Z();
  RNAlgebraic *dx, *dy, *dz;
  RNAlgebraic *dxX = new RNAlgebraic(1.0, 0); dxX->Multiply( d.X());
  RNAlgebraic *dxY = new RNAlgebraic(rz, 0);  dxY->Multiply(-d.Y());
  RNAlgebraic *dxZ = new RNAlgebraic(ry, 0);  dxZ->Multiply( d.Z());
  RNAlgebraic *dyX = new RNAlgebraic(rz, 0);  dyX->Multiply( d.X());
  RNAlgebraic *dyY = new RNAlgebraic(1.0, 0); dyY->Multiply( d.Y());
  RNAlgebraic *dyZ = new RNAlgebraic(rx, 0);  dyZ->Multiply(-d.Z());
  RNAlgebraic *dzX = new RNAlgebraic(ry, 0);  dzX->Multiply(-d.X());
  RNAlgebraic *dzY = new RNAlgebraic(rx, 0);  dzY->Multiply( d.Y());
  RNAlgebraic *dzZ = new RNAlgebraic(1.0, 0); dzZ->Multiply( d.Z());
  dx = new RNAlgebraic(                      dxX);
  dx = new RNAlgebraic(RN_ADD_OPERATION, dx, dxY);
  dx = new RNAlgebraic(RN_ADD_OPERATION, dx, dxZ);
  dy = new RNAlgebraic(                      dyX);
  dy = new RNAlgebraic(RN_ADD_OPERATION, dy, dyY);
  dy = new RNAlgebraic(RN_ADD_OPERATION, dy, dyZ);
  dz = new RNAlgebraic(                      dzX);
  dz = new RNAlgebraic(RN_ADD_OPERATION, dz, dzY);
  dz = new RNAlgebraic(RN_ADD_OPERATION, dz, dzZ);

  // Scale vector
  if (variable_index[FET_SX] >= 0) dx->Multiply(sx);
  if (variable_index[FET_SY] >= 0) dy->Multiply(sy);
  if (variable_index[FET_SZ] >= 0) dz->Multiply(sz);
  
  // Start from transformed origin
  px = new RNAlgebraic(c.X(), 0); 
  py = new RNAlgebraic(c.Y(), 0); 
  pz = new RNAlgebraic(c.Z(), 0); 
  
  // Add rotated vector
  px = new RNAlgebraic(RN_ADD_OPERATION, px, dx);
  py = new RNAlgebraic(RN_ADD_OPERATION, py, dy);
  pz = new RNAlgebraic(RN_ADD_OPERATION, pz, dz);

  // Add translation
  if (variable_index[FET_TX] >= 0) px = new RNAlgebraic(RN_ADD_OPERATION, px, new RNAlgebraic(tx, 0));
  if (variable_index[FET_TY] >= 0) py = new RNAlgebraic(RN_ADD_OPERATION, py, new RNAlgebraic(ty, 0));
  if (variable_index[FET_TZ] >= 0) pz = new RNAlgebraic(RN_ADD_OPERATION, pz, new RNAlgebraic(tz, 0));

  // Return success
  return 1;
}



int FETShape::
ComputeTransformedVectorCoordinates(const R3Vector& vector,
  RNAlgebraic *& nx, RNAlgebraic *& ny, RNAlgebraic *& nz) const
{
  // Get transformation variables
  RNPolynomial rx, ry, rz;
  if (variable_index[FET_RX] >= 0) rx.AddTerm(1.0, variable_index[FET_RX], 1.0, TRUE); 
  if (variable_index[FET_RY] >= 0) ry.AddTerm(1.0, variable_index[FET_RY], 1.0, TRUE); 
  if (variable_index[FET_RZ] >= 0) rz.AddTerm(1.0, variable_index[FET_RZ], 1.0, TRUE); 

  // Get vector after initial transformation
  R3Vector n = vector;
  n.Transform(Transformation());

  // Get rotated vector 
  // dx =     d.X() - rz*d.Y() + ry*d.Z();
  // dy =  rz*d.X()      d.Y() - rx*d.Z();
  // dz = -ry*d.X() + rx*d.Y()      d.Z();
  RNAlgebraic *nxX = new RNAlgebraic(1.0, 0); nxX->Multiply( n.X());
  RNAlgebraic *nxY = new RNAlgebraic(rz, 0);  nxY->Multiply(-n.Y());
  RNAlgebraic *nxZ = new RNAlgebraic(ry, 0);  nxZ->Multiply( n.Z());
  RNAlgebraic *nyX = new RNAlgebraic(rz, 0);  nyX->Multiply( n.X());
  RNAlgebraic *nyY = new RNAlgebraic(1.0, 0); nyY->Multiply( n.Y());
  RNAlgebraic *nyZ = new RNAlgebraic(rx, 0);  nyZ->Multiply(-n.Z());
  RNAlgebraic *nzX = new RNAlgebraic(ry, 0);  nzX->Multiply(-n.X());
  RNAlgebraic *nzY = new RNAlgebraic(rx, 0);  nzY->Multiply( n.Y());
  RNAlgebraic *nzZ = new RNAlgebraic(1.0, 0); nzZ->Multiply( n.Z());
  nx = new RNAlgebraic(                      nxX);
  nx = new RNAlgebraic(RN_ADD_OPERATION, nx, nxY);
  nx = new RNAlgebraic(RN_ADD_OPERATION, nx, nxZ);
  ny = new RNAlgebraic(                      nyX);
  ny = new RNAlgebraic(RN_ADD_OPERATION, ny, nyY);
  ny = new RNAlgebraic(RN_ADD_OPERATION, ny, nyZ);
  nz = new RNAlgebraic(                      nzX);
  nz = new RNAlgebraic(RN_ADD_OPERATION, nz, nzY);
  nz = new RNAlgebraic(RN_ADD_OPERATION, nz, nzZ);

  // Return success
  return 1;
}



