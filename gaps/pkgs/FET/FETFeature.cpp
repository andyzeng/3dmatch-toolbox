////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "FET.h"



////////////////////////////////////////////////////////////////////////
// FETFeature member functions
////////////////////////////////////////////////////////////////////////

FETFeature::
FETFeature(FETReconstruction *reconstruction, int shape_type, 
  const R3Point& position, const R3Vector& direction, const R3Vector& normal,
  RNLength radius, const FETDescriptor& descriptor, const RNRgb& color, RNFlags flags)
  : reconstruction(NULL),
    reconstruction_index(-1),
    shape(NULL),
    shape_index(-1),
    correspondences(),
    shape_type(shape_type),
    position(position),
    direction(direction),
    normal(normal),
    radius(radius),
    salience(1.0),
    distinction(0.0),
    descriptor(descriptor),
    color(color),
    flags(flags),
    generator_type(UNKNOWN_FEATURE_TYPE),
    primitive_marker(-1)
{
  // Insert into reconstruction
  if (reconstruction) reconstruction->InsertFeature(this);
}



FETFeature::
FETFeature(const FETFeature& feature)
  : reconstruction(NULL),
    reconstruction_index(-1),
    shape(NULL),
    shape_index(-1),
    shape_type(feature.shape_type),
    position(feature.position),
    direction(feature.direction),
    normal(feature.normal),
    radius(feature.radius),
    salience(feature.salience),
    distinction(feature.distinction),
    descriptor(feature.descriptor),
    color(feature.color),
    flags(feature.flags),
    generator_type(feature.generator_type),
    primitive_marker(feature.primitive_marker)
{
#if 0
  // Insert into reconstruction
  if (feature.reconstruction) feature.reconstruction->InsertFeature(this);
#endif
}



FETFeature::
~FETFeature(void)
{
  // Delete correspondences
  while (NCorrespondences() > 0) {
    FETCorrespondence *correspondence = Correspondence(NCorrespondences()-1);
    delete correspondence;
  }

  // Remove feature from shape
  if (shape) shape->RemoveFeature(this);

  // Remove feature from reconstruction
  if (reconstruction) reconstruction->RemoveFeature(this);
}



////////////////////////////////////////////////////////////////////////
// Geometric properties
////////////////////////////////////////////////////////////////////////

R3Point FETFeature::
Position(RNBoolean transformed) const
{
  // Return position
  if (!transformed || !shape) return position;
  R3Point p = position;
  shape->Transform(p);
  return p;
}



R3Vector FETFeature::
Direction(RNBoolean transformed) const
{
  // Return direction
  if (!transformed || !shape) return direction;
  R3Vector d = direction;
  shape->Transform(d);
  return d;
}



R3Vector FETFeature::
Normal(RNBoolean transformed) const
{
  // Return normal
  if (!transformed || !shape) return normal;
  R3Vector n = normal;
  shape->Transform(n);
  return n;
}



RNLength FETFeature::
Radius(RNBoolean transformed) const
{
  // Return radius
  return radius;
}



////////////////////////////////////////////////////////////////////////
// Manipulation
////////////////////////////////////////////////////////////////////////

void FETFeature::
SetPosition(const R3Point& position, RNBoolean transformed)
{
  // Set position
  R3Point p = position;
  if (transformed && shape) shape->InverseTransform(p);
  this->position = p;

  // Invalidate bounding boxes
  if (shape) shape->InvalidateBBox();
}



void FETFeature::
SetDirection(const R3Vector& direction, RNBoolean transformed)
{
  // Set direction
  R3Vector d = direction;
  if (transformed && shape) shape->InverseTransform(d);
  this->direction = d;
}



void FETFeature::
SetNormal(const R3Vector& normal, RNBoolean transformed)
{
  // Set normal
  R3Vector n = normal;
  if (transformed && shape) shape->InverseTransform(n);
  this->normal = n;
}



void FETFeature::
SetRadius(RNLength radius, RNBoolean transformed)
{
  // Set radius
  this->radius = radius;
}



void FETFeature::
Transform(const R3Affine& transformation)
{
  // Transform everything
  position.Transform(transformation);
  direction.Transform(transformation);
  normal.Transform(transformation);
  RNScalar scale = transformation.ScaleFactor();
  if (RNIsNotEqual(scale, 1.0)) radius *= scale;

  // Invalidate bounding boxes
  if (shape) shape->InvalidateBBox();
}



////////////////////////////////////////////////////////////////////////
// Correspondence manipulation
////////////////////////////////////////////////////////////////////////

void FETFeature::
InsertCorrespondence(FETCorrespondence *correspondence, int k)
{
  // Just checking
  assert(correspondence->reconstruction == this->reconstruction);
  assert(correspondence->feature_indices[k] == -1);
  assert(correspondence->features[k] == NULL);

  // Insert correspondence
  correspondence->features[k] = this;
  correspondence->feature_indices[k] = correspondences.NEntries();
  correspondences.Insert(correspondence);
}



void FETFeature::
RemoveCorrespondence(FETCorrespondence *correspondence, int k)
{
  // Just checking
  assert(correspondence->reconstruction == this->reconstruction);
  assert(correspondence->feature_indices[k] >= 0);
  assert(correspondence->feature_indices[k] < correspondences.NEntries());
  assert(correspondence->features[k] == this);

  // Remove correspondence
  RNArrayEntry *entry = correspondences.KthEntry(correspondence->feature_indices[k]);
  FETCorrespondence *tail = correspondences.Tail();
  tail->feature_indices[tail->FeatureIndex(this)] = correspondence->feature_indices[k];
  correspondences.EntryContents(entry) = tail;
  correspondences.RemoveTail();
  correspondence->feature_indices[k] = -1;
  correspondence->features[k] = NULL;
}



void FETFeature::
SortCorrespondences(void)
{
  // Sort correspondences
  correspondences.Sort(FETCompareCorrespondences);
}



////////////////////////////////////////////////////////////////////////
// Relationship
////////////////////////////////////////////////////////////////////////

RNLength FETFeature::
EuclideanDistance(const R3Point& point) const
{
  // Return squared distance to point
  switch (shape_type) {
  case POINT_FEATURE_SHAPE:
    return R3Distance(this->position, point);
    break;

  case LINE_FEATURE_SHAPE: {
    R3Line line(this->position, this->direction);
    return R3Distance(line, point);
    break; }

  case PLANE_FEATURE_SHAPE: {
    R3Plane plane(this->position, this->normal);
    return R3Distance(plane, point);
    break; }
  }

  // Unknown shape type
  return RN_UNKNOWN;
}



RNLength FETFeature::
SquaredEuclideanDistance(const R3Point& point) const
{
  // Return squared distance to point
  switch (shape_type) {
  case POINT_FEATURE_SHAPE: {
    RNLength dd = R3SquaredDistance(this->position, point);
    return dd; }

  case LINE_FEATURE_SHAPE: {
    R3Line line(this->position, this->direction);
    RNScalar d = R3Distance(line, point);
    return d * d; }

  case PLANE_FEATURE_SHAPE: {
    R3Plane plane(this->position, this->normal);
    RNScalar d = R3Distance(plane, point);
    return d * d; }
  }

  // Unknown shape type
  return RN_UNKNOWN;
}



////////////////////////////////////////////////////////////////////////
// Display
////////////////////////////////////////////////////////////////////////

void FETFeature::
Draw(void) const
{
  // Check feature type
  if (shape_type == NULL_FEATURE_SHAPE) {
    // Draw point
    glPointSize(5);
    glBegin(GL_POINTS);
    R3LoadNormal(normal);
    R3LoadPoint(position);
    glEnd();
    glPointSize(1);
  }
  else if (shape_type == POINT_FEATURE_SHAPE) {
    // Draw point
    glPointSize(5);
    glBegin(GL_POINTS);
    R3LoadNormal(normal);
    R3LoadPoint(position);
    glEnd();
    glPointSize(1);
  }
  else if (shape_type == LINE_FEATURE_SHAPE) {
    // Draw point
    glPointSize(5);
    glBegin(GL_POINTS);
    R3LoadNormal(normal);
    R3LoadPoint(position);
    glEnd();
    glPointSize(1);

    // Draw line
    glLineWidth(3);
    glBegin(GL_LINES);
    // R3LoadPoint(position - radius * direction);
    R3LoadPoint(position);
    R3LoadPoint(position + radius * direction);
    glEnd();
    glLineWidth(1);
  }
  else if (shape_type == PLANE_FEATURE_SHAPE) {
    int dim = normal.MinDimension();
    R3Vector axis1 = R3xyz_triad[dim] % normal;
    axis1.Normalize();
    R3Vector axis2 = normal % axis1;
    axis2.Normalize();
    RNLength r = 0.5 * radius; // should be 1.0 * radius
    glBegin(GL_POLYGON);
    R3LoadNormal(normal);
    R3LoadPoint(position - r * axis1 - r * axis2);
    R3LoadPoint(position + r * axis1 - r * axis2);
    R3LoadPoint(position + r * axis1 + r * axis2);
    R3LoadPoint(position - r * axis1 + r * axis2);
    glEnd();

#if 0    
    // Draw line for normal
    glLineWidth(5);
    glBegin(GL_LINES);
    R3LoadNormal(normal);
    R3LoadPoint(position);
    R3LoadPoint(position + 0.5 * radius * normal);
    glEnd();
    glLineWidth(1);
#endif
  }
}



////////////////////////////////////////////////////////////////////////
// Input / output functions
////////////////////////////////////////////////////////////////////////

int FETFeature::
ReadAscii(FILE *fp)
{
  // Read feature
  int r, s, dummy;
  unsigned int f;
  RNScalar p[3], d[3], n[3], c[3];
  fscanf(fp, "%d", &r);
  fscanf(fp, "%d", &s);
  fscanf(fp, "%d", &shape_type);
  fscanf(fp, "%lf%lf%lf", &p[0], &p[1], &p[2]);
  fscanf(fp, "%lf%lf%lf", &d[0], &d[1], &d[2]);
  fscanf(fp, "%lf%lf%lf", &n[0], &n[1], &n[2]);
  fscanf(fp, "%lf%lf%lf", &c[0], &c[1], &c[2]);
  fscanf(fp, "%lf%lf%u%d%d%lf", &radius, &salience, &f, &generator_type, &primitive_marker, &distinction);
  for (int k = 0; k < 3; k++) fscanf(fp, "%d", &dummy);
  descriptor.ReadAscii(fp);
  flags.Reset(f);
    
  // Check reconstruction index
  assert(((r == -1) && !reconstruction) || (r == reconstruction_index));
  
  // Set vectors
  position.Reset(p[0], p[1], p[2]);
  direction.Reset(d[0], d[1], d[2]);
  normal.Reset(n[0], n[1], n[2]);
  color.Reset(c[0], c[1], c[2]);
  
  // Insert into shape
  if (reconstruction) {
    if ((s >= 0) && (s < reconstruction->NShapes())) {
      FETShape *shape = reconstruction->Shape(s);
      shape->InsertFeature(this);
    }
  }

  // Return success
  return 1;
}


int FETFeature::
WriteAscii(FILE *fp) const
{
  // Write feature
  int dummy = 0;
  int r = reconstruction_index;
  int s = (shape) ? shape->reconstruction_index : -1;
  unsigned int f = flags;
  fprintf(fp, "%d ", r);
  fprintf(fp, "%d ", s);
  fprintf(fp, "%d ", shape_type);
  fprintf(fp, "%g %g %g ", position[0], position[1], position[2]);
  fprintf(fp, "%g %g %g ", direction[0], direction[1], direction[2]);
  fprintf(fp, "%g %g %g ", normal[0], normal[1], normal[2]);
  fprintf(fp, "%g %g %g ", color[0], color[1], color[2]);
  fprintf(fp, "%g %g %u %d %d %g ", radius, salience, f, generator_type, primitive_marker, distinction);
  for (int k = 0; k < 3; k++) fprintf(fp, "%d ", dummy);
  descriptor.WriteAscii(fp);

  // Return success
  return 1;
}

  

int FETFeature::
ReadBinary(FILE *fp)
{
  // Read feature
  int r, s, dummy;
  RNScalar p[3], d[3], n[3], c[3];
  fread(&r, sizeof(int), 1, fp);
  fread(&s, sizeof(int), 1, fp);
  fread(&shape_type, sizeof(int), 1, fp);
  fread(p, sizeof(RNCoord), 3, fp);
  fread(d, sizeof(RNCoord), 3, fp);
  fread(n, sizeof(RNCoord), 3, fp);
  fread(c, sizeof(RNCoord), 3, fp);
  fread(&radius, sizeof(RNLength), 1, fp);
  fread(&salience, sizeof(RNScalar), 1, fp);
  fread(&flags, sizeof(RNFlags), 1, fp);
  fread(&generator_type, sizeof(int), 1, fp);
  fread(&primitive_marker, sizeof(int), 1, fp);
  fread(&distinction, sizeof(RNScalar), 1, fp);
  for (int k = 0; k < 14; k++) fread(&dummy, sizeof(int), 1, fp);
  descriptor.ReadBinary(fp);

  // Check reconstruction index
  assert(((r == -1) && !reconstruction) || (r == reconstruction_index));

  // Set vectors
  position.Reset(p[0], p[1], p[2]);
  direction.Reset(d[0], d[1], d[2]);
  normal.Reset(n[0], n[1], n[2]);
  color.Reset(c[0], c[1], c[2]);
  
  // Insert into shape
  if (reconstruction) {
    if ((s >= 0) && (s < reconstruction->NShapes())) {
      FETShape *shape = reconstruction->Shape(s);
      shape->InsertFeature(this);
    }
  }

  // Return success
  return 1;
}


int FETFeature::
WriteBinary(FILE *fp) const
{
  // Write feature
  int dummy = 0;
  int r = reconstruction_index;
  int s = (shape) ? shape->reconstruction_index : -1;
  fwrite(&r, sizeof(int), 1, fp);
  fwrite(&s, sizeof(int), 1, fp);
  fwrite(&shape_type, sizeof(int), 1, fp);
  fwrite(position.Coords(), sizeof(RNCoord), 3, fp);
  fwrite(direction.Coords(), sizeof(RNCoord), 3, fp);
  fwrite(normal.Coords(), sizeof(RNCoord), 3, fp);
  fwrite(color.Coords(), sizeof(RNCoord), 3, fp);
  fwrite(&radius, sizeof(RNLength), 1, fp);
  fwrite(&salience, sizeof(RNScalar), 1, fp);
  fwrite(&flags, sizeof(RNFlags), 1, fp);
  fwrite(&generator_type, sizeof(int), 1, fp);
  fwrite(&primitive_marker, sizeof(int), 1, fp);
  fwrite(&distinction, sizeof(RNScalar), 1, fp);
  for (int k = 0; k < 14; k++) fwrite(&dummy, sizeof(int), 1, fp);
  descriptor.WriteBinary(fp);

  // Return success
  return 1;
}

  

////////////////////////////////////////////////////////////////////////
// Compatibility 
////////////////////////////////////////////////////////////////////////

FETCompatibilityParameters::
FETCompatibilityParameters(
  RNLength max_euclidean_distance, 
  RNLength *max_descriptor_distances, 
  RNAngle max_normal_angle, 
  RNScalar min_distinction, 
  RNScalar min_salience, 
  RNBoolean discard_boundaries)
  : min_distinction(min_distinction),
    min_salience(min_salience),
    discard_boundaries(discard_boundaries)
{
  if (max_euclidean_distance == RN_UNKNOWN) max_euclidean_distance_squared = RN_UNKNOWN;
  else max_euclidean_distance_squared = max_euclidean_distance * max_euclidean_distance;

  if (max_normal_angle == RN_UNKNOWN) min_normal_dot_product = RN_UNKNOWN;
  else min_normal_dot_product = cos(max_normal_angle);

  for (int i = 0; i < NUM_FEATURE_TYPES; i++) {
    if (!max_descriptor_distances || (max_descriptor_distances[i] == RN_UNKNOWN)) max_descriptor_distance_squared[i] = RN_UNKNOWN;
    else max_descriptor_distance_squared[i] = max_descriptor_distances[i] * max_descriptor_distances[i];
  }
}



int 
AreFeaturesCompatible(FETFeature *feature1, FETFeature *feature2, void *data)
{
  // Get compatibility data
  FETCompatibilityParameters *compatibility = (FETCompatibilityParameters *) data;
  if (!compatibility) return 1;

  // Check if feature2 is on boundary (allow matching to boundaries if they will be discarded)
  if (compatibility->discard_boundaries) {
    if (feature2->flags & FEATURE_IS_ON_BOUNDARY) return 1;
  }

  // Check shape types
  if (feature1->ShapeType() != feature2->ShapeType()) return 0;

  // Check generator types
  if (((feature1->GeneratorType() == SILHOUETTE_FEATURE_TYPE) && (feature2->GeneratorType() == RIDGE_FEATURE_TYPE)) ||
      ((feature1->GeneratorType() == RIDGE_FEATURE_TYPE) && (feature2->GeneratorType() == SILHOUETTE_FEATURE_TYPE))) {
    // For ridge to match silhouette, viewing vectors should be "opposite"
    R3Vector vector1 = feature1->Position(TRUE) - feature1->Shape()->Viewpoint();
    R3Vector vector2 = feature2->Position(TRUE) - feature2->Shape()->Viewpoint();
    if (vector1.Dot(vector2) > 0) return 0;
  }
  else {
    // For other features, must be same generator type
    if (feature1->GeneratorType() != feature2->GeneratorType()) return 0;
  }

  // Check saliences
  if (compatibility->min_salience != RN_UNKNOWN) {
    if (feature2->Salience() < compatibility->min_salience) return 0;
  }

  // Check orientations
  if (compatibility->min_normal_dot_product != RN_UNKNOWN) {
    if ((feature1->ShapeType() == PLANE_FEATURE_SHAPE) && (feature2->ShapeType() == PLANE_FEATURE_SHAPE)) {
      RNScalar normal_dot_product = feature1->normal.Dot(feature2->normal);
      if (normal_dot_product < compatibility->min_normal_dot_product) return 0;
    }
    else if ((feature1->ShapeType() == LINE_FEATURE_SHAPE) && (feature2->ShapeType() == LINE_FEATURE_SHAPE)) {
      RNScalar direction_dot_product = fabs(feature1->direction.Dot(feature2->direction));
      if (direction_dot_product < compatibility->min_normal_dot_product) return 0;
    }
  }

  // Check descriptors
  int t = feature1->GeneratorType();
  if (t == feature2->GeneratorType()) {
    if (compatibility->max_descriptor_distance_squared[t] != RN_UNKNOWN) {
      RNScalar dd = feature1->descriptor.SquaredDistance(feature2->descriptor);
      if ((dd == RN_UNKNOWN) || (dd > compatibility->max_descriptor_distance_squared[t])) return 0;
    }
  }

#if 0
  // Check distinction
  if (compatibility->min_distinction != RN_UNKNOWN) {
    RNLength squared_euclidean_distance = R3SquaredDistance(feature1->position, feature2->position);
    RNScalar distinction = Distinction(feature1, feature2, squared_euclidean_distance);
    if ((distinction == RN_UNKNOWN) || (distinction <= compatibility->min_distinction)) return 0;
  }
#endif

  // Passed all tests
  return 1;
}



int
FETCompareFeatures(const void *data1, const void *data2)
{
  FETFeature *feature1 = *((FETFeature **) data1);
  FETFeature *feature2 = *((FETFeature **) data2);
  if (feature1->salience > feature2->salience) return -1;
  else if (feature1->salience < feature2->salience) return 1;
  else return 0;
}

