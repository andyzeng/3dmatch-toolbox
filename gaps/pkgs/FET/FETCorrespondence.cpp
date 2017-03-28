////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "FET.h"



////////////////////////////////////////////////////////////////////////
// FETCorrespondence member functions
////////////////////////////////////////////////////////////////////////

FETCorrespondence::
FETCorrespondence(FETReconstruction *reconstruction,
  FETFeature *feature1, FETFeature *feature2,
  RNScalar affinity, int relationship_type)
  : reconstruction(NULL),
    reconstruction_index(-1),
    match(NULL),
    match_index(-1),
    affinity(affinity),
    relationship_type(relationship_type)
{
  // Initialize features
  features[0] = features[1] = NULL;
  feature_indices[0] = feature_indices[1] = -1;

  // Insert into reconstruction
  if (reconstruction) reconstruction->InsertCorrespondence(this);

  // Insert into features
  if (feature1) feature1->InsertCorrespondence(this, 0);
  if (feature2) feature2->InsertCorrespondence(this, 1);
}



FETCorrespondence::
FETCorrespondence(const FETCorrespondence& correspondence)
  : reconstruction(NULL),
    reconstruction_index(-1),
    match(NULL),
    match_index(-1),
    affinity(correspondence.affinity),
    relationship_type(correspondence.relationship_type)
{
  // Initialize features
  features[0] = features[1] = NULL;
  feature_indices[0] = feature_indices[1] = -1;

#if 0
  // Insert into reconstruction
  if (correspondence.reconstruction) correspondence.reconstruction->InsertCorrespondence(this);

  // Insert into features
  if (correspondence.features[0]) correspondence.features[0]->InsertCorrespondence(this, 0);
  if (correspondence.features[1]) correspondence.features[1]->InsertCorrespondence(this, 1);
#endif
}



FETCorrespondence::
~FETCorrespondence(void)
{
  // Remove from features
  if (features[0]) features[0]->RemoveCorrespondence(this, 0);
  if (features[1]) features[1]->RemoveCorrespondence(this, 1);
  
  // Remove from match
  if (match) match->RemoveCorrespondence(this);

  // Remove from reconstruction
  if (reconstruction) reconstruction->RemoveCorrespondence(this);
}



RNLength FETCorrespondence::
EuclideanDistance(void) const
{
  // Check stuff
  if (!features[0]) return RN_UNKNOWN;
  if (!features[1]) return RN_UNKNOWN;
  if (relationship_type != COINCIDENT_RELATIONSHIP) return RN_UNKNOWN;
  FETShape *shape0 = features[0]->shape;
  if (!shape0) return RN_UNKNOWN;
  FETShape *shape1 = features[1]->shape;
  if (!shape1) return RN_UNKNOWN;
  
  // Compute distance from f1->p0
  R3Point position0 = features[0]->Position();
  shape0->Transform(position0);
  shape1->InverseTransform(position0);
  RNScalar d0 = features[1]->EuclideanDistance(position0);

  // Compute distance from f0->p1
  R3Point position1 = features[1]->Position();
  shape1->Transform(position1);
  shape0->InverseTransform(position1);
  RNScalar d1 = features[0]->EuclideanDistance(position1);

  // Return average
  return 0.5 * (d0 + d1);
}



RNLength FETCorrespondence::
SquaredEuclideanDistance(void) const
{
  // Check stuff
  if (!features[0]) return RN_UNKNOWN;
  if (!features[1]) return RN_UNKNOWN;
  if (relationship_type != COINCIDENT_RELATIONSHIP) return RN_UNKNOWN;
  FETShape *shape0 = features[0]->shape;
  if (!shape0) return RN_UNKNOWN;
  FETShape *shape1 = features[1]->shape;
  if (!shape1) return RN_UNKNOWN;
  
  // Compute distance from f1->p0
  R3Point position0 = features[0]->Position();
  shape0->Transform(position0);
  shape1->InverseTransform(position0);
  RNScalar dd0 = features[1]->SquaredEuclideanDistance(position0);

  // Compute distance from f0->p1
  R3Point position1 = features[1]->Position();
  shape1->Transform(position1);
  shape0->InverseTransform(position1);
  RNScalar dd1 = features[0]->SquaredEuclideanDistance(position1);

  // Return average
  return 0.5 * (dd0 + dd1);
}



RNLength FETCorrespondence::
DescriptorDistance(void) const
{
  // Return descriptor distance
  RNScalar dd = SquaredDescriptorDistance();
  if (dd == RN_UNKNOWN) return RN_UNKNOWN;
  else return sqrt(dd);
}



RNLength FETCorrespondence::
SquaredDescriptorDistance(void) const
{
  // Return squared descriptor distance
  if (!features[0]) return RN_UNKNOWN;
  if (!features[1]) return RN_UNKNOWN;
  return features[0]->descriptor.SquaredDistance(features[1]->descriptor);
}



RNAngle FETCorrespondence::
NormalAngle(void) const
{
  // Return angle between normals
  if (!features[0]) return RN_UNKNOWN;
  if (!features[1]) return RN_UNKNOWN;
  FETShape *shape0 = features[0]->shape;
  if (!shape0) return RN_UNKNOWN;
  FETShape *shape1 = features[1]->shape;
  if (!shape1) return RN_UNKNOWN;
  R3Vector normal0 = features[0]->normal;
  R3Vector normal1 = features[1]->normal;
  shape0->Transform(normal0);
  shape1->Transform(normal1);
  RNAngle angle = R3InteriorAngle(normal0, normal1);
  if (relationship_type == PERPENDICULAR_RELATIONSHIP) angle = fabs(RN_PI_OVER_TWO - angle);
  return angle;
}



RNLength FETCorrespondence::
Error(void) const
{
  // Get useful variables
  FETFeature *feature1 = features[0];
  FETFeature *feature2 = features[1];
  if (!feature1 || !feature2) return RN_UNKNOWN;
  FETShape *shape1 = feature1->shape;
  FETShape *shape2 = feature2->shape;
  if (!shape1 || !shape2) return RN_UNKNOWN;
  
  // Get distance on shape1
  R3Point predicted_position, correct_position;
  predicted_position = feature2->position;
  shape2->Transform(predicted_position);
  shape1->InverseTransform(predicted_position);
  correct_position = feature2->position;
  shape2->Transform(correct_position);
  shape1->InverseTransform(correct_position);
  RNLength d1 = R3Distance(predicted_position, correct_position);

  // Get distance on shape2
  predicted_position = feature1->position;
  shape1->Transform(predicted_position);
  shape2->InverseTransform(predicted_position);
  correct_position = feature1->position;
  shape1->Transform(correct_position);
  shape2->InverseTransform(correct_position);
  RNLength d2 = R3Distance(predicted_position, correct_position);

  // Are d1 and d2 always the same?
  // Return average
  return 0.5 * (d1 + d2);
}



////////////////////////////////////////////////////////////////////////
// Input / output functions
////////////////////////////////////////////////////////////////////////

int FETCorrespondence::
ReadAscii(FILE *fp)
{
  // Read correspondence
  int r, m, f0, f1, dummy;
  fscanf(fp, "%d", &r);
  fscanf(fp, "%d", &m);
  fscanf(fp, "%d", &f0);
  fscanf(fp, "%d", &f1);
  fscanf(fp, "%lf", &affinity);
  fscanf(fp, "%d", &relationship_type);
  for (int k = 0; k < 4; k++) fscanf(fp, "%d", &dummy);

  // Check reconstruction index
  assert(((r == -1) && !reconstruction) || (r == reconstruction_index));
  
  // Insert into stuff
  if (reconstruction) {
    // Insert into match
    if (m >= 0) {
      FETMatch *mat = reconstruction->Match(m);
      mat->InsertCorrespondence(this);
    }
  
    // Insert into features
    if (f0 >= 0) {
      FETFeature *feature0 = reconstruction->Feature(f0);
      feature0->InsertCorrespondence(this, 0);
    }
    if (f1 >= 0) {
      FETFeature *feature1 = reconstruction->Feature(f1);
      feature1->InsertCorrespondence(this, 1);
    }
  }

  // Return success
  return 1;
}



int FETCorrespondence::
WriteAscii(FILE *fp) const
{
  // Write correspondence
  int dummy = 0;
  int m = (match) ? (match->reconstruction_index) : -1;
  int f0 = (features[0]) ? features[0]->reconstruction_index : -1;
  int f1 = (features[1]) ? features[1]->reconstruction_index : -1;
  fprintf(fp, "%d ", reconstruction_index);
  fprintf(fp, "%d ", m);
  fprintf(fp, "%d ", f0);
  fprintf(fp, "%d ", f1);
  fprintf(fp, "%g ", affinity);
  fprintf(fp, "%d ", relationship_type);
  for (int k = 0; k < 4; k++) fprintf(fp, "%d ", dummy);

  // Return success
  return 1;
}

  

int FETCorrespondence::
ReadBinary(FILE *fp)
{
  // Read correspondence
  int r, m, f0, f1, dummy;
  fread(&r, sizeof(int), 1, fp);
  fread(&m, sizeof(int), 1, fp);
  fread(&f0, sizeof(int), 1, fp);
  fread(&f1, sizeof(int), 1, fp);
  fread(&affinity, sizeof(RNScalar), 1, fp);
  fread(&relationship_type, sizeof(int), 1, fp);
  for (int k = 0; k < 16; k++) fread(&dummy, sizeof(int), 1, fp);

  // Check reconstruction index
  assert(((r == -1) && !reconstruction) || (r == reconstruction_index));
  
  // Insert into stuff
  if (reconstruction) {
    // Insert into match
    if (m >= 0) {
      FETMatch *match = reconstruction->Match(m);
      match->InsertCorrespondence(this);
    }
  
    // Insert into features
    if (f0 >= 0) {
      FETFeature *feature0 = reconstruction->Feature(f0);
      feature0->InsertCorrespondence(this, 0);
    }
    if (f1 >= 0) {
      FETFeature *feature1 = reconstruction->Feature(f1);
      feature1->InsertCorrespondence(this, 1);
    }
  }

  // Return success
  return 1;
}



int FETCorrespondence::
WriteBinary(FILE *fp) const
{
  // Write correspondence
  int dummy = 0;
  int m = (match) ? (match->reconstruction_index) : -1;
  int f0 = (features[0]) ? features[0]->reconstruction_index : -1;
  int f1 = (features[1]) ? features[1]->reconstruction_index : -1;
  fwrite(&reconstruction_index, sizeof(int), 1, fp);
  fwrite(&m, sizeof(int), 1, fp);
  fwrite(&f0, sizeof(int), 1, fp);
  fwrite(&f1, sizeof(int), 1, fp);
  fwrite(&affinity, sizeof(RNScalar), 1, fp);
  fwrite(&relationship_type, sizeof(int), 1, fp);
  for (int k = 0; k < 16; k++) fwrite(&dummy, sizeof(int), 1, fp);

  // Return success
  return 1;
}



int
FETCompareCorrespondences(const void *data1, const void *data2)
{
  FETCorrespondence *correspondence1 = *((FETCorrespondence **) data1);
  FETCorrespondence *correspondence2 = *((FETCorrespondence **) data2);
  if (correspondence1->affinity > correspondence2->affinity) return -1;
  else if (correspondence1->affinity < correspondence2->affinity) return 1;
  else return 0;
}

