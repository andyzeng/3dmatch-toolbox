////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "FET.h"



////////////////////////////////////////////////////////////////////////
// Constructors
////////////////////////////////////////////////////////////////////////

FETMatch::
FETMatch(FETReconstruction *reconstruction,
  FETShape *shape1, FETShape *shape2,
  const R3Affine& transformation21,
  RNScalar affinity)
  : reconstruction(NULL),
    reconstruction_index(-1),
    current_transformation(transformation21),
    initial_transformation(transformation21),
    ground_truth_transformation(R3identity_affine),
    affinity(affinity)
{
  // Initialize shapes
  shapes[0] = shapes[1] = NULL;
  shape_indices[0] = shape_indices[1] = -1;
  
  // Insert into reconstruction
  if (reconstruction) reconstruction->InsertMatch(this);

  // Insert into shapes
  if (shape1) shape1->InsertMatch(this, 0);
  if (shape2) shape2->InsertMatch(this, 1);
}



FETMatch::
FETMatch(const FETMatch& match)
  : reconstruction(NULL),
    reconstruction_index(-1),
    current_transformation(match.current_transformation),
    initial_transformation(match.initial_transformation),
    ground_truth_transformation(match.ground_truth_transformation),
    affinity(match.affinity)
{
  // Initialize shapes
  shapes[0] = shapes[1] = NULL;
  shape_indices[0] = shape_indices[1] = -1;

#if 0
  // Insert into reconstruction
  if (match.reconstruction) match.reconstruction->InsertMatch(this);

  // Insert into shapes
  if (match.shapes[0]) match.shapes[0]->InsertMatch(this, 0);
  if (match.shapes[1]) match.shapes[1]->InsertMatch(this, 1);
#endif
}



FETMatch::
~FETMatch(void)
{
  // Remove correspondences
  while (NCorrespondences() > 0) {
    FETCorrespondence *correspondence = Correspondence(NCorrespondences()-1);
    RemoveCorrespondence(correspondence);
  }
   
  // Remove from shapes
  if (shapes[0]) shapes[0]->RemoveMatch(this, 0);
  if (shapes[1]) shapes[1]->RemoveMatch(this, 1);

  // Remove from reconstruction
  if (reconstruction) reconstruction->RemoveMatch(this);
}



////////////////////////////////////////////////////////////////////////
// FETMatch manipulation
////////////////////////////////////////////////////////////////////////
  
void FETMatch::
InsertCorrespondence(FETCorrespondence *correspondence)
{
  // Just checking
  assert(correspondence->reconstruction == this->reconstruction);
  assert(correspondence->match_index == -1);
  assert(correspondence->match == NULL);

  // Insert correspondence
  correspondence->match = this;
  correspondence->match_index = correspondences.NEntries();
  correspondences.Insert(correspondence);
}



void FETMatch::
RemoveCorrespondence(FETCorrespondence *correspondence)
{
  // Just checking
  assert(correspondence->reconstruction == this->reconstruction);
  assert(correspondence->match_index >= 0);
  assert(correspondence->match_index < correspondences.NEntries());
  assert(correspondence->match == this);

  // Remove correspondence
  RNArrayEntry *entry = correspondences.KthEntry(correspondence->match_index);
  FETCorrespondence *tail = correspondences.Tail();
  tail->match_index = correspondence->match_index;
  correspondences.EntryContents(entry) = tail;
  correspondences.RemoveTail();
  correspondence->match_index = -1;
  correspondence->match = NULL;
}



////////////////////////////////////////////////////////////////////////
// Input / output functions
////////////////////////////////////////////////////////////////////////

int FETMatch::
ReadAscii(FILE *fp)
{
  // Read match
  int dummy = 0;
  int r, s0, s1, ncorrespondences;
  RNScalar current_matrix[16], ground_truth_matrix[16];
  fscanf(fp, "%d", &r);
  fscanf(fp, "%d", &s0);
  fscanf(fp, "%d", &s1);
  fscanf(fp, "%d", &ncorrespondences);
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
  fscanf(fp, "%lf", &affinity);
  for (int k = 0; k < 4; k++) fscanf(fp, "%d", &dummy);

  // Check reconstruction index
  assert(((r == -1) && !reconstruction) || (r == reconstruction_index));
  
  // Insert into shapes
  if (reconstruction) {
    if (s0 >= 0) {
      FETShape *shape0 = reconstruction->Shape(s0);
      shape0->InsertMatch(this, 0);
    }
    if (s1 >= 0) {
      FETShape *shape1 = reconstruction->Shape(s1);
      shape1->InsertMatch(this, 1);
    }
  }

  // Assign transformation
  current_transformation.Reset(R4Matrix(current_matrix), 0);
  initial_transformation.Reset(R4Matrix(current_matrix), 0);
  ground_truth_transformation.Reset(R4Matrix(ground_truth_matrix), 0);

  // Return success
  return 1;
}



int FETMatch::
WriteAscii(FILE *fp) const
{
  // Write match
  int dummy = 0;
  int r = reconstruction_index;
  int s0 = (shapes[0]) ? shapes[0]->reconstruction_index : -1;
  int s1 = (shapes[1]) ? shapes[1]->reconstruction_index : -1;
  int ncorrespondences = NCorrespondences();
  R4Matrix current_matrix = current_transformation.Matrix();
  R4Matrix ground_truth_matrix = ground_truth_transformation.Matrix();
  fprintf(fp, "%d ", r);
  fprintf(fp, "%d ", s0);
  fprintf(fp, "%d ", s1);
  fprintf(fp, "%d ", ncorrespondences);
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
  fprintf(fp, "%g", affinity);
  for (int k = 0; k < 4; k++) fprintf(fp, "%d ", dummy);
  
  // Return success
  return 1;
}

  

int FETMatch::
ReadBinary(FILE *fp)
{
  // Read match
  int dummy = 0;
  int r, s0, s1, ncorrespondences;
  RNScalar current_matrix[16], ground_truth_matrix[16];
  fread(&r, sizeof(int), 1, fp);
  fread(&s0, sizeof(int), 1, fp);
  fread(&s1, sizeof(int), 1, fp);
  fread(&ncorrespondences, sizeof(int), 1, fp);
  fread(&current_matrix[0], sizeof(RNScalar), 16, fp);
  fread(&ground_truth_matrix[0], sizeof(RNScalar), 16, fp);
  fread(&affinity, sizeof(RNScalar), 1, fp);
  for (int k = 0; k < 16; k++) fread(&dummy, sizeof(int), 1, fp);

  // Check reconstruction index
  assert(((r == -1) && !reconstruction) || (r == reconstruction_index));
  
  // Insert into shapes
  if (reconstruction) {
    if (s0 >= 0) {
      FETShape *shape0 = reconstruction->Shape(s0);
      shape0->InsertMatch(this, 0);
    }
    if (s1 >= 0) {
      FETShape *shape1 = reconstruction->Shape(s1);
      shape1->InsertMatch(this, 1);
    }
  }

  // Assign transformation
  current_transformation.Reset(R4Matrix(current_matrix), 0);
  initial_transformation.Reset(R4Matrix(current_matrix), 0);
  ground_truth_transformation.Reset(R4Matrix(ground_truth_matrix), 0);

  // Return success
  return 1;
}



int FETMatch::
WriteBinary(FILE *fp) const
{
  // Write match
  int dummy = 0;
  int s0 = (shapes[0]) ? shapes[0]->reconstruction_index : -1;
  int s1 = (shapes[1]) ? shapes[1]->reconstruction_index : -1;
  int ncorrespondences = NCorrespondences();
  R4Matrix current_matrix = current_transformation.Matrix();
  R4Matrix ground_truth_matrix = ground_truth_transformation.Matrix();
  fwrite(&reconstruction_index, sizeof(int), 1, fp);
  fwrite(&s0, sizeof(int), 1, fp);
  fwrite(&s1, sizeof(int), 1, fp);
  fwrite(&ncorrespondences, sizeof(int), 1, fp);
  fwrite(&current_matrix[0][0], sizeof(RNScalar), 16, fp);
  fwrite(&ground_truth_matrix[0][0], sizeof(RNScalar), 16, fp);
  fwrite(&affinity, sizeof(RNScalar), 1, fp);
  for (int k = 0; k < 16; k++) fwrite(&dummy, sizeof(int), 1, fp);

  // Return success
  return 1;
}

  

