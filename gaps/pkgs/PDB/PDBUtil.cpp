// Source file for PDB utilities



// Include files

#include "PDB.h"



R3Point 
PDBCentroid(const RNArray<PDBAtom *>& atoms)
{
  // Compute center of mass
  R3Point centroid(0.0, 0.0, 0.0);
  for (int i = 0; i < atoms.NEntries(); i++) 
    centroid += atoms[i]->Position();
  centroid /= atoms.NEntries();

  // Return center of mass
  return centroid;
}



R3Box
PDBBox(const RNArray<PDBAtom *>& atoms)
{
  // Compute bounding box
  R3Box bbox(R3null_box);
  for (int i = 0; i < atoms.NEntries(); i++) 
    bbox.Union(atoms[i]->BBox());

  // Return bounding box
  return bbox;
}



RNLength
PDBMaxDistance(const RNArray<PDBAtom *>& atoms, const R3Point& center)
{
  // Compute average distance between a position on the surface and a center point
  RNLength max_distance = 0.0;
  for (int i = 0; i < atoms.NEntries(); i++) {
    RNLength distance = R3Distance(atoms[i]->Position(), center);
    if (distance > max_distance) max_distance = distance;
  }

  // Return maximum distance
  return max_distance;
}



RNLength
PDBAverageDistance(const RNArray<PDBAtom *>& atoms, const R3Point& center)
{
  // Compute average distance between the atoms and a center point
  RNLength distance = 0.0;
  for (int i = 0; i < atoms.NEntries(); i++) 
    distance += R3Distance(atoms[i]->Position(), center);
  distance /= atoms.NEntries();

  // Return average distance
  return distance;
}



R3Triad 
PDBPrincipleAxes(const RNArray<PDBAtom *>& atoms, const R3Point& centroid) 
{
  // Compute covariance matrix
  RNScalar m[9] = { 0 };
  for (int i = 0; i < atoms.NEntries(); i++) {
    RNScalar x = atoms[i]->Position().X() - centroid.X();
    RNScalar y = atoms[i]->Position().Y() - centroid.Y();
    RNScalar z = atoms[i]->Position().Z() - centroid.Z();
    m[0] += x*x;
    m[4] += y*y;
    m[8] += z*z;
    m[1] += x*y;
    m[3] += x*y;
    m[2] += x*z;
    m[6] += x*z;
    m[5] += y*z;
    m[7] += y*z;
  }

  // Normalize covariance matrix
  for (int i = 0; i < 9; i++) {
    m[i] /= atoms.NEntries();
  }

  // Calculate SVD of second order moments
  RNScalar U[9];
  RNScalar W[3];
  RNScalar Vt[9];
  RNSvdDecompose(3, 3, m, U, W, Vt);  // m == U . DiagonalMatrix(W) . Vt

  // Principle axes are in Vt
  R3Vector axes[3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      axes[i][j] = Vt[3*i+j];
    }
  }
  
  // Normalize all axis vectors
  RNLength length0 = axes[0].Length();
  RNLength length1 = axes[1].Length();
  RNLength length2 = axes[2].Length();
  if (RNIsPositive(length0)) axes[0] /= length0;
  if (RNIsPositive(length1)) axes[1] /= length1;
  if (RNIsPositive(length2)) axes[2] /= length2;

#if 0
  // Compute which side of axes are "heavier" 
  RNScalar sum[3] = { 0, 0, 0 };
  for (int i = 0; i < atoms.NEntries(); i++) {
    for (int j = 0; j < 3; j++) {
      RNScalar dot = axes[j].Dot(atoms[i]->Position() - centroid);
      RNScalar dot_squared = dot * dot;
      if (dot > 0.0) { sum[j] += dot_squared * atoms[i]->Charge(); positive_count[j]++; }
      else { sum[j] -= dot_squared * atoms[i]->Charge(); negative_count[j]++; }
    }
  }

  // Flip axes so that "heavier" on positive side
  for (int j =0; j < 3; j++) {
    if (sum[j] < 0) {
      axes[j].Flip();
    }
  }

  // Compute orthonormal triad of axes
  if (length0 > length1) {
    if (length1 > length2) {
      if (RNIsPositive(length1)) {
        axes[2] = axes[0] % axes[1];
      }
    }
    else {
      if (RNIsPositive(length0) && RNIsPositive(length2) ) {
        axes[1] = axes[2] % axes[0];
      }
    }
  }
  else {
    if (length0 > length2) {
      if (RNIsPositive(length0)) {
        axes[2] = axes[0] % axes[1];
      }
    }
    else {
      if (RNIsPositive(length1) && RNIsPositive(length2) ) {
        axes[0] = axes[1] % axes[2];
      }
    }
  }
#else
  // Count atoms on each side of axes for each element type
  RNScalar sum[3][6] = { { 0 } };
  for (int a = 0; a < atoms.NEntries(); a++) {
    PDBAtom *atom = atoms[a];
    PDBElement *element = atom->Element();
    if (!element) continue;
    int j = element - PDBelements;
    if (j >= 6) continue;
    for (int i = 0; i < 3; i++) {
      RNScalar dot = axes[i].Dot(atom->Position() - centroid);
      if (dot > 0.0) { sum[i][j] += dot * dot; }
      else { sum[i][j] -= dot * dot; }
    }
  }

#if 1
  // Find element type and axis dimension with greatest abs(sum)
  int best_j[3] = { 0, 0, 0 };
  RNScalar best_sum[3] = { 0, 0, 0 };
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 6; j++) {
      if (fabs(sum[i][j]) > fabs(best_sum[i])) {
        best_sum[i] = sum[i][j];
        best_j[i] = j;
      }
    }
  }

#else
  // Find element type and axis dimension with greatest abs(sum)
  int best_j[3];
  RNScalar best_sum[3] = { 0, 0, 0 };
  for (int i = 0; i < 3; i++) {
    for (int j = 5; j >= 0; j--) {
      if (RNIsZero(sum[i][j], 1.0)) continue;
      if (RNIsNegative(sum[i][j], 1.0)) axes[i].Flip(); 
      best_sum[i] = sum[i][j];
      best_j[i] = j;
      break;
    }
  }
#endif

  // Flip axes so that "heavier" on positive side
  RNDimension least_i = 0;
  RNScalar least_abs_best_sum = FLT_MAX;
  for (int i =0; i < 3; i++) {
    if (best_j[i] >= 0) {
      if (best_sum[i] < 0) axes[i].Flip();
      RNScalar abs_best_sum = fabs(best_sum[i]);
      if (abs_best_sum  < least_abs_best_sum) {
        least_abs_best_sum = abs_best_sum;
        least_i = i;
      }
    }
  }


  // Make orthonormal triad
  axes[least_i] = axes[(least_i+1)%3] % axes[(least_i+2)%3];
#endif

  // Return triad of axes
  return R3Triad(axes[0], axes[1], axes[2]);
}



R3Affine
PDBAlignmentTransformation(const RNArray<PDBAtom *>& atoms, RNBoolean align_translation, RNBoolean align_rotation, RNBoolean align_scale)
{
  // Start with identity affine
  R3Affine affine(R3identity_affine);

  // Compute center of mass
  R3Point centroid = PDBCentroid(atoms);

  // Move origin back to center of mass
  if (!align_translation) {
    affine.Translate(centroid.Vector());
  }

  // Scale object by inverse of average distance to center of mass
  if (align_scale) {
    RNScalar scale = PDBAverageDistance(atoms, centroid);
    if (RNIsPositive(scale)) affine.Scale(1.0 / scale);
  }

  // Apply rotation that orients principal axes of object with cartesian axes
  if (align_rotation) {
    R3Triad triad = PDBPrincipleAxes(atoms, centroid);
    affine.Transform(R3Affine(triad.InverseMatrix()));
  }

  // Move center of mass to origin
  affine.Translate(-(centroid.Vector()));
  
  // Return normalization transformation
  return affine;
}



static R4Matrix
SetQuaternion(RNScalar p[4])
{
  R4Matrix m(R4identity_matrix);
  RNScalar l;

  if(p[0]<0){
    p[0]=-p[0];
    p[1]=-p[1];
    p[2]=-p[2];
    p[3]=-p[3];
  }
  l=p[0]*p[0]+p[1]*p[1]+p[2]*p[2]+p[3]*p[3];
  if(l<.000001){return R4identity_matrix;}

  l=sqrt(l);
  p[0]/=l;
  p[1]/=l;
  p[2]/=l;
  p[3]/=l;

  m[0][0]=p[0]*p[0]+p[1]*p[1]-p[2]*p[2]-p[3]*p[3];
  m[0][1]=2*(p[1]*p[2]+p[0]*p[3]);
  m[0][2]=2*(p[1]*p[3]-p[0]*p[2]);

  m[1][0]=2*(p[1]*p[2]-p[0]*p[3]);
  m[1][1]=p[0]*p[0]+p[2]*p[2]-p[1]*p[1]-p[3]*p[3];
  m[1][2]=2*(p[2]*p[3]+p[0]*p[1]);

  m[2][0]=2*(p[1]*p[3]+p[0]*p[2]);
  m[2][1]=2*(p[2]*p[3]-p[0]*p[1]);
  m[2][2]=p[0]*p[0]+p[3]*p[3]-p[1]*p[1]-p[2]*p[2];

  return m;
}



static RNScalar 
ComputeError(const RNArray<PDBAtom *>& atoms1, const RNArray<PDBAtom *>& atoms2, RNScalar* weights, 
  const R4Matrix& rotation, const R3Point& center1, const R3Point& center2, RNScalar s1, RNScalar s2)
{
  // Get number of atoms
  int count = atoms1.NEntries();
  if (atoms2.NEntries() < count) 
    count = atoms2.NEntries();

  RNScalar error = 0;
  for(int i=0;i<count;i++){
    RNScalar w  = (weights) ? weights[i] * weights[i] : 1;
    R3Vector v1= (atoms1[i]->Position()-center1)/s1;
    R3Vector v2= (atoms2[i]->Position()-center2)/s2;
    R3Vector v = v1 - rotation * v2;
    error += v.Dot(v) * w;
  }
  return error;
}



R3Affine
PDBAlignmentTransformation(const RNArray<PDBAtom *>& atoms2, const RNArray<PDBAtom *>& atoms1, RNScalar* weights, 
  RNBoolean align_translation, RNBoolean align_rotation, RNBoolean align_scale)
{
  int i,j,k;

  // Get number of atoms
  int count = atoms1.NEntries();
  if (atoms2.NEntries() < count) 
    count = atoms2.NEntries();

  // Check alignment flags
  if (count < 1) align_translation = 0;
  if (count < 2) align_scale = 0;
  if (count < 3) align_rotation = 0;

  // Compute centers
  R3Point center1(0.0, 0.0, 0.0);
  R3Point center2(0.0, 0.0, 0.0);
  if (align_translation){
    center1 = PDBCentroid(atoms1);
    center2 = PDBCentroid(atoms2);
  }

  // Compute scales
  RNScalar s1 = 1;
  RNScalar s2 = 1;
  if (align_scale){
    s1 = PDBAverageDistance(atoms1, center1);
    s2 = PDBAverageDistance(atoms2, center2);
  }

  // Compute cross-covariance of two point sets
  R4Matrix rotation = R4identity_matrix;
  if (align_rotation) {
    R4Matrix m = R4identity_matrix;
    m[0][0] = m[1][1] = m[2][2] = 0;
    for (i=0; i< count; i++){
      R3Vector p1 = (atoms1[i]->Position() - center1) / s1;
      R3Vector p2 = (atoms2[i]->Position() - center2) / s2;
      RNScalar w  = (weights) ? weights[i] * weights[i] : 1;
      for(j=0;j<3;j++){
        for(k=0;k<3;k++){
          m[j][k]+=p1[j]*p2[k]*w;
        }
      }
    }
    for(j=0;j<3;j++){for(k=0;k<3;k++){m[j][k] /= count;}}

    // Make cross-covariance matrix skew-symmetric
    R4Matrix a = R4identity_matrix;
    for(j=0;j<3;j++){for(k=0;k<3;k++){a[j][k]=m[j][k]-m[k][j];}}
    
    // Compute trace of cross-covariance matrix
    RNScalar trace=m[0][0]+m[1][1]+m[2][2];
    
    // Setup symmetric matrix whose eigenvectors give quaternion terms of optimal rotation
    RNScalar M[16];
    M[0]=trace;
    M[1]=M[4]=a[1][2];
    M[2]=M[8]=a[2][0];
    M[3]=M[12]=a[0][1];
    for(j=0;j<3;j++){
      for(k=0;k<3;k++){M[4*(j+1)+(k+1)]=m[j][k]+m[k][j];}
      M[4*(j+1)+(j+1)]-=trace;
    }

    // Perform SVD to get eigenvectors (quaternion terms of optimal rotation)
    RNScalar U[16];
    RNScalar W[4];
    RNScalar Vt[16];
    RNSvdDecompose(4, 4, M, U, W, Vt);  
    
    // Look at error using all eigenvectors and keep best
    int minI=0;
    R4Matrix temp[4];
    RNScalar e[4];
    for(i=0;i<4;i++){
      RNScalar p[4];
      for(j=0;j<4;j++){p[j]=U[4*j+i];}
      if(p[0]<0){for(j=0;j<4;j++){p[j]=-p[j];}}
      temp[i] = SetQuaternion(p);
      e[i]= ComputeError(atoms1, atoms2, weights, temp[i], center1, center2, s1, s2);
      if (e[i]<e[minI]) minI=i;
    }
    rotation = temp[minI];
  }

  // Compute result
  R3Affine result = R3identity_affine;
  if (align_translation) result.Translate(center1.Vector());
  else result.Translate(center2.Vector());
  if (align_scale) result.Scale(s1/s2);
  if (align_rotation) result.Transform(R3Affine(rotation));
  result.Translate(-(center2.Vector()));

  // Return transformation that takes atoms2 to atoms1
  // Note the reversal in order of atoms2 and atoms1 in arguments
  return result;
}



RNScalar 
PDBMeanAlignmentError(const RNArray<PDBAtom *>& atoms1, const RNArray<PDBAtom *>& atoms2, RNScalar* weights, const R3Affine& affine)
{
  // Get number of atoms
  int count = atoms1.NEntries();
  if (atoms2.NEntries() < count) 
    count = atoms2.NEntries();

  // Get sum of squared errors
  RNScalar total_squared_error = 0;
  RNScalar total_weight = 0;
  for(int i=0;i<count;i++){
    RNScalar w  = (weights) ? weights[i] * weights[i] : 1;
    R3Point p1 = atoms1[i]->Position();
    R3Point p2 = atoms2[i]->Position();
    p1.Transform(affine);
    R3Vector v = p1 - p2;
    total_squared_error += w * v.Dot(v);
    total_weight += w;
  }

  // Return average error
  if (RNIsZero(total_weight)) return 0.0;
  else return sqrt(total_squared_error / total_weight);
}



R3Affine PDBAlignmentTransformation(const PDBModel *model, 
  RNBoolean align_translation, RNBoolean align_rotation, RNBoolean align_scale)
{ 
  // Return transformation that aligns atoms of model onto canonical coordinate system
  return PDBAlignmentTransformation(model->atoms, align_translation, align_rotation, align_scale);
}



R3Affine PDBAlignmentTransformation(const PDBChain *chain, 
  RNBoolean align_translation, RNBoolean align_rotation, RNBoolean align_scale)
{ 
  // Return transformation that aligns atoms of chain onto canonical coordinate system
  return PDBAlignmentTransformation(chain->atoms, align_translation, align_rotation, align_scale);
}



R3Affine PDBAlignmentTransformation(const PDBResidue *residue, 
  RNBoolean align_translation, RNBoolean align_rotation, RNBoolean align_scale)
{ 
  // Return transformation that aligns atoms of residue onto canonical coordinate system
  return PDBAlignmentTransformation(residue->atoms, align_translation, align_rotation, align_scale);
}



void PDBAlignModel(PDBModel *model, 
  RNBoolean align_translation, RNBoolean align_rotation, RNBoolean align_scale) 
{ 
  // Compute and apply transformation that aligns atoms of model onto canoncial coordinate system
  R3Affine affine = PDBAlignmentTransformation(model, align_translation, align_rotation, align_scale);
  model->Transform(affine);
}



void PDBAlignChain(PDBChain *chain, 
  RNBoolean align_translation, RNBoolean align_rotation, RNBoolean align_scale) 
{ 
  // Compute and apply transformation that aligns atoms of chain onto canoncial coordinate system
  R3Affine affine = PDBAlignmentTransformation(chain, align_translation, align_rotation, align_scale);
  chain->Transform(affine);
}



void PDBAlignResidue(PDBResidue *residue, 
  RNBoolean align_translation, RNBoolean align_rotation, RNBoolean align_scale) 
{ 
  // Compute and apply transformation that aligns atoms of residue onto canoncial coordinate system
  R3Affine affine = PDBAlignmentTransformation(residue, align_translation, align_rotation, align_scale);
  residue->Transform(affine);
}



R3Affine PDBAlignmentTransformation(const PDBModel *model1, const PDBModel *model2, 
  RNBoolean align_translation, RNBoolean align_rotation, RNBoolean align_scale) 
{ 
  // Return transformation that aligns atoms of model1 onto those of model2
  return PDBAlignmentTransformation(model1->atoms, model2->atoms, NULL, align_translation, align_rotation, align_scale);
}



R3Affine PDBAlignmentTransformation(const PDBChain *chain1, const PDBChain *chain2, 
  RNBoolean align_translation, RNBoolean align_rotation, RNBoolean align_scale) 
{ 
  // Return transformation that aligns atoms of chain1 onto those of chain2
  return PDBAlignmentTransformation(chain1->atoms, chain2->atoms, NULL, align_translation, align_rotation, align_scale);
}



R3Affine PDBAlignmentTransformation(const PDBResidue *residue1, const PDBResidue *residue2, 
  RNBoolean align_translation, RNBoolean align_rotation, RNBoolean align_scale) 
{ 
  // Return transformation that aligns atoms of residue1 onto those of residue2
  return PDBAlignmentTransformation(residue1->atoms, residue2->atoms, NULL, align_translation, align_rotation, align_scale);
}



void PDBAlignModel(PDBModel *model1, const PDBModel *model2,
  RNBoolean align_translation, RNBoolean align_rotation, RNBoolean align_scale) 
{ 
  // Compute and apply transformation that aligns atoms of model1 onto those of model2
  R3Affine affine = PDBAlignmentTransformation(model1, model2, align_translation, align_rotation, align_scale);
  model1->Transform(affine);
}



void PDBAlignChain(PDBChain *chain1, const PDBChain *chain2, 
  RNBoolean align_translation, RNBoolean align_rotation, RNBoolean align_scale) 
{ 
  // Compute and apply transformation that aligns atoms of chain1 onto those of chain2
  R3Affine affine = PDBAlignmentTransformation(chain1, chain2, align_translation, align_rotation, align_scale);
  chain1->Transform(affine);
}



void PDBAlignResidue(PDBResidue *residue1, const PDBResidue *residue2, 
  RNBoolean align_translation, RNBoolean align_rotation, RNBoolean align_scale) 
{ 
  // Compute and apply transformation that aligns atoms of residue1 onto those of residue2
  R3Affine affine = PDBAlignmentTransformation(residue1, residue2, align_translation, align_rotation, align_scale);
  residue1->Transform(affine);
}






