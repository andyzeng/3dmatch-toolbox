/* Source file for the R3 surfel feature class */



////////////////////////////////////////////////////////////////////////
// INCLUDE FILES
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS/DESTRUCTORS
////////////////////////////////////////////////////////////////////////

R3SurfelFeatureVector::
R3SurfelFeatureVector(int nvalues)
  : values(NULL),
    nvalues(nvalues)
{
  // Allocate values
  if (nvalues > 0) {
    values = new RNScalar [ nvalues ];
    for (int i = 0; i < nvalues; i++) {
      values[i] = RN_UNKNOWN;
    }
  }
}



R3SurfelFeatureVector::
R3SurfelFeatureVector(const R3SurfelFeatureVector& vector)
  : nvalues(vector.nvalues)
{
  // Allocate values
  if (nvalues > 0) {
    values = new RNScalar [ nvalues ];
    for (int i = 0; i < nvalues; i++) {
      values[i] = vector.values[i];
    }
  }
}



R3SurfelFeatureVector::
~R3SurfelFeatureVector(void)
{
  // Delete values
  if (values) delete [] values;
}



////////////////////////////////////////////////////////////////////////
// FEATURE MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

R3SurfelFeatureVector& R3SurfelFeatureVector::
operator=(const R3SurfelFeatureVector& vector)
{
  // Reallocate values
  if (nvalues != vector.nvalues) {
    nvalues = vector.nvalues;
    if (values) delete [] values;
    if (nvalues > 0) values = new RNScalar [ nvalues ];
  }

  // Copy values
  for (int i = 0; i < nvalues; i++) {
    values[i] = vector.values[i];
  }

  // Return this
  return *this;
}



R3SurfelFeatureVector& R3SurfelFeatureVector::
operator+=(const R3SurfelFeatureVector& vector)
{
  // Check number of values
  assert(nvalues == vector.nvalues);

  // Add values
  for (int i = 0; i < nvalues; i++) {
    values[i] += vector.values[i];
  }

  // Return this
  return *this;
}



R3SurfelFeatureVector& R3SurfelFeatureVector::
operator-=(const R3SurfelFeatureVector& vector)
{
  // Check number of values
  assert(nvalues == vector.nvalues);

  // Subtract values
  for (int i = 0; i < nvalues; i++) {
    values[i] -= vector.values[i];
  }

  // Return this
  return *this;
}



R3SurfelFeatureVector& R3SurfelFeatureVector::
operator*=(const R3SurfelFeatureVector& vector)
{
  // Check number of values
  assert(nvalues == vector.nvalues);

  // Multiply values
  for (int i = 0; i < nvalues; i++) {
    values[i] *= vector.values[i];
  }

  // Return this
  return *this;
}



R3SurfelFeatureVector& R3SurfelFeatureVector::
operator/=(const R3SurfelFeatureVector& vector)
{
  // Check number of values
  assert(nvalues == vector.nvalues);

  // Divide values
  for (int i = 0; i < nvalues; i++) {
    if (RNIsNotZero(vector.values[i])) values[i] /= vector.values[i];
  }

  // Return this
  return *this;
}



R3SurfelFeatureVector& R3SurfelFeatureVector::
operator+=(RNScalar a)
{
  // Add values
  for (int i = 0; i < nvalues; i++) {
    values[i] += a;
  }

  // Return this
  return *this;
}



R3SurfelFeatureVector& R3SurfelFeatureVector::
operator-=(RNScalar a)
{
  // Subtract values
  for (int i = 0; i < nvalues; i++) {
    values[i] -= a;
  }

  // Return this
  return *this;
}



R3SurfelFeatureVector& R3SurfelFeatureVector::
operator*=(RNScalar a)
{
  // Multiply values
  for (int i = 0; i < nvalues; i++) {
    values[i] *= a;
  }

  // Return this
  return *this;
}



R3SurfelFeatureVector& R3SurfelFeatureVector::
operator/=(RNScalar a)
{
  // Check a
  if (RNIsZero(a)) return *this;

  // Divide values
  for (int i = 0; i < nvalues; i++) {
    values[i] /= a;
  }

  // Return this
  return *this;
}



R3SurfelFeatureVector 
operator+(const R3SurfelFeatureVector& vector1, const R3SurfelFeatureVector& vector2)
{
  // Add feature vectors
  R3SurfelFeatureVector vector = vector1;
  vector += vector2;
  return vector;
}



R3SurfelFeatureVector 
operator-(const R3SurfelFeatureVector& vector1, const R3SurfelFeatureVector& vector2)
{
  // Subtract feature vectors
  R3SurfelFeatureVector vector = vector1;
  vector -= vector2;
  return vector;
}




R3SurfelFeatureVector 
operator*(const R3SurfelFeatureVector& vector1, const R3SurfelFeatureVector& vector2)
{
  // Multiply feature vectors
  R3SurfelFeatureVector vector = vector1;
  vector *= vector2;
  return vector;
}




R3SurfelFeatureVector 
operator/(const R3SurfelFeatureVector& vector1, const R3SurfelFeatureVector& vector2)
{
  // Divide feature vectors
  R3SurfelFeatureVector vector = vector1;
  vector /= vector2;
  return vector;
}




R3SurfelFeatureVector 
operator+(const R3SurfelFeatureVector& vector1, const RNScalar a)
{
  // Add scalar
  R3SurfelFeatureVector vector = vector1;
  vector += a;
  return vector;
}




R3SurfelFeatureVector 
operator+(const RNScalar a, const R3SurfelFeatureVector& vector1)
{
  // Add scalar
  R3SurfelFeatureVector vector = vector1;
  vector += a;
  return vector;
}




R3SurfelFeatureVector 
operator-(const R3SurfelFeatureVector& vector1, const RNScalar a)
{
  // Subtract scalar
  R3SurfelFeatureVector vector = vector1;
  vector -= a;
  return vector;
}




R3SurfelFeatureVector 
operator-(const RNScalar a, const R3SurfelFeatureVector& vector1)
{
  // Subtract from scalar
  R3SurfelFeatureVector vector;
  vector *= -1.0;
  vector += a;
  return vector;
}




R3SurfelFeatureVector 
operator*(const R3SurfelFeatureVector& vector1, const RNScalar a)
{
  // Multiply by scalar
  R3SurfelFeatureVector vector = vector1;
  vector *= a;
  return vector;
}



R3SurfelFeatureVector 
operator*(const RNScalar a, const R3SurfelFeatureVector& vector1)
{
  // Multiply by scalar
  R3SurfelFeatureVector vector = vector1;
  vector *= a;
  return vector;
}



R3SurfelFeatureVector 
operator/(const R3SurfelFeatureVector& vector1, const RNScalar a)
{
  // Divide by scalar
  R3SurfelFeatureVector vector = vector1;
  vector /= a;
  return vector;
}



void R3SurfelFeatureVector::
Resize(int size)
{
  // Save old stuff
  RNScalar *old_values = values;
  int old_nvalues = nvalues;

  // Set new stuff
  nvalues = size;
  values = NULL;

  // Copy values
  if (nvalues > 0) {
    values = new RNScalar [ nvalues ];
    int overlap = (nvalues > old_nvalues) ? old_nvalues : nvalues;
    for (int i = 0; i < overlap; i++) values[i] = old_values[i];
    for (int i = overlap; i < nvalues; i++) values[i] = RN_UNKNOWN;
  }

  // Delete old values
  if (old_values) delete [] old_values;
}



void R3SurfelFeatureVector::
SetValue(int k, RNScalar value)
{
  // Set value
  assert((k >= 0) && (k < nvalues));
  values[k] = value;
}



void R3SurfelFeatureVector::
Clear(RNScalar value)
{
  // Set all values
  for (int i = 0; i < nvalues; i++) {
    values[i] = value;
  }
}



////////////////////////////////////////////////////////////////////////
// COMPARISON FUNCTIONS
////////////////////////////////////////////////////////////////////////

RNLength R3SurfelFeatureVector::
EuclideanDistanceSquared(const R3SurfelFeatureVector& vector) const
{
  // Compute sum of squared differences
  RNLength sum = 0;
  assert(NValues() == vector.NValues());
  for (int i = 0; i < NValues(); i++) {
    RNScalar delta = Value(i) - vector.Value(i);
    sum += delta * delta;
  }

  // Return squared Euclidean distance
  return sum;
}



RNScalar R3SurfelFeatureVector::
Correlation(const R3SurfelFeatureVector& vector) const
{
  // Compute sum of products (dot product)
  RNLength sum = 0;
  assert(NValues() == vector.NValues());
  for (int i = 0; i < NValues(); i++) {
    sum += Value(i) * vector.Value(i);
  }

  // Return correlation
  return sum;
}



////////////////////////////////////////////////////////////////////////
// DISPLAY FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelFeatureVector::
Print(FILE *fp, const char *prefix, const char *suffix) const
{
  // Check fp
  if (!fp) fp = stdout;

  // Print feature values
  if (prefix) fprintf(fp, "%s", prefix);
  for (int i = 0; i < NValues(); i++) {
    fprintf(fp, "%g ", Value(i));
  }
  if (suffix) fprintf(fp, "%s", suffix);
  fprintf(fp, "\n");
}


