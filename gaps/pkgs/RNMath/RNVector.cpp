// Source file for vector class



// Include files

#include "RNMath.h"



RNVector::
RNVector(void)
  : values(NULL), nvalues(0)
{
}



RNVector::
RNVector(int nvalues, RNScalar *values)
  : values(NULL), nvalues(0)
{
  // Initialize values
  Reset(nvalues, values);
}



RNVector::
RNVector(const RNVector& vector)
  : values(NULL), nvalues(vector.NValues())
{
  // Copy values
  if (nvalues > 0) {
    this->values = new RNScalar [ nvalues ];
    for (int i = 0; i < nvalues; i++) {
      SetValue(i, vector.Value(i));
    }
  }
}



RNVector::
~RNVector(void)
{
  // Delete values
  if (values) delete [] values;
}



RNBoolean RNVector::
IsZero(void) const
{
  // Return if all values are zero
  for (int i = 0; i < nvalues; i++) 
    if (values[i] != 0) return FALSE;
  return TRUE;
}



RNLength RNVector::
Length(void) const
{
  // Compute geometric length of vector
  RNLength sum = 0;
  for (int i = 0; i < nvalues; i++) 
    sum += values[i] * values[i];
  return sqrt(sum);
}



RNVector RNVector::
Subvector(int min_row, int max_row) const
{
  // Build subvector
  RNVector result(max_row - min_row + 1);
  for (int i = 0; i < max_row - min_row + 1; i++) {
    result.SetValue(i, Value(min_row+i));
  }

  // Return subvector
  return result;
}



RNScalar RNVector::
Dot(const RNVector& vector) const
{
  RNLength sum = 0;
  for (int i = 0; i < nvalues; i++) 
    sum += values[i] * vector.values[i];
  return sum;
}


RNBoolean RNVector::
operator==(const RNVector& vector) const
{
  // Return if all values are same
  for (int i = 0; i < nvalues; i++) 
    if (values[i] != vector.values[i]) return FALSE;
  return TRUE;
}



RNBoolean RNVector::
operator!=(const RNVector& vector) const
{
  return !(*this == vector);
}



void RNVector::
Negate(void)
{
  // Negate all values
  for (int i = 0; i < nvalues; i++) 
    values[i] = -values[i];
}



void RNVector::
Normalize(void)
{
  // Scale the vector to have length 1
  RNLength length = Length();
  if (length == 0) return;
  Multiply(1.0/length);
}



void RNVector::
Add(const RNVector& vector)
{
  // Add all values entry-by-entry
  for (int i = 0; i < nvalues; i++) 
    values[i] += vector.values[i];
}



void RNVector::
Subtract(const RNVector& vector)
{
  // Subtract all values entry-by-entry
  for (int i = 0; i < nvalues; i++) 
    values[i] -= vector.values[i];
}




void RNVector::
Multiply(RNScalar a)
{
  // Multiply all values by a
  for (int i = 0; i < nvalues; i++) 
    values[i] *= a;
}



void RNVector::
Reset(int nvalues, RNScalar *values)
{
  // Delete old values
  if (values) delete [] values;
  values = NULL;

  // Copy number of values
  this->nvalues = nvalues;

  // Copy values
  if (nvalues > 0) {
    this->values = new RNScalar [ nvalues ];
    if (values) {
      for (int i = 0; i < nvalues; i++) 
        this->values[i] = values[i];
    }
    else {
      for (int i = 0; i < nvalues; i++) 
        this->values[i] = 0;
    }
  }
}


RNVector& RNVector::
operator=(const RNVector& vector)
{
  // Copy vector
  if (values) delete [] values;
  nvalues = vector.NValues();
  values = new RNScalar [ nvalues ];
  for (int i = 0; i < nvalues; i++) {
    values[i] = vector.values[i];
  }
  return *this;
}



RNVector& RNVector::
operator+=(const RNVector& vector)
{
  Add(vector);
  return *this;
}



RNVector& RNVector::
operator-=(const RNVector& vector)
{
  Subtract(vector);
  return *this;
}



RNVector& RNVector::
operator*=(RNScalar a)
{
  Multiply(a);
  return *this;
}



RNVector& RNVector::
operator/=(RNScalar a)
{
  if (a != 0) Multiply(1/a);
  return *this;
}




RNVector operator-(const RNVector& vector)
{
  RNVector result(vector);
  result.Negate();
  return result;
}



RNVector operator+(const RNVector& vector1, const RNVector& vector2)
{
  RNVector result(vector1);
  result.Add(vector2);
  return result;
}



RNVector operator-(const RNVector& vector1, const RNVector& vector2)
{
  RNVector result(vector1);
  result.Subtract(vector2);
  return result;
}



RNVector operator*(RNScalar a, const RNVector& vector)
{
  RNVector result(vector);
  result.Multiply(a);
  return result;
}



RNVector operator*(const RNVector& vector, RNScalar a)
{
  RNVector result(vector);
  result.Multiply(a);
  return result;
}



RNVector operator/(const RNVector& vector, RNScalar a)
{
  RNVector result(vector);
  if (a != 0) result.Multiply(1/a);
  return result;
}





