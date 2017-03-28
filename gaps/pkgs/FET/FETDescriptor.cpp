////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "FET.h"



////////////////////////////////////////////////////////////////////////
// Descriptor member functions
////////////////////////////////////////////////////////////////////////

FETDescriptor::
FETDescriptor(int nvalues, float *values) 
  : values(NULL),
    nvalues(0)
{
  // Initialize
  Reset(nvalues, values);
}



FETDescriptor::
FETDescriptor(const FETDescriptor& descriptor) 
  : values(NULL),
    nvalues(descriptor.nvalues)
{
  // Copy values
  if (nvalues > 0) {
    values = new float [ nvalues ];
    for (int i = 0; i < nvalues; i++) {
      values[i] = descriptor.values[i];
    }
  }
}



FETDescriptor::
~FETDescriptor(void) 
{
  // Delete values
  if (values) delete [] values;
}



int FETDescriptor::
NValues(void) const
{
  // Return number of values
  return nvalues;
}



float FETDescriptor::
Value(int k) const
{
  // Return kth value
  assert((k >= 0) && (k < nvalues));
  return values[k];
}



float FETDescriptor::
L1Norm(void) const
{
  // Return sum of values
  float sum = 0;
  for (int i = 0; i < nvalues; i++) {
    if (values[i] == RN_UNKNOWN) continue;
    sum += values[i];
  }
  return sum;
}



FETDescriptor& FETDescriptor::
operator=(const FETDescriptor& descriptor) 
{
  // Delete previous values
  if (this->nvalues > 0) {
    assert(this->values);
    if (this->nvalues != descriptor.nvalues) {
      delete [] this->values;
      this->values = NULL;
      this->nvalues = 0;
    }
  }

  // Assign new values
  if (descriptor.nvalues > 0) {
    this->nvalues = descriptor.nvalues;
    this->values = new float [ this->nvalues ];
    for (int i = 0; i < this->nvalues; i++) {
      this->values[i] = descriptor.values[i];
    }
  }

  // Return this
  return *this;
}



void FETDescriptor::
SetValue(int k, float value)
{
  // Set kth value
  assert((k >= 0) && (k < nvalues));
  values[k] = value;
}



void FETDescriptor::
Reset(int nvalues, float *values) 
{
  // Empty
  if (this->values) delete [] this->values;
  this->values = NULL;
  this->nvalues = 0;

  // Fill
  if (nvalues > 0) {
    this->nvalues = nvalues;
    this->values = new float [ nvalues ];
    for (int i = 0; i < nvalues; i++) {
      this->values[i] = (values) ? values[i] : 0.0F;
    }
  }
}



int FETDescriptor::
ReadAscii(FILE *fp)
{
  // Read descriptor values
  fscanf(fp, "%d", &nvalues);
  if (nvalues > 0) {
    values = new float [ nvalues ];
    for (int k = 0; k < nvalues; k++) {
      fscanf(fp, "%f", &values[k]);
    }
  }


  // Return success
  return 1;
}



int FETDescriptor::
WriteAscii(FILE *fp) const
{
  // Write descriptor values
  fprintf(fp, "%d", nvalues);
  if (nvalues > 0) {
    for (int k = 0; k < nvalues; k++) {
      fprintf(fp, "%g ", values[k]);
    }
    fprintf(fp, "\n");
  }

  // Return success
  return 1;
}

  
  
int FETDescriptor::
ReadBinary(FILE *fp)
{
  // Read descriptor values
  fread(&nvalues, sizeof(int), 1, fp);
  if (nvalues > 0) {
    values = new float [ nvalues ];
    for (int k = 0; k < nvalues; k++) {
      fread(&values[k], sizeof(float), 1, fp);
    }
  }


  // Return success
  return 1;
}



int FETDescriptor::
WriteBinary(FILE *fp) const
{
  // Write descriptor values
  fwrite(&nvalues, sizeof(int), 1, fp);
  if (nvalues > 0) {
    for (int k = 0; k < nvalues; k++) {
      fwrite(&values[k], sizeof(float), 1, fp);
    }
  }

  // Return success
  return 1;
}

  
  
float FETDescriptor::
SquaredDistance(const FETDescriptor& descriptor, float unknown_penalty) const
{
  // Return squared distance in descriptor space
  float sum = 0.0;
  assert(nvalues == descriptor.nvalues);
  for (int i = 0; i < nvalues; i++) {
    float delta;
    if (values[i] == RN_UNKNOWN) delta = unknown_penalty;
    else if (descriptor.values[i] == RN_UNKNOWN) delta = unknown_penalty;
    else delta = values[i] - descriptor.values[i];
    sum += delta * delta;
  }
  return sum;
}



float FETDescriptor::
Distance(const FETDescriptor& descriptor, float unknown_penalty) const
{
  // Return distance in descriptor space
  return sqrt(SquaredDistance(descriptor, unknown_penalty));
}



