/* Source file for the R3 surfel feature class */



////////////////////////////////////////////////////////////////////////
// INCLUDE FILES
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS/DESTRUCTORS
////////////////////////////////////////////////////////////////////////

R3SurfelFeature::
R3SurfelFeature(const char *name, RNScalar minimum, RNScalar maximum, RNScalar weight)
  : scene(NULL),
    scene_index(-1),
    name((name) ? strdup(name) : NULL),
    range(minimum, maximum),
    weight(weight),
    data(NULL)
{
}



R3SurfelFeature::
R3SurfelFeature(const R3SurfelFeature& feature)
  : scene(NULL),
    scene_index(-1),
    name((feature.name) ? strdup(feature.name) : NULL),
    range(feature.range),
    weight(feature.weight),
    data(NULL)
{
}



R3SurfelFeature::
~R3SurfelFeature(void)
{
  // Remove from scene
  if (scene) scene->RemoveFeature(this);

  // Delete name
  if (name) free(name);
}



////////////////////////////////////////////////////////////////////////
// PROPERTY FUNCTIONS
////////////////////////////////////////////////////////////////////////

int R3SurfelFeature::
Type(void) const
{
  // Return feature type
  return R3_SURFEL_BASIC_FEATURE_TYPE;
}



////////////////////////////////////////////////////////////////////////
// MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelFeature::
SetName(const char *name) 
{
  // Copy name
  if (this->name) free(this->name);
  if (name) this->name = strdup(name);
}



void R3SurfelFeature::
SetRange(const RNInterval& range) 
{
  // Set range
  this->range = range;
}



void R3SurfelFeature::
SetRange(RNScalar minimum, RNScalar maximum) 
{
  // Set range
  this->range.Reset(minimum, maximum);
}



void R3SurfelFeature::
SetMinimum(RNScalar minimum) 
{
  // Set minimum
  range.SetMin(minimum);
}



void R3SurfelFeature::
SetMaximum(RNScalar maximum) 
{
  // Set maximum
  range.SetMax(maximum);
}



void R3SurfelFeature::
SetWeight(RNScalar weight) 
{
  // Set weight
  this->weight = weight;
}



void R3SurfelFeature::
SetData(void *data) 
{
  // Set user data
  this->data = data;
}



////////////////////////////////////////////////////////////////////////
// DISPLAY FUNCTIONS
////////////////////////////////////////////////////////////////////////

int R3SurfelFeature::
UpdateFeatureVector(R3SurfelObject *object, R3SurfelFeatureVector& vector) const
{
  // Feature was not computed by this function
  RNAbort("Feature should be evaluated by derived class");
  return 0;
}



////////////////////////////////////////////////////////////////////////
// DISPLAY FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelFeature::
Print(FILE *fp, const char *prefix, const char *suffix) const
{
  // Check fp
  if (!fp) fp = stdout;

  // Print feature name
  if (prefix) fprintf(fp, "%s", prefix);
  fprintf(fp, "%s ", Name());
  if (suffix) fprintf(fp, "%s", suffix);
  fprintf(fp, "\n");
}



