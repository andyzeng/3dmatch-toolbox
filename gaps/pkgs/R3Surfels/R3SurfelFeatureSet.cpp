/* Source file for the R3 surfel feature set class */



////////////////////////////////////////////////////////////////////////
// INCLUDE FILES
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS/DESTRUCTORS
////////////////////////////////////////////////////////////////////////

R3SurfelFeatureSet::
R3SurfelFeatureSet(void)
  : features()
{
}



R3SurfelFeatureSet::
R3SurfelFeatureSet(const R3SurfelFeatureSet& set)
  : features()
{
  // Not implemented
  RNAbort("Not implemented");
}



R3SurfelFeatureSet::
~R3SurfelFeatureSet(void)
{
}



////////////////////////////////////////////////////////////////////////
// SET MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelFeatureSet::
InsertFeature(R3SurfelFeature *feature)
{
  // Insert feature
  features.Insert(feature);
}



void R3SurfelFeatureSet::
RemoveFeature(R3SurfelFeature *feature)
{
  // Remove feature
  features.Remove(feature);
}



void R3SurfelFeatureSet::
RemoveFeature(int k)
{
  // Remove feature
  features.RemoveKth(k);
}



void R3SurfelFeatureSet::
Empty(void)
{
  // Remove features
  features.Empty();
}



////////////////////////////////////////////////////////////////////////
// DISPLAY FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelFeatureSet::
Print(FILE *fp, const char *prefix, const char *suffix) const
{
  // Check fp
  if (!fp) fp = stdout;

  // Print features
  // Print all surfel features
  if (prefix) fprintf(fp, "%s", prefix);
  for (int i = 0; i < NFeatures(); i++) {
    R3SurfelFeature *feature = Feature(i);
    fprintf(fp, "%s ", feature->Name());
  }
  if (suffix) fprintf(fp, "%s", suffix);
  fprintf(fp, "\n");
}


