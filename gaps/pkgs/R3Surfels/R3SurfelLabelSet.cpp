/* Source file for the R3 surfel label set class */



////////////////////////////////////////////////////////////////////////
// INCLUDE FILES
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS/DESTRUCTORS
////////////////////////////////////////////////////////////////////////

R3SurfelLabelSet::
R3SurfelLabelSet(void)
  : labels()
{
}



R3SurfelLabelSet::
R3SurfelLabelSet(const R3SurfelLabelSet& set)
  : labels()
{
  // Not implemented
  RNAbort("Not implemented");
}



R3SurfelLabelSet::
~R3SurfelLabelSet(void)
{
}



////////////////////////////////////////////////////////////////////////
// SET MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelLabelSet::
InsertLabel(R3SurfelLabel *label)
{
  // Insert label
  labels.Insert(label);
}



void R3SurfelLabelSet::
RemoveLabel(R3SurfelLabel *label)
{
  // Remove label
  labels.Remove(label);
}



void R3SurfelLabelSet::
RemoveLabel(int k)
{
  // Remove label
  labels.RemoveKth(k);
}



void R3SurfelLabelSet::
Empty(void)
{
  // Remove labels
  labels.Empty();
}



////////////////////////////////////////////////////////////////////////
// DISPLAY FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelLabelSet::
Print(FILE *fp, const char *prefix, const char *suffix) const
{
  // Check fp
  if (!fp) fp = stdout;

  // Print labels
  // Print all surfel labels
  if (prefix) fprintf(fp, "%s", prefix);
  for (int i = 0; i < NLabels(); i++) {
    R3SurfelLabel *label = Label(i);
    fprintf(fp, "%s ", label->Name());
  }
  if (suffix) fprintf(fp, "%s", suffix);
  fprintf(fp, "\n");
}


