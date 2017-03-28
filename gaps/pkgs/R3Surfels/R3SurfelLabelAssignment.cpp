/* Source file for the R3 label assignment class */



////////////////////////////////////////////////////////////////////////
// INCLUDE FILES
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS/DESTRUCTORS
////////////////////////////////////////////////////////////////////////

R3SurfelLabelAssignment::
R3SurfelLabelAssignment(R3SurfelObject *object, R3SurfelLabel* label, RNScalar confidence, int originator)
  : scene(NULL),
    scene_index(-1),
    object(object),
    object_index(-1),
    label(label),
    label_index(-1),
    confidence(confidence),
    originator(originator),
    data(NULL)
{
}



R3SurfelLabelAssignment::
R3SurfelLabelAssignment(const R3SurfelLabelAssignment& assignment)
  : scene(NULL),
    scene_index(-1),
    object(assignment.object),
    object_index(-1),
    label(assignment.label),
    label_index(-1),
    confidence(assignment.confidence),
    originator(assignment.originator),
    data(NULL)
{
}



R3SurfelLabelAssignment::
~R3SurfelLabelAssignment(void)
{
  // Update scene
  if (scene) scene->RemoveLabelAssignment(this);
}



////////////////////////////////////////////////////////////////////////
// PROPERTY MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelLabelAssignment::
SetConfidence(RNScalar confidence)
{
  // Set confidence
  this->confidence = confidence;

  // Mark scene as dirty
  if (scene) scene->SetDirty();
}



void R3SurfelLabelAssignment::
SetOriginator(int originator)
{
  // Set originator
  this->originator = originator;

  // Mark scene as dirty
  if (scene) scene->SetDirty();
}



void R3SurfelLabelAssignment::
SetData(void *data) 
{
  // Set user data
  this->data = data;
}



