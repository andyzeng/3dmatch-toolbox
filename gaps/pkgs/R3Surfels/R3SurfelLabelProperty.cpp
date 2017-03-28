/* Source file for the R3 surfel label property class */



////////////////////////////////////////////////////////////////////////
// INCLUDE FILES
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS/DESTRUCTORS
////////////////////////////////////////////////////////////////////////

R3SurfelLabelProperty::
R3SurfelLabelProperty(int type, R3SurfelLabel *label, RNScalar *operands, int noperands)
  : scene(NULL),
    scene_index(-1),
    label(label),
    operands(NULL),
    noperands(0),
    type(type)
{
  // Check if operands were provided
  if ((noperands > 0) && (operands)) {
    // Copy operands
    this->noperands = noperands;
    this->operands = new RNScalar [ this->noperands ];
    for (int i = 0; i < this->noperands; i++) {
      this->operands[i] = operands[i];
    }
  }
}



R3SurfelLabelProperty::
R3SurfelLabelProperty(const R3SurfelLabelProperty& property)
  : scene(NULL),
    scene_index(-1),
    label(property.label),
    operands(NULL),
    noperands(0),
    type(property.type)
{
  // Copy operands
  if ((property.noperands > 0) && (property.operands)) {
    this->noperands = property.noperands;
    this->operands = new RNScalar [ this->noperands ];
    for (int i = 0; i < this->noperands; i++) {
      this->operands[i] = property.operands[i];
    }
  }
}



R3SurfelLabelProperty::
~R3SurfelLabelProperty(void)
{
  // Remove from scene
  if (scene) scene->RemoveLabelProperty(this);

  // Delete operands
  if (operands) delete [] operands;
}






////////////////////////////////////////////////////////////////////////
// INTERNAL FUNCDTIONS
////////////////////////////////////////////////////////////////////////

int R3SurfelLabelProperty::
NOperands(int type)
{
  // Return number of operands
  switch (type) {
  }

  // Default case
  return 0;
}



