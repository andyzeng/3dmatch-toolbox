/* Source file for the R3 surfel label relationship class */



////////////////////////////////////////////////////////////////////////
// INCLUDE FILES
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS/DESTRUCTORS
////////////////////////////////////////////////////////////////////////

R3SurfelLabelRelationship::
R3SurfelLabelRelationship(int type, const RNArray<R3SurfelLabel *>& labels, RNScalar *operands, int noperands)
  : scene(NULL),
    scene_index(-1),
    labels(labels),
    operands(NULL),
    noperands(0),
    type(type)
{
  // Copy operands, if provided
  if ((noperands > 0) && (operands)) {
    this->noperands = noperands;
    this->operands = new RNScalar [ this->noperands ];
    for (int i = 0; i < this->noperands; i++) {
      this->operands[i] = operands[i];
    }
  }
}



R3SurfelLabelRelationship::
R3SurfelLabelRelationship(int type, R3SurfelLabel *label0, R3SurfelLabel *label1, RNScalar *operands, int noperands)
  : scene(NULL),
    scene_index(-1),
    labels(),
    operands(NULL),
    noperands(0),
    type(type)
{
  // Insert labels
  labels.Insert(label0);
  labels.Insert(label1);

  // Copy operands, if provided
  if ((noperands > 0) && (operands)) {
    this->noperands = noperands;
    this->operands = new RNScalar [ this->noperands ];
    for (int i = 0; i < this->noperands; i++) {
      this->operands[i] = operands[i];
    }
  }
}



R3SurfelLabelRelationship::
R3SurfelLabelRelationship(const R3SurfelLabelRelationship& relationship)
  : scene(NULL),
    scene_index(-1),
    labels(relationship.labels),
    operands(NULL),
    noperands(0),
    type(relationship.type)
{
  // Copy operands
  if ((relationship.noperands > 0) && (relationship.operands)) {
    this->noperands = relationship.noperands;
    this->operands = new RNScalar [ this->noperands ];
    for (int i = 0; i < this->noperands; i++) {
      this->operands[i] = relationship.operands[i];
    }
  }
}



R3SurfelLabelRelationship::
~R3SurfelLabelRelationship(void)
{
  // Remove from scene
  if (scene) scene->RemoveLabelRelationship(this);

  // Delete operands
  if (operands) delete [] operands;
}






////////////////////////////////////////////////////////////////////////
// INTERNAL FUNCDTIONS
////////////////////////////////////////////////////////////////////////

