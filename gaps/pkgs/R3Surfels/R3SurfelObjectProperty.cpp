/* Source file for the R3 surfel object property class */



////////////////////////////////////////////////////////////////////////
// INCLUDE FILES
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS/DESTRUCTORS
////////////////////////////////////////////////////////////////////////

R3SurfelObjectProperty::
R3SurfelObjectProperty(int type, R3SurfelObject *object, RNScalar *operands, int noperands)
  : scene(NULL),
    scene_index(-1),
    object(object),
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
  else {
    // Compute operands
    UpdateOperands();
  }
}



R3SurfelObjectProperty::
R3SurfelObjectProperty(const R3SurfelObjectProperty& property)
  : scene(NULL),
    scene_index(-1),
    object(property.object),
    operands(NULL),
    noperands(0),
    type(property.type)
{
  // Initialize operands
  if ((property.noperands > 0) && (property.operands)) {
    this->noperands = property.noperands;
    this->operands = new RNScalar [ this->noperands ];
    for (int i = 0; i < this->noperands; i++) {
      this->operands[i] = property.operands[i];
    }
  }
}



R3SurfelObjectProperty::
~R3SurfelObjectProperty(void)
{
  // Remove from scene
  if (scene) scene->RemoveObjectProperty(this);

  // Delete operands
  if (operands) delete [] operands;
}



////////////////////////////////////////////////////////////////////////
// DISPLAY FUNCDTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelObjectProperty::
Draw(RNFlags flags) const
{
  // Draw property based on type
  switch (type) {
  case R3_SURFEL_OBJECT_PCA_PROPERTY: 
    if (noperands == 21) {
      R3Point centroid(operands[0], operands[1], operands[2]);
      R3Vector axis1(operands[3], operands[4], operands[5]);
      R3Vector axis2(operands[6], operands[7], operands[8]);
      R3Vector axis3(operands[9], operands[10], operands[11]);
      RNScalar stddev1 = sqrt(operands[12]);
      RNScalar stddev2 = sqrt(operands[13]);
      RNScalar stddev3 = sqrt(operands[14]);
      RNScalar extent01 = operands[15];
      RNScalar extent02 = operands[16];
      RNScalar extent03 = operands[17];
      RNScalar extent11 = operands[18];
      RNScalar extent12 = operands[19];
      RNScalar extent13 = operands[20];

      // Draw stddevs
      glLineWidth(2);
      glBegin(GL_LINES);
      if (flags[R3_SURFEL_COLOR_DRAW_FLAG]) glColor3d(1, 0, 0);
      R3LoadPoint(centroid - stddev1 * axis1);
      R3LoadPoint(centroid + stddev1 * axis1);
      if (flags[R3_SURFEL_COLOR_DRAW_FLAG]) glColor3d(0, 1, 0);
      R3LoadPoint(centroid - stddev2 * axis2);
      R3LoadPoint(centroid + stddev2 * axis2);
      if (flags[R3_SURFEL_COLOR_DRAW_FLAG]) glColor3d(0, 0, 1);
      R3LoadPoint(centroid - stddev3 * axis3);
      R3LoadPoint(centroid + stddev3 * axis3);
      glEnd();

      // Draw extents
      glLineWidth(1);
      glBegin(GL_LINES);
      if (flags[R3_SURFEL_COLOR_DRAW_FLAG]) glColor3d(1, 0, 0);
      R3LoadPoint(centroid + extent01 * axis1);
      R3LoadPoint(centroid + extent11 * axis1);
      if (flags[R3_SURFEL_COLOR_DRAW_FLAG]) glColor3d(0, 1, 0);
      R3LoadPoint(centroid + extent02 * axis2);
      R3LoadPoint(centroid + extent12 * axis2);
      if (flags[R3_SURFEL_COLOR_DRAW_FLAG]) glColor3d(0, 0, 1);
      R3LoadPoint(centroid + extent03 * axis3);
      R3LoadPoint(centroid + extent13 * axis3);
      glEnd();
    }
    break; 
  }
}



////////////////////////////////////////////////////////////////////////
// INTERNAL FUNCDTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelObjectProperty::
UpdateOperands(void)
{
  // Check if already uptodate
  if (noperands > 0) return;

  // Check property type
  switch(type) {
  case R3_SURFEL_OBJECT_PCA_PROPERTY: {
    // PCA Properties
    // noperands = 21
    // operands[0-2]: centroid
    // operands[3-5]: xyz of axis1
    // operands[6-8]: xyz of axis2
    // operands[9-11]: xyz of axis3
    // operands[12-14]: variances
    // operands[15-20]: extents 

    // Allocate operands
    noperands = 21;
    operands = new RNScalar [ noperands ];
    for (int i = 0; i < noperands; i++) operands[i] = 0;

    // Create pointset
    R3SurfelPointSet *pointset = object->PointSet();
    if (!pointset) return;
    if (pointset->NPoints() < 3) { delete pointset; return; }

    // Compute principle axes
    RNScalar variances[3];
    R3Point centroid = pointset->Centroid();
    R3Triad triad = pointset->PrincipleAxes(&centroid, variances);

    // Find extents
    R3Box extent = R3null_box;
    for (int i = 0; i < pointset->NPoints(); i++) {
      R3SurfelPoint *point = pointset->Point(i);
      R3Point position = point->Position();
      R3Vector vector = position - centroid;
      RNScalar x = triad.Axis(0).Dot(vector);
      RNScalar y = triad.Axis(1).Dot(vector);
      RNScalar z = triad.Axis(2).Dot(vector);
      extent.Union(R3Point(x, y, z));
    }

    // Fill operands
    operands[0] = centroid.X();
    operands[1] = centroid.Y();
    operands[2] = centroid.Z();
    operands[3] = triad.Axis(0).X();
    operands[4] = triad.Axis(0).Y();
    operands[5] = triad.Axis(0).Z();
    operands[6] = triad.Axis(1).X();
    operands[7] = triad.Axis(1).Y();
    operands[8] = triad.Axis(1).Z();
    operands[9] = triad.Axis(2).X();
    operands[10] = triad.Axis(2).Y();
    operands[11] = triad.Axis(2).Z();
    operands[12] = variances[0];
    operands[13] = variances[1];
    operands[14] = variances[2];
    operands[15] = extent[0][0];
    operands[16] = extent[0][1];
    operands[17] = extent[0][2];
    operands[18] = extent[1][0];
    operands[19] = extent[1][1];
    operands[20] = extent[1][2];

    // Delete pointset
    delete pointset;
    break; }
  }
}

