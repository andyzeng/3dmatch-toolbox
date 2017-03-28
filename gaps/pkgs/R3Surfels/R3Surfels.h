/* Include file for R3 surfels module */

#ifndef __R3__SURFELS__H__
#define __R3__SURFELS__H__



/* Dependency include files */

#include "R3Shapes/R3Shapes.h"



/* Draw flags */

#define R3_SURFEL_COLOR_DRAW_FLAG     0x0001
#define R3_SURFEL_DEFAULT_DRAW_FLAGS  R3_SURFEL_COLOR_DRAW_FLAG



/* Draw method selection */

// Define only one of these
// #define R3_SURFEL_DRAW_WITH_DISPLAY_LIST
// #define R3_SURFEL_DRAW_WITH_VBO
// #define R3_SURFEL_DRAW_WITH_ARRAYS
#define R3_SURFEL_DRAW_WITH_POINTS



/* Class declarations */

class R3Surfel;
class R3SurfelBlock;
class R3SurfelDatabase;
class R3SurfelConstraint;
class R3SurfelPoint;
class R3SurfelPointSet;
class R3SurfelPointGraph;
class R3SurfelNode;
class R3SurfelNodeSet;
class R3SurfelTree;
class R3SurfelScan;
class R3SurfelFeature;
class R3SurfelFeatureSet;
class R3SurfelFeatureVector;
class R3SurfelObject;
class R3SurfelObjectSet;
class R3SurfelObjectProperty;
class R3SurfelObjectRelationship;
class R3SurfelLabel;
class R3SurfelLabelSet;
class R3SurfelLabelProperty;
class R3SurfelLabelRelationship;
class R3SurfelLabelAssignment;
typedef R3SurfelLabelAssignment R3SurfelObjectAssignment;
class R3SurfelScene;



/* Surfel pkg include files */

#include "R3Surfels/R3Surfel.h"
#include "R3Surfels/R3SurfelBlock.h"
#include "R3Surfels/R3SurfelDatabase.h"
#include "R3Surfels/R3SurfelConstraint.h"
#include "R3Surfels/R3SurfelPoint.h"
#include "R3Surfels/R3SurfelPointSet.h"
#include "R3Surfels/R3SurfelPointGraph.h"
#include "R3Surfels/R3SurfelNode.h"
#include "R3Surfels/R3SurfelNodeSet.h"
#include "R3Surfels/R3SurfelTree.h"
#include "R3Surfels/R3SurfelScan.h"
#include "R3Surfels/R3SurfelFeature.h"
#include "R3Surfels/R3SurfelFeatureSet.h"
#include "R3Surfels/R3SurfelFeatureVector.h"
#include "R3Surfels/R3SurfelFeatureEvaluation.h"
#include "R3Surfels/R3SurfelObject.h"
#include "R3Surfels/R3SurfelObjectSet.h"
#include "R3Surfels/R3SurfelObjectProperty.h"
#include "R3Surfels/R3SurfelObjectRelationship.h"
#include "R3Surfels/R3SurfelLabel.h"
#include "R3Surfels/R3SurfelLabelSet.h"
#include "R3Surfels/R3SurfelLabelProperty.h"
#include "R3Surfels/R3SurfelLabelRelationship.h"
#include "R3Surfels/R3SurfelLabelAssignment.h"
#include "R3Surfels/R3SurfelScene.h"



/* Utility include files */

#include "R3Surfels/R3SurfelUtils.h"



/* Initialization functions */

int R3InitSurfels(void);
void R3StopSurfels(void);



#endif








