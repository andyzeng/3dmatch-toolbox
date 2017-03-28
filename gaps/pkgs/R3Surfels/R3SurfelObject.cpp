/* Source file for the R3 surfel object class */



////////////////////////////////////////////////////////////////////////
// INCLUDE FILES
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// PRIVATE GLOBAL VARIABLES
////////////////////////////////////////////////////////////////////////

static int R3surfel_next_object_identifier = 0;



////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS/DESTRUCTORS
////////////////////////////////////////////////////////////////////////

R3SurfelObject::
R3SurfelObject(const char *name)
  : scene(NULL),
    scene_index(-1),
    parent(NULL),
    parts(),
    properties(),
    relationships(),
    assignments(),
    nodes(),
    feature_vector(),
    name((name) ? strdup(name) : NULL),
    identifier(-1),
    complexity(0),
    bbox(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX),
    data(NULL)
{
}



R3SurfelObject::
R3SurfelObject(const R3SurfelObject& object)
  : scene(NULL),
    scene_index(-1),
    parent(NULL),
    parts(),
    properties(),
    relationships(),
    assignments(),
    nodes(object.nodes),
    feature_vector(object.feature_vector),
    name((object.name) ? strdup(object.name) : NULL),
    identifier(-1),
    complexity(object.complexity),
    bbox(object.bbox),
    data(NULL)
{
}



R3SurfelObject::
~R3SurfelObject(void)
{
  // Delete object properties
  while (NObjectProperties() > 0) {
    R3SurfelObjectProperty *property = ObjectProperty(NObjectProperties()-1);
    delete property;
  }

  // Delete object relationships
  while (NObjectRelationships() > 0) {
    R3SurfelObjectRelationship *relationship = ObjectRelationship(NObjectRelationships()-1);
    delete relationship;
  }

  // Delete object assignments
  while (NLabelAssignments() > 0) {
    R3SurfelLabelAssignment *assignment = LabelAssignment(NLabelAssignments()-1);
    delete assignment;
  }

  // Remove nodes from object
  while (NNodes() > 0) {
    R3SurfelNode *node = Node(NNodes()-1);
    RemoveNode(node);
  }

  // Remove object from scene
  if (scene) scene->RemoveObject(this);

  // Delete name
  if (name) free(name);
}



////////////////////////////////////////////////////////////////////////
// PROPERTY FUNCTIONS
////////////////////////////////////////////////////////////////////////

int R3SurfelObject::
Identifier(void) const
{
  // Update ID 
  if (identifier < 0) {
    ((R3SurfelObject *) this)->identifier = R3surfel_next_object_identifier++;
  }

  // Return identifier 
  return identifier;
}



const R3Box& R3SurfelObject::
BBox(void) const
{
  // Return bounding box of object
  if (bbox[0][0] == FLT_MAX) 
    ((R3SurfelObject *) this)->UpdateBBox();
  return bbox;
}



const R3SurfelFeatureVector& R3SurfelObject::
FeatureVector(void) const
{
  // Update feature vector
  if (feature_vector.NValues() == 0) {
    if (scene->NFeatures() > 0) {
      ((R3SurfelObject *) this)->UpdateFeatureVector();
    }
  }

  // Return feature vector
  return feature_vector;
}



int R3SurfelObject::
PartHierarchyLevel(void) const
{
  // Return level in part hierarchy (root is 0)
  int level = 0;
  R3SurfelObject *ancestor = parent;
  while (ancestor) { level++; ancestor = ancestor->parent; }
  return level;
}



////////////////////////////////////////////////////////////////////////
// SCAN ACCESS FUNCTIONS
////////////////////////////////////////////////////////////////////////

R3SurfelScan *R3SurfelObject::
Scan(void) const
{
  // Search for scan amongst object's nodes
  R3SurfelScan *scan = NULL;
  for (int i = 0; i < NNodes(); i++) {
    R3SurfelNode *node = Node(i);
    R3SurfelScan *s = node->Scan();
    if (s && (s != scan)) {
      if (scan) return NULL;
      else scan = s;
    }
  }

  // Return scan, or NULL if not found
  return scan;
}



////////////////////////////////////////////////////////////////////////
// PROPERTY ACCESS FUNCTIONS
////////////////////////////////////////////////////////////////////////

R3SurfelObjectProperty *R3SurfelObject::
FindObjectProperty(int type) const
{
  // Search for property of type
  for (int i = 0; i < NObjectProperties(); i++) {
    R3SurfelObjectProperty *property = ObjectProperty(i);
    if (property->Type() == type) return property;
  }

  // Property not found
  return NULL;
}



////////////////////////////////////////////////////////////////////////
// POINT ACCESS FUNCTIONS
////////////////////////////////////////////////////////////////////////

R3SurfelPointSet *R3SurfelObject::
PointSet(RNBoolean leaf_level) const
{
  // Allocate point set
  R3SurfelPointSet *pointset = new R3SurfelPointSet();
  if (!pointset) {
    fprintf(stderr, "Unable to allocate point set\n");
    return NULL;
  }

  // Insert points
  if (leaf_level) {
    // Insert points from all blocks of all leaf nodes
    for (int i = 0; i < NNodes(); i++) {
      R3SurfelNode *node = Node(i);
      RNArray<R3SurfelNode *> stack;
      stack.Insert(node);
      while (!stack.IsEmpty()) {
        node = stack.Tail();
        stack.RemoveTail();
        if (node->NParts() > 0) {
          for (int j = 0; j < node->NParts(); j++) {
            R3SurfelNode *part = node->Part(j);
            stack.Insert(part);
          }
        }
        else {
          for (int j = 0; j < node->NBlocks(); j++) {
            R3SurfelBlock *block = node->Block(j);
            pointset->InsertPoints(block);
          }
        }
      }
    }
  }
  else {
    // Insert points from all nodes at highest level
    for (int i = 0; i < NNodes(); i++) {
      R3SurfelNode *node = Node(i);
      for (int j = 0; j < node->NBlocks(); j++) {
        R3SurfelBlock *block = node->Block(j);
        pointset->InsertPoints(block);
      }
    }
  }
  
  // Return pointset
  return pointset;
}



////////////////////////////////////////////////////////////////////////
// ASSIGNMENT ACCESS FUNCTIONS
////////////////////////////////////////////////////////////////////////

R3SurfelLabelAssignment *R3SurfelObject::
GroundTruthLabelAssignment(void) const
{
  // Return ground truth assignment (or NULL if there is none)
  return BestLabelAssignment(R3_SURFEL_LABEL_ASSIGNMENT_GROUND_TRUTH_ORIGINATOR);
}



R3SurfelLabelAssignment *R3SurfelObject::
HumanLabelAssignment(void) const
{
  // Return human generated assignment (or NULL if there is none)
  return BestLabelAssignment(R3_SURFEL_LABEL_ASSIGNMENT_HUMAN_ORIGINATOR);
}



R3SurfelLabelAssignment *R3SurfelObject::
PredictedLabelAssignment(void) const
{
  // Return best predicted assignment (or NULL if there is none)
  return BestLabelAssignment(R3_SURFEL_LABEL_ASSIGNMENT_MACHINE_ORIGINATOR);
}



R3SurfelLabelAssignment *R3SurfelObject::
CurrentLabelAssignment(void) const
{
  // Return best assignment (either human or predicted)
  R3SurfelLabelAssignment *human_assignment = HumanLabelAssignment();
  if (human_assignment) return human_assignment;
  R3SurfelLabelAssignment *predicted_assignment = PredictedLabelAssignment();
  if (predicted_assignment) return predicted_assignment;
  return NULL;
}



R3SurfelLabelAssignment *R3SurfelObject::
BestLabelAssignment(int originator) const
{
  // Return best assignment (or NULL if there is none)
  R3SurfelLabelAssignment *best_assignment = NULL;
  RNScalar best_confidence = -FLT_MAX;
  for (int i = 0; i < assignments.NEntries(); i++) {
    R3SurfelLabelAssignment *assignment = assignments.Kth(i);
    if (assignment->Originator() == originator) {
      RNScalar confidence = assignment->Confidence();
      if (confidence > best_confidence) {
        best_assignment = assignment;
        best_confidence = confidence;
      }
    }
  }

  // Return best assignment
  return best_assignment;
}



R3SurfelLabel *R3SurfelObject::
GroundTruthLabel(void) const
{
  // Return ground truth label
  R3SurfelLabelAssignment *assignment = GroundTruthLabelAssignment();
  if (assignment) return assignment->Label();
  return NULL;
}



R3SurfelLabel *R3SurfelObject::
HumanLabel(void) const
{
  // Return human generated label
  R3SurfelLabelAssignment *assignment = HumanLabelAssignment();
  if (assignment) return assignment->Label();
  return NULL;
}



R3SurfelLabel *R3SurfelObject::
PredictedLabel(void) const
{
  // Return predicted label
  R3SurfelLabelAssignment *assignment = PredictedLabelAssignment();
  if (assignment) return assignment->Label();
  return NULL;
}



R3SurfelLabel *R3SurfelObject::
CurrentLabel(void) const
{
  // Return best label (either human or predicted)
  R3SurfelLabel *human_label = HumanLabel();
  if (human_label) return human_label;
  R3SurfelLabel *predicted_label = PredictedLabel();
  if (predicted_label) return predicted_label;
  return NULL;
}



R3SurfelLabel *R3SurfelObject::
BestLabel(int originator) const
{
  // Return best label with given origniator
  R3SurfelLabelAssignment *assignment = BestLabelAssignment(originator);
  if (assignment) return assignment->Label();
  return NULL;
}



////////////////////////////////////////////////////////////////////////
// PROPERTY MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelObject::
SetParent(R3SurfelObject *parent)
{
  // Just checking
  assert(parent);
  assert(this->parent);
  assert(scene == this->parent->scene);
  assert(parent->scene == this->parent->scene);
  if (parent == this->parent) return;

  // Invalidate bounding boxes starting at current parent
  R3SurfelObject *ancestor = this->parent;
  while (ancestor) {
    if (ancestor->bbox[0][0] == FLT_MAX) break;
    ancestor->bbox[0][0] = FLT_MAX;
    ancestor = ancestor->parent;
  }

  // Invalidate bounding boxes starting at new parent
  ancestor = parent;
  while (ancestor) {
    if (ancestor->bbox[0][0] == FLT_MAX) break;
    ancestor->bbox[0][0] = FLT_MAX;
    ancestor = ancestor->parent;
  }

  // Update hierarchy
  this->parent->parts.Remove(this);
  parent->parts.Insert(this);
  this->parent = parent;

  // Mark scene as dirty
  if (scene) scene->SetDirty();
}



void R3SurfelObject::
SetName(const char *name)
{
  // Delete previous name
  if (this->name) free(this->name);
  this->name = (name) ? strdup(name) : NULL;

  // Mark scene as dirty
  if (scene) scene->SetDirty();
}



void R3SurfelObject::
SetIdentifier(int identifier)
{
  // Set identifier
  this->identifier = identifier;

  // Update next identifier
  if (identifier >= R3surfel_next_object_identifier) {
    R3surfel_next_object_identifier = identifier+1;
  }

  // Mark scene as dirty
  if (scene) scene->SetDirty();
}



void R3SurfelObject::
SetFeatureVector(const R3SurfelFeatureVector& vector)
{
  // Copy feature vector
  this->feature_vector = vector;

  // Mark scene as dirty
  if (scene) scene->SetDirty();
}



void R3SurfelObject::
SetData(void *data) 
{
  // Set user data
  this->data = data;
}



////////////////////////////////////////////////////////////////////////
// NODE MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelObject::
InsertNode(R3SurfelNode *node)
{
  // Just checking
  assert(node);
  assert(!nodes.FindEntry(node));
  assert(node->object == NULL);

  // Update node
  node->object = this;

  // Insert node
  nodes.Insert(node);

  // Update complexity
  complexity += node->Complexity();

  // Invalidate bounding box
  R3SurfelObject *ancestor = this;
  while (ancestor) {
    if (ancestor->bbox[0][0] == FLT_MAX) break;
    ancestor->bbox[0][0] = FLT_MAX;
    ancestor = ancestor->parent;
  }

  // Update node
  node->UpdateAfterInsert(this);

  // Mark scene as dirty
  if (scene) scene->SetDirty();
}



void R3SurfelObject::
RemoveNode(R3SurfelNode *node)
{
  // Just checking
  assert(node);
  assert(nodes.FindEntry(node));
  assert(node->object == this);

  // Update node
  node->UpdateBeforeRemove(this);

  // Update node
  node->object = NULL;

  // Remove node
  nodes.Remove(node);

  // Update complexity
  complexity -= node->Complexity();

  // Invalidate bounding box
  R3SurfelObject *ancestor = this;
  while (ancestor) {
    if (ancestor->bbox[0][0] == FLT_MAX) break;
    ancestor->bbox[0][0] = FLT_MAX;
    ancestor = ancestor->parent;
  }

  // Mark scene as dirty
  if (scene) scene->SetDirty();
}



///////////////////////////////////////////////////////////////////////
// SURFEL MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelObject::
SetMarks(RNBoolean mark)
{
  // Set mark for all nodes
  for (int i = 0; i < NNodes(); i++) {
    R3SurfelNode *node = Node(i);
    node->SetMarks(mark);
  }
}



///////////////////////////////////////////////////////////////////////
// MEMORY MANGEMENT FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelObject::
ReadBlocks(void)
{
  // Read blocks in object
  for (int i = 0; i < NNodes(); i++) {
    R3SurfelNode *node = Node(i);
    node->ReadBlocks();
  }
}



void R3SurfelObject::
ReleaseBlocks(void)
{
  // Release blocks in object
  for (int i = 0; i < NNodes(); i++) {
    R3SurfelNode *node = Node(i);
    node->ReleaseBlocks();
  }
}



RNBoolean R3SurfelObject::
AreBlocksResident(void) const
{
  // Return whether blocks are in memory
  for (int i = 0; i < NNodes(); i++) {
    R3SurfelNode *node = Node(i);
    if (!node->AreBlocksResident()) return FALSE;
  }

  // All blocks are resident
  return TRUE;
}



////////////////////////////////////////////////////////////////////////
// DISPLAY FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelObject::
Draw(RNFlags flags) const
{
  // Draw all nodes
  for (int i = 0; i < NNodes(); i++) {
    R3SurfelNode *node = Node(i);
    node->Draw(flags);
  }
}



void R3SurfelObject::
Print(FILE *fp, const char *prefix, const char *suffix) const
{
  // Check fp
  if (!fp) fp = stdout;

  // Print object
  if (prefix) fprintf(fp, "%s", prefix);
  fprintf(fp, "%d %s", SceneIndex(), (Name()) ? Name() : "-");
  if (suffix) fprintf(fp, "%s", suffix);
  fprintf(fp, "\n");

  // Print all nodes
  for (int i = 0; i < NNodes(); i++) {
    R3SurfelNode *node = Node(i);
    node->Print(fp, prefix, suffix);
  }
  fprintf(fp, "\n");
}




////////////////////////////////////////////////////////////////////////
// STRUCTURE UPDATE FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelObject::
UpdateAfterInsert(R3SurfelScene *scene)
{
  // Invalidate bounding boxes
  R3SurfelObject *ancestor = parent;
  while (ancestor) {
    if (ancestor->bbox[0][0] == FLT_MAX) break;
    ancestor->bbox[0][0] = FLT_MAX;
    ancestor = ancestor->parent;
  }
}



void R3SurfelObject::
UpdateBeforeRemove(R3SurfelScene *scene)
{
  // Invalidate properties
  R3SurfelObject *ancestor = parent;
  while (ancestor) {
    if (ancestor->bbox[0][0] == FLT_MAX) break;
    ancestor->bbox[0][0] = FLT_MAX;
    ancestor = ancestor->parent;
  }
}



void R3SurfelObject::
UpdateAfterInsertObjectProperty(R3SurfelObjectProperty *property)
{
  // Just checking
  assert(property->Scene());
  assert(property->Scene() == scene);

  // Insert property
  properties.Insert(property);
}



void R3SurfelObject::
UpdateBeforeRemoveObjectProperty(R3SurfelObjectProperty *property)
{
  // Just checking
  assert(property->Scene());
  assert(property->Scene() == scene);

  // Remove property
  properties.Remove(property);
}



void R3SurfelObject::
UpdateAfterInsertObjectRelationship(R3SurfelObjectRelationship *relationship)
{
  // Just checking
  assert(relationship->Scene());
  assert(relationship->Scene() == scene);

  // Insert relationship
  relationships.Insert(relationship);
}



void R3SurfelObject::
UpdateBeforeRemoveObjectRelationship(R3SurfelObjectRelationship *relationship)
{
  // Just checking
  assert(relationship->Scene());
  assert(relationship->Scene() == scene);

  // Remove relationship
  relationships.Remove(relationship);
}



////////////////////////////////////////////////////////////////////////
// ASSIGNMENT MANIPULATION UPDATE FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelObject::
UpdateAfterInsertLabelAssignment(R3SurfelLabelAssignment *assignment)
{
  // Check assignment
  assert(assignment->Scene());
  assert(assignment->Scene() == scene);
  assert(assignment->Object() == this);

  // Insert assignment
  assignment->object_index = assignments.NEntries();
  assignments.Insert(assignment);
}



void R3SurfelObject::
UpdateBeforeRemoveLabelAssignment(R3SurfelLabelAssignment *assignment)
{
  // Check assignment
  assert(assignment->Scene());
  assert(assignment->Scene() == scene);
  assert(assignment->Object() == this);

  // Remove assignment
  RNArrayEntry *entry = assignments.KthEntry(assignment->object_index);
  R3SurfelLabelAssignment *tail = assignments.Tail();
  tail->object_index = assignment->object_index;
  assignments.EntryContents(entry) = tail;
  assignments.RemoveTail();
  assignment->object_index = -1;
}


////////////////////////////////////////////////////////////////////////
// BLOCK MANIPULATION UPDATE FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelObject::
UpdateAfterInsertBlock(R3SurfelNode *node, R3SurfelBlock *block)
{
  // Update complexity
  complexity += block->NSurfels();

  // Invalidate properties
  R3SurfelObject *ancestor = this;
  while (ancestor) {
    if (ancestor->bbox[0][0] == FLT_MAX) break;
    ancestor->bbox[0][0] = FLT_MAX;
    ancestor = ancestor->parent;
  }
}



void R3SurfelObject::
UpdateBeforeRemoveBlock(R3SurfelNode *node, R3SurfelBlock *block)
{
  // Update complexity
  complexity -= block->NSurfels();

  // Invalidate properties
  R3SurfelObject *ancestor = this;
  while (ancestor) {
    if (ancestor->bbox[0][0] == FLT_MAX) break;
    ancestor->bbox[0][0] = FLT_MAX;
    ancestor = ancestor->parent;
  }
}



void R3SurfelObject::
UpdateAfterTransform(R3SurfelNode *node)
{
  // Invalidate bboxes
  R3SurfelObject *ancestor = this;
  while (ancestor) {
    if (ancestor->bbox[0][0] == FLT_MAX) break;
    ancestor->bbox[0][0] = FLT_MAX;
    ancestor = ancestor->parent;
  }
}



////////////////////////////////////////////////////////////////////////
// UPDATE FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelObject::
UpdateProperties(void)
{
  // Update properties
  double dummy = 0;
  dummy += Complexity();
  dummy += BBox().Min().X();
  if (dummy == 927612.21242) {
    printf("Amazing!\n");
  }
}



void R3SurfelObject::
UpdateFeatureVector(void)
{
  // Check if need update
  if (!scene) return;
  if (scene->NFeatures() == 0) return;
  if (feature_vector.NValues() == scene->NFeatures()) return;

  // Create feature vector
  R3SurfelFeatureVector vector(scene->NFeatures());

  // Evaluate features
  for (int i = 0; i < scene->NFeatures(); i++) {
    R3SurfelFeature *feature = scene->Feature(i);
    feature->UpdateFeatureVector(this, vector);
  }

  // Set feature vector
  SetFeatureVector(vector);
}



void R3SurfelObject::
UpdateBBox(void)
{
  // Check if bounding box is uptodate
  if (bbox[0][0] != FLT_MAX) return;

  // Initialize bounding box
  bbox = R3null_box;

  // Union bounding boxes of nodes
  for (int i = 0; i < NNodes(); i++) {
    R3SurfelNode *node = Node(i);
    bbox.Union(node->BBox());
  }

  // Union bounding boxes of parts
  for (int i = 0; i < NParts(); i++) {
    R3SurfelObject *part = Part(i);
    bbox.Union(part->BBox());
  }
}








