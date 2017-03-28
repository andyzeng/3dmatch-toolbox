/* Source file for the R3 label class */



////////////////////////////////////////////////////////////////////////
// INCLUDE FILES
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// PRIVATE GLOBAL VARIABLES
////////////////////////////////////////////////////////////////////////

static int R3surfel_next_label_identifier = 0;



////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS/DESTRUCTORS
////////////////////////////////////////////////////////////////////////

R3SurfelLabel::
R3SurfelLabel(const char *name)
  : scene(NULL),
    scene_index(-1),
    parent(NULL),
    parts(),
    properties(),
    relationships(),
    assignments(),
    name((name) ? strdup(name) : NULL),
    identifier(-1),
    assignment_keystroke(-1),
    color(0.5,0.5,0.5),
    data(NULL)
{
}



R3SurfelLabel::
R3SurfelLabel(const R3SurfelLabel& label)
  : scene(NULL),
    scene_index(-1),
    parent(NULL),
    parts(),
    properties(),
    relationships(),
    assignments(),
    name((label.name) ? strdup(label.name) : NULL),
    identifier(-1),
    assignment_keystroke(label.assignment_keystroke),
    color(label.color),
    data(NULL)
{
}



R3SurfelLabel::
~R3SurfelLabel(void)
{
  // Delete label properties
  while (NLabelProperties() > 0) {
    R3SurfelLabelProperty *property = LabelProperty(NLabelProperties()-1);
    delete property;
  }

  // Delete label relationships
  while (NLabelRelationships() > 0) {
    R3SurfelLabelRelationship *relationship = LabelRelationship(NLabelRelationships()-1);
    delete relationship;
  }

  // Delete label assignments
  while (NLabelAssignments() > 0) {
    R3SurfelLabelAssignment *assignment = LabelAssignment(NLabelAssignments()-1);
    delete assignment;
  }

  // Delete label from scene
  if (scene) scene->RemoveLabel(this);

  // Delete name
  if (name) free(name);
}



////////////////////////////////////////////////////////////////////////
// PROPERTY ACCESS FUNCTIONS
////////////////////////////////////////////////////////////////////////

R3SurfelLabelProperty *R3SurfelLabel::
FindLabelProperty(int type) const
{
  // Search for property of type
  for (int i = 0; i < NLabelProperties(); i++) {
    R3SurfelLabelProperty *property = LabelProperty(i);
    if (property->Type() == type) return property;
  }

  // Property not found
  return NULL;
}



////////////////////////////////////////////////////////////////////////
// PROPERTY FUNCTIONS
////////////////////////////////////////////////////////////////////////

int R3SurfelLabel::
Identifier(void) const
{
  // Update ID 
  if (identifier < 0) {
    ((R3SurfelLabel *) this)->identifier = R3surfel_next_label_identifier++;
  }

  // Return identifier 
  return identifier;
}



int R3SurfelLabel::
PartHierarchyLevel(void) const
{
  // Return level in part hierarchy (root is 0)
  int level = 0;
  R3SurfelLabel *ancestor = parent;
  while (ancestor) { level++; ancestor = ancestor->parent; }
  return level;
}



////////////////////////////////////////////////////////////////////////
// PROPERTY MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelLabel::
SetParent(R3SurfelLabel *parent)
{
  // Just checking
  assert(parent);
  assert(this->parent);
  assert(scene == this->parent->scene);
  assert(parent->scene == this->parent->scene);
  if (parent == this->parent) return;

  // Update hierarchy
  this->parent->parts.Remove(this);
  parent->parts.Insert(this);
  this->parent = parent;

  // Mark scene as dirty
  if (scene) scene->SetDirty();
}



void R3SurfelLabel::
SetName(const char *name)
{
  // Delete previous name
  if (this->name) free(this->name);

  // Set new name
  this->name = (name) ? strdup(name) : NULL;

  // Mark scene as dirty
  if (scene) scene->SetDirty();
}



void R3SurfelLabel::
SetIdentifier(int identifier)
{
  // Set identifier
  this->identifier = identifier;

  // Update next identifier
  if (identifier >= R3surfel_next_label_identifier) {
    R3surfel_next_label_identifier = identifier+1;
  }

  // Mark scene as dirty
  if (scene) scene->SetDirty();
}



void R3SurfelLabel::
SetAssignmentKeystroke(int key)
{
  // Set key that user can press to assign this label
  this->assignment_keystroke = key;

  // Mark scene as dirty
  if (scene) scene->SetDirty();
}



void R3SurfelLabel::
SetColor(const RNRgb& color)
{
  // Set color
  this->color = color;

  // Mark scene as dirty
  // XXX if (scene) scene->SetDirty();
}



void R3SurfelLabel::
SetData(void *data) 
{
  // Set user data
  this->data = data;
}



////////////////////////////////////////////////////////////////////////
// STRUCTURE UPDATE FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelLabel::
UpdateAfterInsertLabelProperty(R3SurfelLabelProperty *property)
{
  // Just checking
  assert(property->Scene());
  assert(property->Scene() == scene);

  // Insert property 
  properties.Insert(property);
}



void R3SurfelLabel::
UpdateBeforeRemoveLabelProperty(R3SurfelLabelProperty *property)
{
  // Just checking
  assert(property->Scene());
  assert(property->Scene() == scene);

  // Remove property 
  properties.Remove(property);
}



void R3SurfelLabel::
UpdateAfterInsertLabelRelationship(R3SurfelLabelRelationship *relationship)
{
  // Just checking
  assert(relationship->Scene());
  assert(relationship->Scene() == scene);

  // Insert relationship 
  relationships.Insert(relationship);
}



void R3SurfelLabel::
UpdateBeforeRemoveLabelRelationship(R3SurfelLabelRelationship *relationship)
{
  // Just checking
  assert(relationship->Scene());
  assert(relationship->Scene() == scene);

  // Remove relationship 
  relationships.Remove(relationship);
}



void R3SurfelLabel::
UpdateAfterInsertLabelAssignment(R3SurfelLabelAssignment *assignment)
{
  // Just checking
  assert(assignment->Scene());
  assert(assignment->Scene() == scene);
  assert(assignment->Label() == this);

  // Insert assignment
  assignment->label_index = assignments.NEntries();
  assignments.Insert(assignment);
}



void R3SurfelLabel::
UpdateBeforeRemoveLabelAssignment(R3SurfelLabelAssignment *assignment)
{
  // Just checking
  assert(assignment->Scene());
  assert(assignment->Scene() == scene);
  assert(assignment->Label() == this);

  // Remove assignment
  RNArrayEntry *entry = assignments.KthEntry(assignment->label_index);
  R3SurfelLabelAssignment *tail = assignments.Tail();
  tail->label_index = assignment->label_index;
  assignments.EntryContents(entry) = tail;
  assignments.RemoveTail();
  assignment->label_index = -1;
}



////////////////////////////////////////////////////////////////////////
// DISPLAY FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelLabel::
Print(FILE *fp, const char *prefix, const char *suffix) const
{
  // Check fp
  if (!fp) fp = stdout;

  // Print label
  if (prefix) fprintf(fp, "%s", prefix);
  fprintf(fp, "%d %s", scene_index, (name) ? name : "-");
  if (suffix) fprintf(fp, "%s", suffix);
  fprintf(fp, "\n");
}



