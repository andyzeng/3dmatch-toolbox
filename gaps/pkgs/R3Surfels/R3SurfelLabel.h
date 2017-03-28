/* Include file for the R3 surfel label class */



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelLabel {
public:
  //////////////////////////////////////////
  //// CONSTRUCTOR/DESTRUCTOR FUNCTIONS ////
  //////////////////////////////////////////

  R3SurfelLabel(const char *name = NULL);
  R3SurfelLabel(const R3SurfelLabel& label);

  // Destructor function
  virtual ~R3SurfelLabel(void);


  ////////////////////////////
  //// PROPERTY FUNCTIONS ////
  ////////////////////////////

  // Name property functions
  const char *Name(void) const;
  int Identifier(void) const;
  int AssignmentKeystroke(void) const;
  const RNRgb& Color(void) const;
  void *Data(void) const;


  //////////////////////////
  //// ACCESS FUNCTIONS ////
  //////////////////////////

  // Scene access functions
  R3SurfelScene *Scene(void) const;
  int SceneIndex(void) const;

  // Part access functions
  int NParts(void) const;
  R3SurfelLabel *Part(int k) const;
  R3SurfelLabel *Parent(void) const;
  int PartHierarchyLevel(void) const;

  // Property access functions
  int NLabelProperties(void) const;
  R3SurfelLabelProperty *LabelProperty(int k) const;
  R3SurfelLabelProperty *FindLabelProperty(int type) const;

  // Relationship access functions
  int NLabelRelationships(void) const;
  R3SurfelLabelRelationship *LabelRelationship(int k) const;

  // Assignment access functions
  int NLabelAssignments(void) const;
  R3SurfelLabelAssignment *LabelAssignment(int k) const;


  ////////////////////////////////
  //// PROPERTY MANIPULATION FUNCTIONS ////
  ////////////////////////////////

  // Manipulation functions
  virtual void SetParent(R3SurfelLabel *parent);
  virtual void SetName(const char *name);
  virtual void SetIdentifier(int identifier);
  virtual void SetAssignmentKeystroke(int key);
  virtual void SetColor(const RNRgb& color);
  virtual void SetData(void *data);


  ///////////////////////////
  //// DISPLAY FUNCTIONS ////
  ///////////////////////////

  // Print function
  virtual void Print(FILE *fp = NULL, const char *prefix = NULL, const char *suffix = NULL) const;


  ////////////////////////////////////////////////////////////////////////
  // INTERNAL STUFF BELOW HERE
  ////////////////////////////////////////////////////////////////////////

protected:
  // Label update functions
  void UpdateAfterInsert(R3SurfelScene *scene);
  void UpdateBeforeRemove(R3SurfelScene *scene);

  // Property update functions
  void UpdateAfterInsertLabelProperty(R3SurfelLabelProperty *property);
  void UpdateBeforeRemoveLabelProperty(R3SurfelLabelProperty *property);

  // Relationship update functions
  void UpdateAfterInsertLabelRelationship(R3SurfelLabelRelationship *relationship);
  void UpdateBeforeRemoveLabelRelationship(R3SurfelLabelRelationship *relationship);

  // Assignment update functions
  void UpdateAfterInsertLabelAssignment(R3SurfelLabelAssignment *assignment);
  void UpdateBeforeRemoveLabelAssignment(R3SurfelLabelAssignment *assignment);

protected:
  // Internal instance data
  friend class R3SurfelScene;
  R3SurfelScene *scene;
  int scene_index;
  R3SurfelLabel *parent;
  RNArray<R3SurfelLabel *> parts;
  RNArray<R3SurfelLabelProperty *> properties;
  RNArray<R3SurfelLabelRelationship *> relationships;
  RNArray<R3SurfelLabelAssignment *> assignments;
  char *name;
  int identifier;
  int assignment_keystroke;
  RNRgb color;
  void *data;
};



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline const char *R3SurfelLabel::
Name(void) const
{
  // Return name
  return name;
}



inline int R3SurfelLabel::
AssignmentKeystroke(void) const
{
  // Return assignment key
  return assignment_keystroke;
}



inline const RNRgb& R3SurfelLabel::
Color(void) const
{
  // Return color
  return color;
}



inline void *R3SurfelLabel::
Data(void) const
{
  // Return user data
  return data;
}



inline R3SurfelScene *R3SurfelLabel::
Scene(void) const
{
  // Return scene this label is in
  return scene;
}



inline int R3SurfelLabel::
SceneIndex(void) const
{
  // Return index in list of labels associated with scene
  return scene_index;
}



inline int R3SurfelLabel::
NParts(void) const
{
  // Return number of parts in hierarchy
  return parts.NEntries();
}



inline R3SurfelLabel *R3SurfelLabel::
Part(int k) const
{
  // Return kth part in hierarchy
  return parts.Kth(k);
}



inline R3SurfelLabel *R3SurfelLabel::
Parent(void) const
{
  // Return parent in hierarchy
  return parent;
}



inline int R3SurfelLabel::
NLabelProperties(void) const
{
  // Return number of properties
  return properties.NEntries();
}



inline R3SurfelLabelProperty *R3SurfelLabel::
LabelProperty(int k) const
{
  // Return kth property
  return properties[k];
}



inline int R3SurfelLabel::
NLabelRelationships(void) const
{
  // Return number of relationships
  return relationships.NEntries();
}



inline R3SurfelLabelRelationship *R3SurfelLabel::
LabelRelationship(int k) const
{
  // Return kth relationship
  return relationships[k];
}



inline int R3SurfelLabel::
NLabelAssignments(void) const
{
  // Return number of assignments
  return assignments.NEntries();
}



inline R3SurfelLabelAssignment *R3SurfelLabel::
LabelAssignment(int k) const
{
  // Return kth assignment
  return assignments[k];
}



inline void R3SurfelLabel::
UpdateAfterInsert(R3SurfelScene *scene)
{
}



inline void R3SurfelLabel::
UpdateBeforeRemove(R3SurfelScene *scene)
{
}



