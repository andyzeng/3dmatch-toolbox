/* Include file for the R3 surfel object class */



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelObject {
public:
  //////////////////////////////////////////
  //// CONSTRUCTOR/DESTRUCTOR FUNCTIONS ////
  //////////////////////////////////////////

  // Constructor functions
  R3SurfelObject(const char *name = NULL);
  R3SurfelObject(const R3SurfelObject& object);

  // Destructor function
  virtual ~R3SurfelObject(void);


  ////////////////////////////
  //// PROPERTY FUNCTIONS ////
  ////////////////////////////

  // Name property functions
  const char *Name(void) const;
  int Identifier(void) const;

  // Complexity property functions
  RNScalar Complexity(void) const;

  // Geometric property functions
  const R3Box& BBox(void) const;
  R3Point Centroid(void) const;

  // Feature vector functions
  const R3SurfelFeatureVector& FeatureVector(void) const;

  // User data property functions
  void *Data(void) const;


  //////////////////////////
  //// ACCESS FUNCTIONS ////
  //////////////////////////

  // Scene access functions
  R3SurfelScene *Scene(void) const;
  int SceneIndex(void) const;

  // Node access functions
  int NNodes(void) const;
  R3SurfelNode *Node(int k) const;

  // Part access functions
  int NParts(void) const;
  R3SurfelObject *Part(int k) const;
  R3SurfelObject *Parent(void) const;
  int PartHierarchyLevel(void) const;

  // Scan access functions
  R3SurfelScan *Scan(void) const;

  // Object property access functions
  int NObjectProperties(void) const;
  R3SurfelObjectProperty *ObjectProperty(int k) const;
  R3SurfelObjectProperty *FindObjectProperty(int type) const;

  // Object relationship access functions
  int NObjectRelationships(void) const;
  R3SurfelObjectRelationship *ObjectRelationship(int k) const;

  // Assignment access functions
  int NLabelAssignments(void) const;
  R3SurfelLabelAssignment *LabelAssignment(int k) const;

  // More assignment access functions
  R3SurfelLabelAssignment *GroundTruthLabelAssignment(void) const;
  R3SurfelLabelAssignment *HumanLabelAssignment(void) const;
  R3SurfelLabelAssignment *PredictedLabelAssignment(void) const;
  R3SurfelLabelAssignment *CurrentLabelAssignment(void) const;
  R3SurfelLabelAssignment *BestLabelAssignment(int originator) const;

  // More label access functions
  R3SurfelLabel *GroundTruthLabel(void) const;
  R3SurfelLabel *HumanLabel(void) const;
  R3SurfelLabel *PredictedLabel(void) const;
  R3SurfelLabel *CurrentLabel(void) const;
  R3SurfelLabel *BestLabel(int originator) const;

  // Point access functions
  R3SurfelPointSet *PointSet(RNBoolean leaf_level = FALSE) const;


  /////////////////////////////////////////
  //// PROPERTY MANIPULATION FUNCTIONS ////
  /////////////////////////////////////////

  // Name manipulation functions
  virtual void SetParent(R3SurfelObject *parent);
  virtual void SetName(const char *name);
  virtual void SetIdentifier(int identifier);

  // Feature vector manipulation functions
  virtual void SetFeatureVector(const R3SurfelFeatureVector& vector);

  // User data manipulation functions
  virtual void SetData(void *data);


  /////////////////////////////////////////
  //// STRUCTURE MANIPULATION FUNCTIONS ////
  /////////////////////////////////////////

  // Node manipulation functions
  void InsertNode(R3SurfelNode *node);
  void RemoveNode(R3SurfelNode *node);


  ///////////////////////////////////////
  //// SURFEL MANIPULATION FUNCTIONS ////
  ///////////////////////////////////////

  // Surfel manipulation functions
  virtual void SetMarks(RNBoolean mark = TRUE);


  /////////////////////////////////////
  //// MEMORY MANAGEMENT FUNCTIONS ////
  /////////////////////////////////////

  // Block memory management
  void ReadBlocks(void);
  void ReleaseBlocks(void);
  RNBoolean AreBlocksResident(void) const;


  ///////////////////////////
  //// DISPLAY FUNCTIONS ////
  ///////////////////////////

  // Draw function
  virtual void Draw(RNFlags flags = R3_SURFEL_DEFAULT_DRAW_FLAGS) const;

  // Print function
  virtual void Print(FILE *fp = NULL, const char *prefix = NULL, const char *suffix = NULL) const;


  ////////////////////////////////////////////////////////////////////////
  // INTERNAL STUFF BELOW HERE
  ////////////////////////////////////////////////////////////////////////

public:
  // Update functions
  void UpdateProperties(void);
  void UpdateFeatureVector(void);

  // Node update functions
  void UpdateAfterInsertBlock(R3SurfelNode *node, R3SurfelBlock *block);
  void UpdateBeforeRemoveBlock(R3SurfelNode *node, R3SurfelBlock *block);
  void UpdateAfterTransform(R3SurfelNode *node);

protected:
  // Object update functions
  void UpdateAfterInsert(R3SurfelScene *scene);
  void UpdateBeforeRemove(R3SurfelScene *scene);

  // Property update functions
  void UpdateAfterInsertObjectProperty(R3SurfelObjectProperty *property);
  void UpdateBeforeRemoveObjectProperty(R3SurfelObjectProperty *property);

  // Relationship update functions
  void UpdateAfterInsertObjectRelationship(R3SurfelObjectRelationship *relationship);
  void UpdateBeforeRemoveObjectRelationship(R3SurfelObjectRelationship *relationship);

  // Assignment update functions
  void UpdateAfterInsertLabelAssignment(R3SurfelLabelAssignment *assignment);
  void UpdateBeforeRemoveLabelAssignment(R3SurfelLabelAssignment *assignment);

  // Property update functions
  void UpdateBBox();

protected:
  // Internal instance data
  friend class R3SurfelScene;
  R3SurfelScene *scene;
  int scene_index;
  R3SurfelObject *parent;
  RNArray<R3SurfelObject *> parts;
  RNArray<R3SurfelObjectProperty *> properties;
  RNArray<R3SurfelObjectRelationship *> relationships;
  RNArray<R3SurfelLabelAssignment *> assignments;
  RNArray<R3SurfelNode *> nodes;
  R3SurfelFeatureVector feature_vector;
  char *name;
  int identifier;
  RNScalar complexity;
  R3Box bbox;
  void *data;
};



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline const char *R3SurfelObject::
Name(void) const
{
  // Return name
  return name;
}



inline RNScalar R3SurfelObject::
Complexity(void) const
{
  // Return compexity
  return complexity;
}



inline R3Point R3SurfelObject::
Centroid(void) const
{
  // Return centroid of object
  return BBox().Centroid();
}



inline void *R3SurfelObject::
Data(void) const
{
  // Return user data
  return data;
}



inline R3SurfelScene *R3SurfelObject::
Scene(void) const
{
  // Return scene this object is in
  return scene;
}



inline int R3SurfelObject::
SceneIndex(void) const
{
  // Return index in list of objects associated with scene
  return scene_index;
}


inline int R3SurfelObject::
NNodes(void) const
{
  // Return number of nodes
  return nodes.NEntries();
}



inline R3SurfelNode *R3SurfelObject::
Node(int k) const
{
  // Return kth node
  return nodes[k];
}



inline int R3SurfelObject::
NParts(void) const
{
  // Return number of parts in hierarchy
  return parts.NEntries();
}



inline R3SurfelObject *R3SurfelObject::
Part(int k) const
{
  // Return kth part in hierarchy
  return parts.Kth(k);
}



inline R3SurfelObject *R3SurfelObject::
Parent(void) const
{
  // Return parent in hierarchy
  return parent;
}



inline int R3SurfelObject::
NObjectProperties(void) const
{
  // Return number of properties
  return properties.NEntries();
}



inline R3SurfelObjectProperty *R3SurfelObject::
ObjectProperty(int k) const
{
  // Return kth property
  return properties[k];
}



inline int R3SurfelObject::
NObjectRelationships(void) const
{
  // Return number of relationships
  return relationships.NEntries();
}



inline R3SurfelObjectRelationship *R3SurfelObject::
ObjectRelationship(int k) const
{
  // Return kth relationship
  return relationships[k];
}



inline int R3SurfelObject::
NLabelAssignments(void) const
{
  // Return number of assignments
  return assignments.NEntries();
}



inline R3SurfelLabelAssignment *R3SurfelObject::
LabelAssignment(int k) const
{
  // Return kth assignment
  return assignments[k];
}



