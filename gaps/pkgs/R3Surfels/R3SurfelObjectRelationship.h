/* Include file for the R3 surfel object relationship class */



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelObjectRelationship {
public:
  //////////////////////////////////////////
  //// CONSTRUCTOR/DESTRUCTOR FUNCTIONS ////
  //////////////////////////////////////////

  // Constructor functions
  R3SurfelObjectRelationship(int type = 0, R3SurfelObject *object0 = NULL, R3SurfelObject *object1 = NULL, 
    RNScalar *operands = NULL, int noperands = 0);
  R3SurfelObjectRelationship(int type, const RNArray<R3SurfelObject *>& objects, 
    RNScalar *operands = NULL, int noperands = 0);
  R3SurfelObjectRelationship(const R3SurfelObjectRelationship& relationship);

  // Destructor function
  virtual ~R3SurfelObjectRelationship(void);


  ////////////////////////////
  //// PROPERTY FUNCTIONS ////
  ////////////////////////////

  // Relationship type
  int Type(void) const;


  ////////////////////////////
  //// ACCESS FUNCTIONS ////
  ////////////////////////////

  // Scene access functions
  R3SurfelScene *Scene(void) const;
  int SceneIndex(void) const;

  // Object access functions 
  int NObjects(void) const;
  R3SurfelObject *Object(int k) const;

  // Operand access functions 
  int NOperands(void) const;
  RNScalar Operand(int k) const;


  ////////////////////////////////////////////////////////////////////////
  // INTERNAL STUFF BELOW HERE
  ////////////////////////////////////////////////////////////////////////

protected:
  // Update functions
  void UpdateAfterInsert(R3SurfelScene *scene);
  void UpdateBeforeRemove(R3SurfelScene *scene);

  // Operand functions
  void UpdateOperands(void);


protected:
  friend class R3SurfelScene;
  R3SurfelScene *scene;
  int scene_index;
  RNArray<R3SurfelObject *> objects;
  RNScalar *operands;
  int noperands;
  int type;
};



////////////////////////////////////////////////////////////////////////
// PREDEFINED OBJECT RELATIONSHIP TYPES
////////////////////////////////////////////////////////////////////////

enum {
  R3_SURFEL_OBJECT_NULL_RELATIONSHIP,
  R3_SURFEL_OBJECT_SIMILARITY_RELATIONSHIP,
  R3_SURFEL_OBJECT_NUM_RELATIONSHIPS
};



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline int R3SurfelObjectRelationship::
Type(void) const
{
  // Return relationship type
  return type;
}


inline R3SurfelScene *R3SurfelObjectRelationship::
Scene(void) const
{
  // Return scene
  return scene;
}



inline int R3SurfelObjectRelationship::
SceneIndex(void) const
{
  // Return index of this relationship in scene
  return scene_index;
}



inline int R3SurfelObjectRelationship::
NObjects(void) const
{
  // Return number of objects
  return objects.NEntries();
}



inline R3SurfelObject *R3SurfelObjectRelationship::
Object(int k) const
{
  // Return object
  return objects[k];
}



inline int R3SurfelObjectRelationship::
NOperands(void) const
{
  // Return number of operands
  return noperands;
}



inline RNScalar R3SurfelObjectRelationship::
Operand(int k) const
{
  // Return operand
  assert((k >= 0) && (k < NOperands()));
  return operands[k];
}



inline void R3SurfelObjectRelationship::
UpdateAfterInsert(R3SurfelScene *scene)
{
}



inline void R3SurfelObjectRelationship::
UpdateBeforeRemove(R3SurfelScene *scene)
{
}



