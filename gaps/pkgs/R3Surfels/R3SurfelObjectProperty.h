/* Include file for the R3 surfel object property class */



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelObjectProperty {
public:
  //////////////////////////////////////////
  //// CONSTRUCTOR/DESTRUCTOR FUNCTIONS ////
  //////////////////////////////////////////

  // Constructor functions
  R3SurfelObjectProperty(int type = 0, R3SurfelObject *object = NULL, 
    RNScalar *operands = NULL, int noperands = 0);
  R3SurfelObjectProperty(const R3SurfelObjectProperty& property);

  // Destructor function
  virtual ~R3SurfelObjectProperty(void);


  ////////////////////////////
  //// PROPERTY FUNCTIONS ////
  ////////////////////////////

  // Property type
  int Type(void) const;


  ////////////////////////////
  //// ACCESS FUNCTIONS ////
  ////////////////////////////

  // Scene access functions
  R3SurfelScene *Scene(void) const;
  int SceneIndex(void) const;

  // Object access functions 
  R3SurfelObject *Object(void) const;

  // Operand access functions 
  int NOperands(void) const;
  RNScalar Operand(int k) const;


  ///////////////////////////
  //// DISPLAY FUNCTIONS ////
  ///////////////////////////

  // Draw function
  virtual void Draw(RNFlags flags = R3_SURFEL_DEFAULT_DRAW_FLAGS) const;



  ////////////////////////////////////////////////////////////////////////
  // INTERNAL STUFF BELOW HERE
  ////////////////////////////////////////////////////////////////////////

  // Update functions
  void UpdateOperands(void);


protected:
  // Update functions
  void UpdateAfterInsert(R3SurfelScene *scene);
  void UpdateBeforeRemove(R3SurfelScene *scene);

protected:
  friend class R3SurfelScene;
  R3SurfelScene *scene;
  int scene_index;
  R3SurfelObject *object;
  RNScalar *operands;
  int noperands;
  int type;
};



////////////////////////////////////////////////////////////////////////
// PREDEFINED OBJECT PROPERTY TYPES
////////////////////////////////////////////////////////////////////////

enum {
  R3_SURFEL_OBJECT_NULL_PROPERTY,
  R3_SURFEL_OBJECT_PCA_PROPERTY,
  R3_SURFEL_OBJECT_NUM_PROPERTIES
};



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline int R3SurfelObjectProperty::
Type(void) const
{
  // Return property type
  return type;
}


inline R3SurfelScene *R3SurfelObjectProperty::
Scene(void) const
{
  // Return scene
  return scene;
}



inline int R3SurfelObjectProperty::
SceneIndex(void) const
{
  // Return index of this property in scene
  return scene_index;
}



inline R3SurfelObject *R3SurfelObjectProperty::
Object(void) const
{
  // Return object
  return object;
}



inline int R3SurfelObjectProperty::
NOperands(void) const
{
  // Return number of operands
  return noperands;
}



inline RNScalar R3SurfelObjectProperty::
Operand(int k) const
{
  // Return operand
  assert((k >= 0) && (k < NOperands()));
  return operands[k];
}



inline void R3SurfelObjectProperty::
UpdateAfterInsert(R3SurfelScene *scene)
{
}



inline void R3SurfelObjectProperty::
UpdateBeforeRemove(R3SurfelScene *scene)
{
}



