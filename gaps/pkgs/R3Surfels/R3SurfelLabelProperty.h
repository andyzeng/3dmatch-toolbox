/* Include file for the R3 surfel label property class */



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelLabelProperty {
public:
  //////////////////////////////////////////
  //// CONSTRUCTOR/DESTRUCTOR FUNCTIONS ////
  //////////////////////////////////////////

  // Constructor functions
  R3SurfelLabelProperty(int type = 0, R3SurfelLabel *label = NULL, 
    RNScalar *operands = NULL, int noperands = 0);
  R3SurfelLabelProperty(const R3SurfelLabelProperty& property);

  // Destructor function
  virtual ~R3SurfelLabelProperty(void);


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

  // Label access functions 
  R3SurfelLabel *Label(void) const;

  // Operand access functions 
  int NOperands(void) const;
  RNScalar Operand(int k) const;


  ////////////////////////////////////////////////////////////////////////
  // INTERNAL STUFF BELOW HERE
  ////////////////////////////////////////////////////////////////////////

  static int NOperands(int type);

protected:
  // Update functions
  void UpdateAfterInsert(R3SurfelScene *scene);
  void UpdateBeforeRemove(R3SurfelScene *scene);

protected:
  friend class R3SurfelScene;
  R3SurfelScene *scene;
  int scene_index;
  R3SurfelLabel *label;
  RNScalar *operands;
  int noperands;
  int type;
};



////////////////////////////////////////////////////////////////////////
// PREDEFINED LABEL PROPERTY TYPES
////////////////////////////////////////////////////////////////////////

enum {
  R3_SURFEL_LABEL_NULL_PROPERTY,
  R3_SURFEL_LABEL_NUM_PROPERTYS
};



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline int R3SurfelLabelProperty::
Type(void) const
{
  // Return property type
  return type;
}


inline R3SurfelScene *R3SurfelLabelProperty::
Scene(void) const
{
  // Return scene
  return scene;
}



inline int R3SurfelLabelProperty::
SceneIndex(void) const
{
  // Return index of this property in scene
  return scene_index;
}



inline R3SurfelLabel *R3SurfelLabelProperty::
Label(void) const
{
  // Return label
  return label;
}



inline int R3SurfelLabelProperty::
NOperands(void) const
{
  // Return number of operands
  return noperands;
}



inline RNScalar R3SurfelLabelProperty::
Operand(int k) const
{
  // Return operand
  assert((k >= 0) && (k < NOperands()));
  return operands[k];
}



inline void R3SurfelLabelProperty::
UpdateAfterInsert(R3SurfelScene *scene)
{
}



inline void R3SurfelLabelProperty::
UpdateBeforeRemove(R3SurfelScene *scene)
{
}



