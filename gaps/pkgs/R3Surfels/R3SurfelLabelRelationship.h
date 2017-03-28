/* Include file for the R3 surfel label relationship class */



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelLabelRelationship {
public:
  //////////////////////////////////////////
  //// CONSTRUCTOR/DESTRUCTOR FUNCTIONS ////
  //////////////////////////////////////////

  // Constructor functions
  R3SurfelLabelRelationship(int type = 0, R3SurfelLabel *label0 = NULL, R3SurfelLabel *label1 = NULL, 
    RNScalar *operands = NULL, int noperands = 0);
  R3SurfelLabelRelationship(int type, const RNArray<R3SurfelLabel *>& labels, 
    RNScalar *operands = NULL, int noperands = 0);
  R3SurfelLabelRelationship(const R3SurfelLabelRelationship& relationship);

  // Destructor function
  virtual ~R3SurfelLabelRelationship(void);


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

  // Label access functions 
  int NLabels(void) const;
  R3SurfelLabel *Label(int k) const;

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

protected:
  friend class R3SurfelScene;
  R3SurfelScene *scene;
  int scene_index;
  RNArray<R3SurfelLabel *> labels;
  RNScalar *operands;
  int noperands;
  int type;
};



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline int R3SurfelLabelRelationship::
Type(void) const
{
  // Return relationship type
  return type;
}


inline R3SurfelScene *R3SurfelLabelRelationship::
Scene(void) const
{
  // Return scene
  return scene;
}



inline int R3SurfelLabelRelationship::
SceneIndex(void) const
{
  // Return index of this relationship in scene
  return scene_index;
}



inline int R3SurfelLabelRelationship::
NLabels(void) const
{
  // Return number of labels
  return labels.NEntries();
}



inline R3SurfelLabel *R3SurfelLabelRelationship::
Label(int k) const
{
  // Return label
  return labels[k];
}



inline int R3SurfelLabelRelationship::
NOperands(void) const
{
  // Return number of operands
  return noperands;
}



inline RNScalar R3SurfelLabelRelationship::
Operand(int k) const
{
  // Return operand
  assert((k >= 0) && (k < NOperands()));
  return operands[k];
}



inline void R3SurfelLabelRelationship::
UpdateAfterInsert(R3SurfelScene *scene)
{
}



inline void R3SurfelLabelRelationship::
UpdateBeforeRemove(R3SurfelScene *scene)
{
}



