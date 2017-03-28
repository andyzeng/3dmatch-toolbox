/* Include file for the R3 surfel label assignment class */



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelLabelAssignment {
public:
  //////////////////////////////////////////
  //// CONSTRUCTOR/DESTRUCTOR FUNCTIONS ////
  //////////////////////////////////////////

  // Constructor functions
  R3SurfelLabelAssignment(R3SurfelObject *object = NULL, R3SurfelLabel *label = NULL, double confidence = 0, int originator = 0);
  R3SurfelLabelAssignment(const R3SurfelLabelAssignment& assignment);

  // Destructor function
  virtual ~R3SurfelLabelAssignment(void);


  ////////////////////////////
  //// PROPERTY FUNCTIONS ////
  ////////////////////////////

  // Confidence property functions
  double Confidence(void) const;

  // Other properties
  int Originator(void) const;

  // User data property functions
  void *Data(void) const;


  //////////////////////////
  //// ACCESS FUNCTIONS ////
  //////////////////////////

  // Scene access functions
  R3SurfelScene *Scene(void) const;
  int SceneIndex(void) const;

  // Object access functions
  R3SurfelObject *Object(void) const;
  int ObjectIndex(void) const;

  // Label access functions
  R3SurfelLabel *Label(void) const;
  int LabelIndex(void) const;


  /////////////////////////////////
  //// MANIPULUATION FUNCTIONS ////
  /////////////////////////////////

  // Property manipulation functions
  void SetConfidence(double confidence);
  void SetOriginator(int originator);

  // User data manipulation functions
  virtual void SetData(void *data);


  ////////////////////////////////////////////////////////////////////////
  // INTERNAL STUFF FROM HERE DOWN
  ////////////////////////////////////////////////////////////////////////

protected:
  // Update functions
  void UpdateAfterInsert(R3SurfelScene *scene);
  void UpdateBeforeRemove(R3SurfelScene *scene);

protected:
  friend class R3SurfelScene;
  R3SurfelScene *scene;
  int scene_index;
  friend class R3SurfelObject;
  R3SurfelObject *object;
  int object_index;
  friend class R3SurfelLabel;
  R3SurfelLabel *label;
  int label_index;
  double confidence;
  int originator;
  void *data;
};



////////////////////////////////////////////////////////////////////////
// CONSTANT DEFINITIONS
////////////////////////////////////////////////////////////////////////

// Originator types

typedef enum {
  R3_SURFEL_LABEL_ASSIGNMENT_HUMAN_ORIGINATOR,
  R3_SURFEL_LABEL_ASSIGNMENT_MACHINE_ORIGINATOR,
  R3_SURFEL_LABEL_ASSIGNMENT_GROUND_TRUTH_ORIGINATOR,
  R3_SURFEL_LABEL_ASSIGNMENT_NUM_ORIGINATORS
} R3SurfelLabelAssignmentOriginator;



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline double R3SurfelLabelAssignment::
Confidence(void) const
{
  // Return confidence
  return confidence;
}



inline int R3SurfelLabelAssignment::
Originator(void) const
{
  // Return originator
  return originator;
}



inline void *R3SurfelLabelAssignment::
Data(void) const
{
  // Return user data
  return data;
}



inline R3SurfelScene *R3SurfelLabelAssignment::
Scene(void) const
{
  // Return scene
  return scene;
}



inline int R3SurfelLabelAssignment::
SceneIndex(void) const
{
  // Return index in scene's list
  return scene_index;
}



inline R3SurfelObject *R3SurfelLabelAssignment::
Object(void) const
{
  // Return object
  return object;
}



inline int R3SurfelLabelAssignment::
ObjectIndex(void) const
{
  // Return index in object's list
  return object_index;
}



inline R3SurfelLabel *R3SurfelLabelAssignment::
Label(void) const
{
  // Return label
  return label;
}



inline int R3SurfelLabelAssignment::
LabelIndex(void) const
{
  // Return index in label's list
  return label_index;
}



inline void R3SurfelLabelAssignment::
UpdateAfterInsert(R3SurfelScene *scene)
{
}



inline void R3SurfelLabelAssignment::
UpdateBeforeRemove(R3SurfelScene *scene)
{
}



