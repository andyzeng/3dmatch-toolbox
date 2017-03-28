/* Include file for the R3 surfel feature class */



/* Class definition */

class R3SurfelFeature {
public:
  //////////////////////////////////////////////
  //// CONSTRUCTOR AND DESTRUCTOR FUNCTIONS ////
  //////////////////////////////////////////////

  // Constructor functions
  R3SurfelFeature(const char *name = NULL, 
    RNScalar minimum = -FLT_MAX, RNScalar maximum = FLT_MAX, 
    RNScalar weight = 1);
  R3SurfelFeature(const R3SurfelFeature& feature);

  // Destructor functions
  virtual ~R3SurfelFeature(void);


  //////////////////////////
  //// ACCESS FUNCTIONS ////
  //////////////////////////

  // Scene access functions
  R3SurfelScene *Scene(void) const;
  int SceneIndex(void) const;

  ////////////////////////////
  //// PROPERTY FUNCTIONS ////
  ////////////////////////////

  // Name properties
  const char *Name(void) const;

  // Range properties
  const RNInterval& Range(void) const;
  RNScalar Minimum(void) const;
  RNScalar Maximum(void) const;

  // Weight property
  RNScalar Weight(void) const;

  // Feature type
  virtual int Type(void) const;

  // User data
  void *Data(void) const;


  ///////////////////////////////
  //// MANIPULTION FUNCTIONS ////
  ///////////////////////////////

  // Name manipulation
  void SetName(const char *name);

  // Range manipulation
  virtual void SetRange(const RNInterval& range);
  virtual void SetRange(RNScalar minimum, RNScalar maximum);
  virtual void SetMinimum(RNScalar minimum);
  virtual void SetMaximum(RNScalar maximum);

  // Weight manipulation
  virtual void SetWeight(RNScalar weight);

  // User data manipulation
  virtual void SetData(void *data);


  //////////////////////////////
  //// EVALUATION FUNCTIONS ////
  //////////////////////////////

  // Compute feature(s) for object
  virtual int UpdateFeatureVector(R3SurfelObject *object, R3SurfelFeatureVector& vector) const;
  

  ///////////////////////////
  //// DISPLAY FUNCTIONS ////
  ///////////////////////////

  // Print function
  virtual void Print(FILE *fp = NULL, const char *prefix = NULL, const char *suffix = NULL) const;


  ////////////////////////////////////////////////////////////////////////
  // INTERNAL STUFF BELOW HERE
  ////////////////////////////////////////////////////////////////////////

private:
  friend class R3SurfelScene;
  R3SurfelScene *scene;
  int scene_index;
  char *name;
  RNInterval range;
  RNScalar weight;
  int type;
  void *data;
};



////////////////////////////////////////////////////////////////////////
// FEATURE TYPE DEFINITIONS
////////////////////////////////////////////////////////////////////////

enum {
  R3_SURFEL_BASIC_FEATURE_TYPE,
  R3_SURFEL_POINTSET_FEATURE_TYPE,
  R3_SURFEL_VOLUME_GRID_FEATURE_TYPE,
  R3_SURFEL_OVERHEAD_GRID_FEATURE_TYPE,
  R3_SURFEL_NUM_FEATURE_TYPES
};



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline R3SurfelScene *R3SurfelFeature::
Scene(void) const
{
  // Return scene this feature is in
  return scene;
}



inline int R3SurfelFeature::
SceneIndex(void) const
{
  // Return index in list of features associated with scene
  return scene_index;
}



inline const char *R3SurfelFeature::
Name(void) const
{
  // Return name
  return name;
}



inline const RNInterval& R3SurfelFeature::
Range(void) const
{
  // Return range
  return range;
}



inline RNScalar R3SurfelFeature::
Minimum(void) const
{
  // Return minimum
  return range.Min();
}



inline RNScalar R3SurfelFeature::
Maximum(void) const
{
  // Return maximum
  return range.Max();
}



inline RNScalar R3SurfelFeature::
Weight(void) const
{
  // Return weight
  return weight;
}



inline void *R3SurfelFeature::
Data(void) const
{
  // Return user data
  return data;
}



