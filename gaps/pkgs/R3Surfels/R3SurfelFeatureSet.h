/* Include file for the R3 surfel feature set class */



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelFeatureSet {
public:
  //////////////////////////////////////////
  //// CONSTRUCTOR/DESTRUCTOR FUNCTIONS ////
  //////////////////////////////////////////

  // Constructor functions
  R3SurfelFeatureSet(void);
  R3SurfelFeatureSet(const R3SurfelFeatureSet& set);

  // Destructor function
  virtual ~R3SurfelFeatureSet(void);


  //////////////////////////
  //// ACCESS FUNCTIONS ////
  //////////////////////////

  // Feature access functions
  int NFeatures(void) const;
  R3SurfelFeature *Feature(int k) const;
  R3SurfelFeature *operator[](int k) const;


  /////////////////////////////////////////
  //// MANIPULATION FUNCTIONS ////
  /////////////////////////////////////////

  // Function insertion/removal functions
  virtual void InsertFeature(R3SurfelFeature *feature);
  virtual void RemoveFeature(R3SurfelFeature *feature);
  virtual void RemoveFeature(int k);
  virtual void Empty(void);


  ///////////////////////////
  //// DISPLAY FUNCTIONS ////
  ///////////////////////////

  // Print function
  virtual void Print(FILE *fp = NULL, const char *prefix = NULL, const char *suffix = NULL) const;


  ////////////////////////////////////////////////////////////////////////
  // INTERNAL STUFF BELOW HERE
  ////////////////////////////////////////////////////////////////////////

private:
  RNArray<R3SurfelFeature *> features;
};



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline int R3SurfelFeatureSet::
NFeatures(void) const
{
  // Return number of features
  return features.NEntries();
}



inline R3SurfelFeature *R3SurfelFeatureSet::
Feature(int k) const
{
  // Return kth feature
  return features.Kth(k);
}



inline R3SurfelFeature *R3SurfelFeatureSet::
operator[](int k) const
{
  // Return kth feature
  return Feature(k);
}





