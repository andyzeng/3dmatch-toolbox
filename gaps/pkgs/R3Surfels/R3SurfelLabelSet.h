/* Include file for the R3 surfel label set class */



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelLabelSet {
public:
  //////////////////////////////////////////
  //// CONSTRUCTOR/DESTRUCTOR FUNCTIONS ////
  //////////////////////////////////////////

  // Constructor functions
  R3SurfelLabelSet(void);
  R3SurfelLabelSet(const R3SurfelLabelSet& set);

  // Destructor function
  virtual ~R3SurfelLabelSet(void);


  //////////////////////////
  //// ACCESS FUNCTIONS ////
  //////////////////////////

  // Label access functions
  int NLabels(void) const;
  R3SurfelLabel *Label(int k) const;
  R3SurfelLabel *operator[](int k) const;


  /////////////////////////////////////////
  //// MANIPULATION FUNCTIONS ////
  /////////////////////////////////////////

  // Function insertion/removal functions
  virtual void InsertLabel(R3SurfelLabel *label);
  virtual void RemoveLabel(R3SurfelLabel *label);
  virtual void RemoveLabel(int k);
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
  RNArray<R3SurfelLabel *> labels;
};



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline int R3SurfelLabelSet::
NLabels(void) const
{
  // Return number of labels
  return labels.NEntries();
}



inline R3SurfelLabel *R3SurfelLabelSet::
Label(int k) const
{
  // Return kth label
  return labels.Kth(k);
}



inline R3SurfelLabel *R3SurfelLabelSet::
operator[](int k) const
{
  // Return kth label
  return Label(k);
}





