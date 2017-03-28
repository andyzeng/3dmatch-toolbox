/* Include file for the R3 surfel object set class */



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelObjectSet {
public:
  //////////////////////////////////////////
  //// CONSTRUCTOR/DESTRUCTOR FUNCTIONS ////
  //////////////////////////////////////////

  // Constructor functions
  R3SurfelObjectSet(void);
  R3SurfelObjectSet(const R3SurfelObjectSet& set);

  // Destructor function
  virtual ~R3SurfelObjectSet(void);

  ////////////////////////////
  //// PROPERTY FUNCTIONS ////
  ////////////////////////////

  // Geometric property functions
  R3Point Centroid(void) const;
  const R3Box& BBox(void) const;


  //////////////////////////
  //// ACCESS FUNCTIONS ////
  //////////////////////////

  // Object access functions
  int NObjects(void) const;
  R3SurfelObject *Object(int k) const;
  R3SurfelObject *operator[](int k) const;
  int ObjectIndex(R3SurfelObject *object) const;


  /////////////////////////////////////////
  //// MANIPULATION FUNCTIONS ////
  /////////////////////////////////////////

  // Object manipulation functions
  virtual void InsertObject(R3SurfelObject *object);
  virtual void RemoveObject(R3SurfelObject *object);
  virtual void RemoveObject(int k);
  virtual void Empty(void);


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

private:
  RNArray<R3SurfelObject *> objects;
  R3Box bbox;
};



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline R3Point R3SurfelObjectSet::
Centroid(void) const
{
  // Return centroid of set
  return BBox().Centroid();
}



inline int R3SurfelObjectSet::
NObjects(void) const
{
  // Return number of objects
  return objects.NEntries();
}



inline R3SurfelObject *R3SurfelObjectSet::
Object(int k) const
{
  // Return kth object
  return objects.Kth(k);
}



inline R3SurfelObject *R3SurfelObjectSet::
operator[](int k) const
{
  // Return kth object
  return Object(k);
}



inline int R3SurfelObjectSet::
ObjectIndex(R3SurfelObject *object) const
{
  // Return index of object (or -1 if not found)
  RNArrayEntry *entry = objects.FindEntry(object);
  return (entry) ? objects.EntryIndex(entry) : -1;
}



