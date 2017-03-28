/* Include file for the R3 surfel node set class */



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelNodeSet {
public:
  //////////////////////////////////////////
  //// CONSTRUCTOR/DESTRUCTOR FUNCTIONS ////
  //////////////////////////////////////////

  // Constructor functions
  R3SurfelNodeSet(void);
  R3SurfelNodeSet(const R3SurfelNodeSet& set);

  // Destructor function
  virtual ~R3SurfelNodeSet(void);


  ////////////////////////////
  //// PROPERTY FUNCTIONS ////
  ////////////////////////////

  // Complexity property functions
  RNScalar Complexity(void) const;

  // Geometric property functions
  R3Point Centroid(void) const;
  const R3Box& BBox(void) const;


  //////////////////////////
  //// ACCESS FUNCTIONS ////
  //////////////////////////

  // Node access functions
  int NNodes(void) const;
  R3SurfelNode *Node(int k) const;
  R3SurfelNode *operator[](int k) const;
  int NodeIndex(R3SurfelNode *node) const;


  /////////////////////////////////////////
  //// MANIPULATION FUNCTIONS ////
  /////////////////////////////////////////

  // High-level manipulation functions
  virtual void InsertNodes(R3SurfelTree *tree);
  virtual void InsertNodes(R3SurfelTree *tree, 
    const R3Point& xycenter, RNLength xyradius, 
    RNCoord zmin = -FLT_MAX, RNCoord zmax = FLT_MAX,
    RNScalar center_resolution = 0, RNScalar perimeter_resolution = 0,
    RNScalar focus_exponent = 10);
  virtual void InsertNodes(R3SurfelTree *tree, R3SurfelNode *node,
    const R3Point& xycenter, RNLength xyradius, 
    RNCoord zmin = -FLT_MAX, RNCoord zmax = FLT_MAX,
    RNScalar center_resolution = 0, RNScalar perimeter_resolution = 0,
    RNScalar focus_exponent = 10);

  // Low-level manipulation functions
  virtual void InsertNode(R3SurfelNode *node);
  virtual void RemoveNode(R3SurfelNode *node);
  virtual void RemoveNode(int k);
  virtual void Empty(void);


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

private:
  RNArray<R3SurfelNode *> nodes;
  RNScalar complexity;
  R3Box bbox;
};



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline RNScalar R3SurfelNodeSet::
Complexity(void) const
{
  // Return complexity of set
  return complexity;
}



inline R3Point R3SurfelNodeSet::
Centroid(void) const
{
  // Return centroid of set
  return bbox.Centroid();
}



inline const R3Box& R3SurfelNodeSet::
BBox(void) const
{
  // Return bounding box of set
  return bbox;
}



inline int R3SurfelNodeSet::
NNodes(void) const
{
  // Return number of nodes
  return nodes.NEntries();
}



inline R3SurfelNode *R3SurfelNodeSet::
Node(int k) const
{
  // Return kth node
  return nodes.Kth(k);
}



inline R3SurfelNode *R3SurfelNodeSet::
operator[](int k) const
{
  // Return kth node
  return Node(k);
}



inline int R3SurfelNodeSet::
NodeIndex(R3SurfelNode *node) const
{
  // Return index of node (or -1 if not found)
  RNArrayEntry *entry = nodes.FindEntry(node);
  return (entry) ? nodes.EntryIndex(entry) : -1;
}



