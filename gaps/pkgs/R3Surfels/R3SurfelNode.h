/* Include file for the R3 surfel node class */



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelNode {
public:
  //////////////////////////////////////////
  //// CONSTRUCTOR/DESTRUCTOR FUNCTIONS ////
  //////////////////////////////////////////

  // Constructor functions
  R3SurfelNode(const char *name = NULL);
  R3SurfelNode(const R3SurfelNode& node);

  // Destructor function
  virtual ~R3SurfelNode(void);


  ////////////////////////////
  //// PROPERTY FUNCTIONS ////
  ////////////////////////////

  // Resolution property functions
  RNScalar Complexity(void) const;
  RNScalar Resolution(void) const;
  RNScalar AverageRadius(void) const;

  // Geometric property functions
  const R3Box& BBox(void) const;
  R3Point Centroid(void) const;

  // Surfel property functions
  RNBoolean HasActive(void) const;
  RNBoolean HasNormals(void) const;
  RNBoolean HasAerial(void) const;
  RNBoolean HasTerrestrial(void) const;

  // Name property functions
  const char *Name(void) const;

  // User data property functions
  void *Data(void) const;


  //////////////////////////
  //// ACCESS FUNCTIONS ////
  //////////////////////////

  // Object access functions
  R3SurfelObject *Object(RNBoolean search_ancestors = FALSE) const;

  // Scan access functions
  R3SurfelScan *Scan(RNBoolean search_ancestors = TRUE) const;

  // Tree access functions
  R3SurfelTree *Tree(void) const;
  int TreeIndex(void) const;

  // Part access functions
  int NParts(void) const;
  R3SurfelNode *Part(int k) const;
  R3SurfelNode *Parent(void) const;
  int TreeLevel(void) const;

  // Block access functions
  int NBlocks(void) const;
  R3SurfelBlock *Block(int k) const;

  // Point access functions
  R3SurfelPointSet *PointSet(RNBoolean leaf_level = FALSE) const;


  /////////////////////////////////////////
  //// PROPERTY MANIPULATION FUNCTIONS ////
  /////////////////////////////////////////

  // Name manipulation functions
  virtual void SetName(const char *name);

  // User data manipulation functions
  virtual void SetData(void *data);


  //////////////////////////////////////////
  //// STRUCTURE MANIPULATION FUNCTIONS ////
  //////////////////////////////////////////

  // Hierarchy manipulation functions
  virtual void SetParent(R3SurfelNode *parent);

  // Block manipulation functions
  virtual void InsertBlock(R3SurfelBlock *block);
  virtual void RemoveBlock(R3SurfelBlock *block);


  ///////////////////////////////////////
  //// SURFEL MANIPULATION FUNCTIONS ////
  ///////////////////////////////////////

  // Surfel manipulation functions
  virtual void Transform(const R3Affine& transformation);
  virtual void SetMarks(RNBoolean mark = TRUE);


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
  virtual int DrawResidentAncestor(RNFlags flags = R3_SURFEL_DEFAULT_DRAW_FLAGS) const;
  virtual int DrawResidentDescendents(RNFlags flags = R3_SURFEL_DEFAULT_DRAW_FLAGS) const;

  // Print function
  virtual void Print(FILE *fp = NULL, const char *prefix = NULL, const char *suffix = NULL) const;


  ////////////////////////////////////////////////////////////////////////
  // INTERNAL STUFF BELOW HERE
  ////////////////////////////////////////////////////////////////////////

public:
  // Update functions
  void UpdateProperties(void);

protected:
  // Object update functions
  void UpdateAfterInsert(R3SurfelObject *object);
  void UpdateBeforeRemove(R3SurfelObject *object);

  // Scan update functions
  void UpdateAfterInsert(R3SurfelScan *scan);
  void UpdateBeforeRemove(R3SurfelScan *scan);

  // Tree update functions
  void UpdateAfterInsert(R3SurfelTree *tree);
  void UpdateBeforeRemove(R3SurfelTree *tree);

  // Property update functions
  virtual void UpdateBBox();
  virtual void UpdateComplexity();
  virtual void UpdateResolution();
  virtual void UpdateFlags();

  // Update surfels
  virtual void UpdateSurfelNormals();

protected:
  // Internal data
  friend class R3SurfelScene;
  friend class R3SurfelObject;
  R3SurfelObject *object;
  friend class R3SurfelScan;
  R3SurfelScan *scan;
  friend class R3SurfelTree;
  R3SurfelTree *tree;
  int tree_index;
  R3SurfelNode *parent;
  RNArray<R3SurfelNode *> parts;
  RNArray<R3SurfelBlock *> blocks;
  RNScalar complexity;
  RNScalar resolution;
  R3Box bbox;
  char *name;
  RNFlags flags;
  void *data;
};



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline R3Point R3SurfelNode::
Centroid(void) const
{
  // Return centroid of node
  return BBox().Centroid();
}



inline const char *R3SurfelNode::
Name(void) const
{
  // Return name
  return name;
}



inline void *R3SurfelNode::
Data(void) const
{
  // Return user data
  return data;
}



inline R3SurfelTree *R3SurfelNode::
Tree(void) const
{
  // Return tree this node is in
  return tree;
}



inline int R3SurfelNode::
TreeIndex(void) const
{
  // Return index in list of nodes associated with tree
  return tree_index;
}


inline int R3SurfelNode::
NParts(void) const
{
  // Return number of parts in tree
  return parts.NEntries();
}



inline R3SurfelNode *R3SurfelNode::
Part(int k) const
{
  // Return kth part in tree
  return parts.Kth(k);
}



inline R3SurfelNode *R3SurfelNode::
Parent(void) const
{
  // Return parent in tree
  return parent;
}



inline int R3SurfelNode::
NBlocks(void) const
{
  // Return number of blocks
  return blocks.NEntries();
}



inline R3SurfelBlock *R3SurfelNode::
Block(int k) const
{
  // Return kth block
  return blocks[k];
}



inline void R3SurfelNode::
UpdateAfterInsert(R3SurfelObject *object)
{
}



inline void R3SurfelNode::
UpdateBeforeRemove(R3SurfelObject *object)
{
}




inline void R3SurfelNode::
UpdateAfterInsert(R3SurfelScan *scan)
{
}



inline void R3SurfelNode::
UpdateBeforeRemove(R3SurfelScan *scan)
{
}







