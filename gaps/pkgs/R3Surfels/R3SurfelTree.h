/* Include file for the R3 surfel tree class */



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelTree {
public:
  //////////////////////////////////////////
  //// CONSTRUCTOR/DESTRUCTOR FUNCTIONS ////
  //////////////////////////////////////////

  // Constructor functions
  R3SurfelTree(void);
  R3SurfelTree(const R3SurfelTree& tree);

  // Destructor function
  virtual ~R3SurfelTree(void);


  ////////////////////////////
  //// PROPERTY FUNCTIONS ////
  ////////////////////////////

  // Geometric property functions
  const R3Box& BBox(void) const;
  R3Point Centroid(void) const;


  //////////////////////////
  //// ACCESS FUNCTIONS ////
  //////////////////////////

  // Scene access function
  R3SurfelScene *Scene(void) const;

  // Database access functions
  R3SurfelDatabase *Database(void) const;

  // Node access functions
  int NNodes(void) const;
  R3SurfelNode *Node(int k) const;
  R3SurfelNode *FindNodeByName(const char *node_name) const;
  R3SurfelNode *RootNode(void) const;


  //////////////////////////////////////////
  //// STRUCTURE MANIPULATION FUNCTIONS ////
  //////////////////////////////////////////

  // Node manipulation functions
  void InsertNode(R3SurfelNode *node, R3SurfelNode *parent);
  void RemoveNode(R3SurfelNode *node);


  /////////////////////////////////////
  //// NODE MANIPULATION FUNCTIONS ////
  /////////////////////////////////////

  // Node splitting based on complexity/size
  virtual int SplitNode(R3SurfelNode *node,  
    int max_parts_per_node = 64, int max_blocks_per_node = 64, 
    RNScalar max_leaf_complexity = 1024*1024, RNScalar max_block_complexity = 1024*1024,
    RNLength max_leaf_extent = 0, RNLength max_block_extent = 0, int max_levels = 64);
  virtual int SplitNodes(R3SurfelNode *node,  
    int max_parts_per_node = 64, int max_blocks_per_node = 64, 
    RNScalar max_leaf_complexity = 1024*1024, RNScalar max_block_complexity = 1024*1024,
    RNLength max_leaf_extent = 0, RNLength max_block_extent = 0, int max_levels = 64);
  virtual int SplitNodes(int max_parts_per_node = 64, int max_blocks_per_node = 64, 
    RNScalar max_leaf_complexity = 1024*1024, RNScalar max_block_complexity = 1024*1024,
    RNLength max_leaf_extent = 0, RNLength max_block_extent = 0, int max_levels = 64);

  // Node splitting based on pointset
  virtual int SplitNodes(R3SurfelPointSet& pointset,
    RNArray<R3SurfelNode *> *nodesA = NULL, RNArray<R3SurfelNode *> *nodesB = NULL);

  // Leaf node splitting based on constraint
  virtual int SplitLeafNodes(R3SurfelNode *node, const R3SurfelConstraint& constraint, 
    RNArray<R3SurfelNode *> *nodesA = NULL, RNArray<R3SurfelNode *> *nodesB = NULL);
  virtual int SplitLeafNodes(const R3SurfelConstraint& constraint, 
    RNArray<R3SurfelNode *> *nodesA = NULL, RNArray<R3SurfelNode *> *nodesB = NULL);

  // Node insertion based on multiresolution 
  int CreateMultiresolutionNodes(RNScalar min_complexity = 8, RNScalar min_resolution = 1.0, RNScalar min_multiresolution_factor = 0.25);


  //////////////////////////////////////
  //// BLOCK MANIPULATION FUNCTIONS ////
  //////////////////////////////////////

  // Block splitting based on complexity/size
  virtual int SplitBlocks(R3SurfelNode *node, RNScalar max_complexity, RNScalar max_extent);

  // Block spliting based on pointset
  virtual int SplitBlocks(R3SurfelNode *node, R3SurfelPointSet& pointset, 
    RNArray<R3SurfelBlock *> *blocksA = NULL, RNArray<R3SurfelBlock *> *blocksB = NULL);

  // Block splitting based on constraint
  virtual int SplitBlock(R3SurfelNode *node, R3SurfelBlock *block, const R3SurfelConstraint& constraint,  
    R3SurfelBlock **blockA = NULL, R3SurfelBlock **blockB = NULL);
  virtual int SplitBlocks(R3SurfelNode *node, const R3SurfelConstraint& constraint, 
    RNArray<R3SurfelBlock *> *blocksA = NULL, RNArray<R3SurfelBlock *> *blocksB = NULL);

  // Muiltiresolution block creation
  virtual int CreateMultiresolutionBlocks(R3SurfelNode *node, RNScalar multiresolution_factor = 0.25, RNScalar max_complexity = 0, RNScalar max_resolution = 0);
  virtual int CreateMultiresolutionBlocks(RNScalar multiresolution_factor = 0.25, RNScalar max_complexity = 0);


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

protected:
  // Update functions
  void UpdateAfterInsert(R3SurfelScene *scene);
  void UpdateBeforeRemove(R3SurfelScene *scene);

protected:
  // Scene stuff
  friend class R3SurfelScene;
  R3SurfelScene *scene;

  // Database stuff
  R3SurfelDatabase *database;

  // Node stuff
  RNArray<R3SurfelNode *> nodes;
};



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline R3SurfelScene *R3SurfelTree::
Scene(void) const
{
  // Return scene
  return scene;
}



inline R3SurfelDatabase *R3SurfelTree::
Database(void) const
{
  // Return database
  return database;
}



inline const R3Box& R3SurfelTree::
BBox(void) const
{
  // Return total number of surfels
  R3SurfelNode *root = RootNode();
  if (!root) return R3null_box;
  return root->BBox();
}



inline R3Point R3SurfelTree::
Centroid(void) const
{
  // Return centroid of scebe
  return BBox().Centroid();
}



inline int R3SurfelTree::
NNodes(void) const
{
  // Return number of nodes
  return nodes.NEntries();
}



inline R3SurfelNode *R3SurfelTree::
Node(int k) const
{
  // Return kth node
  return nodes.Kth(k);
}



inline R3SurfelNode *R3SurfelTree::
RootNode(void) const
{
  // Return root node
  return nodes[0];
}



inline void R3SurfelTree::
UpdateAfterInsert(R3SurfelScene *scene)
{
}



inline void R3SurfelTree::
UpdateBeforeRemove(R3SurfelScene *scene)
{
}


