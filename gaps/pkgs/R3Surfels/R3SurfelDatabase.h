/* Include file for the R3 surfel database class */



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelDatabase {
public:
  //////////////////////////////////////////
  //// CONSTRUCTOR/DESTRUCTOR FUNCTIONS ////
  //////////////////////////////////////////

  // Constructor functions
  R3SurfelDatabase(void);
  R3SurfelDatabase(const R3SurfelDatabase& database);

  // Destructor function
  virtual ~R3SurfelDatabase(void);


  ////////////////////////////
  //// PROPERTY FUNCTIONS ////
  ////////////////////////////

  // Property functions
  int NSurfels(void) const;
  const char *Name(void) const;

  // Geometric property functions
  const R3Box& BBox(void) const;
  R3Point Centroid(void) const;


  //////////////////////////
  //// ACCESS FUNCTIONS ////
  //////////////////////////

  // Tree access functions
  R3SurfelTree *Tree(void) const;

  // Block access functions
  int NBlocks(void) const;
  R3SurfelBlock *Block(int k) const;


  //////////////////////////////////////////
  //// PROPERTY MANIPULATION FUNCTIONS ////
  //////////////////////////////////////////

  // Property manipulation functions
  void SetName(const char *name);


  //////////////////////////////////////////
  //// STRUCTURE MANIPULATION FUNCTIONS ////
  //////////////////////////////////////////

  // Block manipulation functions
  virtual void InsertBlock(R3SurfelBlock *block);
  virtual void RemoveBlock(R3SurfelBlock *block);
  virtual void RemoveAndDeleteBlock(R3SurfelBlock *block);

  // High-level block manipulation functions
  virtual int InsertSubsetBlocks(R3SurfelBlock *block, 
    const RNArray<const R3Surfel *>& subsetA, const RNArray<const R3Surfel *>& subsetB, 
    R3SurfelBlock **blockA, R3SurfelBlock **blockB);


  ///////////////////////////////////////
  //// SURFEL MANIPULATION FUNCTIONS ////
  ///////////////////////////////////////

  // Surfel manipulation functions
  void SetMarks(RNBoolean mark = TRUE);


  /////////////////////////////////////
  //// MEMORY MANAGEMENT FUNCTIONS ////
  /////////////////////////////////////

  // Memory management functions
  int ReadBlock(R3SurfelBlock *block);
  int ReleaseBlock(R3SurfelBlock *block);
  int SyncBlock(R3SurfelBlock *block);
  RNBoolean IsBlockResident(R3SurfelBlock *block) const;
  unsigned long ResidentSurfels(void) const;


  ///////////////////////
  //// I/O FUNCTIONS ////
  ///////////////////////

  // I/O functions
  virtual int OpenFile(const char *filename, const char *rwaccess = NULL);
  virtual int SyncFile(void);
  virtual int CloseFile(void);
  virtual RNBoolean IsOpen(void) const;

  // I/O functions for other file formats
  virtual int ReadFile(const char *filename);
  virtual int WriteFile(const char *filename) const;


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
  // Block I/O functions
  virtual int InternalReadBlock(R3SurfelBlock *block);
  virtual int InternalReleaseBlock(R3SurfelBlock *block);
  virtual int InternalSyncBlock(R3SurfelBlock *block);

  // Internal functions
  virtual int WriteHeader(FILE *fp, int swap_endian);

protected:
  FILE *fp;
  char *filename;
  char *rwaccess;
  unsigned int major_version;
  unsigned int minor_version;
  unsigned int swap_endian;
  unsigned long long file_blocks_offset;
  unsigned int file_blocks_count;
  RNArray<R3SurfelBlock *> blocks;
  int nsurfels;
  R3Box bbox;
  char *name;
  friend class R3SurfelTree;
  R3SurfelTree *tree;
  unsigned long resident_surfels;
};



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline int R3SurfelDatabase::
NSurfels(void) const
{
  // Return total number of surfels in all blocks
  return nsurfels;
}



inline const R3Box& R3SurfelDatabase::
BBox(void) const
{
  // Return bounding box of database
  return bbox;
}



inline R3Point R3SurfelDatabase::
Centroid(void) const
{
  // Return centroid of database
  return BBox().Centroid();
}



inline const char *R3SurfelDatabase::
Name(void) const
{
  // Return name
  return name;
}



inline R3SurfelTree *R3SurfelDatabase::
Tree(void) const
{
  // Return tree
  return tree;
}



inline int R3SurfelDatabase::
NBlocks(void) const
{
  // Return number of blocks
  return blocks.NEntries();
}



inline R3SurfelBlock *R3SurfelDatabase::
Block(int k) const
{
  // Return kth block
  return blocks[k];
}


inline RNBoolean R3SurfelDatabase::
IsOpen(void) const
{
  // Return whether database is open
  return (fp) ? TRUE : FALSE;
}



inline RNBoolean R3SurfelDatabase::
IsBlockResident(R3SurfelBlock *block) const
{
  // Return whether block is resident in memory
  return (block->surfels) ? TRUE : FALSE;
}



inline unsigned long R3SurfelDatabase::
ResidentSurfels(void) const
{
  // Return number of resident surfels
  return resident_surfels;
}



inline int R3SurfelDatabase::
ReadBlock(R3SurfelBlock *block)
{
  // Check whether block needs to be read
  if (block->file_read_count == 0) {
    if (!InternalReadBlock(block)) return 0;
  }

  // Increment reference count
  block->file_read_count++;

  // Return success
  return 1;
}



inline int R3SurfelDatabase::
ReleaseBlock(R3SurfelBlock *block)
{
  // Check whether block needs to be written
  if (block->file_read_count == 1) {
    if (!InternalReleaseBlock(block)) return 0;
  }

  // Decrement reference count
  block->file_read_count--;

  // Check if delete pending
  if (block->file_read_count == 0) {
    if (block->flags[R3_SURFEL_BLOCK_DELETE_PENDING_FLAG]) {
      RemoveBlock(block);
      delete block;
    }
  }

  // Return success
  return 1;
}



inline int R3SurfelDatabase::
SyncBlock(R3SurfelBlock *block)
{
  // Check whether block needs to be written
  if (block->IsDirty()) {
    if (!InternalSyncBlock(block)) return 0;
    block->SetDirty(FALSE);
  }

  // Return success
  return 1;
}



