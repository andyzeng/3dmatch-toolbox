/* Source file for the R3 surfel database class */



////////////////////////////////////////////////////////////////////////
// INCLUDE FILES
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// PRINT DEBUG CONTROL
////////////////////////////////////////////////////////////////////////

// #define PRINT_DEBUG



////////////////////////////////////////////////////////////////////////
// Versioning variables
////////////////////////////////////////////////////////////////////////

static unsigned int current_major_version = 3;
static unsigned int current_minor_version = 1;



////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS/DESTRUCTORS
////////////////////////////////////////////////////////////////////////

R3SurfelDatabase::
R3SurfelDatabase(void)
  : fp(NULL),
    filename(NULL),
    rwaccess(NULL),
    major_version(current_major_version),
    minor_version(current_minor_version),
    swap_endian(0),
    file_blocks_offset(0),
    file_blocks_count(0),
    blocks(),
    nsurfels(0),
    bbox(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX),
    name(NULL),
    tree(NULL),
    resident_surfels(0)
{
}



R3SurfelDatabase::
R3SurfelDatabase(const R3SurfelDatabase& database)
  : fp(NULL),
    filename(NULL),
    rwaccess(NULL),
    major_version(current_major_version),
    minor_version(current_minor_version),
    swap_endian(0),
    file_blocks_offset(0),
    file_blocks_count(0),
    blocks(),
    nsurfels(0),
    bbox(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX),
    name(strdup(database.name)),
    tree(NULL),
    resident_surfels(0)
{
  RNAbort("Not implemented");
}



R3SurfelDatabase::
~R3SurfelDatabase(void)
{
  // Delete tree
  if (tree) delete tree;

  // Delete filename
  if (filename) free(filename);

  // Delete rwaccess
  if (rwaccess) free(rwaccess);

  // Delete name
  if (name) free(name);
}



////////////////////////////////////////////////////////////////////////
// PROPERTY MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelDatabase::
SetName(const char *name)
{
  // Set node name
  if (this->name) delete this->name;
  this->name = strdup(name);
}



////////////////////////////////////////////////////////////////////////
// SURFEL MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelDatabase::
SetMarks(RNBoolean mark)
{
  // Mark all surfels
  for (int i = 0; i < NBlocks(); i++) {
    R3SurfelBlock *block = Block(i);
    block->SetMarks(mark);
  }
}



////////////////////////////////////////////////////////////////////////
// BLOCK MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelDatabase::
InsertBlock(R3SurfelBlock *block)
{
  // Just checking
  assert(block->database == NULL);
  assert(block->database_index == -1);
  assert(block->file_surfels_offset == 0);
  assert(block->file_surfels_count == 0);
  assert(block->file_read_count == 0);

  // Update block database info
  block->database = this;
  block->database_index = blocks.NEntries();
  block->file_surfels_offset = 0;
  block->file_surfels_count = 0;
  block->file_read_count = (block->surfels) ? 1 : 0;
  block->SetDirty(TRUE);

  // Insert block
  blocks.Insert(block);

  // Update bounding box
  bbox.Union(block->BBox());

  // Update number of surfels
  nsurfels += block->NSurfels();

  // Update block
  block->UpdateAfterInsert(this);

  // Update resident surfels
  if (block->surfels) resident_surfels += block->NSurfels();

#ifdef PRINT_DEBUG
  // Print debug message
  printf("Inserted Block %6d : %6d %9ld : %9.3f %9.3f %9.3f\n", 
    block->database_index, block->nsurfels, resident_surfels,
    block->Centroid().X(), block->Centroid().Y(), block->Centroid().Z()); 
  fflush(stdout);
#endif
}



void R3SurfelDatabase::
RemoveBlock(R3SurfelBlock *block)
{
  // Just checking
  assert(block->file_read_count == 0);
  assert(block->database == this);
  assert(block->node == NULL);
    
  // Update resident surfels
  if (block->surfels) resident_surfels -= block->NSurfels();
    
  // Update block
  block->UpdateBeforeRemove(this);
    
  // Find block
  RNArrayEntry *entry = blocks.KthEntry(block->database_index);
  if (!entry) return;
  R3SurfelBlock *tail = blocks.Tail();
  blocks.EntryContents(entry) = tail;
  tail->database_index = block->database_index;
  blocks.RemoveTail();
    
  // Reset block database info
  block->database = NULL;
  block->database_index = -1;
  block->file_surfels_offset = 0;
  block->file_surfels_count = 0;
  block->file_read_count = 0;
  block->SetDirty(FALSE);
    
  // Update number of surfels
  nsurfels -= block->NSurfels();

  // Does not update bounding box
  // XXX

#ifdef PRINT_DEBUG
  // Print debug message
  printf("Removed Block  %6d : %6d %9ld : %9.3f %9.3f %9.3f\n", 
         block->database_index, block->nsurfels, resident_surfels,
         block->Centroid().X(), block->Centroid().Y(), block->Centroid().Z()); 
  fflush(stdout);
#endif
}



void R3SurfelDatabase::
RemoveAndDeleteBlock(R3SurfelBlock *block)
{
  // Check if still referenced
  if (block->file_read_count == 0) {
    // Block is not referenced, can simply delete it
    RemoveBlock(block);
    delete block;
  }
  else {
    // Block is referenced, mark for delete later
    block->flags.Add(R3_SURFEL_BLOCK_DELETE_PENDING_FLAG);
  }
}



////////////////////////////////////////////////////////////////////////
// HIGH-LEVEL MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

int R3SurfelDatabase::
InsertSubsetBlocks(R3SurfelBlock *block, 
  const RNArray<const R3Surfel *>& subset1, const RNArray<const R3Surfel *>& subset2, 
  R3SurfelBlock **blockA, R3SurfelBlock **blockB)
{
  // Check subset1
  if (subset1.IsEmpty()) {
    if (blockA) *blockA = NULL;
    if (blockB) *blockB = block;
    return 0;
  }

  // Check subset2
  if (subset2.IsEmpty()) {
    if (blockA) *blockA = block;
    if (blockB) *blockB = NULL;
    return 0;
  }

  // Create new blocks
  R3SurfelBlock *block1 = new R3SurfelBlock(subset1, block->Origin());
  R3SurfelBlock *block2 = new R3SurfelBlock(subset2, block->Origin());
    
  // Insert new blocks 
  InsertBlock(block1);
  InsertBlock(block2);

  // Update file offsets
  if ((block->file_surfels_offset > 0) && (block->file_surfels_count > 0)) {
    block1->file_surfels_offset = block->file_surfels_offset;
    block1->file_surfels_count = block1->NSurfels();
    block2->file_surfels_offset = block->file_surfels_offset + block1->NSurfels() * sizeof(R3Surfel);
    block2->file_surfels_count = block2->NSurfels();
    block->file_surfels_offset = 0;
    block->file_surfels_count = 0;
  }

  // Update file read counts ???
  if (block->file_read_count > 0) {
    block1->file_read_count = block->file_read_count;
    block2->file_read_count = block->file_read_count;
  }
    
  // Update block properties
  block1->UpdateProperties();
  block2->UpdateProperties();

  // Release blocks
  ReleaseBlock(block1);
  ReleaseBlock(block2);
      
  // Return new blocks
  if (blockA) *blockA = block1;
  if (blockB) *blockB = block2;

  // Return success
  return 1;
}
  
  

////////////////////////////////////////////////////////////////////////
// DISPLAY FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelDatabase::
Draw(RNFlags flags) const
{
  // Draw blocks
  for (int i = 0; i < blocks.NEntries(); i++) {
    blocks[i]->Draw(flags);
  }
}



void R3SurfelDatabase::
Print(FILE *fp, const char *prefix, const char *suffix) const
{
  // Check fp
  if (!fp) fp = stdout;

  // Print name
  fprintf(fp, "%s%s%s\n", (prefix) ? prefix : "", (name) ? name : "Database", (suffix) ? suffix : "");

  // Add indent to prefix
  char indent_prefix[1024];
  sprintf(indent_prefix, "%s  ", (prefix) ? prefix : "");

  // Print blocks
  for (int i = 0; i < NBlocks(); i++) {
    blocks[i]->Print(fp, indent_prefix, suffix);
  }
}



////////////////////////////////////////////////////////////////////////
// I/O FUNCTIONS
////////////////////////////////////////////////////////////////////////

int R3SurfelDatabase::
ReadFile(const char *filename)
{
  // Parse input filename extension
  const char *extension;
  if (!(extension = strrchr(filename, '.'))) {
    printf("Filename %s has no extension (e.g., .ply)\n", filename);
    return 0;
  }

  // Read file of appropriate type
  if (!strncmp(extension, ".list", 5)) {
    // Open file
    FILE *fp;
    if (!(fp = fopen(filename, "r"))) {
      fprintf(stderr, "Unable to open file %s\n", filename);
      return 0;
    }

    // Read file names
    char buffer[4096];
    while (fscanf(fp, "%s", buffer) == (unsigned int) 1) {
      // Create block
      R3SurfelBlock *block = new R3SurfelBlock();
      if (!block) {
        fprintf(stderr, "Unable to create block for %s\n", buffer);
        return 0;
      }
    
      // Read block
      if (!block->ReadFile(buffer)) { 
        delete block; 
        return 0; 
      }

      // Update properties
      block->UpdateProperties();

      // Insert block
      InsertBlock(block);

      // Release block
      ReleaseBlock(block);
    }

    // Close file
    fclose(fp);
  }
  else { 
    // Create block
    R3SurfelBlock *block = new R3SurfelBlock();
    if (!block) {
      fprintf(stderr, "Unable to create block\n");
      return 0;
    }
    
    // Read file
    if (!block->ReadFile(filename)) { 
      delete block; 
      return 0; 
    }
    
    // Update properties
    block->UpdateProperties();

    // Insert block
    InsertBlock(block);

    // Return success
    return 1;
  }

  // Should never get here
  return 0;
}



int R3SurfelDatabase::
WriteFile(const char *filename) const
{
  // Parse input filename extension
  const char *extension;
  if (!(extension = strrchr(filename, '.'))) {
    printf("Filename %s has no extension (e.g., .xyz)\n", filename);
    return 0;
  }

  // Write file of appropriate type
  if (NBlocks() == 1) {
    R3SurfelDatabase *database = (R3SurfelDatabase *) this;
    R3SurfelBlock *block = Block(0);
    if (!database->ReadBlock(block)) return 0;
    if (!block->WriteFile(filename)) return 0;
    if (!database->ReleaseBlock(block)) return 0;
  }
  else {
    fprintf(stderr, "Invalid file extension %s for database with more than one block\n", extension);
    return 0;
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// I/O UTILITY FUNCTIONS
////////////////////////////////////////////////////////////////////////

static void
swap2(void *values, int count)
{
  // Swap endian of 2-byte data type
  unsigned short *y = (unsigned short *) values;
  for (int i = 0; i < count; i++) {
    unsigned short x = y[i];
    y[i] = (x<<8) | (x>>8);
  }
}



static void
swap4(void *values, int count)
{
  // Swap endian of 4-byte data type
  unsigned int *y = (unsigned int *) values;
  for (int i = 0; i < count; i++) {
    unsigned int x = y[i];
    y[i] = 
       (x<<24) | 
      ((x<<8) & 0x00FF0000) | 
      ((x>>8) & 0x0000FF00) | 
       (x>>24);
  }
}



static void
swap8(void *values, int count)
{
  // Swap endian
  unsigned long long *y = (unsigned long long *) values;
  for (int i = 0; i < count; i++) {
    unsigned long long x = y[i];
    y[i] = 
       (x<<56) | 
      ((x<<40) & 0x00FF000000000000ULL) |
      ((x<<24) & 0x0000FF0000000000ULL) |
      ((x<<8)  & 0x000000FF00000000ULL) |
      ((x>>8)  & 0x00000000FF000000ULL) |
      ((x>>24) & 0x0000000000FF0000ULL) |
      ((x>>40) & 0x000000000000FF00ULL) |
       (x>>56);
  }
}



static int 
ReadChar(FILE *fp, char *ptr, int count, int /* swap_endian */)
{
  // Read the values
  if (fread(ptr, sizeof(char), count, fp) != (size_t) count) {
    fprintf(stderr, "Unable to read char from database file\n");
    return 0;
  }

  // Return success
  return 1;
}



#if 0
static int 
ReadShort(FILE *fp, RNInt16 *ptr, int count, int swap_endian)
{
  // Read the values
  if (fread(ptr, sizeof(RNInt16), count, fp) != (size_t) count) {
    fprintf(stderr, "Unable to read short from database file\n");
    return 0;
  }

  // Swap endian
  if (swap_endian) swap2(ptr, count);

  // Return success
  return 1;
}
#endif



#if 0
static int 
ReadUnsignedShort(FILE *fp, RNUInt16 *ptr, int count, int swap_endian)
{
  // Read the values
  if (fread(ptr, sizeof(RNUInt16), count, fp) != (size_t) count) {
    fprintf(stderr, "Unable to read unsigned short from database file\n");
    return 0;
  }

  // Swap endian
  if (swap_endian) swap2(ptr, count);

  // Return success
  return 1;
}
#endif


static int 
ReadInt(FILE *fp, int *ptr, int count, int swap_endian)
{
  // Read the values
  if (fread(ptr, sizeof(int), count, fp) != (size_t) count) {
    fprintf(stderr, "Unable to read integer from database file\n");
    return 0;
  }

  // Swap endian
  if (swap_endian) swap4(ptr, count);

  // Return success
  return 1;
}



static int 
ReadUnsignedInt(FILE *fp, unsigned int *ptr, int count, int swap_endian)
{
  // Read the values
  if (fread(ptr, sizeof(unsigned int), count, fp) != (size_t) count) {
    fprintf(stderr, "Unable to read unsigned integer from database file\n");
    return 0;
  }

  // Swap endian
  if (swap_endian) swap4(ptr, count);

  // Return success
  return 1;
}



#if 0
static int
ReadFloat(FILE *fp, float *ptr, int count, int swap_endian)
{
  // Read the values
  if (fread(ptr, sizeof(float), count, fp) != (size_t) count) {
    fprintf(stderr, "Unable to read float from database file\n");
    return 0;
  }

  // Swap endian
  if (swap_endian) swap4(ptr, count);

  // Return success
  return 1;
}
#endif



static int
ReadDouble(FILE *fp, double *ptr, int count, int swap_endian)
{
  // Read the values
  if (fread(ptr, sizeof(double), count, fp) != (size_t) count) {
    fprintf(stderr, "Unable to read double from database file\n");
    return 0;
  }

  // Swap endian
  if (swap_endian) swap8(ptr, count);

  // Return success
  return 1;
}



static int
ReadUnsignedLongLong(FILE *fp, unsigned long long *ptr, int count, int swap_endian)
{
  // Read the values
  if (fread(ptr, sizeof(unsigned long long), count, fp) != (size_t) count) {
    fprintf(stderr, "Unable to read unsigned long long from database file\n");
    return 0;
  }

  // Swap endian
  if (swap_endian) swap8(ptr, count);

  // Return success
  return 1;
}



static int
ReadSurfel(FILE *fp, R3Surfel *ptr, int count, int swap_endian, 
  unsigned int major_version, unsigned int minor_version)
{
  // Check database version
  if ((major_version == current_major_version) && (minor_version == current_minor_version)) {
    // Read surfels all at once into struct
    int sofar = 0;
    while (sofar < count) {
      size_t status = fread(ptr, sizeof(R3Surfel), count - sofar, fp);
      if (status > 0) sofar += status;
      else { fprintf(stderr, "Unable to read surfel from database file\n"); return 0; }
    }
  }
  else {
    // Read surfels one by one and element by element
    if (major_version < 2) {
      for (int i = 0; i < count; i++) {
        float position[3];
        unsigned char color_and_flags[4];
        fread(position, sizeof(float), 3, fp);
        fread(color_and_flags, sizeof(unsigned char), 4, fp);
        ptr[i].SetCoords(position);
        ptr[i].SetColor(color_and_flags);
        ptr[i].SetFlags(color_and_flags[3]);
      }
    }
  }

  // Swap endian
  if (swap_endian) {
    for (int i = 0; i < count; i++) {
      float *coords = ptr[i].PositionPtr(); swap4(coords, 3);
      RNInt16 *normals = ptr[i].NormalPtr(); swap2(normals, 3);
      RNUInt16 *radius = ptr[i].RadiusPtr(); swap2(radius, 1);
    }
  }

  // Return success
  return 1;
}



static int
WriteChar(FILE *fp, char *ptr, int count, int /* swap_endian */)
{
  // Write the values
  if (fwrite(ptr, sizeof(char), count, fp) != (size_t) count) {
    fprintf(stderr, "Unable to write integer to database file\n");
    return 0;
  }

  // Return success
  return 1;
}



#if 0
static int
WriteShort(FILE *fp, RNInt16 *ptr, int count, int swap_endian)
{
  // Swap endian
  if (swap_endian) swap2(ptr, count);

  // Write the values
  if (fwrite(ptr, sizeof(RNInt16), count, fp) != (size_t) count) {
    fprintf(stderr, "Unable to write short to database file\n");
    return 0;
  }

  // Swap endian back
  if (swap_endian) swap2(ptr, count);

  // Return success
  return 1;
}
#endif



#if 0
static int
WriteUnsignedShort(FILE *fp, RNUInt16 *ptr, int count, int swap_endian)
{
  // Swap endian
  if (swap_endian) swap2(ptr, count);

  // Write the values
  if (fwrite(ptr, sizeof(RNUInt16), count, fp) != (size_t) count) {
    fprintf(stderr, "Unable to write unsigned short to database file\n");
    return 0;
  }

  // Swap endian back
  if (swap_endian) swap2(ptr, count);

  // Return success
  return 1;
}
#endif



static int
WriteInt(FILE *fp, int *ptr, int count, int swap_endian)
{
  // Swap endian
  if (swap_endian) swap4(ptr, count);

  // Write the values
  if (fwrite(ptr, sizeof(int), count, fp) != (size_t) count) {
    fprintf(stderr, "Unable to write integer to database file\n");
    return 0;
  }

  // Swap endian back
  if (swap_endian) swap4(ptr, count);

  // Return success
  return 1;
}



static int
WriteUnsignedInt(FILE *fp, unsigned int *ptr, int count, int swap_endian)
{
  // Swap endian
  if (swap_endian) swap4(ptr, count);

  // Write the values
  if (fwrite(ptr, sizeof(unsigned int), count, fp) != (size_t) count) {
    fprintf(stderr, "Unable to write integer to database file\n");
    return 0;
  }

  // Swap endian back
  if (swap_endian) swap4(ptr, count);

  // Return success
  return 1;
}



#if 0
static int
WriteFloat(FILE *fp, float *ptr, int count, int swap_endian)
{
  // Swap endian
  if (swap_endian) swap4(ptr, count);

  // Write the values
  if (fwrite(ptr, sizeof(float), count, fp) != (size_t) count) {
    fprintf(stderr, "Unable to write float to database file\n");
    return 0;
  }

  // Swap endian back
  if (swap_endian) swap4(ptr, count);

  // Return success
  return 1;
}
#endif



static int
WriteDouble(FILE *fp, double *ptr, int count, int swap_endian)
{
  // Swap endian
  if (swap_endian) swap8(ptr, count);

  // Write the values
  if (fwrite(ptr, sizeof(double), count, fp) != (size_t) count) {
    fprintf(stderr, "Unable to write double to database file\n");
    return 0;
  }

  // Swap endian back
  if (swap_endian) swap8(ptr, count);

  // Return success
  return 1;
}



static int
WriteUnsignedLongLong(FILE *fp, unsigned long long *ptr, int count, int swap_endian)
{
  // Swap endian
  if (swap_endian) swap8(ptr, count);

  // Write the values
  if (fwrite(ptr, sizeof(unsigned long long), count, fp) != (size_t) count) {
    fprintf(stderr, "Unable to write unsigned long long to database file\n");
    return 0;
  }

  // Swap endian back
  if (swap_endian) swap8(ptr, count);

  // Return success
  return 1;
}



static int
WriteSurfel(FILE *fp, R3Surfel *ptr, int count, int swap_endian, 
  unsigned int major_version, unsigned int minor_version)
{
  // Swap endian
  if (swap_endian) {
    for (int i = 0; i < count; i++) {
      float *coords = ptr[i].PositionPtr(); swap4(coords, 3);
      RNInt16 *normals = ptr[i].NormalPtr(); swap2(normals, 3);
      RNUInt16 *radius = ptr[i].RadiusPtr(); swap2(radius, 1);
    }
  }

  // Clear surfel marks
  for (int i = 0; i < count; i++) ptr[i].SetMark(FALSE);

  // Write current version of surfel
  int sofar = 0;
  while (sofar < count) {
    size_t status = fwrite(ptr, sizeof(R3Surfel), count - sofar, fp);
    if (status > 0) sofar += status;
    else { fprintf(stderr, "Unable to write surfel to database file\n"); return 0; }
  }

  // Swap endian back
  if (swap_endian) {
    for (int i = 0; i < count; i++) {
      float *coords = ptr[i].PositionPtr(); swap4(coords, 3);
      RNInt16 *normals = ptr[i].NormalPtr(); swap2(normals, 3);
      RNUInt16 *radius = ptr[i].RadiusPtr(); swap2(radius, 1);
    }
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// BLOCK I/O FUNCTIONS
////////////////////////////////////////////////////////////////////////

int R3SurfelDatabase::
InternalReadBlock(R3SurfelBlock *block)
{
  // Check number of surfels
  if (block->NSurfels() == 0) return 1;

  // Just checking
  assert(fp);
  assert(block->database == this);
  assert(block->file_surfels_offset > 0);
  assert(block->file_surfels_count >= (unsigned int) block->nsurfels);

  // Allocate surfels
  block->surfels = new R3Surfel [ block->nsurfels ];
  if (!block->surfels) {
    fprintf(stderr, "Unable to allocate surfels\n");
    return 0;
  }
  
  // Read surfels
  RNFileSeek(fp, block->file_surfels_offset, RN_FILE_SEEK_SET);
  if (!ReadSurfel(fp, block->surfels, block->nsurfels, swap_endian, major_version, minor_version)) return 0;
  
  // Update resident surfels
  resident_surfels += block->NSurfels();

#ifdef PRINT_DEBUG
  // Print debug message
  printf("Read Block     %6d : %6d %9ld : %9.3f %9.3f %9.3f\n", 
         block->database_index, block->nsurfels, resident_surfels,
         block->Centroid().X(), block->Centroid().Y(), block->Centroid().Z()); 
  fflush(stdout);
#endif

  // Return success
  return 1;
}



int R3SurfelDatabase::
InternalReleaseBlock(R3SurfelBlock *block)
{
  // Just checking
  assert(fp);
  assert(block->database == this);

  // Write block
  if (!SyncBlock(block)) return 0;
    
#ifdef R3_SURFEL_DRAW_WITH_DISPLAY_LIST
  // Delete opengl display lists
  if (block->opengl_id > 0) {
    glDeleteLists(block->opengl_id, 2);
    block->opengl_id = 0;
  }
#endif

#ifdef DRAW_WITH_VBO
  // Delete opengl vertex buffer object
  if (block->opengl_id > 0) {
    glDeleteBuffers(1, &block->opengl_id);
    block->opengl_id = 0;
  }
#endif

  // Delete surfels
  if (block->surfels) {
    delete [] block->surfels;
    block->surfels = NULL;
  }
      
  // Update resident surfels
  resident_surfels -= block->NSurfels();

#ifdef PRINT_DEBUG
  // Print debug message
  printf("Released Block %6d : %6d %9ld : %9.3f %9.3f %9.3f\n", 
         block->database_index, block->nsurfels, resident_surfels,
         block->Centroid().X(), block->Centroid().Y(), block->Centroid().Z()); 
  fflush(stdout);
#endif

  // Return success
  return 1;
}



int R3SurfelDatabase::
InternalSyncBlock(R3SurfelBlock *block)
{
  // Check number of surfels
  if (block->NSurfels() == 0) return 1;

  // Check file rwaccess
  if (!strstr(rwaccess, "+")) {
    fprintf(stderr, "Unable to write block to read-only file\n");
    return 0;
  }

  // Check database version
  if ((major_version != current_major_version) || (minor_version != current_minor_version)) {
    fprintf(stderr, "Unable to write block to database with different version\n");
    return 0;
  }

  // Just checking
  assert(fp);
  assert(block->database == this);

  // Check if surfels can be put at original offset in file
  if ((block->file_surfels_offset > 0) && ((unsigned int) block->nsurfels <= block->file_surfels_count)) {
    // Surfels fit at original offset in file
    RNFileSeek(fp, block->file_surfels_offset, RN_FILE_SEEK_SET);
  }
  else {
    // Surfels must be put at end of file
    RNFileSeek(fp, 0, SEEK_END);
    block->file_surfels_offset = RNFileTell(fp);
    block->file_surfels_count = block->nsurfels;
  }

  // Write surfels to file
  if (!WriteSurfel(fp, block->surfels, block->nsurfels, swap_endian, major_version, minor_version)) return 0;

#ifdef PRINT_DEBUG
  // Print debug message
  printf("Synced Block %6d : %6d %9ld : %9.3f %9.3f %9.3f\n", 
         block->database_index, block->nsurfels, resident_surfels,
         block->Centroid().X(), block->Centroid().Y(), block->Centroid().Z()); 
  fflush(stdout);
#endif

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// FILE I/O FUNCTIONS
////////////////////////////////////////////////////////////////////////

int R3SurfelDatabase::
WriteHeader(FILE *fp, int swap_endian)
{
  // Get convenient variables
  unsigned int endian_test = 1;
  unsigned int nblocks = blocks.NEntries();
  char magic[32] = { '\0' };
  strncpy(magic, "R3SurfelDatabase", 32);
  char buffer[1024] = { '\0' };

  // Write header
  RNFileSeek(fp, 0, RN_FILE_SEEK_SET);
  if (!WriteChar(fp, magic, 32, swap_endian)) return 0;
  if (!WriteUnsignedInt(fp, &endian_test, 1, swap_endian)) return 0;
  if (!WriteUnsignedInt(fp, &endian_test, 1, swap_endian)) return 0;
  if (!WriteUnsignedInt(fp, &major_version, 1, swap_endian)) return 0;
  if (!WriteUnsignedInt(fp, &minor_version, 1, swap_endian)) return 0;
  if (!WriteUnsignedLongLong(fp, &file_blocks_offset, 1, swap_endian)) return 0;
  if (!WriteUnsignedInt(fp, &file_blocks_count, 1, swap_endian)) return 0;
  if (!WriteUnsignedInt(fp, &nblocks, 1, swap_endian)) return 0;
  if (!WriteInt(fp, &nsurfels, 1, swap_endian)) return 0;
  if (!WriteDouble(fp, &bbox[0][0], 6, swap_endian)) return 0;
  if (!WriteChar(fp, buffer, 1024, swap_endian)) return 0;

  // Return success
  return 1;
}



int R3SurfelDatabase::
OpenFile(const char *filename, const char *rwaccess)
{
  // Remember file name
  if (this->filename) free(this->filename);
  this->filename = strdup(filename);

  // Parse rwaccess
  if (this->rwaccess) free(this->rwaccess);
  if (!rwaccess) this->rwaccess = strdup("w+b");
  else if (strstr(rwaccess, "w")) this->rwaccess = strdup("w+b");
  else if (strstr(rwaccess, "+")) this->rwaccess = strdup("r+b");
  else this->rwaccess = strdup("rb"); 

  // Open file
  fp = fopen(filename, this->rwaccess);
  if (!fp) {
    fprintf(stderr, "Unable to open database file %s with rwaccess %s\n", filename, rwaccess);
    return 0;
  }

  // Check if file is new
  if (!strcmp(this->rwaccess, "w+b")) {
    // File is new -- write header
    if (!WriteHeader(fp, 0)) return 0;
  }
  else {
    // Read unique string
    char buffer[1024]; 
    if (!ReadChar(fp, buffer, 32, 0)) return 0;
    if (strcmp(buffer, "R3SurfelDatabase")) {
      fprintf(stderr, "Incorrect header (%s) in database file %s\n", buffer, filename);
      return 0;
    }

    // Read endian test
    unsigned int endian_test1, endian_test2;
    if (!ReadUnsignedInt(fp, &endian_test1, 1, 0)) return 0;
    if (endian_test1 != 1) swap_endian = 1;
    if (!ReadUnsignedInt(fp, &endian_test2, 1, swap_endian)) return 0;
    if (endian_test2 != 1) {
      fprintf(stderr, "Incorrect endian (%x) in database file %s\n", endian_test1, filename);
      return 0;
    }

    // Read version
    if (!ReadUnsignedInt(fp, &major_version, 1, swap_endian)) return 0;
    if (!ReadUnsignedInt(fp, &minor_version, 1, swap_endian)) return 0;
    if ((major_version < 1) || (major_version > 3)) {
      fprintf(stderr, "Incorrect version (%d.%d) in database file %s\n", major_version, minor_version, filename);
      return 0;
    }
  
    // Read rest of header
    unsigned int nblocks;
    if (!ReadUnsignedLongLong(fp, &file_blocks_offset, 1, swap_endian)) return 0;
    if (!ReadUnsignedInt(fp, &file_blocks_count, 1, swap_endian)) return 0;
    if (!ReadUnsignedInt(fp, &nblocks, 1, swap_endian)) return 0;
    if (!ReadInt(fp, &nsurfels, 1, swap_endian)) return 0;
    if (!ReadDouble(fp, &bbox[0][0], 6, swap_endian)) return 0;
    if (!ReadChar(fp, buffer, 1024, swap_endian)) return 0;

    // Read blocks
    RNFileSeek(fp, file_blocks_offset, RN_FILE_SEEK_SET);
    for (unsigned int i = 0; i < nblocks; i++) {
      R3SurfelBlock *block = new R3SurfelBlock();
      unsigned int block_flags;
      if (!ReadUnsignedLongLong(fp, &block->file_surfels_offset, 1, swap_endian)) return 0;
      if (!ReadUnsignedInt(fp, &block->file_surfels_count, 1, swap_endian)) return 0;
      if (!ReadInt(fp, &block->nsurfels, 1, swap_endian)) return 0;
      if (!ReadDouble(fp, &block->origin[0], 3, swap_endian)) return 0;
      if (!ReadDouble(fp, &block->bbox[0][0], 6, swap_endian)) return 0;
      if (!ReadDouble(fp, &block->resolution, 1, swap_endian)) return 0;
      if (!ReadUnsignedInt(fp, &block_flags, 1, swap_endian)) return 0;
      if (!ReadChar(fp, buffer, 64, swap_endian)) return 0;
      block->flags = block_flags;
      block->SetDirty(FALSE);
      block->database = this;
      block->database_index = blocks.NEntries();
      blocks.Insert(block);
    }
  }

  // Return success
  return 1;
}



int R3SurfelDatabase::
SyncFile(void)
{
  // Check if file is read-only
  if (!strcmp(rwaccess, "rb")) return 1;

  // Sync blocks
  for (int i = 0; i < blocks.NEntries(); i++) {
    R3SurfelBlock *block = blocks.Kth(i);
    if (!SyncBlock(block)) return 0;
  }

  // Update blocks offset
  unsigned int nblocks = blocks.NEntries();
  if (nblocks  > file_blocks_count) {
    RNFileSeek(fp, 0, RN_FILE_SEEK_END);
    file_blocks_offset = RNFileTell(fp);
    file_blocks_count = nblocks;
  }

  // Write blocks
  char buffer[128] = { '\0' };
  RNFileSeek(fp, file_blocks_offset, RN_FILE_SEEK_SET);
  for (int i = 0; i < blocks.NEntries(); i++) {
    R3SurfelBlock *block = blocks.Kth(i);
    unsigned int block_flags = block->flags; 
    if (!WriteUnsignedLongLong(fp, &block->file_surfels_offset, 1, swap_endian)) return 0;
    if (!WriteUnsignedInt(fp, &block->file_surfels_count, 1, swap_endian)) return 0;
    if (!WriteInt(fp, &block->nsurfels, 1, swap_endian)) return 0;
    if (!WriteDouble(fp, &block->origin[0], 3, swap_endian)) return 0;
    if (!WriteDouble(fp, &block->bbox[0][0], 6, swap_endian)) return 0;
    if (!WriteDouble(fp, &block->resolution, 1, swap_endian)) return 0;
    if (!WriteUnsignedInt(fp, &block_flags, 1, swap_endian)) return 0;
    if (!WriteChar(fp, buffer, 64, swap_endian)) return 0;
  }

  // Write header again (now that the offset values have been filled in)
  if (!WriteHeader(fp, swap_endian)) return 0;

  // Return success
  return 1;
}



int R3SurfelDatabase::
CloseFile(void)
{
  // Sync file
  if (!SyncFile()) return 0;

  // Close file
  fclose(fp);
  fp = NULL;

  // Reset filename
  if (filename) free(filename);
  filename = NULL;

  // Reset rwaccess
  if (rwaccess) free(rwaccess);
  rwaccess = NULL;

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// LP2 I/O FUNCTIONS
////////////////////////////////////////////////////////////////////////

#if 0

struct LPPointID {
  int originalTileID;
  int withinTileIndex;
};

struct LPLidarPoint {
  LPPointID id;
  float pos[3];
  unsigned char color[4];
  int intensity;
  unsigned char scanner;
  double time;
};

struct LPTileHeader {
  R3Point offset;
  R2Box bbox;
  int numPoints;
  char futureUse[516];
};


struct LPSuperTileHeader : public LPTileHeader {
  int numSubtiles;
};

struct LPFileHeader {
  char signature[16];
  unsigned int version;
  unsigned int numTiles;
  char futureUse[512];
};


#endif

