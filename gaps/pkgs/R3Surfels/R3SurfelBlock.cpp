/* Source file for the R3 surfel block class */



////////////////////////////////////////////////////////////////////////
// INCLUDE FILES
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS/DESTRUCTORS
////////////////////////////////////////////////////////////////////////

R3SurfelBlock::
R3SurfelBlock(void)
  : surfels(NULL),
    nsurfels(0),
    origin(0,0,0),
    bbox(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX),
    resolution(0),
    flags(0),
    database(NULL),
    database_index(-1),
    file_surfels_offset(0),
    file_surfels_count(0),
    file_read_count(0),
    node(NULL),
    opengl_id(0)
{
}



R3SurfelBlock::
R3SurfelBlock(const R3SurfelBlock& block)
  : surfels(NULL),
    nsurfels(block.nsurfels),
    origin(block.origin),
    bbox(block.bbox),
    resolution(block.resolution),
    flags(block.flags & R3_SURFEL_BLOCK_PROPERTY_FLAGS),
    database(NULL),
    database_index(-1),
    file_surfels_offset(0),
    file_surfels_count(0),
    file_read_count(0),
    node(NULL),
    opengl_id(0)
{
  // Copy surfels
  surfels = new R3Surfel [ nsurfels ];
  for (int i = 0; i < nsurfels; i++) {
    surfels[i] = block.surfels[i];
  }
}



R3SurfelBlock::
R3SurfelBlock(const R3SurfelPointSet *set)
  : surfels(NULL),
    nsurfels(set->NPoints()),
    origin(set->Centroid()),
    bbox(set->BBox()),
    resolution(0),
    flags(R3_SURFEL_BLOCK_BBOX_UPTODATE_FLAG),
    database(NULL),
    database_index(-1),
    file_surfels_offset(0),
    file_surfels_count(0),
    file_read_count(0),
    node(NULL),
    opengl_id(0)
{
  // Copy surfels
  surfels = new R3Surfel [ nsurfels ];
  for (int i = 0; i < set->NPoints(); i++) {
    const R3SurfelPoint *point = set->Point(i);
    R3Point position = point->Position() - origin.Vector();
    R3Vector normal = point->Normal();
    R3Surfel *surfel = &surfels[i];
    surfel->SetCoords(position[0], position[1], position[2]);
    surfel->SetNormal(normal[0], normal[1], normal[2]);
    surfel->SetColor(point->Color());
    surfel->SetRadius(point->Radius());
    surfel->SetFlags(point->Flags() & ~R3_SURFEL_MARKED_FLAG);
  }
}



R3SurfelBlock::
R3SurfelBlock(const R3SurfelPointSet *set, const R3Point& origin)
  : surfels(NULL),
    nsurfels(set->NPoints()),
    origin(origin),
    bbox(set->BBox()),
    resolution(0),
    flags(R3_SURFEL_BLOCK_BBOX_UPTODATE_FLAG),
    database(NULL),
    database_index(-1),
    file_surfels_offset(0),
    file_surfels_count(0),
    file_read_count(0),
    node(NULL),
    opengl_id(0)
{
  // Copy surfels
  surfels = new R3Surfel [ nsurfels ];
  for (int i = 0; i < set->NPoints(); i++) {
    const R3SurfelPoint *point = set->Point(i);
    R3Point position = point->Position() - origin.Vector();
    R3Vector normal = point->Normal();
    R3Surfel *surfel = &surfels[i];
    surfel->SetCoords(position[0], position[1], position[2]);
    surfel->SetNormal(normal[0], normal[1], normal[2]);
    surfel->SetColor(point->Color());
    surfel->SetRadius(point->Radius());
    surfel->SetFlags(point->Flags() & ~R3_SURFEL_MARKED_FLAG);
  }
}



R3SurfelBlock::
R3SurfelBlock(const R3Surfel *surfels, int nsurfels, const R3Point& origin)
  : surfels(NULL),
    nsurfels(nsurfels),
    origin(origin),
    bbox(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX),
    resolution(0),
    flags(0),
    database(NULL),
    database_index(-1),
    file_surfels_offset(0),
    file_surfels_count(0),
    file_read_count(0),
    node(NULL),
    opengl_id(0)
{
  // Copy surfels
  this->surfels = new R3Surfel [ nsurfels ];
  for (int i = 0; i < nsurfels; i++) {
    this->surfels[i] = surfels[i];
  }
}



R3SurfelBlock::
R3SurfelBlock(const RNArray<const R3Surfel *>& array, const R3Point& origin)
  : surfels(NULL),
    nsurfels(array.NEntries()),
    origin(origin),
    bbox(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX),
    resolution(0),
    flags(0),
    database(NULL),
    database_index(-1),
    file_surfels_offset(0),
    file_surfels_count(0),
    file_read_count(0),
    node(NULL),
    opengl_id(0)
{
  // Copy surfels
  surfels = new R3Surfel [ nsurfels ];
  for (int i = 0; i < nsurfels; i++) {
    this->surfels[i] = *(array[i]);
  }
}



R3SurfelBlock::
R3SurfelBlock(const R3Point *points, int npoints)
  : surfels(NULL),
    nsurfels(npoints),
    origin(R3zero_point),
    bbox(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX),
    resolution(0),
    flags(0),
    database(NULL),
    database_index(-1),
    file_surfels_offset(0),
    file_surfels_count(0),
    file_read_count(0),
    node(NULL),
    opengl_id(0)
{
  // Check number of points
  if (npoints == 0) return;

  // Compute origin
  origin = R3Centroid(npoints, (R3Point *) points);

  // Compute bounding box
  for (int i = 0; i < npoints; i++) {
    bbox.Union(points[i]);
  }

  // Copy surfels
  surfels = new R3Surfel [ nsurfels ];
  for (int i = 0; i < nsurfels; i++) {
    R3Point point = points[i];
    point -= origin.Vector();
    R3Surfel& surfel = this->surfels[i];
    surfel.SetCoords(point.X(), point.Y(), point.Z());
    surfel.SetRadius(bbox.DiagonalLength() / npoints);
    surfel.SetColor(128, 128, 128);
  }
}



R3SurfelBlock::
R3SurfelBlock(const RNArray<R3Point *>& points)
  : surfels(NULL),
    nsurfels(points.NEntries()),
    origin(R3zero_point),
    bbox(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX),
    resolution(0),
    flags(0),
    database(NULL),
    database_index(-1),
    file_surfels_offset(0),
    file_surfels_count(0),
    file_read_count(0),
    node(NULL),
    opengl_id(0)
{
  // Check points
  if (points.IsEmpty()) return;
  
  // Compute origin
  origin = R3Centroid(points);

  // Compute bounding box
  for (int i = 0; i < points.NEntries(); i++) {
    bbox.Union(*(points[i]));
  }

  // Copy surfels
  surfels = new R3Surfel [ nsurfels ];
  for (int i = 0; i < nsurfels; i++) {
    R3Point point = *(points[i]);
    point -= origin.Vector();
    R3Surfel& surfel = this->surfels[i];
    surfel.SetCoords(point.X(), point.Y(), point.Z());
    surfel.SetRadius(bbox.DiagonalLength() / points.NEntries());
    surfel.SetColor(128, 128, 128);
  }
}



R3SurfelBlock::
~R3SurfelBlock(void)
{
  // Just checking
  assert(!database);
  assert(database_index == -1);
  assert(file_read_count == 0);
  assert(!node);

  // Delete surfels
  if (surfels) delete [] surfels;

#ifdef DRAW_WITH_DISPLAY_LIST
  // Delete opengl display lists
  if (opengl_id > 0) glDeleteLists(opengl_id, 2);
#endif

#ifdef DRAW_WITH_VBO
  glDeleteBuffers(1, &opengl_id);
#endif
}



////////////////////////////////////////////////////////////////////////
// PROPERTY FUNCTIONS
////////////////////////////////////////////////////////////////////////

const R3Box& R3SurfelBlock::
BBox(void) const
{
  // Update bbox
  if (!flags[R3_SURFEL_BLOCK_BBOX_UPTODATE_FLAG]) {
    R3SurfelBlock *block = (R3SurfelBlock *) this;
    block->UpdateBBox();
  }

  // Return bbox
  return bbox;
}



RNScalar R3SurfelBlock::
Resolution(void) const
{
  // Update resolution
  if (!flags[R3_SURFEL_BLOCK_RESOLUTION_UPTODATE_FLAG]) {
    R3SurfelBlock *block = (R3SurfelBlock *) this;
    block->UpdateResolution();
  }

  // Return resolution
  return resolution;
}



RNBoolean R3SurfelBlock::
HasActive(void) const
{
  // Update active flags
  if (!flags[R3_SURFEL_BLOCK_FLAGS_UPTODATE_FLAG]) {
    R3SurfelBlock *block = (R3SurfelBlock *) this;
    block->UpdateFlags();
  }

   // Return whether block has active 
  return flags[R3_SURFEL_BLOCK_HAS_ACTIVE_FLAG];
}



RNBoolean R3SurfelBlock::
HasNormals(void) const
{
  // Update normals flags
  if (!flags[R3_SURFEL_BLOCK_FLAGS_UPTODATE_FLAG]) {
    R3SurfelBlock *block = (R3SurfelBlock *) this;
    block->UpdateFlags();
  }

   // Return whether block has normals 
  return flags[R3_SURFEL_BLOCK_HAS_NORMALS_FLAG];
}



RNBoolean R3SurfelBlock::
HasAerial(void) const
{
  // Update aerial/terrestrial flags
  if (!flags[R3_SURFEL_BLOCK_FLAGS_UPTODATE_FLAG]) {
    R3SurfelBlock *block = (R3SurfelBlock *) this;
    block->UpdateFlags();
  }

   // Return whether block has aerial scanner points
  return flags[R3_SURFEL_BLOCK_HAS_AERIAL_FLAG];
}



RNBoolean R3SurfelBlock::
HasTerrestrial(void) const
{
  // Update aerial/terrestrial flags
  if (!flags[R3_SURFEL_BLOCK_FLAGS_UPTODATE_FLAG]) {
    R3SurfelBlock *block = (R3SurfelBlock *) this;
    block->UpdateFlags();
  }

  // Return whether block has terrestrial scanner points
  return flags[R3_SURFEL_BLOCK_HAS_TERRESTRIAL_FLAG];
}



RNBoolean R3SurfelBlock::
IsDirty(void) const
{
  // Return whether block is dirty
  return flags[R3_SURFEL_BLOCK_DIRTY_FLAG];
}



void R3SurfelBlock::
SetDirty(RNBoolean dirty)
{
  // Set whether block is dirty
  if (dirty) flags.Add(R3_SURFEL_BLOCK_DIRTY_FLAG);
  else flags.Remove(R3_SURFEL_BLOCK_DIRTY_FLAG);
}



////////////////////////////////////////////////////////////////////////
// PROPERTY UPDATE FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelBlock::
SetOrigin(const R3Point& origin)
{
  // Set origin
  this->origin = origin;
}



void R3SurfelBlock::
SetSurfelPosition(int surfel_index, const R3Point& position)
{
  // Check if surfels are resident
  if (!surfels) {
    fprintf(stderr, "Unable to set surfel position for non-resident block\n");
    abort();
  }

  // Set surfel position
  float x = position[0] - origin[0];
  float y = position[1] - origin[1];
  float z = position[2] - origin[2];
  surfels[surfel_index].SetCoords(x, y, z);

  // Mark properties out of date
  flags.Remove(R3_SURFEL_BLOCK_BBOX_UPTODATE_FLAG);
  flags.Remove(R3_SURFEL_BLOCK_RESOLUTION_UPTODATE_FLAG);

  // Remember that block is dirty
  SetDirty();
}



void R3SurfelBlock::
SetSurfelNormal(int surfel_index, const R3Vector& normal)
{
  // Check if surfels are resident
  if (!surfels) {
    fprintf(stderr, "Unable to set surfel position for non-resident block\n");
    abort();
  }

  // Set surfel normal
  surfels[surfel_index].SetNormal(normal.X(), normal.Y(), normal.Z());

  // Remember that block is dirty
  SetDirty();
}



void R3SurfelBlock::
SetSurfelRadius(int surfel_index, RNLength radius)
{
  // Check if surfels are resident
  if (!surfels) {
    fprintf(stderr, "Unable to set surfel position for non-resident block\n");
    abort();
  }

  // Set surfel normal
  surfels[surfel_index].SetRadius(radius);

  // Remember that block is dirty
  SetDirty();
}



void R3SurfelBlock::
SetSurfelColor(int surfel_index, const RNRgb& color)
{
  // Check if surfels are resident
  if (!surfels) {
    fprintf(stderr, "Unable to set surfel position for non-resident block\n");
    abort();
  }

  // Set surfel color
  surfels[surfel_index].SetColor(color);

  // Remember that block is dirty
  SetDirty();
}



void R3SurfelBlock::
SetSurfelActive(int surfel_index, RNBoolean active)
{
  // Check if surfels are resident
  if (!surfels) {
    fprintf(stderr, "Unable to set surfel position for non-resident block\n");
    abort();
  }

  // Set surfel active
  surfels[surfel_index].SetActive(active);

  // Remember that block is dirty
  SetDirty();
}



void R3SurfelBlock::
SetSurfelAerial(int surfel_index, RNBoolean aerial)
{
  // Check if surfels are resident
  if (!surfels) {
    fprintf(stderr, "Unable to set surfel position for non-resident block\n");
    abort();
  }

  // Set surfel aerial
  surfels[surfel_index].SetAerial(aerial);

  // Remember that block is dirty
  SetDirty();
}



void R3SurfelBlock::
SetSurfelFlags(int surfel_index, unsigned char flags)
{
  // Check if surfels are resident
  if (!surfels) {
    fprintf(stderr, "Unable to set surfel property for non-resident block\n");
    abort();
  }

  // Set surfel mark
  surfels[surfel_index].SetFlags(flags);

  // Remember that block is dirty
  SetDirty();
}



void R3SurfelBlock::
SetSurfelSilhouetteBoundary(int surfel_index, RNBoolean boundary)
{
  // Check if surfels are resident
  if (!surfels) {
    fprintf(stderr, "Unable to set surfel property for non-resident block\n");
    abort();
  }

  // Set surfel mark
  surfels[surfel_index].SetSilhouetteBoundary(boundary);

  // Remember that block is dirty
  SetDirty();
}



void R3SurfelBlock::
SetSurfelShadowBoundary(int surfel_index, RNBoolean boundary)
{
  // Check if surfels are resident
  if (!surfels) {
    fprintf(stderr, "Unable to set surfel property for non-resident block\n");
    abort();
  }

  // Set surfel mark
  surfels[surfel_index].SetShadowBoundary(boundary);

  // Remember that block is dirty
  SetDirty();
}



void R3SurfelBlock::
SetSurfelBorderBoundary(int surfel_index, RNBoolean boundary)
{
  // Check if surfels are resident
  if (!surfels) {
    fprintf(stderr, "Unable to set surfel property for non-resident block\n");
    abort();
  }

  // Set surfel mark
  surfels[surfel_index].SetBorderBoundary(boundary);

  // Remember that block is dirty
  SetDirty();
}



void R3SurfelBlock::
SetMarks(RNBoolean mark)
{
  // Check surfels
  if (!mark && !surfels) return;

  // Set mark for all surfels
  for (int i = 0; i < nsurfels; i++) {
    surfels[i].SetMark(mark);
  }
}



void R3SurfelBlock::
Transform(const R3Affine& transformation) 
{
  // Check if transformation is identity
  if (transformation.IsIdentity()) return;

  // Get scale factor
  RNScalar scale = transformation.ScaleFactor();
  if (RNIsEqual(scale, 1.0)) scale = 1.0;
  if (scale == 0) return;

  // Update resolution
  if (resolution > 0) resolution /= scale * scale;

  // Transform origin
  R3Point old_origin = origin;
  origin.Transform(transformation);

  // Read block
  if (database) database->ReadBlock(this);

  // Transform surfels 
  bbox = R3null_box;
  for (int i = 0; i < NSurfels(); i++) {
    R3Surfel *surfel = &surfels[i];
    R3Point position(surfel->X() + old_origin[0], surfel->Y() + old_origin[1], surfel->Z() + old_origin[2]);
    R3Vector normal(surfel->NX(), surfel->NY(), surfel->NZ());
    position.Transform(transformation);
    normal.Transform(transformation);
    surfel->SetCoords(position.X() - origin.X(), position.Y() - origin.Y(), position.Z() - origin.Z());
    surfel->SetNormal(normal.X(), normal.Y(), normal.Z());
    surfel->SetRadius(scale * surfel->Radius());
    bbox.Union(position);
  }

  // Remember that block is dirty
  SetDirty();

  // Release block
  if (database) database->ReleaseBlock(this);

  // Update database
  // ???
}



////////////////////////////////////////////////////////////////////////
// PROPERTY UPDATE FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelBlock::
UpdateProperties(void)
{
  // Read block
  if (database) database->ReadBlock(this);

  // Update properties
  double dummy = 0;
  dummy += BBox().Min().X();
  dummy += Resolution();
  dummy += (HasAerial()) ? 1 : 2;
  if (dummy == 927612.21242) {
    printf("Amazing!\n");
  }

  // Update normals
  UpdateSurfelNormals();

  // Release block
  if (database) database->ReleaseBlock(this);
}



void R3SurfelBlock::
UpdateBBox(void)
{
  // Read block
  if (database) database->ReadBlock(this);

  // Update bounding box
  bbox = R3null_box;
  for (int i = 0; i < nsurfels; i++) {
    const float *p = surfels[i].Coords();
    bbox.Union(R3Point(p[0], p[1], p[2]));
  }

  // Release block
  if (database) database->ReleaseBlock(this);

  // Translate bounding box by origin
  bbox.Translate(origin.Vector());

  // Mark bbox uptodate
  flags.Add(R3_SURFEL_BLOCK_BBOX_UPTODATE_FLAG);
}



void R3SurfelBlock::
UpdateResolution(void)
{
  // Initialize resolution
  resolution = 0;

#if 0
  // Estimate resoluton based on surfel radii
  if (nsurfels > 0) {
    // Read block
    if (database) database->ReadBlock(this);

    // Sum surfel radii
    int nsamples = 1000;
    if (nsurfels < nsamples) nsamples = nsurfels;
    int step = nsurfels / nsamples;
    RNLength total_radius = 0;
    for (int i = 0; i < nsamples; i++) {
      R3Surfel *surfel = &surfels[i*step];
      RNLength radius = surfel->Radius();
      total_radius += radius;
    }

    // Resolution is samples / area
    if (RNIsZero(total_radius)) return;
    RNLength average_radius = total_radius / nsamples;
    RNScalar area = RN_PI * average_radius * average_radius;
    resolution = 1.0 / area;

    // Release block
    if (database) database->ReleaseBlock(this);
  }
#else
  // Estimate resolution based on number of surfels per cross-section of bounding box
  R3Box box = BBox();
  int dim = box.ShortestAxis();
  RNLength length1 = box.AxisLength((dim+1)%3);
  RNLength length2 = box.AxisLength((dim+2)%3);
  RNArea area = length1 * length2;
  resolution = (area > 0) ? sqrt(nsurfels / area) : 0;
#endif

  // Mark resolution uptodate
  flags.Add(R3_SURFEL_BLOCK_RESOLUTION_UPTODATE_FLAG);
}



void R3SurfelBlock::
UpdateFlags(void)
{
  // Read block
  if (database) database->ReadBlock(this);

  // Reset flags
  flags.Remove(R3_SURFEL_BLOCK_HAS_ACTIVE_FLAG);
  flags.Remove(R3_SURFEL_BLOCK_HAS_NORMALS_FLAG);
  flags.Remove(R3_SURFEL_BLOCK_HAS_AERIAL_FLAG);
  flags.Remove(R3_SURFEL_BLOCK_HAS_TERRESTRIAL_FLAG);

  // Update flags
  for (int i = 0; i < nsurfels; i++) {
    if (surfels[i].IsActive()) flags.Add(R3_SURFEL_BLOCK_HAS_ACTIVE_FLAG);
    if (surfels[i].HasNormal()) flags.Add(R3_SURFEL_BLOCK_HAS_NORMALS_FLAG);
    if (surfels[i].IsAerial()) flags.Add(R3_SURFEL_BLOCK_HAS_AERIAL_FLAG);
    else flags.Add(R3_SURFEL_BLOCK_HAS_TERRESTRIAL_FLAG);
  }

  // Release block
  if (database) database->ReleaseBlock(this);

  // Mark flags uptodate
  flags.Add(R3_SURFEL_BLOCK_FLAGS_UPTODATE_FLAG);
}



////////////////////////////////////////////////////////////////////////
// SURFEL UPDATE FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelBlock::
UpdateSurfelNormals(void) 
{
  // Update normals of all surfels in block
  R3SurfelPointSet pointset(this);
  pointset.UpdateNormals();
}



////////////////////////////////////////////////////////////////////////
// DRAW FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelBlock::
Draw(RNFlags flags) const
{
  // Get convenient variables
  int c = flags[R3_SURFEL_COLOR_DRAW_FLAG];

  // Push translation to origin
  glPushMatrix();
  glTranslated(origin[0], origin[1], origin[2]);

#ifdef R3_SURFEL_DRAW_WITH_DISPLAY_LIST
  // Create a display list id to use to detect errors
  static GLuint error_id = 0;
  if (error_id == 0) error_id = glGenLists(1);

  // Create display list for block
  if (opengl_id == 0) {
    R3SurfelBlock *block = (R3SurfelBlock *) this;
    block->opengl_id = error_id;
    glGetError();
    GLuint id = glGenLists(2);
    if ((id > 0) && (glGetError() == GL_NO_ERROR)) {
      glNewList(id, GL_COMPILE);
#     if 1
        // Draw without color using arrays
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(3, GL_FLOAT, sizeof(R3Surfel), surfels);
        glDrawArrays(GL_POINTS, 0, NSurfels());
        glDisableClientState(GL_VERTEX_ARRAY);
#     else
        // Draw without color using glBegin and glEnd
        glBegin(GL_POINTS);
        for (int i = 0; i < NSurfels(); i++) {
          const R3Surfel& surfel = surfels[i];
          glVertex3fv(surfel.Coords());
        }
        glEnd();
#     endif
      glEndList();
      if (glGetError() == GL_NO_ERROR) {
        glNewList(id+1, GL_COMPILE);
#       if 0
          // Draw with color using arrays
          glEnableClientState(GL_VERTEX_ARRAY);
          glEnableClientState(GL_COLOR_ARRAY);
          glVertexPointer(3, GL_FLOAT, sizeof(R3Surfel), surfels);
          glColorPointer(3, GL_UNSIGNED_BYTE, sizeof(R3Surfel), surfels[0].Color());
          glDrawArrays(GL_POINTS, 0, NSurfels());
          glDisableClientState(GL_VERTEX_ARRAY);
          glDisableClientState(GL_COLOR_ARRAY);
#       else
          // Draw with color using glBegin and glEnd
          glBegin(GL_POINTS);
          for (int i = 0; i < NSurfels(); i++) {
            const R3Surfel& surfel = surfels[i];
            glColor3ubv(surfel.Color());
            glVertex3fv(surfel.Coords());
          }
          glEnd();
#       endif
        glEndList();
        if (glGetError() == GL_NO_ERROR) {
          block->opengl_id = id;
        }
      }
      if (opengl_id == error_id) {
        glDeleteLists(id, 2);
      }
    }
  }

  // Draw surfels
  if (opengl_id != error_id) {
    // Use display list if available
    glCallList((c) ? opengl_id + 1 : opengl_id);
  }
  else {
    // Draw surfels the slow way
    if (c) {
      // Draw surfels (with color)
      glBegin(GL_POINTS);
      for (int i = 0; i < NSurfels(); i++) {
        const R3Surfel& surfel = surfels[i];
        glColor3ubv(surfel.Color());
        glVertex3fv(surfel.Coords());
      }
      glEnd();
    }
    else {
      // Draw surfels without color
      glBegin(GL_POINTS);
      for (int i = 0; i < NSurfels(); i++) {
        const R3Surfel& surfel = surfels[i];
        glVertex3fv(surfel.Coords());
      }
      glEnd();
    }
  }
#endif

#ifdef R3_SURFEL_DRAW_WITH_VBO
  // Create a VBO id to use to detect errors
  static GLuint error_buffer_id = 0;
  if (error_buffer_id == 0) {
    glGenBuffers(1, &error_buffer_id);
  }

  // Create a VBO for block
  if (opengl_id == 0) {
    glGetError();
    GLuint buffer_id;
    opengl_id = error_buffer_id;
    glGenBuffers(1, &buffer_id);
    if (buffer_id > 0) {
      glBindBuffer(GL_ARRAY_BUFFER, buffer_id);
      glBufferData(GL_ARRAY_BUFFER, NSurfels() * sizeof(R3Surfel), surfels, GL_STATIC_DRAW);
      if (glGetError() == GL_NO_ERROR) {
        opengl_id = buffer_id;
      }
    }
  }

  // Draw surfels
  glEnableClientState(GL_VERTEX_ARRAY);
  if (c) glEnableClientState(GL_COLOR_ARRAY);
  if (opengl_id != error_buffer_id) {
    // Draw surfels using VBO arrays
    glBindBuffer(GL_ARRAY_BUFFER, opengl_id);
    glVertexPointer(3, GL_FLOAT, sizeof(R3Surfel), 0);
    if (c) glColorPointer(3, GL_UNSIGNED_BYTE, sizeof(R3Surfel), 12);
  }
  else {
    // Draw surfels using client-side arrays
    glVertexPointer(3, GL_FLOAT, sizeof(R3Surfel), surfels);
    if (c) glColorPointer(3, GL_UNSIGNED_BYTE, sizeof(R3Surfel), surfels[0].Color());
  }
  glDrawArrays(GL_POINTS, 0, NSurfels());
  glDisableClientState(GL_VERTEX_ARRAY);
  if (c) glDisableClientState(GL_COLOR_ARRAY);
#endif

#ifdef R3_SURFEL_DRAW_WITH_ARRAYS
  // Draw surfels using arrays
  glEnableClientState(GL_VERTEX_ARRAY);
  if (c) glEnableClientState(GL_COLOR_ARRAY);
  glVertexPointer(3, GL_FLOAT, sizeof(R3Surfel), surfels);
  if (c) glColorPointer(3, GL_UNSIGNED_BYTE, sizeof(R3Surfel), surfels[0].Color());
  glDrawArrays(GL_POINTS, 0, NSurfels());
  glDisableClientState(GL_VERTEX_ARRAY);
  if (c) glDisableClientState(GL_COLOR_ARRAY);
#endif

#ifdef R3_SURFEL_DRAW_WITH_POINTS
  // Draw surfels using traditional glBegin and glEnd
  if (c) {
    // Draw points with color
    glBegin(GL_POINTS);
    for (int i = 0; i < NSurfels(); i++) {
      const R3Surfel& surfel = surfels[i];
      glColor3ubv(surfel.Color());
      glVertex3fv(surfel.Coords());
    }
    glEnd();
  }
  else {
    // Draw points without color
    glBegin(GL_POINTS);
    for (int i = 0; i < NSurfels(); i++) {
      const R3Surfel& surfel = surfels[i];
      glVertex3fv(surfel.Coords());
    }
    glEnd();
  }
#endif

  // Pop translation to origin
  glPopMatrix();
}



void R3SurfelBlock::
Print(FILE *fp, const char *prefix, const char *suffix) const
{
  // Check fp
  if (!fp) fp = stdout;

  // Print block
  if (prefix) fprintf(fp, "%s", prefix);
  fprintf(fp, "%d %d", DatabaseIndex(), NSurfels());
  if (suffix) fprintf(fp, "%s", suffix);
  fprintf(fp, "\n");
}



////////////////////////////////////////////////////////////////////////
// I/O FUNCTIONS
////////////////////////////////////////////////////////////////////////

int R3SurfelBlock::
ReadFile(const char *filename)
{
  // Parse input filename extension
  const char *extension;
  if (!(extension = strrchr(filename, '.'))) {
    printf("Filename %s has no extension (e.g., .ply)\n", filename);
    return 0;
  }

  // Read file of appropriate type
  if (!strncmp(extension, ".obj", 4)) {
    return ReadOBJFile(filename);
  }
  else if (!strncmp(extension, ".xyz", 4)) {
    return ReadXYZFile(filename);
  }
  else if (!strncmp(extension, ".bin", 4)) {
    return ReadBinaryFile(filename);
  }
  else if (!strncmp(extension, ".upc", 4)) {
    return ReadUPCFile(filename);
  }
  else { 
    fprintf(stderr, "Unable to read file %s (unrecognized extension: %s)\n", filename, extension); 
    return 0; 
  }

  // Should never get here
  return 0;
}



int R3SurfelBlock::
WriteFile(const char *filename) const
{
  // Parse input filename extension
  const char *extension;
  if (!(extension = strrchr(filename, '.'))) {
    printf("Filename %s has no extension (e.g., .ply)\n", filename);
    return 0;
  }

  // Write file of appropriate type
  if (!strncmp(extension, ".xyz", 4)) {
    return WriteXYZFile(filename);
  }
  else if (!strncmp(extension, ".bin", 4)) {
    return WriteBinaryFile(filename);
  }
  else { 
    fprintf(stderr, "Unable to write file %s (unrecognized extension: %s)\n", filename, extension); 
    return 0; 
  }

  // Should never get here
  return 0;
}



////////////////////////////////////////////////////////////////////////
// OBJ I/O FUNCTIONS
////////////////////////////////////////////////////////////////////////

int R3SurfelBlock::
ReadOBJFile(const char *filename)
{
  // Open file
  FILE *fp;
  if (!(fp = fopen(filename, "r"))) {
    fprintf(stderr, "Unable to open file %s\n", filename);
    return 0;
  }

  // Read file
  if (!ReadOBJ(fp)) {
    fprintf(stderr, "Unable to read OBJ file %s\n", filename);
    return 0;
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



int R3SurfelBlock::
ReadOBJ(FILE *fp)
{
  // Get original file offset
  long int file_offset = RNFileTell(fp);

  // Count the number of surfels and compute centroid
  int count = 0;
  float cx = 0;
  float cy = 0;
  float cz = 0;
  char buffer[4096];
  while (fgets(buffer, 4096, fp)) {
    if (buffer[0] == 'v') {
      // Parse surfel data
      char keyword[64];
      float x, y, z;
      if (sscanf(buffer, "%s%f%f%f", keyword, &x, &y, &z) == (unsigned int) 4) {
        cx += x;
        cy += y;
        cz += z;
        count++;
      }
    }
  }

  // Check number of points
  if (count == 0) return 0;

  // Comput centroid of points
  cx /= count;
  cy /= count;
  cz /= count;

  // Set origin to centroid of points
  origin.Reset(cx, cy, cz);

  // Rewind file to original file offset
  RNFileSeek(fp, file_offset, SEEK_SET);

  // Allocate surfels
  surfels = new R3Surfel [ count ];
  if (!surfels) {
    fprintf(stderr, "Unable to allocate surfel block\n");
    return 0;
  }

  // Read surfels
  nsurfels = 0;
  while (fgets(buffer, 4096, fp)) {
    // Check number of surfels
    if (nsurfels >= count) break;

    // Check if point
    if (buffer[0] == 'v') {
      // Parse surfel data
      char keyword[64];
      float x, y, z;
      if (sscanf(buffer, "%s%f%f%f", keyword, &x, &y, &z) != (unsigned int) 4) {
        fprintf(stderr, "Unable to read point %d out of %d into surfel block\n", nsurfels, count);
        return 0;
      }

      // Assign surfel
      surfels[nsurfels].SetCoords(x - cx, y - cy, z - cz);
      nsurfels++;
    }
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// XYZ I/O FUNCTIONS
////////////////////////////////////////////////////////////////////////

int R3SurfelBlock::
ReadXYZFile(const char *filename)
{
  // Open file
  FILE *fp;
  if (!(fp = fopen(filename, "r"))) {
    fprintf(stderr, "Unable to open file %s\n", filename);
    return 0;
  }

  // Read file
  if (!ReadXYZ(fp)) {
    fprintf(stderr, "Unable to read XYZ file %s\n", filename);
    return 0;
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



int R3SurfelBlock::
WriteXYZFile(const char *filename) const
{
  // Open file
  FILE *fp;
  if (!(fp = fopen(filename, "w"))) {
    fprintf(stderr, "Unable to open file %s\n", filename);
    return 0;
  }

  // Write file
  if (!WriteXYZ(fp)) {
    fprintf(stderr, "Unable to write XYZ file %s\n", filename);
    return 0;
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



int R3SurfelBlock::
ReadXYZ(FILE *fp)
{
  // Get original file offset
  long int file_offset = RNFileTell(fp);

  // Count the number of surfels
  int count = 0;
  char buffer[4096];
  while (fgets(buffer, 4096, fp)) {
    // Check if blank line
    char *bufferp = buffer;
    while (*bufferp && isspace(*bufferp)) bufferp++;
    if (!*bufferp) continue;
    count++;
  }

  // Rewind file to original file offset
  RNFileSeek(fp, file_offset, SEEK_SET);

  // Allocate surfels
  surfels = new R3Surfel [ count ];
  if (!surfels) {
    fprintf(stderr, "Unable to allocate surfel block\n");
    return 0;
  }

  // Read surfels
  nsurfels = 0;
  while (fgets(buffer, 4096, fp)) {
    // Check if blank line
    char *bufferp = buffer;
    while (*bufferp && isspace(*bufferp)) bufferp++;
    if (!*bufferp) continue;

    // Check number of surfels
    if (nsurfels >= count) break;

    // Parse surfel data
    float x, y, z;
    unsigned int r, g, b;
    if (sscanf(buffer, "%f%f%f%u%u%u", &x, &y, &z, &r, &g, &b) != (unsigned int) 6) {
      r = 255; g = 0; b = 0;
      if (sscanf(buffer, "%f%f%f", &x, &y, &z) != (unsigned int) 3) {
        fprintf(stderr, "Unable to read point %d out of %d into surfel block\n", nsurfels, count);
        return 0;
      }
    }

    // Assign surfel
    surfels[nsurfels].SetCoords(x, y, z);
    surfels[nsurfels].SetColor(r, g, b);
    nsurfels++;
  }

  // Return success
  return 1;
}



int R3SurfelBlock::
WriteXYZ(FILE *fp) const
{
  // Write surfels
  for (int i = 0; i < NSurfels(); i++) {
    const R3Surfel *surfel = Surfel(i);
    const float *position = surfel->Coords();
    const unsigned char *color = surfel->Color();
    fprintf(fp, "%g %g %g ", position[0], position[1], position[2]);
    fprintf(fp, "%u %u %u ", color[0], color[1], color[2]);
    fprintf(fp, "0 0\n");
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// BINARY I/O FUNCTIONS
////////////////////////////////////////////////////////////////////////

int R3SurfelBlock::
ReadBinaryFile(const char *filename)
{
  // Open file
  FILE *fp;
  if (!(fp = fopen(filename, "rb"))) {
    fprintf(stderr, "Unable to open file %s\n", filename);
    return 0;
  }

  // Read file
  if (!ReadBinary(fp)) {
    fprintf(stderr, "Unable to read surfel file %s\n", filename);
    return 0;
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



int R3SurfelBlock::
WriteBinaryFile(const char *filename) const
{
  // Open file
  FILE *fp;
  if (!(fp = fopen(filename, "wb"))) {
    fprintf(stderr, "Unable to open file %s\n", filename);
    return 0;
  }

  // Write file
  if (!WriteBinary(fp)) {
    fprintf(stderr, "Unable to write surfel file %s\n", filename);
    return 0;
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



int R3SurfelBlock::
ReadBinary(FILE *fp)
{
  // Read number of surfels
  if (fread(&nsurfels, sizeof(int), 1, fp) != (size_t) 1) {
    fprintf(stderr, "Unable to read number of surfels\n");
    return 0;
  }

  // Read bounding box
  if (fread(&bbox, sizeof(R3Box), 1, fp) != (size_t) 1) {
    fprintf(stderr, "Unable to read bounding box of surfel block\n");
    return 0;
  }

  // Read resolution
  if (fread(&resolution, sizeof(RNLength), 1, fp) != (size_t) 1) {
    fprintf(stderr, "Unable to read resolution of surfel block\n");
    return 0;
  }

  // Read flags
  if (fread(&flags, sizeof(RNFlags), 1, fp) != (size_t) 1) {
    fprintf(stderr, "Unable to read flags of surfel block\n");
    return 0;
  }

  // Allocate memory for surfels
  surfels = new R3Surfel [ nsurfels ];
  if (!surfels) {
    fprintf(stderr, "Unable to allocate surfel block\n");
    return 0;
  }

  // Read surfels
  int count = 0;
  while (count < nsurfels) {
    int status = fread(&surfels[count], sizeof(R3Surfel), nsurfels - count, fp);
    if (status <= 0) {
      fprintf(stderr, "Unable to read surfel block\n");
      return 0;
    }
    count += status;
  }

  // Return success
  return 1;
}



int R3SurfelBlock::
WriteBinary(FILE *fp) const
{
  // Write number of surfels
  int nsurfels = NSurfels();
  if (fwrite(&nsurfels, sizeof(int), 1, fp) != (size_t) 1) {
    fprintf(stderr, "Unable to write number of surfels\n");
    return 0;
  }

  // Write bounding box
  if (fwrite(&bbox, sizeof(R3Box), 1, fp) != (size_t) 1) {
    fprintf(stderr, "Unable to write bounding box of surfel block\n");
    return 0;
  }

  // Write resolution
  if (fwrite(&resolution, sizeof(R3Box), 1, fp) != (size_t) 1) {
    fprintf(stderr, "Unable to write resolution of surfel block\n");
    return 0;
  }

  // Write flags
  RNFlags tmp = flags; tmp.Remove(R3_SURFEL_BLOCK_DATABASE_FLAGS);
  if (fwrite(&tmp, sizeof(RNFlags), 1, fp) != (size_t) 1) {
    fprintf(stderr, "Unable to write flags of surfel block\n");
    return 0;
  }

  // Write surfels
  for (int i = 0; i < NSurfels(); i++) {
    if (fwrite(&surfels[i], sizeof(R3Surfel), 1, fp) != (size_t) 1) return 0;
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// UPC I/O FUNCTIONS
////////////////////////////////////////////////////////////////////////

struct UPCHeader {
  char signature[4]; // Signature of the file format (always UPCf)
  RNUChar8 versionMajor; // The major version number for the file
  RNUChar8 versionMinor; // The minor version number for the file
  RNUInt16 headerSize; // Size of the header block
  RNInt64 numOfPts; // The number of points within the file
  RNScalar64 xScale; // The scale used in the x-coord
  RNScalar64 yScale; // The scale used in the y-coord
  RNScalar64 zScale; // The scale used in the z-coord
  RNScalar64 xOffset; // The offset used in the x-coord
  RNScalar64 yOffset; // The offset used in the y-coord
  RNScalar64 zOffset; // The offset used in the z-coord
};

struct UPCPoint {
  RNScalar64 gpsTimeOfWeek; // The GPS time of week of the point
  RNUChar8 sensorNum[2]; // The laser sensor number used for the point
  RNUChar8 julianDay[3]; // The day the point was collected
  RNUChar8 flightLine[3]; // The flight line number of the point
  RNInt32 x; // The recorded x-coord of the point
  RNInt32 y; // The recorded y-coord of the point
  RNInt32 z; // The recorded z-coord of the point
  RNUChar8 intensity; // The intensity of the point
  RNUInt16 red; // The red component of the point
  RNUInt16 green; // The green component of the point
  RNUInt16 blue; // The blue component of the point
  RNUChar8 returnNum; // The return number of the point
};



int R3SurfelBlock::
ReadUPCFile(const char *filename)
{
  // Open file
  FILE *fp;
  if (!(fp = fopen(filename, "rb"))) {
    fprintf(stderr, "Unable to open file %s\n", filename);
    return 0;
  }

  // Read file
  if (!ReadUPC(fp)) {
    fprintf(stderr, "Unable to read surfel file %s\n", filename);
    return 0;
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



static int 
ReadUPCPreamble(FILE *fp)
{
  // Read preamble (some upc files have junk at front!)
  unsigned long offset = 0;
  char signature[8] = { '\0' };
  while (TRUE) {
    // Read/check signature
    if (fread(signature, sizeof(char), 4, fp) != (size_t) 4) return 0;
    if (!strcmp(signature, "UPCf")) break;
    else offset += 4;
  }
   
  // Seek to start of header
  RNFileSeek(fp, offset, SEEK_SET);

  // Return success
  return 1;
}



static int 
ReadUPCHeader(FILE *fp, UPCHeader& header)
{
  // Read preamble
  if (!ReadUPCPreamble(fp)) {
    fprintf(stderr, "Unable to read signature of UPC file\n");
    return 0;
  }

  // Read signature
  if (fread(&header.versionMajor, 1, 4, fp) != (size_t) 4) {
    fprintf(stderr, "Unable to read UPC header signature\n");
    return 0;
  }

  // Read versionMajor
  if (fread(&header.versionMajor, 1, 1, fp) != (size_t) 1) {
    fprintf(stderr, "Unable to read UPC header versionMajor\n");
    return 0;
  }

  // Read versionMinor
  if (fread(&header.versionMinor, 1, 1, fp) != (size_t) 1) {
    fprintf(stderr, "Unable to read UPC header versionMinor\n");
    return 0;
  }

  // Read headerSize
  if (fread(&header.headerSize, 2, 1, fp) != (size_t) 1) {
    fprintf(stderr, "Unable to read UPC header size\n");
    return 0;
  }

  // Read number of points
  if (fread(&header.numOfPts, 8, 1, fp) != (size_t) 1) {
    fprintf(stderr, "Unable to read number of points in header of UPC file\n");
    return 0;
  }

  // Read scales and offsets
  if (fread(&header.xScale, 8, 6, fp) != (size_t) 6) {
    fprintf(stderr, "Unable to read scales and offsets in header of UPC file\n");
    return 0;
  }

  // Return success
  return 1;
}



int R3SurfelBlock::
ReadUPC(FILE *fp)
{
  // Read header
  UPCHeader header;
  if (!ReadUPCHeader(fp, header)) return 0;

  // Compute scale and offset
  double xoffset = header.xOffset;
  double yoffset = header.yOffset;
  double zoffset = header.zOffset;
  double xscale = header.xScale;
  double yscale = header.yScale;
  double zscale = header.zScale;
 
  // XXX THIS IS A HACK FOR OTTAWA XXX
  xoffset -= 444965;
  yoffset -= 5029450;

  // Set origin
  origin = R3Point(xoffset, yoffset, zoffset);

  // Allocate memory for surfels
  int target_count = (int) header.numOfPts;
  surfels = new R3Surfel [ target_count ];
  if (!surfels) {
    fprintf(stderr, "Unable to allocate surfel block\n");
    return 0;
  }

  // Allocate memory for UPC points
  const int upc_point_size = 36;
  char *upc_points = new char [ target_count * upc_point_size ];

  // Read upc points
  int upc_count = 0;
  while (upc_count < target_count) {
    int status = fread(&upc_points[upc_count], upc_point_size, target_count - upc_count, fp);
    if (status <= 0) break;
    upc_count += status;
  }

  // Assign surfels
  nsurfels = 0;
  for (int i = 0; i < upc_count; i++) {
    char *upc_point = &upc_points[i * upc_point_size];
    double x = *((RNInt32 *) &upc_point[16]) * xscale;
    double y = *((RNInt32 *) &upc_point[20]) * yscale;
    double z = *((RNInt32 *) &upc_point[24]) * zscale;
    RNUInt16 r = *((RNUInt16 *) &upc_point[29]);
    RNUInt16 g = *((RNUInt16 *) &upc_point[31]);
    RNUInt16 b = *((RNUInt16 *) &upc_point[33]);
    if (r > 255) r = 255;
    if (g > 255) g = 255;
    if (b > 255) b = 255;
    surfels[nsurfels].SetCoords(x, y, z);
    surfels[nsurfels].SetColor(r, g, b);
    surfels[nsurfels].SetAerial(upc_point[9] == '0');
    nsurfels++;
  }

  // Delete memory for UPC points
  delete [] upc_points;

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// UPDATE FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelBlock::
UpdateAfterInsert(R3SurfelNode *node)
{
  // Update node
  this->node = node;
}



void R3SurfelBlock::
UpdateBeforeRemove(R3SurfelNode *node)
{
  // Update node
  this->node = NULL;
}



