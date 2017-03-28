/* Source file for the R3 surfel set class */



////////////////////////////////////////////////////////////////////////
// INCLUDE FILES
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS/DESTRUCTORS
////////////////////////////////////////////////////////////////////////

R3SurfelPointSet::
R3SurfelPointSet(void)
  : points(NULL),
    npoints(0),
    nallocated(0),
    bbox(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX)
{
}



R3SurfelPointSet::
R3SurfelPointSet(const R3SurfelPointSet& set)
  : points(NULL),
    npoints(0),
    nallocated(0),
    bbox(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX)
{
  // Insert surfels from set
  InsertPoints(&set);
}



R3SurfelPointSet::
R3SurfelPointSet(R3SurfelBlock *block)
  : points(NULL),
    npoints(0),
    nallocated(0),
    bbox(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX)
{
  // Insert surfels from block
  InsertPoints(block);
}



R3SurfelPointSet::
~R3SurfelPointSet(void)
{
  // Empty set
  Empty();
}




////////////////////////////////////////////////////////////////////////
// PROPERTY FUNCTIONS
////////////////////////////////////////////////////////////////////////

R3Point R3SurfelPointSet::
Centroid(void) const
{
  // Return centroid of set
  R3Point sum(0, 0, 0);
  if (npoints == 0) return sum;
  for (int i = 0; i < npoints; i++) {
    R3SurfelPoint& point = points[i];
    sum[0] += point.X();
    sum[1] += point.Y();
    sum[2] += point.Z();
  }
  return R3Point(sum[0]/npoints, sum[1]/npoints, sum[2]/npoints); 
}



R3Triad R3SurfelPointSet::
PrincipleAxes(const R3Point *center, RNScalar *variances) const
{
  // Check points
  if (npoints == 0) {
    if (variances) { 
      variances[0] = 0.0;
      variances[1] = 0.0;
      variances[2] = 0.0;
    }
    return R3xyz_triad;
  }

  // Get centroid
  R3Point centroid = (center) ? *center : Centroid();

  // Compute covariance matrix
  RNScalar m[9] = { 0 };
  for (int i = 0; i < npoints; i++) {
    R3SurfelPoint& point = points[i];
    RNScalar x = point.X() - centroid[0];
    RNScalar y = point.Y() - centroid[1];
    RNScalar z = point.Z() - centroid[2];
    m[0] += x*x;
    m[4] += y*y;
    m[8] += z*z;
    m[1] += x*y;
    m[3] += x*y;
    m[2] += x*z;
    m[6] += x*z;
    m[5] += y*z;
    m[7] += y*z;
  }

  // Normalize covariance matrix
  for (int i = 0; i < 9; i++) m[i] /= npoints;

  // Compute eigenvalues and eigenvectors
  RNScalar U[9];
  RNScalar W[3];
  RNScalar Vt[9];
  RNSvdDecompose(3, 3, m, U, W, Vt);  // m == U . DiagonalMatrix(W) . Vt

  // Copy principle axes into more convenient form
  // W has eigenvalues (greatest to smallest) and Vt has eigenvectors (normalized)
  R3Vector axes[3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      axes[i][j] = Vt[3*i+j];
    }
  }

  // Flip axes so that "heavier" on positive side for first two dimensions
  int positive_count[3] = { 0, 0, 0 };
  int negative_count[3] = { 0, 0, 0 };
  for (int i = 0; i < npoints; i++) {
    R3SurfelPoint& point = points[i];
    RNScalar x = point.X() - centroid[0];
    RNScalar y = point.Y() - centroid[1];
    RNScalar z = point.Z() - centroid[2];
    R3Vector vector(x, y, z);
    for (int j = 0; j < 2; j++) {
      RNScalar dot = axes[j].Dot(vector);
      if (dot > 0.0) positive_count[j]++;
      else negative_count[j]++;
    }
  }
  for (int j =0; j < 2; j++) {
    if (positive_count[j] < negative_count[j]) {
      axes[j].Flip();
    }
  }

  // Set third axis to form orthonormal triad with other two
  axes[2] = axes[0] % axes[1];

  // Just checking
  assert(RNIsEqual(axes[0].Length(), 1.0, RN_BIG_EPSILON));
  assert(RNIsEqual(axes[1].Length(), 1.0, RN_BIG_EPSILON));
  assert(RNIsEqual(axes[2].Length(), 1.0, RN_BIG_EPSILON));
  assert(RNIsZero(axes[0].Dot(axes[1]), RN_BIG_EPSILON));
  assert(RNIsZero(axes[1].Dot(axes[2]), RN_BIG_EPSILON));
  assert(RNIsZero(axes[0].Dot(axes[2]), RN_BIG_EPSILON));

  // Return variances (eigenvalues)
  if (variances) {
    variances[0] = W[0];
    variances[1] = W[1];
    variances[2] = W[2];
  }

  // Return triad
  return R3Triad(axes[0], axes[1], axes[2]);
}



////////////////////////////////////////////////////////////////////////
// SET MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelPointSet::
InsertPoints(R3SurfelBlock *block)
{
  // Check block
  if (block->NSurfels() == 0) return;

  // Update bounding box
  bbox.Union(block->BBox());

  // Allocate space for points
  AllocatePoints(npoints + block->NSurfels());

  // Read block
  if (block->database) block->database->ReadBlock(block);

  // Copy surfels
  for (int i = 0; i < block->NSurfels(); i++) {
    const R3Surfel *surfel = block->Surfel(i);
    points[npoints].Reset(block, surfel);
    npoints++;
  }

  // Release block
  if (block->database) block->database->ReleaseBlock(block);
}




void R3SurfelPointSet::
InsertPoints(R3SurfelBlock *block, const R2Box& constraint_box)
{
  // Check block
  if (block->NSurfels() == 0) return;

  // Check and update bounding box (conservatively)
  R3Box intersection_box(constraint_box[0][0], constraint_box[0][1], -FLT_MAX, constraint_box[1][0], constraint_box[1][1], FLT_MAX);
  intersection_box.Intersect(block->BBox());
  if (intersection_box.IsEmpty()) return;
  bbox.Union(intersection_box);

  // Translate constraint by block origin (and store in floats)
  const R3Point& origin = block->Origin();
  float xmin = constraint_box[0][0] - origin[0];
  float ymin = constraint_box[0][1] - origin[1];
  float xmax = constraint_box[1][0] - origin[0];
  float ymax = constraint_box[1][1] - origin[1];

  // Allocate space for points
  AllocatePoints(npoints + block->NSurfels());

  // Read block
  if (block->database) block->database->ReadBlock(block);

  // Copy points
  for (int i = 0; i < block->NSurfels(); i++) {
    const R3Surfel *surfel = block->Surfel(i);
    if (surfel->X() < xmin) continue;
    if (surfel->Y() < ymin) continue;
    if (surfel->X() > xmax) continue;
    if (surfel->Y() > ymax) continue;
    points[npoints].Reset(block, surfel);
    npoints++;
  }

  // Release block
  if (block->database) block->database->ReleaseBlock(block);
}




void R3SurfelPointSet::
InsertPoints(R3SurfelBlock *block, const R3Box& constraint_box)
{
  // Check block
  if (block->NSurfels() == 0) return;

  // Check and update bounding box (conservatively)
  R3Box intersection_box = constraint_box;
  intersection_box.Intersect(block->BBox());
  if (intersection_box.IsEmpty()) return;
  bbox.Union(intersection_box);

  // Translate constraint to block coordinate system (and store in floats)
  const R3Point& origin = block->Origin();
  float xmin = constraint_box[0][0] - origin[0];
  float ymin = constraint_box[0][1] - origin[1];
  float zmin = constraint_box[0][2] - origin[2];
  float xmax = constraint_box[1][0] - origin[0];
  float ymax = constraint_box[1][1] - origin[1];
  float zmax = constraint_box[1][2] - origin[2];

  // Allocate space for points
  AllocatePoints(npoints + block->NSurfels());

  // Read block
  if (block->database) block->database->ReadBlock(block);

  // Copy points
  for (int i = 0; i < block->NSurfels(); i++) {
    const R3Surfel *surfel = block->Surfel(i);
    if (surfel->X() < xmin) continue;
    if (surfel->Y() < ymin) continue;
    if (surfel->Z() < zmin) continue;
    if (surfel->X() > xmax) continue;
    if (surfel->Y() > ymax) continue;
    if (surfel->Z() > zmax) continue;
    points[npoints].Reset(block, surfel);
    npoints++;
  }

  // Release block
  if (block->database) block->database->ReleaseBlock(block);
}




void R3SurfelPointSet::
InsertPoints(R3SurfelBlock *block, const R3Point& center, RNLength radius, RNCoord zmin, RNCoord zmax)
{
  // Check block
  if (block->NSurfels() == 0) return;

  // Check and update bounding box (conservatively)
  R3Box constraint_box(center[0] - radius, center[1] - radius, zmin, center[0] + radius, center[1] + radius, zmax);
  R3Box intersection_box = constraint_box;
  intersection_box.Intersect(block->BBox());
  if (intersection_box.IsEmpty()) return;
  bbox.Union(intersection_box);

  // Translate constraint to block coordinate system (and store in floats)
  const R3Point& origin = block->Origin();
  float xc = center[0] - origin[0];
  float yc = center[1] - origin[1];
  float zlo = zmin - origin[2];
  float zhi = zmax - origin[2];
  float rr = radius * radius;

  // Allocate space for points
  AllocatePoints(npoints + block->NSurfels());

  // Read block
  if (block->database) block->database->ReadBlock(block);

  // Copy points
  for (int i = 0; i < block->NSurfels(); i++) {
    const R3Surfel *surfel = block->Surfel(i);
    if (surfel->Z() < zlo) continue;
    if (surfel->Z() > zhi) continue;
    float dx = surfel->X() - xc;
    float dy = surfel->Y() - yc;
    float dd = dx*dx + dy*dy;
    if (dd > rr) continue;
    points[npoints].Reset(block, surfel);
    npoints++;
  }

  // Release block
  if (block->database) block->database->ReleaseBlock(block);
}




void R3SurfelPointSet::
InsertPoints(R3SurfelBlock *block, const R3SurfelConstraint& constraint)
{
  // Check block
  if (block->NSurfels() == 0) return;
  if (!constraint.Check(block->BBox())) return;

  // Allocate space for points
  AllocatePoints(npoints + block->NSurfels());

  // Read block
  if (block->database) block->database->ReadBlock(block);

  // Copy points
  for (int i = 0; i < block->NSurfels(); i++) {
    const R3Surfel *surfel = block->Surfel(i);
    if (!constraint.Check(block, surfel)) continue;
    points[npoints].Reset(block, surfel);
    bbox.Union(points[npoints].Position());
    npoints++;
  }

  // Release block
  if (block->database) block->database->ReleaseBlock(block);
}




void R3SurfelPointSet::
InsertPoints(const R3SurfelPointSet *set)
{
  // Check set
  if (set->NPoints() == 0) return;

  // Check and update bounding box (conservatively)
  bbox.Union(set->BBox());

  // Allocate space for points
  AllocatePoints(npoints + set->NPoints());

  // Copy points
  for (int i = 0; i < set->NPoints(); i++) {
    points[npoints] = set->points[i];
    npoints++;
  }
}



void R3SurfelPointSet::
InsertPoints(const R3SurfelPointSet *set, const R2Box& constraint_box)
{
  // Check set
  if (set->NPoints() == 0) return;

  // Check and update bounding box (conservatively)
  R3Box intersection_box(constraint_box[0][0], constraint_box[0][1], -FLT_MAX, constraint_box[1][0], constraint_box[1][1], FLT_MAX);
  intersection_box.Intersect(set->BBox());
  if (intersection_box.IsEmpty()) return;
  bbox.Union(intersection_box);

  // Allocate space for points
  AllocatePoints(npoints + set->NPoints());

  // Copy points
  for (int i = 0; i < set->NPoints(); i++) {
    const R3SurfelPoint *point = set->Point(i);
    R3Point position = point->Position();
    if (!R2Intersects(constraint_box, R2Point(position.X(), position.Y()))) continue;
    points[npoints] = *point;
    npoints++;
  }
}



void R3SurfelPointSet::
InsertPoints(const R3SurfelPointSet *set, const R3Box& constraint_box)
{
  // Check set
  if (set->NPoints() == 0) return;

  // Check and update bounding box (conservatively)
  R3Box intersection_box = constraint_box;
  intersection_box.Intersect(set->BBox());
  if (intersection_box.IsEmpty()) return;
  bbox.Union(intersection_box);

  // Allocate space for points
  AllocatePoints(npoints + set->NPoints());

  // Copy points
  for (int i = 0; i < set->NPoints(); i++) {
    const R3SurfelPoint *point = set->Point(i);
    if (!R3Intersects(constraint_box, point->Position())) continue;
    points[npoints] = *point;
    npoints++;
  }
}



void R3SurfelPointSet::
InsertPoints(const R3SurfelPointSet *set, const R3Point& center, RNLength radius, RNCoord zmin, RNCoord zmax)
{
  // Check set
  if (set->NPoints() == 0) return;

  // Check and update bounding box (conservatively)
  R3Box constraint_box(center[0] - radius, center[1] - radius, zmin, center[0] + radius, center[1] + radius, zmax);
  R3Box intersection_box = constraint_box;
  intersection_box.Intersect(set->BBox());
  if (intersection_box.IsEmpty()) return;
  bbox.Union(intersection_box);

  // Allocate space for points
  AllocatePoints(npoints + set->NPoints());

  // Copy points
  RNScalar rr = radius * radius;
  for (int i = 0; i < set->NPoints(); i++) {
    const R3SurfelPoint *point = set->Point(i);
    const R3Point& position = point->Position();
    if (position.Z() < zmin) continue;
    if (position.Z() > zmax) continue;
    RNScalar dx = point->X() - center.X();
    RNScalar dy = point->Y() - center.Y();
    RNScalar dd = dx*dx + dy*dy;
    if (dd > rr) continue;
    points[npoints] = *point;
    npoints++;
  }
}



void R3SurfelPointSet::
InsertPoints(const R3SurfelPointSet *set, const R3SurfelConstraint& constraint)
{
  // Check set
  if (set->NPoints() == 0) return;
  if (!constraint.Check(set->BBox())) return;

  // Allocate space for points
  AllocatePoints(npoints + set->NPoints());

  // Insert surfels
  for (int i = 0; i < set->NPoints(); i++) {
    const R3SurfelPoint *point = set->Point(i);
    if (!constraint.Check(point->Position())) continue;
    bbox.Union(point->Position());
    points[npoints] = *point;
    npoints++;
  }
}



void R3SurfelPointSet::
InsertPoint(const R3SurfelPoint& point)
{
  // AllocatePoints points
  if (npoints == nallocated) {
    if (npoints > 0) AllocatePoints(2 * npoints);
    else AllocatePoints(4);
  }

  // Insert point
  points[npoints++] = point;;

  // Update bounding box
  bbox.Union(point.Position());
}



void R3SurfelPointSet::
RemovePoint(const R3SurfelPoint *point)
{
  // Copy last point over point
  RemovePoint(PointIndex(point));
}



void R3SurfelPointSet::
RemovePoint(int k)
{
  // Copy last point over kth point
  assert((k >= 0) && (k < npoints));
  points[k] = points[npoints-1];
  npoints--;
}



void R3SurfelPointSet::
AllocatePoints(int n)
{
  // Check if already big enough
  if (n <= nallocated) return;

  // Allocate enough memory to store n points
  R3SurfelPoint *next_points = new R3SurfelPoint [ n ];

  // Copy and delete old points
  if (points) {
    for (int i = 0; i < npoints; i++) next_points[i] = points[i];
    delete [] points;
  }

  // Update set
  points = next_points;
  nallocated = n;
}



void R3SurfelPointSet::
Empty(void)
{
  // Delete points
  if (points) delete [] points;

  // Reset everything
  nallocated = 0;
  npoints = 0;
  points = NULL;
  bbox = R3null_box;
}



void R3SurfelPointSet::
Subtract(const R3SurfelPointSet *set)
{
  // Clear marks in this point set
  for (int i = 0; i < npoints; i++) {
    points[i].SetMark(FALSE);
  }

  // Set marks in other point set
  for (int i = 0; i < set->npoints; i++) {
    set->points[i].SetMark(TRUE);
  }

  // Remove marked points from this point set
  for (int i = 0; i < npoints; i++) {
    if (!points[i].IsMarked()) continue;
    points[i] = points[npoints-1];
    npoints--;
    i--;
  }

  // Update bounding box (conservatively)
  // Do nothing
}



void R3SurfelPointSet::
Intersect(const R3SurfelPointSet *set)
{
  // Set marks in this point set
  for (int i = 0; i < npoints; i++) {
    points[i].SetMark(TRUE);
  }

  // Clear marks in other point set
  for (int i = 0; i < set->npoints; i++) {
    set->points[i].SetMark(FALSE);
  }

  // Remove marked points from this point set
  for (int i = 0; i < npoints; i++) {
    if (!points[i].IsMarked()) continue;
    points[i] = points[npoints-1];
    npoints--;
    i--;
  }

  // Update bounding box (conservatively)
  // Do nothing
}



void R3SurfelPointSet::
Union(const R3SurfelPointSet *set)
{
  // Set marks in other point set
  for (int i = 0; i < set->npoints; i++) {
    set->points[i].SetMark(TRUE);
  }

  // Clear marks in this point set
  for (int i = 0; i < npoints; i++) {
    points[i].SetMark(FALSE);
  }

  // Count marked points from other point set
  int count = 0;
  for (int i = 0; i < set->npoints; i++) {
    if (!set->points[i].IsMarked()) continue;
    count++;
  }

  // Allocate space for points
  AllocatePoints(npoints + count);

  // Insert marked points from other point set
  for (int i = 0; i < set->npoints; i++) {
    if (!set->points[i].IsMarked()) continue;
    points[npoints] = set->points[i];
    npoints++;
  }

  // Update bounding box (conservatively)
  bbox.Union(set->BBox());
}



R3SurfelPointSet& R3SurfelPointSet::
operator=(const R3SurfelPointSet& set)
{
  // Empty this pointset
  Empty();

  // Insert points from other pointset
  InsertPoints(&set);

  // Return this
  return *this;
}



void R3SurfelPointSet::
SetMarks(RNBoolean mark)
{
  // Set mark for all points
  for (int i = 0; i < npoints; i++) {
    points[i].SetMark(mark);
  }
}



RNArray<R3SurfelBlock *> *R3SurfelPointSet::
Blocks(void) const
{
  // Check pointset 
  if (npoints == 0) return NULL;

  // Create array of blocks
  RNArray<R3SurfelBlock *> *blocks = new RNArray<R3SurfelBlock *>();
  if (!blocks) {
    fprintf(stderr, "Unable to allocate array of blocks\n");
    return NULL;
  }

  // Fill array of blocks
  R3SurfelBlock *last_block = NULL;
  for (int i = 0; i < npoints; i++) {
    R3SurfelPoint *point = &points[i];
    R3SurfelBlock *block = point->Block();
    if (!block) continue;
    if (block == last_block) continue;
    if (blocks->FindEntry(block)) continue;
    blocks->Insert(block);
    last_block = block;
  }

  // Check array of blocks
  if (blocks->IsEmpty()) {
    delete blocks;
    return NULL;
  }

  // Return array of blocks
  return blocks;
}



RNArray<R3SurfelNode *> *R3SurfelPointSet::
Nodes(void) const
{
  // Check pointset 
  if (npoints == 0) return NULL;

  // Create array of nodes
  RNArray<R3SurfelNode *> *nodes = new RNArray<R3SurfelNode *>();
  if (!nodes) {
    fprintf(stderr, "Unable to allocate array of nodes\n");
    return NULL;
  }

  // Fill array of nodes
  R3SurfelNode *last_node = NULL;
  for (int i = 0; i < npoints; i++) {
    R3SurfelPoint *point = &points[i];
    R3SurfelBlock *block = point->Block();
    if (!block) continue;
    R3SurfelNode *node = block->Node();
    if (!node) continue;
    if (node == last_node) continue;
    if (nodes->FindEntry(node)) continue;
    nodes->Insert(node);
    last_node = node;
  }    

  // Check array of nodes
  if (nodes->IsEmpty()) {
    delete nodes;
    return NULL;
  }

  // Return array of nodes
  return nodes;
}



RNArray<R3SurfelObject *> *R3SurfelPointSet::
Objects(void) const
{
  // Check pointset 
  if (npoints == 0) return NULL;

  // Create array of objects
  RNArray<R3SurfelObject *> *objects = new RNArray<R3SurfelObject *>();
  if (!objects) {
    fprintf(stderr, "Unable to allocate array of objects\n");
    return NULL;
  }

  // Fill array of objects
  R3SurfelObject *last_object = NULL;
  for (int i = 0; i < npoints; i++) {
    R3SurfelPoint *point = &points[i];
    R3SurfelBlock *block = point->Block();
    if (!block) continue;
    R3SurfelNode *node = block->Node();
    if (!node) continue;
    R3SurfelObject *object = node->Object();
    if (!object) continue;
    if (object == last_object) continue;
    if (objects->FindEntry(object)) continue;
    objects->Insert(object);
    last_object = object;
  }    

  // Check array of objects
  if (objects->IsEmpty()) {
    delete objects;
    return NULL;
  }

  // Return array of objects
  return objects;
}



////////////////////////////////////////////////////////////////////////
// DRAW FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelPointSet::
Draw(RNFlags flags) const
{
  // Draw surfels
  if (flags[R3_SURFEL_COLOR_DRAW_FLAG]) {
    // Draw surfels with color
    glBegin(GL_POINTS);
    for (int i = 0; i < NPoints(); i++) {
      const R3SurfelPoint *point = Point(i);
      glColor3ubv(point->Color());
      glVertex3dv(point->Position().Coords());
    }
    glEnd();
  }
  else {
    // Draw surfels without color
    glBegin(GL_POINTS);
    for (int i = 0; i < NPoints(); i++) {
      const R3SurfelPoint *point = Point(i);
      glVertex3dv(point->Position().Coords());
    }
    glEnd();
  }
}



////////////////////////////////////////////////////////////////////////
// I/O FUNCTIONS
////////////////////////////////////////////////////////////////////////

int R3SurfelPointSet::
ReadFile(const char *filename)
{
  // Create block
  R3SurfelBlock *block = new R3SurfelBlock();
  if (!block) {
    fprintf(stderr, "Unable to allocate block for %s\n", filename);
    return 0;
  }

  // Read block
  if (!block->ReadFile(filename)) return 0;

  // Insert points from block
  InsertPoints(block);

  // Return success
  return 1;
}



int R3SurfelPointSet::
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
  else { 
    fprintf(stderr, "Unable to write file %s (unrecognized extension: %s)\n", filename, extension); 
    return 0; 
  }

  // Should never get here
  return 0;
}



int R3SurfelPointSet::
WriteXYZFile(const char *filename) const
{
  // Open file
  FILE *fp;
  if (!(fp = fopen(filename, "w"))) {
    fprintf(stderr, "Unable to open file %s\n", filename);
    return 0;
  }

  // Write points
  for (int i = 0; i < NPoints(); i++) {
    const R3SurfelPoint *point = Point(i);
    R3Point position = point->Position();
    fprintf(fp, "%g %g %g\n", position.X(), position.Y(), position.Z());
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// UPDATE FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelPointSet::
UpdateNormals(RNScalar max_neighborhood_radius, int max_neighborhood_points) const
{
  // Declare variables (fill it only if needed)
  R3Kdtree<R3SurfelPoint *> *kdtree = NULL;
  RNArray<R3SurfelPoint *> neighbors;
  R3Point *positions = NULL;
  R3Point *viewpoint = NULL;

  // Compute normals for all points that don't already have them
  for (int i = 0; i < NPoints(); i++) {
    R3SurfelPoint *point = Point(i);
    if (point->HasNormal()) continue;

    // Allocate kdtree
    if (!kdtree) {
      RNArray<R3SurfelPoint *> points;
      for (int j = 0; j < NPoints(); j++) points.Insert(Point(j));
      if (!kdtree) kdtree = new R3Kdtree<R3SurfelPoint *>(points, SurfelPointPosition, NULL);
      if (!kdtree) RNAbort("Unable to allocate kdtree to update normals");
    }

    // Find neighbors with kdtree
    neighbors.Truncate(0);
    kdtree->FindClosest(point, 0, max_neighborhood_radius, max_neighborhood_points, neighbors);
    if (neighbors.NEntries() < 3) continue;

    // Allocate array of positions
    if (!positions) {
      positions = new R3Point [max_neighborhood_points + 1];
      if (!positions) RNAbort("Unable to allocate positions to update normals");
    }

    // Create array of positions for neighborhood
    int npositions = 0;
    positions[npositions++] = point->Position();
    for (int j = 0; j < neighbors.NEntries(); j++) {
      R3SurfelPoint *neighbor = neighbors.Kth(j);
      positions[npositions++] = neighbor->Position();
    }

    // Compute normal with PCA of neighborhood
    R3Point centroid = R3Centroid(npositions, positions);
    R3Triad triad = R3PrincipleAxes(centroid, npositions, positions);
    R3Vector normal = triad[2];

    // Compute scan viewpoint
    if (!viewpoint) {
      static R3Point estimated_viewpoint(0, 0, 0);
      estimated_viewpoint = Centroid();
      viewpoint = &estimated_viewpoint;
    }

    // Orient normal towards scan viewpoint
    R3Plane plane(centroid, normal);
    if (R3SignedDistance(plane, *viewpoint) < 0) normal.Flip();

    // Assign normal
    point->SetNormal(normal);
  }

  // Delete data
  if (positions) delete [] positions;
  if (kdtree) delete kdtree;
}




