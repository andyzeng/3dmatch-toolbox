/* Source file for the R3 surfel node set class */



////////////////////////////////////////////////////////////////////////
// INCLUDE FILES
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS/DESTRUCTORS
////////////////////////////////////////////////////////////////////////

R3SurfelNodeSet::
R3SurfelNodeSet(void)
  : nodes(),
    complexity(0),
    bbox(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX)
{
}



R3SurfelNodeSet::
R3SurfelNodeSet(const R3SurfelNodeSet& set)
  : nodes(set.nodes),
    complexity(set.complexity),
    bbox(set.bbox)
{
}



R3SurfelNodeSet::
~R3SurfelNodeSet(void)
{
}



////////////////////////////////////////////////////////////////////////
// SET MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

static RNLength 
SquaredXYDistance(const R3Point& point, const R3Box& box)
{
  // Find axial distances in XY plane
  RNLength dx, dy;
  if (RNIsGreater(point.X(), box.XMax())) dx = point.X() - box.XMax();
  else if (RNIsLess(point.X(), box.XMin())) dx = box.XMin()- point.X();
  else dx = 0.0;
  if (RNIsGreater(point.Y(), box.YMax())) dy = point.Y() - box.YMax();
  else if (RNIsLess(point.Y(), box.YMin())) dy = box.YMin()- point.Y();
  else dy = 0.0;
    
  // Return square of distance in XY between point and closest point in box 
  if (dy == 0.0) return dx*dx;
  else if (dx == 0.0) return dy*dy;
  else return dx*dx + dy*dy;
}



void R3SurfelNodeSet::
InsertNodes(R3SurfelTree *tree, R3SurfelNode *node,
  const R3Point& xycenter, RNLength xyradius, 
  RNCoord zmin, RNCoord zmax,
  RNScalar center_resolution, RNScalar perimeter_resolution,
  RNScalar focus_exponent)
{
  // Check if node is in infinite cylinder
  if (node->BBox().ZMin() > zmax) return;
  if (node->BBox().ZMax() < zmin) return;
  RNLength squared_xydistance = SquaredXYDistance(xycenter, node->BBox());
  if (squared_xydistance > xyradius * xyradius) return;

  // Check if this node is a leaf
  if (node->NParts() == 0) {
    // Simply insert leaf node
    InsertNode(node);
    return;
  }

  // Check if we are selecting interior nodes based on resolution
  if (center_resolution > 0) {
    // Compute the target resolution at this node position
    RNScalar target_resolution = center_resolution;
    if ((xyradius > 0) && (perimeter_resolution > 0)) {
      RNLength xydistance = sqrt(squared_xydistance);
      RNScalar t = 1 - xydistance / xyradius;
      target_resolution = perimeter_resolution + 
        (center_resolution - perimeter_resolution) * pow(t, focus_exponent);
    }

    // Check if this node is within a factor of two of target resolution
    if (node->Resolution() > 0.5 * target_resolution) {
      // Insert all parts of this node and don't recurse 
      for (int i = 0; i < node->NParts(); i++) InsertNode(node->Part(i));
      return;
    }
  }

  // Recurse to parts 
  for (int i = 0; i < node->NParts(); i++) {
    R3SurfelNode *part = node->Part(i);
    InsertNodes(tree, part, xycenter, xyradius, zmin, zmax, center_resolution, perimeter_resolution);
  }
}



void R3SurfelNodeSet::
InsertNodes(R3SurfelTree *tree, 
  const R3Point& xycenter, RNLength xyradius, 
  RNCoord zmin, RNCoord zmax,
  RNScalar center_resolution, RNScalar perimeter_resolution,
  RNScalar focus_exponent)
{
  // Insert nodes recursively starting at root
  InsertNodes(tree, tree->RootNode(), 
    xycenter, xyradius, zmin, zmax, 
    center_resolution, perimeter_resolution, focus_exponent);
}



void R3SurfelNodeSet::
InsertNodes(R3SurfelTree *tree)
{
  // Insert all leaf nodes 
  for (int i = 0; i < tree->NNodes(); i++) {
    R3SurfelNode *node = tree->Node(i);
    if (node->NParts() > 0) continue;
    InsertNode(node);
  }
}



void R3SurfelNodeSet::
InsertNode(R3SurfelNode *node)
{
  // Insert node
  nodes.Insert(node);

  // Update complexity
  complexity += node->Complexity();

  // Update bounding box
  bbox.Union(node->BBox());
}



void R3SurfelNodeSet::
RemoveNode(R3SurfelNode *node)
{
  // Remove node
  RNArrayEntry *entry = nodes.FindEntry(node);
  if (entry) RemoveNode(nodes.EntryIndex(entry));
}



void R3SurfelNodeSet::
RemoveNode(int k)
{
  // Update complexity
  complexity -= nodes[k]->Complexity();

  // Copy last node over node
  RNArrayEntry *entry = nodes.KthEntry(k);
  assert(entry);
  R3SurfelNode *tail = nodes.Tail();
  nodes.EntryContents(entry) = tail;
  nodes.RemoveTail();

  // Update bounding box
  // XXX
}



void R3SurfelNodeSet::
Empty(void)
{
  // Remove nodes
  nodes.Empty();

  // Update complexity
  complexity = 0;

  // Update bounding box
  bbox = R3null_box;
}



///////////////////////////////////////////////////////////////////////
// MEMORY MANGEMENT FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelNodeSet::
ReadBlocks(void)
{
  // Read nodes in set
  for (int i = 0; i < NNodes(); i++) {
    R3SurfelNode *node = Node(i);
    node->ReadBlocks();
  }
}



void R3SurfelNodeSet::
ReleaseBlocks(void)
{
  // Release nodes in set
  for (int i = 0; i < NNodes(); i++) {
    R3SurfelNode *node = Node(i);
    node->ReleaseBlocks();
  }
}



RNBoolean R3SurfelNodeSet::
AreBlocksResident(void) const
{
  // Return whether nodes are in memory
  for (int i = 0; i < NNodes(); i++) {
    R3SurfelNode *node = Node(i);
    if (!node->AreBlocksResident()) return FALSE;
  }

  // All nodes are resident
  return TRUE;
}



////////////////////////////////////////////////////////////////////////
// DISPLAY FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelNodeSet::
Draw(RNFlags flags) const
{
  // Draw nodes
  for (int i = 0; i < NNodes(); i++) {
    R3SurfelNode *node = Node(i);
    node->Draw(flags);
  }
}



void R3SurfelNodeSet::
Print(FILE *fp, const char *prefix, const char *suffix) const
{
  // Check fp
  if (!fp) fp = stdout;

  // Print nodes
  // Print all surfel nodes
  if (prefix) fprintf(fp, "%s", prefix);
  for (int i = 0; i < NNodes(); i++) {
    R3SurfelNode *node = Node(i);
    fprintf(fp, "%d ", node->TreeIndex());
  }
  if (suffix) fprintf(fp, "%s", suffix);
  fprintf(fp, "\n");
}


