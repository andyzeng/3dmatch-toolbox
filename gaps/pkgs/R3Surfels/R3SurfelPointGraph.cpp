/* Source file for the R3 surfel graph class */



////////////////////////////////////////////////////////////////////////
// INCLUDE FILES
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS/DESTRUCTORS
////////////////////////////////////////////////////////////////////////

R3SurfelPointGraph::
R3SurfelPointGraph(void)
  : set(),
    neighbors(NULL),
    max_neighbors(0),
    max_distance(-1)
{
}



R3SurfelPointGraph::
R3SurfelPointGraph(const R3SurfelPointGraph& graph)
  : set(graph.set),
    neighbors(NULL),
    max_neighbors(graph.max_neighbors),
    max_distance(graph.max_distance)
{
  // Copy neighbors
  neighbors = new RNArray<R3SurfelPoint *> [ NPoints() ];
  for (int i = 0; i < NPoints(); i++) {
    neighbors[i].Resize(graph.neighbors[i].NEntries());
    for (int j = 0; j < graph.neighbors[i].NEntries(); j++) {
      neighbors[i].Insert(graph.neighbors[i][j]);
    }
  }
}



R3SurfelPointGraph::
R3SurfelPointGraph(const R3SurfelPointSet& set, int max_neighbors, RNLength max_distance)
  : set(set),
    neighbors(NULL),
    max_neighbors(max_neighbors),
    max_distance(max_distance)
{
  // Allocate neighbors
  neighbors = new RNArray<R3SurfelPoint *> [ NPoints() ];

  // Find neighbors with kdtree
  RNArray<R3SurfelPoint *> points;
  for (int i = 0; i < NPoints(); i++) points.Insert(Point(i));
  R3Kdtree<R3SurfelPoint *> kdtree(points, SurfelPointPosition, NULL);
  for (int i = 0; i < NPoints(); i++) {
    kdtree.FindClosest(Point(i), 0, max_distance, max_neighbors, neighbors[i]);
  }
}



R3SurfelPointGraph::
~R3SurfelPointGraph(void)
{
  // Delete neighbors
  if (neighbors) {
    delete [] neighbors;
    neighbors = NULL;
  }
}



////////////////////////////////////////////////////////////////////////
// POINT PROPERTIES
////////////////////////////////////////////////////////////////////////

R3Vector R3SurfelPointGraph::
PointNormal(int point_index, RNBoolean fast_and_approximate) const
{
  // Get point index
  if (point_index < 0) return R3zero_vector;
  if (NNeighbors(point_index) < 2) return R3zero_vector;
  R3SurfelPoint *point = Point(point_index);

  // Check algorithm
  R3Vector normal;
  if (fast_and_approximate) {
    // Compute normal with vector cross product
    R3Point positions[3];
    int index1 = (int) (RNRandomScalar() * NNeighbors(point_index));
    int index2 = (index1 + NNeighbors(point_index)/2) % NNeighbors(point_index);
    const R3SurfelPoint *neighbor1 = Neighbor(point_index, index1);
    const R3SurfelPoint *neighbor2 = Neighbor(point_index, index2);
    positions[0] = point->Position();
    positions[1] = neighbor1->Position();
    positions[2] = neighbor2->Position();
    R3Vector v1 = positions[1] - positions[0];
    R3Vector v2 = positions[2] - positions[0];
    normal = v1 % v2;
    normal.Normalize();
  }
  else {
    // Create neighborhood points
    int npositions = NNeighbors(point_index) + 1;
    R3Point *positions = new R3Point [npositions];
    positions[0] = point->Position();
    for (int j = 0; j < NNeighbors(point_index); j++) {
      const R3SurfelPoint *neighbor = Neighbor(point_index, j);
      positions[j+1] = neighbor->Position();
    }

    // Compute normal of neighborhood points with PCA 
    R3Point centroid = R3Centroid(npositions, positions);
    R3Triad triad = R3PrincipleAxes(centroid, npositions, positions);

    // Delete neighborhood points
    delete [] positions;

    // Return normal
    normal = triad[2];
  }

  // Flip normal so that positive in max dimension
  int dim = normal.MaxDimension();
  if (normal[dim] < 0) normal.Flip();

  // Return normal
  return normal;
}



////////////////////////////////////////////////////////////////////////
// DRAW FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelPointGraph::
Draw(RNFlags flags) const
{
  // Draw surfels
  set.Draw(flags);

  // Draw lines between neighbors
  glBegin(GL_LINES);
  for (int i = 0; i < NPoints(); i++) {
    R3SurfelPoint *point0 = Point(i);
    for (int j = 0; j < NNeighbors(i); j++) {
      R3SurfelPoint *point1 = Neighbor(i,j);
      R3LoadPoint(point0->Position());
      R3LoadPoint(point1->Position());
    }
  }
  glEnd();
}



////////////////////////////////////////////////////////////////////////
// TEST FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelPointGraph::
UpdateNormals(void) const
{
  // Declare variables (fill it only if needed)
  R3Point *positions = NULL;

  // Compute normals for all points that don't already have them
  for (int i = 0; i < NPoints(); i++) {
    R3SurfelPoint *point = Point(i);
    if (point->HasNormal()) continue;

    // Allocate array of positions
    if (!positions) {
      positions = new R3Point [MaxNeighbors() + 1];
      if (!positions) RNAbort("Unable to allocate positions to update normals");
    }

    // Create array of positions for neighborhood
    int npositions = 0;
    positions[npositions++] = point->Position();
    for (int j = 0; j < NNeighbors(i); j++) {
      R3SurfelPoint *neighbor = Neighbor(i, j);
      positions[npositions++] = neighbor->Position();
    }

    // Compute normal with PCA of neighborhood
    R3Point centroid = R3Centroid(npositions, positions);
    R3Triad triad = R3PrincipleAxes(centroid, npositions, positions);
    R3Vector normal = triad[2];

    // Assign normal
    point->SetNormal(normal);
  }

  // Delete data
  if (positions) delete [] positions;
}




void R3SurfelPointGraph::
RemoveOutlierEdges(RNScalar max_zscore)
{
  // Allocate temporary memory for edge length statistics
  RNScalar *means = new RNScalar [ NPoints() ];
  RNScalar *stddevs = new RNScalar [ NPoints() ];

  // Compute edge length statistics
  for (int i = 0; i < NPoints(); i++) {
    // Compute sum
    RNLength sum = 0;
    R3SurfelPoint *point0 = Point(i);
    R3Point position0 = point0->Position();
    for (int j = 0; j < NNeighbors(i); j++) {
      R3SurfelPoint *point1 = Neighbor(i,j);
      R3Point position1 = point1->Position();
      RNLength edge_length = R3Distance(position0, position1);
      sum += edge_length;
    }
    
    // Compute mean
    means[i] = (NNeighbors(i)) ? sum / NNeighbors(i) : 0;

    // Compute sum of squared residuals
    RNLength ssd = 0;
    for (int j = 0; j < NNeighbors(i); j++) {
      R3SurfelPoint *point1 = Neighbor(i,j);
      R3Point position1 = point1->Position();
      RNLength edge_length = R3Distance(position0, position1);
      RNLength delta = edge_length - means[i];
      ssd += delta * delta;
    }

    // Compute standard deviation
    RNScalar variance = (NNeighbors(i)) ? ssd / NNeighbors(i) : 0; 
    stddevs[i] = sqrt(variance);
  }

  // Remove outlier edges
  for (int i = 0; i < NPoints(); i++) {
    if (stddevs[i] == 0) continue;
    R3SurfelPoint *point0 = Point(i);
    R3Point position0 = point0->Position();
    for (int j = 0; j < neighbors[i].NEntries(); j++) {
      R3SurfelPoint *point1 = neighbors[i].Kth(j);
      R3Point position1 = point1->Position();
      RNLength edge_length = R3Distance(position0, position1);
      RNScalar zscore = (edge_length - means[i]) / stddevs[i];
      if (zscore > max_zscore) {
        // Remove edge
        RNArrayEntry *entry = neighbors[i].KthEntry(j);
        neighbors[i].EntryContents(entry) = neighbors[i].Tail();
        neighbors[i].RemoveTail();
        j--;
      }
    }
  }    

  // Delete temporary memory for edge length statistics
  delete [] means;
  delete [] stddevs;
}




