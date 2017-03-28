/* Include file for the R3 surfel graph class */



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelPointGraph {
public:
  //////////////////////////////////////////
  //// CONSTRUCTOR/DESTRUCTOR FUNCTIONS ////
  //////////////////////////////////////////

  // Constructor functions
  R3SurfelPointGraph(void);
  R3SurfelPointGraph(const R3SurfelPointGraph& graph);
  R3SurfelPointGraph(const R3SurfelPointSet& set, int max_neighbors = 16, RNLength max_distance = 1);

  // Destructor functions
  virtual ~R3SurfelPointGraph(void);


  //////////////////////////////////
  //// GRAPH PROPERTY FUNCTIONS ////
  //////////////////////////////////

  // Geometric property functions
  R3Point Centroid(void) const;
  const R3Box& BBox(void) const;

  // Graph property functions
  int MaxNeighbors(void) const;
  RNLength MaxDistance(void) const;


  ////////////////////////////////
  //// POINT ACCESS FUNCTIONS ////
  ////////////////////////////////

  // Point access functions
  int NPoints(void) const;
  R3SurfelPoint *Point(int k) const;
  R3SurfelPoint *operator[](int k) const;

  // Surfel neighbor access functions
  int NNeighbors(int surfel_index) const;
  R3SurfelPoint *Neighbor(int surfel_index, int neighbor_index) const;


  //////////////////////////////////
  //// POINT PROPERTY FUNCTIONS ////
  //////////////////////////////////

  int PointIndex(const R3SurfelPoint *point) const;
  R3Vector PointNormal(const R3SurfelPoint *point, RNBoolean fast_and_approximate = FALSE) const;
  R3Vector PointNormal(int point_index, RNBoolean fast_and_approximate = FALSE) const;


  ////////////////////////////////
  //// MANIPULATION FUNCTIONS ////
  ////////////////////////////////

  // Point manipulation functions
  void SetMarks(RNBoolean mark);


  ///////////////////////////
  //// DISPLAY FUNCTIONS ////
  ///////////////////////////

  // Draw functions
  virtual void Draw(RNFlags flags = R3_SURFEL_DEFAULT_DRAW_FLAGS) const;


  ////////////////////////////////////////////////////////////////////////
  // INTERNAL STUFF BELOW HERE
  ////////////////////////////////////////////////////////////////////////

  // Update functions
  void UpdateNormals(void) const;

  // Test function
  void RemoveOutlierEdges(RNScalar zscore);


private:
  R3SurfelPointSet set;
  RNArray<R3SurfelPoint *> *neighbors;
  int max_neighbors;
  RNLength max_distance;
};



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline int R3SurfelPointGraph::
PointIndex(const R3SurfelPoint *point) const
{
  // Return index of point
  return set.PointIndex(point);
}



inline R3Point R3SurfelPointGraph::
Centroid(void) const
{
  // Return centroid of graph
  return set.Centroid();
}



inline const R3Box& R3SurfelPointGraph::
BBox(void) const
{
  // Return bounding box of graph
  return set.BBox();
}



inline int R3SurfelPointGraph::
MaxNeighbors(void) const
{
  // Return maximum number of neighbors
  return max_neighbors;
}




inline RNLength R3SurfelPointGraph::
MaxDistance(void) const
{
  // Return maximum distance between neighbors
  return max_distance;
}



inline int R3SurfelPointGraph::
NPoints(void) const
{
  // Return number of surfels
  return set.NPoints();
}



inline R3SurfelPoint *R3SurfelPointGraph::
Point(int k) const
{
  // Return kth surfel
  return set.Point(k);
}



inline R3SurfelPoint *R3SurfelPointGraph::
operator[](int k) const
{
  // Return kth surfel
  return Point(k);
}



inline int R3SurfelPointGraph::
NNeighbors(int surfel_index) const
{
  // Return number of neighbors
  return neighbors[surfel_index].NEntries();
}



inline R3SurfelPoint *R3SurfelPointGraph::
Neighbor(int surfel_index, int neighbor_index) const
{
  // Return neighbor
  return neighbors[surfel_index][neighbor_index];
}



inline R3Vector R3SurfelPointGraph::
PointNormal(const R3SurfelPoint *point, RNBoolean fast_and_approximate) const
{
  // Return normal at point
  int point_index = PointIndex(point);
  if (point_index < 0) return R3zero_vector;
  return PointNormal(point_index, fast_and_approximate);
}



inline void R3SurfelPointGraph::
SetMarks(RNBoolean mark)
{
  // Set marks
  set.SetMarks(mark);
}



