/* Include file for the R3 surfel set class */



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelPointSet {
public:
  // Constructor functions
  R3SurfelPointSet(void);
  R3SurfelPointSet(const R3SurfelPointSet& set);
  R3SurfelPointSet(R3SurfelBlock *block);
  virtual ~R3SurfelPointSet(void);

  // Surfel access functions
  int NPoints(void) const;
  R3SurfelPoint *Point(int k) const;
  R3SurfelPoint *operator[](int k) const;
  int PointIndex(const R3SurfelPoint *point) const;

  // Shape property functions
  R3Point Centroid(void) const;
  const R3Box& BBox(void) const;
  R3Triad PrincipleAxes(const R3Point *centroid = NULL, RNScalar *variances = NULL) const;

  // Membership manipulation functions
  virtual void InsertPoints(R3SurfelBlock *block);
  virtual void InsertPoints(R3SurfelBlock *block, const R3Box& box);
  virtual void InsertPoints(R3SurfelBlock *block, const R2Box& box);
  virtual void InsertPoints(R3SurfelBlock *block, const R3Point& center, RNLength radius, RNCoord zmin = -FLT_MAX, RNCoord zmax = FLT_MAX);
  virtual void InsertPoints(R3SurfelBlock *block, const R3SurfelConstraint& constraint);
  virtual void InsertPoints(const R3SurfelPointSet *set);
  virtual void InsertPoints(const R3SurfelPointSet *set, const R3Box& box);
  virtual void InsertPoints(const R3SurfelPointSet *set, const R2Box& box);
  virtual void InsertPoints(const R3SurfelPointSet *set, const R3Point& center, RNLength radius, RNCoord zmin = -FLT_MAX, RNCoord zmax = FLT_MAX);
  virtual void InsertPoints(const R3SurfelPointSet *set, const R3SurfelConstraint& constraint);
  virtual void InsertPoint(const R3SurfelPoint& point);
  virtual void RemovePoint(const R3SurfelPoint *point);
  virtual void RemovePoint(int k);
  virtual void AllocatePoints(int n);

  // Set manipulation functions
  virtual void Empty(void);
  virtual void Subtract(const R3SurfelPointSet *set);
  virtual void Intersect(const R3SurfelPointSet *set);
  virtual void Union(const R3SurfelPointSet *set);
  virtual R3SurfelPointSet& operator=(const R3SurfelPointSet& set);

  // Point manipulation functions
  virtual void SetMarks(RNBoolean mark = TRUE);

  // Draw functions
  virtual void Draw(RNFlags flags = R3_SURFEL_DEFAULT_DRAW_FLAGS) const;

  // I/O functions
  virtual int ReadFile(const char *filename);
  virtual int WriteFile(const char *filename) const;
  virtual int WriteXYZFile(const char *filename) const;


  ////////////////////////////////////////////////////////////////////////
  // INTERNAL STUFF BELOW HERE
  ////////////////////////////////////////////////////////////////////////

  // Update functions
  void UpdateNormals(RNScalar max_neighborhood_radius = 1.0, int max_neighborhood_points = 8) const;

  // Get arrays of things containing points in pointset
  RNArray<R3SurfelBlock *> *Blocks(void) const;
  RNArray<R3SurfelNode *> *Nodes(void) const;
  RNArray<R3SurfelObject *> *Objects(void) const;

private:
  R3SurfelPoint *points;
  int npoints;
  int nallocated;
  R3Box bbox;
};



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline int R3SurfelPointSet::
NPoints(void) const
{
  // Return number of surfels
  return npoints;
}



inline R3SurfelPoint *R3SurfelPointSet::
Point(int k) const
{
  // Return kth surfel
  return &points[k];
}



inline R3SurfelPoint *R3SurfelPointSet::
operator[](int k) const
{
  // Return kth surfel
  return Point(k);
}



inline int R3SurfelPointSet::
PointIndex(const R3SurfelPoint *point) const
{
  // Return index of point
  int index = point - points;
  assert((index >= 0) && (index < npoints));
  return index;
}



#if 0
inline R3Point R3SurfelPointSet::
Centroid(void) const
{
  // Return centroid of set
  return bbox.Centroid();
}
#endif



inline const R3Box& R3SurfelPointSet::
BBox(void) const
{
  // Return bounding box of set
  return bbox;
}





