/* Include file for the R3 surfel constraint class */



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelConstraint {
public:
  // Constructor functions
  R3SurfelConstraint(void);
  virtual ~R3SurfelConstraint(void);

  // Surfel check functions
  virtual int Check(const R3SurfelObject *object) const;
  virtual int Check(const R3SurfelNode *node) const;
  virtual int Check(const R3SurfelBlock *block) const;
  virtual int Check(const R3SurfelBlock *block, const R3Surfel *surfel) const;
  virtual int Check(const R3Box& box) const;
  virtual int Check(const R3Point& point) const;
};



////////////////////////////////////////////////////////////////////////
// Check() RETURN RESULTS
////////////////////////////////////////////////////////////////////////

enum {
  R3_SURFEL_CONSTRAINT_FAIL,
  R3_SURFEL_CONSTRAINT_MAYBE,
  R3_SURFEL_CONSTRAINT_PASS
};



////////////////////////////////////////////////////////////////////////
// Comparison types
////////////////////////////////////////////////////////////////////////

enum {
  R3_SURFEL_CONSTRAINT_NOT_EQUAL,
  R3_SURFEL_CONSTRAINT_EQUAL,
  R3_SURFEL_CONSTRAINT_GREATER,
  R3_SURFEL_CONSTRAINT_GREATER_OR_EQUAL,
  R3_SURFEL_CONSTRAINT_LESS,
  R3_SURFEL_CONSTRAINT_LESS_OR_EQUAL
};



////////////////////////////////////////////////////////////////////////
// Value types
////////////////////////////////////////////////////////////////////////

enum {
  R3_SURFEL_CONSTRAINT_OPERAND,
  R3_SURFEL_CONSTRAINT_VALUE,
  R3_SURFEL_CONSTRAINT_X,
  R3_SURFEL_CONSTRAINT_Y,
  R3_SURFEL_CONSTRAINT_Z
};



////////////////////////////////////////////////////////////////////////
// EXAMPLE DERIVED CLASSES
////////////////////////////////////////////////////////////////////////

class R3SurfelCoordinateConstraint : public R3SurfelConstraint {
public:
  // Constructor functions
  R3SurfelCoordinateConstraint(RNDimension dimension, const RNInterval& interval);

  // Surfel check functions
  virtual int Check(const R3Point& point) const;
  virtual int Check(const R3Box& box) const;

private:
  RNDimension dimension;
  RNInterval interval;
};



class R3SurfelNormalConstraint : public R3SurfelConstraint {
public:
  // Constructor functions
  R3SurfelNormalConstraint(const R3Vector& direction, RNAngle max_angle);

  // Surfel check functions
  virtual int Check(const R3SurfelBlock *block, const R3Surfel *surfel) const;

private:
  float direction[3];
  float min_dot;
};



class R3SurfelBoxConstraint : public R3SurfelConstraint {
public:
  // Constructor functions
  R3SurfelBoxConstraint(const R3Box& box);

  // Surfel check functions
  virtual int Check(const R3Point& point) const;
  virtual int Check(const R3Box& box) const;

private:
  R3Box box;
};



class R3SurfelCylinderConstraint : public R3SurfelConstraint {
public:
  // Constructor functions
  R3SurfelCylinderConstraint(const R3Point& center, RNLength radius, RNCoord zmin = -FLT_MAX, RNCoord zmax = FLT_MAX);

  // Surfel check functions
  virtual int Check(const R3Point& point) const;
  virtual int Check(const R3Box& box) const;

private:
  R3Point center;
  RNLength radius_squared;
  RNCoord zmin, zmax;
};



class R3SurfelSphereConstraint : public R3SurfelConstraint {
public:
  // Constructor functions
  R3SurfelSphereConstraint(const R3Sphere& sphere);

  // Surfel check functions
  virtual int Check(const R3Point& point) const;
  virtual int Check(const R3Box& box) const;

private:
  R3Sphere sphere;
};



class R3SurfelHalfspaceConstraint : public R3SurfelConstraint {
public:
  // Constructor functions
  R3SurfelHalfspaceConstraint(const R3Halfspace& halfspace);

  // Surfel check functions
  virtual int Check(const R3Point& point) const;
  virtual int Check(const R3Box& box) const;

private:
  R3Halfspace halfspace;
};



class R3SurfelLineConstraint : public R3SurfelConstraint {
public:
  // Constructor functions
  R3SurfelLineConstraint(const R3Line& line, 
    RNLength tolerance = 0);

  // Surfel check functions
  virtual int Check(const R3Point& point) const;
  virtual int Check(const R3Box& box) const;

private:
  R3Line line;
  RNLength tolerance;
};



class R3SurfelPlaneConstraint : public R3SurfelConstraint {
public:
  // Constructor functions
  R3SurfelPlaneConstraint(const R3Plane& plane, 
    RNBoolean below = TRUE, RNBoolean on = TRUE, RNBoolean above = TRUE, 
    RNLength tolerance = 0);

  // Surfel check functions
  virtual int Check(const R3Point& point) const;
  virtual int Check(const R3Box& box) const;

private:
  R3Plane plane;
  RNBoolean below;
  RNBoolean on;
  RNBoolean above;
  RNLength tolerance;
};



class R3SurfelGridConstraint : public R3SurfelConstraint {
public:
  // Constructor functions
  R3SurfelGridConstraint(const R3Grid *grid, 
    int comparison_type = R3_SURFEL_CONSTRAINT_LESS,
    int surfel_value_type = R3_SURFEL_CONSTRAINT_OPERAND, 
    int grid_value_type = R3_SURFEL_CONSTRAINT_VALUE, 
    RNScalar surfel_operand = RN_EPSILON, 
    RNScalar grid_operand = 0,
    RNScalar epsilon = 0);

  // Surfel check functions
  virtual int Check(const R3Point& point) const;
  virtual int Check(const R3Box& box) const;

private:
  const R3Grid *grid;
  int comparison_type;
  int surfel_value_type;
  int grid_value_type;
  RNScalar surfel_operand;
  RNScalar grid_operand;
  RNScalar epsilon;
};



class R3SurfelPlanarGridConstraint : public R3SurfelConstraint {
public:
  // Constructor functions
  R3SurfelPlanarGridConstraint(const R3PlanarGrid *grid, 
    RNLength max_offplane_distance = RN_EPSILON, 
    int comparison_type = R3_SURFEL_CONSTRAINT_LESS,
    int surfel_value_type = R3_SURFEL_CONSTRAINT_OPERAND, 
    int grid_value_type = R3_SURFEL_CONSTRAINT_VALUE, 
    RNScalar surfel_operand = RN_EPSILON, 
    RNScalar grid_operand = 0,
    RNScalar epsilon = 0);

  // Surfel check functions
  virtual int Check(const R3Point& point) const;
  virtual int Check(const R3Box& box) const;

private:
  const R3PlanarGrid *grid;
  RNLength max_offplane_distance;
  int comparison_type;
  int surfel_value_type;
  int grid_value_type;
  RNScalar surfel_operand;
  RNScalar grid_operand;
  RNScalar epsilon;
};



class R3SurfelOverheadGridConstraint : public R3SurfelConstraint {
public:
  // Constructor functions
  R3SurfelOverheadGridConstraint(const R2Grid *grid, 
    int comparison_type = R3_SURFEL_CONSTRAINT_LESS,
    int surfel_value_type = R3_SURFEL_CONSTRAINT_OPERAND, 
    int grid_value_type = R3_SURFEL_CONSTRAINT_VALUE, 
    RNScalar surfel_operand = RN_EPSILON, 
    RNScalar grid_operand = 0,
    RNScalar epsilon = 0);

  // Surfel check functions
  virtual int Check(const R3Box& box) const;
  virtual int Check(const R3Point& point) const;

private:
  const R2Grid *grid;
  int comparison_type;
  int surfel_value_type;
  int grid_value_type;
  RNScalar surfel_operand;
  RNScalar grid_operand;
  RNScalar epsilon;
};



class R3SurfelMeshConstraint : public R3SurfelConstraint {
public:
  // Constructor functions
  R3SurfelMeshConstraint(R3Mesh *mesh, 
    const R3Affine& surfels_to_mesh,
    RNLength max_distance = 0.25);

  // Surfel check functions
  virtual int Check(const R3Point& point) const;
  virtual int Check(const R3Box& box) const;

private:
  R3Mesh *mesh;
  R3Affine surfels_to_mesh;
  RNLength max_distance;
  R3MeshSearchTree *tree;
};



class R3SurfelObjectConstraint : public R3SurfelConstraint {
public:
  // Constructor functions
  R3SurfelObjectConstraint(R3SurfelObject *target_object = NULL, 
    RNBoolean converse = FALSE);

  // Surfel check functions
  virtual int Check(const R3SurfelObject *object) const;
  virtual int Check(const R3SurfelNode *node) const;
  virtual int Check(const R3SurfelBlock *block) const;
  virtual int Check(const R3SurfelBlock *block, const R3Surfel *surfel) const;

private:
  R3SurfelObject *target_object; // Null means any object
  RNBoolean converse;
};



class R3SurfelLabelConstraint : public R3SurfelConstraint {
public:
  // Constructor functions
  R3SurfelLabelConstraint(R3SurfelLabel *target_label = NULL, RNBoolean converse = FALSE);

  // Surfel check functions
  virtual int Check(const R3SurfelObject *object) const;
  virtual int Check(const R3SurfelNode *node) const;
  virtual int Check(const R3SurfelBlock *block) const;
  virtual int Check(const R3SurfelBlock *block, const R3Surfel *surfel) const;

private:
  R3SurfelLabel *target_label; // Null means any label
  RNBoolean converse;
};



class R3SurfelSourceConstraint : public R3SurfelConstraint {
public:
  // Constructor functions
  R3SurfelSourceConstraint(RNBoolean include_aerial = TRUE, RNBoolean include_terrestrial = FALSE);

  // Surfel check functions
  virtual int Check(const R3SurfelNode *node) const;
  virtual int Check(const R3SurfelBlock *block) const;
  virtual int Check(const R3SurfelBlock *block, const R3Surfel *surfel) const;

private:
  RNBoolean include_aerial;
  RNBoolean include_terrestrial;
};



class R3SurfelBoundaryConstraint : public R3SurfelConstraint {
public:
  // Constructor functions
  R3SurfelBoundaryConstraint(RNBoolean include_interior = TRUE, RNBoolean include_border = FALSE,
    RNBoolean include_silhouette = FALSE, RNBoolean include_shadow = FALSE);

  // Surfel check functions
  virtual int Check(const R3SurfelBlock *block, const R3Surfel *surfel) const;

private:
  RNBoolean include_interior;
  RNBoolean include_border;
  RNBoolean include_silhouette;
  RNBoolean include_shadow;
};



class R3SurfelMarkConstraint : public R3SurfelConstraint {
public:
  // Constructor functions
  R3SurfelMarkConstraint(RNBoolean include_marked = TRUE, RNBoolean include_unmarked = FALSE);

  // Surfel check functions
  virtual int Check(const R3SurfelBlock *block) const;
  virtual int Check(const R3SurfelBlock *block, const R3Surfel *surfel) const;

private:
  RNBoolean include_marked;
  RNBoolean include_unmarked;
};



class R3SurfelMultiConstraint : public R3SurfelConstraint {
public:
  // Constructor functions
  R3SurfelMultiConstraint(void);

  // Constraint insertion/removal functions
  void InsertConstraint(const R3SurfelConstraint *constraint);
  void RemoveConstraint(const R3SurfelConstraint *constraint);

  // Surfel check functions
  virtual int Check(const R3SurfelObject *object) const;
  virtual int Check(const R3SurfelNode *node) const;
  virtual int Check(const R3SurfelBlock *block) const;
  virtual int Check(const R3SurfelBlock *block, const R3Surfel *surfel) const;
  virtual int Check(const R3Box& box) const;
  virtual int Check(const R3Point& point) const;

private:
  RNArray<const R3SurfelConstraint *> constraints;
};






