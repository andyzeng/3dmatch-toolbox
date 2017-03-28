// Include file for the surfel scene processing utilities



////////////////////////////////////////////////////////////////////////
// Point set creation
////////////////////////////////////////////////////////////////////////

R3SurfelPointSet *CreatePointSet(R3SurfelPointSet *pointset, 
  const R3SurfelConstraint *constraint = NULL);
R3SurfelPointSet *CreatePointSet(R3SurfelScene *scene, 
  R3SurfelNode *source_node = NULL,
  const R3SurfelConstraint *constraint = NULL);
R3SurfelPointSet *CreatePointSet(R3SurfelScene *scene, R3Point& origin, 
  RNLength max_radius = 4, RNLength min_height = 0.25, RNLength max_height = 16, 
  RNLength max_spacing = 0.25, RNVolume min_volume = 0.1, RNVolume max_volume = 1000, 
  int min_points = 0);



////////////////////////////////////////////////////////////////////////
// Point graph creation
////////////////////////////////////////////////////////////////////////

R3SurfelPointGraph *CreatePointGraph(R3SurfelPointSet *pointset, 
  const R3SurfelConstraint *constraint = NULL,
  int max_neighbors = 16, RNLength max_distance = 0.5);
R3SurfelPointGraph *CreatePointGraph(R3SurfelScene *scene, 
  R3SurfelNode *source_node = NULL, const R3SurfelConstraint *constraint = NULL, 
  int max_neighbors = 16, RNLength max_distance = 0.5);



////////////////////////////////////////////////////////////////////////
// Block creation
////////////////////////////////////////////////////////////////////////

R3SurfelBlock *CreateBlock(R3SurfelScene *scene, 
  R3SurfelPointSet *pointset, 
  RNBoolean copy_surfels = FALSE);
R3SurfelBlock *CreateBlock(R3SurfelScene *scene, 
  R3SurfelNode *source_node = NULL, const R3SurfelConstraint *constraint = NULL, 
  RNBoolean copy_surfels = FALSE); 



////////////////////////////////////////////////////////////////////////
// Node creation
////////////////////////////////////////////////////////////////////////

R3SurfelNode *CreateNode(R3SurfelScene *scene, 
  R3SurfelPointSet *pointset, 
  R3SurfelNode *parent_node = NULL, const char *node_name = NULL,
  RNBoolean copy_surfels = FALSE);
R3SurfelNode *CreateNode(R3SurfelScene *scene,
  R3SurfelNode *source_node = NULL, const R3SurfelConstraint *constraint = NULL,
  R3SurfelNode *parent_node = NULL, const char *node_name = NULL,
  RNBoolean copy_surfels = FALSE);



////////////////////////////////////////////////////////////////////////
// Object creation
////////////////////////////////////////////////////////////////////////

R3SurfelObject *CreateObject(R3SurfelScene *scene, 
  R3SurfelPointSet *pointset, 
  R3SurfelObject *parent_object = NULL, const char *object_name = NULL, 
  R3SurfelNode *parent_node = NULL, const char *node_name = NULL,
  RNBoolean copy_surfels = FALSE);
R3SurfelObject *CreateObject(R3SurfelScene *scene, 
  R3SurfelNode *source_node = NULL, const R3SurfelConstraint *constraint = NULL,
  R3SurfelObject *parent_object = NULL, const char *object_name = NULL, 
  R3SurfelNode *parent_node = NULL, const char *node_name = NULL,
  RNBoolean copy_surfels = FALSE);

int SplitObject(R3SurfelObject *object, R3SurfelPointSet *pointset,
  R3SurfelObject **resultA = NULL, R3SurfelObject **resultB = NULL);
int SplitObject(R3SurfelObject *object, const R3SurfelConstraint *constraint = NULL,
  R3SurfelObject **resultA = NULL, R3SurfelObject **resultB = NULL);


////////////////////////////////////////////////////////////////////////
// Grid creation 
////////////////////////////////////////////////////////////////////////

R3Grid *CreateGrid(R3SurfelPointSet *pointset,  
  RNLength grid_spacing = 0.1, int max_resolution = 1024);
R3Grid *CreateGrid(R3SurfelScene *scene, 
  R3SurfelNode *source_node = NULL,
  const R3SurfelConstraint *constraint = NULL, 
  RNLength grid_spacing = 0.1, int max_resolution = 1024);
R3Grid *CreateGrid(R3SurfelScene *scene, const R3Box& bbox, 
  RNLength grid_spacing = 0.1, int max_resolution = 1024);
R3Grid *CreateGrid(R3SurfelScene *scene, 
  RNLength grid_spacing = 0.1, int max_resolution = 1024);



////////////////////////////////////////////////////////////////////////
// Normal computation
////////////////////////////////////////////////////////////////////////

R3Vector *CreateNormals(R3SurfelPointGraph *graph, 
  RNBoolean fast_and_approximate = FALSE);
R3Vector *CreateNormals(R3SurfelPointSet *pointset, 
  int max_neighbors = 16, RNLength max_distance = 0.5);



////////////////////////////////////////////////////////////////////////
// Connected point set computation (using graph)
////////////////////////////////////////////////////////////////////////

R3SurfelPointSet *CreateConnectedPointSet(R3SurfelPointGraph *graph, int seed_index);
R3SurfelPointSet *CreateConnectedPointSet(R3SurfelPointGraph *graph, R3SurfelPoint *seed_point);
R3SurfelPointSet *CreateConnectedPointSet(R3SurfelPointGraph *graph, const R3Point& seed_position);



////////////////////////////////////////////////////////////////////////
// Connected point set computation (using pointset/grid)
////////////////////////////////////////////////////////////////////////

R3SurfelPointSet *CreateConnectedPointSet(R3SurfelPointSet *pointset,
  RNScalar min_volume = 0.1, RNScalar max_volume = 100, 
  RNLength grid_spacing = 0.1, int max_grid_resolution = 1024);
R3SurfelPointSet *CreateConnectedPointSet(R3SurfelPointSet *pointset, const R3Point &seed_position,
  RNScalar min_volume = 0., RNScalar max_volume = 100, 
  RNLength grid_spacing = 0.1, int max_grid_resolution = 1024);
R3SurfelPointSet *CreateConnectedPointSet(R3SurfelPointSet *pointset, 
  const R3Point& seed_origin, RNScalar seed_radius, 
  RNScalar seed_min_height, RNScalar seed_max_height, 
  RNScalar min_volume, RNScalar max_volume, 
  RNLength grid_spacing, int max_grid_resolution);



////////////////////////////////////////////////////////////////////////
// Plane estimation
////////////////////////////////////////////////////////////////////////

R3Plane FitPlane(R3SurfelPointSet *pointset);

R3Plane EstimateSupportPlane(R3SurfelPointSet *pointset, 
  RNLength accuracy = 0.1, RNScalar *npoints = NULL);
R3Plane EstimateSupportPlane(R3SurfelScene *scene, 
  R3SurfelNode *source_node = NULL, const R3SurfelConstraint *constraint = NULL,
  RNLength accuracy = 0.1, RNScalar *npoints = NULL);
R3Plane EstimateSupportPlane(R3SurfelScene *scene, 
  const R3Point& center_point, RNLength radius = 2,
  RNLength accuracy = 0.1, RNScalar *npoints = NULL);
RNCoord EstimateSupportZ(R3SurfelScene *scene, 
  const R3Point& center_point, RNLength radius = 2,
  RNLength accuracy = 0.1, RNScalar *npoints = NULL);

R3Plane FitSupportPlane(R3SurfelPointSet *pointset, 
  RNLength accuracy = 0.1, RNScalar *npoints = NULL);
R3Plane FitSupportPlane(R3SurfelScene *scene, 
  R3SurfelNode *source_node = NULL, const R3SurfelConstraint *constraint = NULL,
  RNLength accuracy = 0.1, RNScalar *npoints = NULL);
R3Plane FitSupportPlane(R3SurfelScene *scene, 
  const R3Point& center_point, RNLength radius = 2,
  RNLength accuracy = 0.1, RNScalar *npoints = NULL);



////////////////////////////////////////////////////////////////////////
// Planar grid computation
////////////////////////////////////////////////////////////////////////

RNArray<R3PlanarGrid *> *
CreatePlanarGrids(R3SurfelPointGraph *graph, 
  RNLength max_offplane_distance = 0.5, RNAngle max_normal_angle = 0.5,
  RNArea min_area = 1, RNScalar min_density = 10, int min_points = 100,
  RNLength grid_spacing = 0.25, RNScalar accuracy_factor = 1);

RNArray<R3PlanarGrid *> *
CreatePlanarGrids(R3SurfelScene *scene, 
  R3SurfelNode *source_node = NULL, const R3SurfelConstraint *constraint = NULL, 
  int max_neighbors = 16, RNLength max_neighbor_distance = 0.5, 
  RNLength max_offplane_distance = 0.5, RNAngle max_normal_angle = 0.5,
  RNArea min_area = 1, RNScalar min_density = 10, int min_points = 100,
  RNLength grid_spacing = 0.25, RNScalar accuracy_factor = 1, 
  RNLength chunk_size = 0);

RNArray<R3SurfelObject *> *
CreatePlanarObjects(R3SurfelScene *scene, 
  R3SurfelNode *source_node = NULL, const R3SurfelConstraint *constraint = NULL, 
  R3SurfelObject *parent_object = NULL, R3SurfelNode *parent_node = NULL, RNBoolean copy_surfels = TRUE,
  int max_neighbors = 16, RNLength max_neighbor_distance = 0.5, 
  RNLength max_offplane_distance = 0.5, RNAngle max_normal_angle = 0.5,
  RNArea min_area = 1, RNScalar min_density = 10, int min_points = 100,
  RNLength grid_spacing = 0.25, RNScalar accuracy_factor = 1,
  RNLength chunk_size = 0);


////////////////////////////////////////////////////////////////////////
// Node set computation
////////////////////////////////////////////////////////////////////////

R3SurfelNodeSet *CreateNodeSet(R3SurfelScene *scene,
  R3SurfelNode *source_node = NULL, const R3SurfelConstraint *constraint = NULL,
  const R3Point& xycenter = R3zero_point, RNLength xyradius = 0,
  RNScalar center_resolution = 0, RNScalar perimeter_resolution = 0,
  RNScalar focus_exponent = 0);



////////////////////////////////////////////////////////////////////////
// Object set creation
////////////////////////////////////////////////////////////////////////

R3SurfelObjectSet *CreateObjectSet(R3SurfelScene *scene, 
  const R3SurfelConstraint *constraint = NULL);



////////////////////////////////////////////////////////////////////////
// High-level object creation
////////////////////////////////////////////////////////////////////////

RNArray<R3SurfelObject *> *
CreateClusterObjects(R3SurfelScene *scene, R3SurfelPointGraph *graph,
  R3SurfelObject *parent_object = NULL, R3SurfelNode *parent_node = NULL, 
  RNLength max_offplane_distance = 0.1, RNAngle max_normal_angle = 0.1,
  int min_points_per_object = 25);

RNArray<R3SurfelObject *> *
CreateClusterObjects(R3SurfelScene *scene, 
  R3SurfelNode *source_node = NULL, const R3SurfelConstraint *constraint = NULL, 
  R3SurfelObject *parent_object = NULL, R3SurfelNode *parent_node = NULL, 
  int max_neighbors = 16, RNLength max_neighbor_distance = 0.5, 
  RNLength max_offplane_distance = 0.1, RNAngle max_normal_angle = 0.1,
  int min_points_per_object = 25, 
  RNLength chunk_size = 0);


////////////////////////////////////////////////////////////////////////
// Basic geometric funcitons
////////////////////////////////////////////////////////////////////////

RNLength XYDistance(const R3Point& point1, const R3Point& point2);
RNLength XYDistanceSquared(const R3Point& point1, const R3Point& point2);
RNLength XYDistance(const R3Point& point, const R3Box& box);
RNLength XYDistanceSquared(const R3Point& point, const R3Box& box);


