/* Source file for the surfel alignment utilities */



////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// Parameters
////////////////////////////////////////////////////////////////////////

static const int default_model_min_npoints = 16;
static const int default_model_max_npoints = 1024;
static const RNLength default_model_radius = 4.0;
static const RNLength default_model_max_height = 5.0;
static const RNCoord default_model_zmin = -FLT_MAX;
static const RNCoord default_model_zmax = FLT_MAX;
static const RNScalar default_model_min_volume = 0.1;
static const RNScalar default_model_max_volume = 10000;
static const RNLength default_model_max_connectivity_gap = 0.25;

static const RNLength default_correspondence_max_distance = 0.5;
static const RNLength default_coverage_sigma = 0.25;

static const int default_initial_guess_iterations = 4;
static const RNLength default_initial_guess_max_translation = 2;
static const RNAngle default_initial_guess_max_rotation = 1.0;

static const int default_icp_iterations = 8;
static const RNLength default_icp_start_distance = 2.0;
static const RNLength default_icp_end_distance = 0.5;




////////////////////////////////////////////////////////////////////////
// Type definitions
////////////////////////////////////////////////////////////////////////

struct Point {
  R3Point position;
  R3Vector normal;
};

struct Model {
  Model(void);
  ~Model(void);
  char name[1024];
  R3Mesh *mesh;
  R3SurfelObject *object;
  RNArray<Point *> *points;
  R3Kdtree<Point *> *kdtree;
  R3Affine pca_transformation;
  R3Point centroid;
  R3Point origin;
  RNLength radius;
};



////////////////////////////////////////////////////////////////////////
// Point set creation functions
////////////////////////////////////////////////////////////////////////

RNArray<Point *> *
CreateSamplePoints(R3Mesh *mesh, 
  int max_npoints = default_model_max_npoints, 
  RNLength max_height = default_model_max_height)
{
  // Allocate points
  RNArray<Point *> *points = new RNArray<Point *>();
  if (!points) {
    fprintf(stderr, "Unable to allocate array of points\n");
    return NULL;
  }

  // Get convenient variables
  R3Box bbox = mesh->BBox();

  // Count cumulative area of faces
  RNArea total_area = 0;
  RNArea *cumulative_area = new RNArea [ mesh->NFaces() ];
  for (int i = 0; i < mesh->NFaces(); i++) {
    R3MeshFace *face = mesh->Face(i);
    RNArea face_area = mesh->FaceArea(face);
    total_area += face_area;
    cumulative_area[i] = total_area;;
  }
    
  // Generate points
  RNSeedRandomScalar();
  while (points->NEntries() < max_npoints) {
    // Generate a random number
    RNScalar r = RNRandomScalar() * total_area;

    // Find face ID
    int min_face_id = 0;
    int max_face_id = mesh->NFaces()-1;
    int face_id = (min_face_id + max_face_id) / 2;
    while (min_face_id < max_face_id) {
      if (min_face_id + 1 >= max_face_id) { 
        if (r < cumulative_area[min_face_id]) { 
          face_id = min_face_id; 
          break; 
        }
        else { 
          face_id = max_face_id; 
          break; 
        }
      }
      else {
        if (cumulative_area[face_id] < r) {
          min_face_id = face_id + 1;
          face_id = (min_face_id + max_face_id) / 2;
        }
        else if (cumulative_area[face_id] > r) {
          max_face_id = face_id;
          face_id = (min_face_id + max_face_id) / 2;
        }
        else {
          break;
        }
      }
    }

    // Find point on face
    R3MeshFace *face = mesh->Face(face_id);
    R3MeshVertex *v0 = mesh->VertexOnFace(face, 0);
    R3MeshVertex *v1 = mesh->VertexOnFace(face, 1);
    R3MeshVertex *v2 = mesh->VertexOnFace(face, 2);
    const R3Point& p0 = mesh->VertexPosition(v0);
    const R3Point& p1 = mesh->VertexPosition(v1);
    const R3Point& p2 = mesh->VertexPosition(v2);
    const R3Vector& n0 = mesh->VertexNormal(v0);
    const R3Vector& n1 = mesh->VertexNormal(v1);
    const R3Vector& n2 = mesh->VertexNormal(v2);
    RNScalar r1 = sqrt(RNRandomScalar());
    RNScalar r2 = RNRandomScalar();
    RNScalar t0 = (1.0 - r1);
    RNScalar t1 = r1 * (1.0 - r2);
    RNScalar t2 = r1 * r2;
    R3Point position = t0*p0 + t1*p1 + t2*p2;
    if (position.Z() - bbox.ZMin() > max_height) continue;
    R3Vector normal = t0*n0 + t1*n1 + t2*n2;

    // Create point
    Point *point = new Point();
    point->position = position;
    point->normal = normal;
    points->Insert(point);
  }

  // Return points
  return points;
}



RNArray<Point *> *
CreateSamplePoints(R3SurfelPointSet *pointset, 
  int npoints = default_model_max_npoints, 
  RNCoord zmin = default_model_zmin, 
  RNCoord zmax = default_model_zmax)
{
  // Check pointset
  if (!pointset) return NULL;
  if (pointset->NPoints() == 0) return NULL;

  // Compute normals
  R3Vector *normals = CreateNormals(pointset);
  if (!normals) return NULL;

  // Allocate points
  RNArray<Point *> *points = new RNArray<Point *>();
  if (!points) {
    fprintf(stderr, "Unable to allocate array of points\n");
    delete normals;
    return NULL;
  }

  // Insert points
  if (npoints > pointset->NPoints()) {
    // Insert all points
    for (int index = 0; index < pointset->NPoints(); index++) {
      const R3SurfelPoint *surfel_point = pointset->Point(index);
      R3Point position = surfel_point->Position();
      if (position.Z() < zmin) continue;
      if (position.Z() > zmax) continue;
      Point *point = new Point();
      point->position = position;
      point->normal = normals[index];
      points->Insert(point);
    }
  }
  else {
    // Sample points from set
    while (points->NEntries() < npoints) {
      // Generate a random number
      int index = (int) (RNRandomScalar() * pointset->NPoints());
      const R3SurfelPoint *surfel_point = pointset->Point(index);
      R3Point position = surfel_point->Position();
      if (position.Z() < zmin) continue;
      if (position.Z() > zmax) continue;
      Point *point = new Point();
      point->position = position;
      point->normal = normals[index];
      points->Insert(point);
    }
  }

  // Delete normals
  delete normals;

  // Return points
  return points;
}



RNArray<Point *> *
CreateSamplePoints(R3SurfelObject *object, 
  int npoints = default_model_max_npoints, 
  RNCoord zmin = default_model_zmin,
  RNCoord zmax = default_model_zmax)
{
  // Read object blocks
  object->ReadBlocks();

  // Create surfel point set
  R3SurfelPointSet *pointset = new R3SurfelPointSet();
  if (!pointset) { object->ReleaseBlocks(); return NULL; }
  for (int i = 0; i < object->NNodes(); i++) {
    R3SurfelNode *node = object->Node(i);
    for (int j = 0; j < node->NBlocks(); j++) {
      R3SurfelBlock *block = node->Block(j);
      pointset->InsertPoints(block);
    }
  }

  // Create points
  RNArray<Point *> *points = CreateSamplePoints(pointset, npoints, zmin, zmax);
  if (!points) {
    fprintf(stderr, "Unable to allocate array of points\n");
    delete pointset;
    object->ReleaseBlocks();
    return NULL;
  }

  // Delete surfel point set
  delete pointset;

  // Release object blocks
  object->ReleaseBlocks();

  // Return points
  return points;
}



RNArray<Point *> *
CreateSamplePoints(R3SurfelScene *scene, const R3Point& center_point, 
  RNScalar radius = default_model_radius, 
  RNCoord zmin = default_model_zmin, 
  RNCoord zmax = default_model_zmax, 
  int min_npoints = default_model_min_npoints,
  int max_npoints = default_model_max_npoints,
  RNScalar min_volume = default_model_min_volume, 
  RNScalar max_volume = default_model_max_volume, 
  RNScalar max_gap = default_model_max_connectivity_gap)
{
  // Create pointset
  R3SurfelCylinderConstraint cylinder_constraint(center_point, radius, zmin, zmax);
  R3SurfelPointSet *pointset1 = CreatePointSet(scene, NULL, &cylinder_constraint);
  if (!pointset1) return NULL; 
  if (pointset1->NPoints() < min_npoints) { delete pointset1; return NULL; }
  R3SurfelPointSet *pointset = CreateConnectedPointSet(pointset1, center_point, min_volume, max_volume, max_gap);
  delete pointset1;
  if (!pointset)  return NULL; 
  if (pointset->NPoints() < min_npoints) { delete pointset; return NULL; }


  // Create points
  RNArray<Point *> *points = CreateSamplePoints(pointset, max_npoints, zmin, zmax);
  if (!points) {
    fprintf(stderr, "Unable to allocate array of points\n");
    delete pointset;
    return NULL;
  }

  // Delete surfel point set
  delete pointset;

  // Return points
  return points;
}



////////////////////////////////////////////////////////////////////////
// Point set alignment functions
////////////////////////////////////////////////////////////////////////

static int 
ArePointsCompatible(Point *point1, Point *point2, void *)
{
  // Parameters
  const RNAngle max_normal_angle = RN_PI / 16.0;
  RNScalar max_normal_dot = cos(max_normal_angle);

  // Check point normals
  const R3Vector& normal1 = point1->normal;
  const R3Vector& normal2 = point2->normal;
  RNScalar dot = fabs(normal1.Dot(normal2));
  if (dot < max_normal_dot) return 0;

  // Passed all tests
  return 1;
}



static int
CreateCorrespondences(
  const RNArray<Point *>& points1, const RNArray<Point *>& points2, 
  R3Kdtree<Point *> *tree1, R3Kdtree<Point *> *tree2, 
  const R3Affine& affine12, const R3Affine& affine21,
  R3Point *correspondences1, R3Point *correspondences2, int max_correspondences, 
  RNLength max_distance = default_correspondence_max_distance)
{
  // Initialize number of correspondences
  assert(max_correspondences >= points1.NEntries() + points2.NEntries());
  int ncorrespondences = 0;

#if 0
  // Compute correspondences for points1 -> points2
  for (int i = 0; i < points1.NEntries(); i++) {
    Point point1 = *(points1[i]);
    point1.position.Transform(affine12);
    point1.normal.Transform(affine12);
    const Point *closest = tree2->FindClosest(&point1, 0.0, max_distance, ArePointsCompatible, NULL);
    if (closest) {
      if (ncorrespondences >= max_correspondences) break;
      correspondences1[ncorrespondences] = points1[i]->position;
      correspondences2[ncorrespondences] = closest->position;
      ncorrespondences++;
    }
  }
#endif

#if 1
  // Compute correspondences for points2 -> points1
  for (int i = 0; i < points2.NEntries(); i++) {
    Point point2 = *(points2[i]);
    point2.position.Transform(affine21);
    point2.normal.Transform(affine21);
    const Point *closest = tree1->FindClosest(&point2, 0.0, max_distance, ArePointsCompatible, NULL);
    if (closest) {
      if (ncorrespondences >= max_correspondences) break;
      correspondences1[ncorrespondences] = closest->position;
      correspondences2[ncorrespondences] = points2[i]->position;
      ncorrespondences++;
    }
  }
#endif

  // Return number of correspondences
  return ncorrespondences;
}



static R3Affine
PCAAlignmentTransformation(const RNArray<R3Point *>& points, R3Point *origin = NULL)
{
  // Just checking
  if (points.NEntries() == 0) return R3identity_affine;

  // Get center
  R3Point center = (origin) ? *origin : R3Centroid(points);

  // Compute 2D covariance matrix
  RNScalar m[4] = { 0 };
  for (int i = 0; i < points.NEntries(); i++) {
    const R3Point& p = *(points[i]);
    RNScalar x = p[0] - center[0];
    RNScalar y = p[1] - center[1];
    m[0] += x*x;
    m[1] += x*y;
    m[2] += x*y;
    m[3] += y*y;
  }

  // Normalize covariance matrix
  for (int i = 0; i < 4; i++) m[i] /= points.NEntries();

  // Compute eigenvalues and eigenvectors
  RNScalar U[4];
  RNScalar W[2];
  RNScalar Vt[4];
  RNSvdDecompose(2, 2, m, U, W, Vt);  // m == U . DiagonalMatrix(W) . Vt

  // Extract principle axis direction from first eigenvector
  R3Vector axis(Vt[0], Vt[1], 0);

  // Flip principle axis so that "heavier" on positive side
  int positive_count = 0;
  int negative_count = 0;
  for (int i = 0; i < points.NEntries(); i++) {
    R3Point& p = *(points[i]);
    RNScalar x = p[0] - center[0];
    RNScalar y = p[1] - center[1];
    R3Vector vertex_vector(x, y, 0);
    RNScalar dot = axis.Dot(vertex_vector);
    if (dot > 0.0) positive_count++;
    else negative_count++;
  }
  if (positive_count < negative_count) {
    axis.Flip();
  }

  // Create transformation
  R3Affine affine = R3identity_affine;
  affine.Rotate(axis, R3posx_vector);
  affine.Translate(-center.Vector());

  // Return transformation
  return affine;
}


static R3Affine
CorrespondenceAlignmentTransformation(int ncorrespondences,
  R3Point *correspondences1, R3Point *correspondences2,
  R3Point *origin1 = NULL, R3Point *origin2 = NULL)
{
  // Compute centers
  R3Point center1 = (origin1) ? *origin1 : R3Centroid(ncorrespondences, correspondences1);
  R3Point center2 = (origin2) ? *origin2 : R3Centroid(ncorrespondences, correspondences2);

  // Compute covariance matrix
  RNScalar m[4] = { 0 };
  for (int i = 0; i < ncorrespondences; i++){
    R3Vector p1 = correspondences1[i] - center1;
    R3Vector p2 = correspondences2[i] - center2;
    m[0] += p1[0]*p2[0];
    m[1] += p1[0]*p2[1];
    m[2] += p1[1]*p2[0];
    m[3] += p1[1]*p2[1];
  }

  // Normalize covariance matrix
  for (int j = 0; j < 4; j++) m[j] /= ncorrespondences;

  // Calculate SVD of covariance matrix
  RNScalar Um[4];
  RNScalar Wm[2];
  RNScalar Vmt[4];
  RNSvdDecompose(2, 2, m, Um, Wm, Vmt);

  // https://sakai.rutgers.edu/access/content/group/7bee3f05-9013-4fc2-8743-3c5078742791/material/svd_ls_rotation.pdf
  R3Matrix Ut(Um[0], Um[2], 0, Um[1], Um[3], 0, 0, 0, 1); 
  R3Matrix V(Vmt[0], Vmt[2], 0, Vmt[1], Vmt[3], 0, 0, 0, 1); 
  R3Matrix VUt = V * Ut;
  R3Matrix D = R3identity_matrix;
  D[1][1] = R3MatrixDet2(VUt[0][0], VUt[0][1], VUt[1][0], VUt[1][1]);
  R3Matrix R = V * D * Ut;
  R4Matrix rotation = R4identity_matrix;
  rotation[0][0] = R[0][0];
  rotation[0][1] = R[0][1];
  rotation[1][0] = R[1][0];
  rotation[1][1] = R[1][1];

  // Compute matrix21
  // XXX Why rotation.Inverse()? XXX
  R4Matrix matrix21 = R4identity_matrix;
  matrix21.Translate(center1.Vector());
  matrix21.Transform(rotation.Inverse());
  matrix21.Translate(-(center2.Vector()));

  // Return resulting matrix that takes points2 to points1
  return matrix21;
}



static R3Affine
ICPAlignmentTransformation(
  const RNArray<Point *>& points1, const RNArray<Point *>& points2, 
  R3Kdtree<Point *> *kdtree1 = NULL, R3Kdtree<Point *> *kdtree2 = NULL, 
  R3Point *correspondences1 = NULL, R3Point *correspondences2 = NULL,
  R3Point *origin1 = NULL, R3Point *origin2 = NULL,
  const R3Affine& initial_affine21 = R3identity_affine, 
  int max_iterations = default_icp_iterations, 
  RNLength start_distance = default_icp_start_distance, 
  RNLength end_distance = default_icp_end_distance,
  int *result_ncorrespondences = NULL, RNBoolean *result_converged = NULL)
{
  // Initialize transformation
  R3Affine affine21 = initial_affine21;

  // Allocate kdtrees
  R3Kdtree<Point *> *tree1 = (kdtree1) ? kdtree1 : new R3Kdtree<Point *>(points1);
  R3Kdtree<Point *> *tree2 = (kdtree2) ? kdtree2 : new R3Kdtree<Point *>(points2);

  // Allocate correspondence buffers
  int max_correspondences = points1.NEntries() + points2.NEntries();
  R3Point *correspondences_buffer1 = new R3Point [2 * max_correspondences];
  R3Point *correspondences_buffer2 = new R3Point [2 * max_correspondences];
  assert(correspondences_buffer1 && correspondences_buffer2);
  int ncorrespondences = 0;

  // Iterate until max_iterations or converged
  int iteration = 0;
  RNBoolean done = FALSE;
  for (iteration = 0; iteration < max_iterations; iteration++) {
    // Get arrays of correspondences
    R3Point *corr1 = &correspondences_buffer1[iteration%2 * max_correspondences];
    R3Point *corr2 = &correspondences_buffer2[iteration%2 * max_correspondences];

    // Compute max distance
    // Linearly ramp down from start_distance to end_distance
    // Spend second half of the iterations at end distance
    RNLength max_distance = start_distance;
    if (max_iterations > 0) max_distance += (end_distance - start_distance) * iteration / (max_iterations-1);
    if (RNIsLessOrEqual(max_distance, end_distance)) max_distance = end_distance;

    // Update correspondences for aligning transformation
    ncorrespondences = CreateCorrespondences(
      points1, points2, 
      tree1, tree2,
      affine21.Inverse(), affine21, 
      corr1, corr2, 
      max_correspondences, max_distance);

    // Compute centers to align
    R3Point center1 = R3Centroid(ncorrespondences, corr1);
    R3Point center2 = R3Centroid(ncorrespondences, corr2);
    center1[2] = (origin1) ? origin1->Z() : tree1->BBox().ZMin();
    center2[2] = (origin2) ? origin2->Z() : tree2->BBox().ZMin();

    // Update aligning transformation for correspondences
    affine21 = CorrespondenceAlignmentTransformation(ncorrespondences, corr1, corr2, &center1, &center2);

    // Check for convergence
    done = FALSE;
    if ((RNIsEqual(max_distance, end_distance)) && (iteration > 0)) {
      done = TRUE;
      R3Point *prev_corr1 = &correspondences_buffer1[(1-(iteration%2)) * ncorrespondences];
      R3Point *prev_corr2 = &correspondences_buffer2[(1-(iteration%2)) * ncorrespondences];
      for (int i = 0; i < ncorrespondences; i++) {
        if (!R3Contains(corr1[i], prev_corr1[i])) { done = FALSE; break; }
        if (!R3Contains(corr2[i], prev_corr2[i])) { done = FALSE; break; }
      }
    }
    if (done) break;
  }

  // Copy correspondences1 into return value
  if (correspondences1) {
    R3Point *corr1 = &correspondences_buffer1[iteration%2 * ncorrespondences];
    for (int i = 0; i < ncorrespondences; i++) correspondences1[i] = corr1[i];
  }

  // Copy correspondences2 into return value
  if (correspondences2) {
    R3Point *corr2 = &correspondences_buffer2[iteration%2 * ncorrespondences];
    for (int i = 0; i < ncorrespondences; i++) correspondences2[i] = corr2[i];
  }

  // Copy converged into return value
  if (result_ncorrespondences) *result_ncorrespondences = ncorrespondences;
  if (result_converged) *result_converged = done;

  // Delete correspondence buffers
  delete [] correspondences_buffer1;
  delete [] correspondences_buffer2;

  // Delete kdtrees
  if (tree1 && !kdtree1) delete tree1;
  if (tree2 && !kdtree2) delete tree2;

  // Return transformation
  return affine21;
}



static void
ScoreAlignmentTransformation(
  const RNArray<Point *>& points1, const RNArray<Point *>& points2, 
  R3Kdtree<Point *> *tree1, R3Kdtree<Point *> *tree2, 
  R3Point *correspondences1, R3Point *correspondences2, int ncorrespondences, 
  const R3Affine& affine12, const R3Affine& affine21, 
  RNLength coverage_sigma = default_coverage_sigma,
  RNScalar *coverage1 = NULL, RNScalar *coverage2 = NULL, 
  RNLength *rmsd = NULL)
{
  // Check number of correspondences
  if ((points1.NEntries() == 0) || (points2.NEntries() == 0) ||
      (ncorrespondences == 0) || !correspondences1 || !correspondences2) {
    if (coverage1) *coverage1 = 0;
    if (coverage2) *coverage2 = 0;
    if (rmsd) *rmsd = RN_INFINITY;
    return;
  }

  // Initialize stuff
  RNLength coverage_factor = -1.0 / (2.0 * coverage_sigma * coverage_sigma);
  RNLength max_squared_distance = 9 * coverage_sigma * coverage_sigma;

  // Compute RMSD
  if (rmsd) {
    // Compute SSD
    RNScalar ssd = 0;
    for (int i = 0; i < ncorrespondences; i++) {
      const R3Point& point1 = correspondences1[i];
      R3Point point2 = correspondences2[i];
      point2.Transform(affine21);
      ssd += R3SquaredDistance(point1, point2);
    }

    // Fill in RMSD in return value
    *rmsd = sqrt(ssd / ncorrespondences);
  }

  // Compute coverage1
  if (coverage1) {
    // Compute support
    RNScalar support1 = 0;
    for (int i = 0; i < points1.NEntries(); i++) {
      const R3Point& point1 = points1[i]->position;
      RNLength closest_squared_distance = max_squared_distance;
      for (int j = 0; j < ncorrespondences; j++) {
        R3Point point2 = correspondences2[j];
        point2.Transform(affine21);
        RNLength squared_distance = R3SquaredDistance(point2, point1);
        if (squared_distance < closest_squared_distance) {
          closest_squared_distance = squared_distance;
        }
      }
      if (closest_squared_distance >= max_squared_distance) continue;
      support1 += exp(closest_squared_distance * coverage_factor);
    }

    // Fill in coverage1 in return value
    *coverage1 = (RNScalar) support1 / (RNScalar) points1.NEntries();
  }

  // Compute coverage2
  if (coverage2) {
    // Compute support
    RNScalar support2 = 0;
    for (int i = 0; i < points2.NEntries(); i++) {
      const R3Point& point2 = points2[i]->position;
      RNLength closest_squared_distance = max_squared_distance;
      for (int j = 0; j < ncorrespondences; j++) {
        R3Point point1 = correspondences1[j];
        point1.Transform(affine12);
        RNLength squared_distance = R3SquaredDistance(point1, point2);
        if (squared_distance < closest_squared_distance) {
          closest_squared_distance = squared_distance;
        }
      }
      if (closest_squared_distance >= max_squared_distance) continue;
      support2 += exp(closest_squared_distance * coverage_factor);
    }

    // Fill in coverage2 in return value
    *coverage2 = (RNScalar) support2 / (RNScalar) points2.NEntries();
  }
}



////////////////////////////////////////////////////////////////////////
// Model creation functions
////////////////////////////////////////////////////////////////////////

Model::
Model(void)
 : mesh(NULL), 
   object(NULL), 
   points(NULL), 
   kdtree(NULL), 
   centroid(0,0,0), 
   radius(0) 
{ 
  // Initialize name
  name[0]='\0'; 
}



Model::
~Model(void)
{ 
  // Delete stuff
  if (mesh) delete mesh;
  if (kdtree) delete kdtree;
  if (points) {
    for (int i = 0; i < points->NEntries(); i++) delete points->Kth(i);
    delete points;
  }
}



Model *
CreateModel(const RNArray<Point *>& points, RNCoord zmin = RN_UNKNOWN,
  const char *name = NULL, R3SurfelObject *object = NULL, R3Mesh *mesh = NULL) 
{
  // Allocate model
  Model *model = new Model();
  if (!model) {
    fprintf(stderr, "Unable to allocate model for surfels\n");
    return NULL;
  }

  // Assign object
  model->object = object;
  model->mesh = mesh;

  // Create points
  model->points = new RNArray<Point *>(points);
  if (!model->points) { 
    delete model; 
    return NULL; 
  }

  // Create kd tree
  model->kdtree = new R3Kdtree<Point *>(*(model->points));
  if (!model->kdtree) {
    delete model->points;
    delete model;
    return NULL;
  }

  // Create array of positions 
  RNArray<R3Point *> positions;
  for (int i = 0; i < model->points->NEntries(); i++) {
    Point *point = model->points->Kth(i);
    positions.Insert(&(point->position));
  }

  // Compute centroid
  model->centroid = R3Centroid(positions);

  // Compute origin
  model->origin = model->centroid; 
  model->origin[2] = (zmin != RN_UNKNOWN) ? zmin : model->kdtree->BBox().ZMin();

  // Compute radius
  model->radius = R3AverageDistance(model->centroid, positions);

  // Create PCA transformation
  model->pca_transformation = PCAAlignmentTransformation(positions, &(model->origin));

  // Remember filename
  if (name) strncpy(model->name, name, 1024);
  else model->name[0] = '\0';

  // Return model
  return model;
}

  
  
Model *
CreateSurfelModel(R3SurfelScene *scene, 
  const R3Point& center_point, RNLength radius,
  RNCoord zmin, RNCoord zmax,
  int min_npoints = default_model_min_npoints,
  int max_npoints = default_model_max_npoints,
  RNScalar min_volume = default_model_min_volume, 
  RNScalar max_volume = default_model_max_volume, 
  RNScalar max_gap = default_model_max_connectivity_gap)
{
  // Create name
  char name[1024];
  sprintf(name, "%.3f_%.3f_%.3f", center_point.X(), center_point.Y(), center_point.Z());

  // Create sample points
  RNArray<Point *> *points = CreateSamplePoints(scene, center_point, radius, zmin, zmax, 
    min_npoints, max_npoints, min_volume, max_volume, max_gap);
  if (!points) return NULL; 

  // Create model
  Model *model = CreateModel(*points, zmin, name);

  // Delete array of points (not points themselves)
  delete points;

  // Return model
  return model;
}

  

#if 0
  
Model *
CreateSurfelModel(R3SurfelScene *scene, 
  const R2Point& center, 
  RNLength radius = default_model_radius, 
  RNLength height = default_model_max_height,
  int min_npoints = default_model_min_npoints,
  int max_npoints = default_model_max_npoints)
  )
{
  // Extract pointset
  R3Point origin(center[0], center[1], 0);
  R3SurfelCylinderConstraint cylinder_constraint(origin, radius);
  R3SurfelPointSet *pointset1 = CreatePointSet(scene, NULL, &cylinder_constraint);
  if (!pointset1) return NULL;

  // Check minimum number of points
  if ((min_npoints > 0) && (pointset1->NPoints() < min_npoints)) {
    delete pointset1;
    return NULL;
  }

  // Move origin to support plane
  RNScalar support_count = 0;
  R3Plane support_plane = EstimateSupportPlane(pointset1, 0.1, &support_count);
  if ((support_count > 16) && (support_plane[2] > 0.5)) {
    RNScalar support_z = -(origin[0]*support_plane[0] + origin[1]*support_plane[1] + support_plane[3]) / support_plane[2];
    origin[2] = support_z;
  }
  else {
    origin[2] = pointset1->BBox().ZMin();
  }

  // Remove points that are too low or too high
  R3SurfelCylinderConstraint z_constraint(origin, radius, origin[2] + 0.25, origin[2] + height);
  R3SurfelPointSet *pointset2 = CreatePointSet(pointset1, &z_constraint);
  delete pointset1;
  if (!pointset2) return NULL;

  // Check minimum number of points
  if ((min_npoints > 0) && (pointset2->NPoints() < min_npoints)) {
    delete pointset2;
    return NULL;
  }

  // Remove points that are not connected
  R3Point seed = origin + R3posz_vector;
  R3SurfelPointSet *pointset = CreateConnectedPointSet(pointset2, seed, 
    default_model_min_volume, default_model_max_volume, default_model_max_connectivity_gap);
  delete pointset2;
  if (!pointset) return NULL;

  // Check minimum number of points
  if ((min_npoints > 0) && (pointset->NPoints() < min_npoints)) {
    delete pointset;
    return NULL;
  }

  // Create points
  RNArray<Point *> *points = CreateSamplePoints(pointset, max_npoints);
  if (!points) { delete pointset; return NULL; }

  // Create model
  Model *model = CreateModel(*points, origin[2], "scene");
  if (!model) { delete points; delete pointset; return NULL; }

  // Delete surfel point set
  delete pointset;

  // Delete array of points (not points themselves)
  delete points;

  // Return model
  return model;
}

#endif



Model *
CreateSurfelModel(R3SurfelScene *scene, 
  const R2Point& center, 
  RNLength radius = default_model_radius, 
  RNLength height = default_model_max_height,
  const R2Grid *zsupport_grid = NULL,
  RNCoord zsupport_coordinate = RN_UNKNOWN,
  int min_npoints = default_model_min_npoints,
  int max_npoints = default_model_max_npoints,
  RNScalar min_volume = default_model_min_volume, 
  RNScalar max_volume = default_model_max_volume, 
  RNScalar max_gap = default_model_max_connectivity_gap)
{
  // Estimate z coordinate
  RNScalar z = RN_UNKNOWN;

  // Check zsupport grid
  if ((z == RN_UNKNOWN) && (zsupport_grid)) {
    z = zsupport_grid->WorldValue(center);
    if (z == R2_GRID_UNKNOWN_VALUE) z = RN_UNKNOWN;
    if (z == 0) z = RN_UNKNOWN;
  }
  
  // Check zsupport coordinate
  if ((z == RN_UNKNOWN) && (zsupport_coordinate != RN_UNKNOWN)) {
    z = zsupport_coordinate;
  }

  // Check estimated zsupport plane
  if (z == RN_UNKNOWN) {
    RNScalar support_count = 0;
    R3Plane support_plane = EstimateSupportPlane(scene, R3Point(center.X(), center.Y(), 0), 4.0, 0.1, &support_count);
    if ((support_count > 16) && (support_plane[2] > 0.5)) {
      z = -(center[0]*support_plane[0] + center[1]*support_plane[1] + support_plane[3]) / support_plane[2];
    }
  }

  // Check if could not find any way to estimate z
  if (z == RN_UNKNOWN) return NULL;

  // Create origin with computed z coordinate
  R3Point origin(center.X(), center.Y(), z);

  // Create model
  return CreateSurfelModel(scene, origin, radius, z + 0.25, z + height, 
    min_npoints, max_npoints, min_volume, max_volume, max_gap);
}



Model *
CreateObjectModel(R3SurfelObject *object) 
{
  // Create name
  char name[1024];
  strncpy(name, object->Name(), 1024);

  // Create sample points
  RNArray<Point *> *points = CreateSamplePoints(object);
  if (!points) return NULL; 

  // Create model
  Model *model = CreateModel(*points, RN_UNKNOWN, name, object);

  // Delete array of points (not points themselves)
  delete points;

  // Return model
  return model;
}

  
  
Model *
CreateMeshModel(R3Mesh *mesh) 
{
  // Create name
  char name[1024];
  strncpy(name, mesh->Name(), 1024);

  // Create sample points
  RNArray<Point *> *points = CreateSamplePoints(mesh);
  if (!points) return NULL; 

  // Create model
  Model *model = CreateModel(*points, RN_UNKNOWN, name, NULL, mesh);

  // Delete array of points (not points themselves)
  delete points;

  // Return model
  return model;
}

  
  
Model *
CreateMeshModel(const char *filename) 
{
  // Allocate mesh
  R3Mesh *mesh = new R3Mesh();
  if (!mesh) {
    fprintf(stderr, "Unable to allocate mesh for %s\n", filename);
    return NULL;
  }

  // Read mesh
  if (!mesh->ReadFile(filename)) {
    fprintf(stderr, "Unable to read mesh from %s\n", filename);
    delete mesh;
    return NULL;
  }

  // Return model
  return CreateMeshModel(mesh);
}

  
  
RNArray<Model *> *
CreateMeshModels(const char *list_name)
{
  // Allocate array of models
  RNArray<Model *> *models = new RNArray<Model *>();
  if (!models) {
    fprintf(stderr, "Unable to allocate array of models\n");
    return NULL;
  }

  // Open list file
  FILE *fp = fopen(list_name, "r");
  if (!fp) {
    fprintf(stderr, "Unable to open mesh list file: %s\n", list_name);
    return NULL;
  }

  // Read list file
  char mesh_name[1024];
  while (fscanf(fp, "%s", mesh_name) == (unsigned int) 1) {
    Model *model = CreateMeshModel(mesh_name); 
    if (!model) return NULL;
    models->Insert(model);
  }

  // Close list file
  fclose(fp);
  
  // Return models
  return models;
};



////////////////////////////////////////////////////////////////////////
// Model alignment functions
////////////////////////////////////////////////////////////////////////

#if 0 // Not used

static int
CreateCorrespondences(Model *model1, Model *model2, 
  const R3Affine& affine12, const R3Affine& affine21,
  R3Point *correspondences1, R3Point *correspondences2, 
  int max_correspondences, RNLength max_distance)
{
  return CreateCorrespondences(
    *(model1->points), *(model2->points),
    model1->kdtree, model2->kdtree, 
    affine12, affine21,
    correspondences1, correspondences2, 
    max_correspondences, max_distance);
}



static R3Affine
ICPAlignmentTransformation(Model *model1, Model *model2,
  R3Point *correspondences1 = NULL, R3Point *correspondences2 = NULL,
  const R3Affine& initial_affine21 = R3identity_affine, 
  int max_iterations = 8, RNLength start_distance = 2.0, RNLength end_distance = 0.5,
  int *result_ncorrespondences = NULL, RNBoolean *result_converged = NULL)
{
  return ICPAlignmentTransformation(
    *(model1->points), *(model2->points),
    model1->kdtree, model2->kdtree, 
    correspondences1, correspondences2, 
    &(model1->origin), &(model2->origin),
    initial_affine21,
    max_iterations, start_distance, end_distance,
    result_ncorrespondences, result_converged);
}



static void
ScoreAlignmentTransformation(Model *model1, Model *model2,
  const R3Affine& affine12, const R3Affine& affine21, 
  RNLength coverage_sigma = default_coverage_sigma,
  RNScalar *result_coverage1 = NULL, RNScalar *result_coverage2 = NULL, 
  RNLength *result_rmsd = NULL, int *result_ncorrespondences = NULL)
{
  // Allocate correspondences
  int max_correspondences = model1->points->NEntries() + model2->points->NEntries();
  R3Point *correspondences1 = new R3Point [ max_correspondences ];
  R3Point *correspondences2 = new R3Point [ max_correspondences ];
  
  // Create correspondences
  RNLength max_distance = 3.0 * coverage_sigma;
  int ncorrespondences = CreateCorrespondences(
    model1, model2, 
    affine12, affine21, 
    correspondences1, correspondences2, 
    max_correspondences, max_distance);

  // Compute scores
  ScoreAlignmentTransformation(
    *(model1->points), *(model2->points),
    model1->kdtree, model2->kdtree, 
    correspondences1, correspondences2, ncorrespondences, 
    affine12, affine21, coverage_sigma, 
    result_coverage1, result_coverage2, result_rmsd);

  // Delete correspondences
  delete [] correspondences1;
  delete [] correspondences2;

  // Return result
  if (result_ncorrespondences) *result_ncorrespondences = ncorrespondences;
}

#endif



#if 0
static int
FitModel(Model *model1, Model *model2,
  const R3Affine& initial_affine21,
  R3Affine *result_affine21 = NULL,   
  RNScalar *result_coverage1 = NULL, RNScalar *result_coverage2 = NULL, 
  RNScalar *result_rmsd = NULL, int *result_ncorrespondences = NULL,
  int icp_iterations = default_icp_iterations,
  RNLength icp_start_distance = default_icp_start_distance,
  RNLength icp_end_distance = default_icp_end_distance,
  RNScalar coverage_sigma = default_coverage_sigma)
{
  // Check number of points
  if (model1->points->NEntries() == 0) return 0;
  if (model2->points->NEntries() == 0) return 0;

  // Allocate correspondence buffers
  int max_correspondences = model1->points->NEntries() + model2->points->NEntries();
  R3Point *correspondences1 = new R3Point [ max_correspondences ];
  R3Point *correspondences2 = new R3Point [ max_correspondences ];
  int ncorrespondences = 0;

  // Refine transformation with ICP
  R3Affine affine21 = ICPAlignmentTransformation(
    *(model1->points), *(model2->points),
    model1->kdtree, model2->kdtree, 
    correspondences1, correspondences2, 
    &(model1->origin), &(model2->origin), 
    initial_affine21, 
    icp_iterations, icp_start_distance, icp_end_distance, 
    &ncorrespondences);

  // Score transformation
  ScoreAlignmentTransformation(
    *(model1->points), *(model2->points),
    model1->kdtree, model2->kdtree,
    correspondences1, correspondences2, ncorrespondences, 
    affine21.Inverse(), affine21, coverage_sigma,
    result_coverage1, result_coverage2, result_rmsd);

  // Return best alignment transformation
  if (result_ncorrespondences) *result_ncorrespondences = ncorrespondences;
  if (result_affine21) *result_affine21 = affine21;

  // Delete correspondences
  delete [] correspondences1;
  delete [] correspondences2;

  // Return success
  return 1;
}
#endif



static int
FitModel(Model *model1, Model *model2,
  R3Affine *result_affine21 = NULL,   
  RNScalar *result_coverage1 = NULL, RNScalar *result_coverage2 = NULL, 
  RNScalar *result_rmsd = NULL, int *result_ncorrespondences = NULL,
  int initial_guess_iterations = default_initial_guess_iterations,
  int icp_iterations_per_guess = default_icp_iterations,
  RNLength icp_start_distance = default_icp_start_distance,
  RNLength icp_end_distance = default_icp_end_distance,
  RNScalar coverage_sigma = default_coverage_sigma,
  RNScalar max_translation = default_initial_guess_max_translation,
  RNScalar max_rotation = default_initial_guess_max_rotation)
{
  // Check number of points
  if (model1->points->NEntries() == 0) return 0;
  if (model2->points->NEntries() == 0) return 0;

  // Allocate correspondence buffers
  int max_correspondences = model1->points->NEntries() + model2->points->NEntries();
  R3Point *correspondences1 = new R3Point [ max_correspondences ];
  R3Point *correspondences2 = new R3Point [ max_correspondences ];

  // Search for best alignment
  RNScalar best_rmsd = FLT_MAX;
  RNScalar best_coverage1 = 0;
  RNScalar best_coverage2 = 0;
  int best_ncorrespondences = 0;
  R3Affine best_affine21 = R3identity_affine;
  for (int i = 0; i < initial_guess_iterations; i++) {
    // Initialize scores
    RNScalar rmsd = FLT_MAX;
    RNScalar coverage1 = 0;
    RNScalar coverage2 = 0;
    int ncorrespondences = 0;

    // Create initial transformation
    R3Affine initial21 = R3identity_affine;

    // Translate by a random vector
    RNCoord tx = 2 * max_translation * RNRandomScalar() - max_translation;
    RNCoord ty = 2 * max_translation * RNRandomScalar() - max_translation;
    RNCoord tz = 2 * max_translation * RNRandomScalar() - max_translation;
    R3Vector translation(tx, ty, tz);
    initial21.Translate(translation);

    // Rotate 180 degrees half the time
    if (RNRandomScalar() < 0.5) {
      initial21.Translate(model1->origin.Vector());
      initial21.ZRotate(RN_PI);
      initial21.Translate(-(model1->origin.Vector()));
    }

    // Rotate 90 degrees a quarter of the time
    if (RNRandomScalar() < 0.25) {
      initial21.Translate(model1->origin.Vector());
      initial21.ZRotate(RN_PI_OVER_TWO);
      initial21.Translate(-(model1->origin.Vector()));
    }

    // Rotate by a random amount half of the time
    if (RNRandomScalar() < 0.5) {
      RNScalar angle = 2 * max_rotation * RNRandomScalar() - max_rotation;
      initial21.Translate(model1->origin.Vector());
      initial21.ZRotate(angle);
      initial21.Translate(-(model1->origin.Vector()));
    }

    // Apply pca transformation
    R3Affine pca = R3identity_affine;  
    pca.Transform(model1->pca_transformation.Inverse());
    pca.Transform(model2->pca_transformation);
    initial21.Transform(pca);

    // Refine transformation with ICP
    R3Affine icp21 = ICPAlignmentTransformation(
      *(model1->points), *(model2->points),
      model1->kdtree, model2->kdtree, 
      correspondences1, correspondences2, 
      &(model1->origin), &(model2->origin), 
      initial21, icp_iterations_per_guess, icp_start_distance, icp_end_distance, 
      &ncorrespondences);

    // Check the number of correspondences
    if (ncorrespondences == 0) continue;

    // Score transformation
    ScoreAlignmentTransformation(
      *(model1->points), *(model2->points),
      model1->kdtree, model2->kdtree,
      correspondences1, correspondences2, ncorrespondences, 
      icp21.Inverse(), icp21, 
      coverage_sigma,
      &coverage1, &coverage2, &rmsd);

    // printf("%d : %6d %12.6f %12.6f %12.6f\n", i, ncorrespondences, rmsd, coverage1, coverage2);

    // Remember if best
    if (coverage2 > best_coverage2) {
      best_rmsd = rmsd;
      best_coverage1 = coverage1;
      best_coverage2 = coverage2;
      best_ncorrespondences = ncorrespondences;
      best_affine21 = icp21;
    }
  }

  // Return best alignment transformation
  if (result_ncorrespondences) *result_ncorrespondences = best_ncorrespondences;
  if (result_affine21) *result_affine21 = best_affine21;
  if (result_coverage1) *result_coverage1 = best_coverage1;
  if (result_coverage2) *result_coverage2 = best_coverage2;
  if (result_rmsd) *result_rmsd = best_rmsd;

  // Delete correspondences
  delete [] correspondences1;
  delete [] correspondences2;

  // Return success
  return 1;
}



Model *
BestFitModel(Model *model1, const RNArray<Model *>& models,
  R3Affine *result_affine21 = NULL,   
  RNScalar *result_coverage1 = NULL, RNScalar *result_coverage2 = NULL, 
  RNScalar *result_rmsd = NULL, int *result_ncorrespondences = NULL,
  int initial_guess_iterations = default_initial_guess_iterations,
  int icp_iterations_per_guess = default_icp_iterations,
  RNLength icp_start_distance = default_icp_start_distance,
  RNLength icp_end_distance = default_icp_end_distance,
  RNScalar coverage_sigma = default_coverage_sigma,
  RNScalar max_translation = default_initial_guess_max_translation,
  RNScalar max_rotation = default_initial_guess_max_rotation)
{
  // Fit each model 
  int best_index = -1;
  Model *best_model = NULL;
  R3Affine best_affine21;
  RNLength best_coverage1 = 0;
  RNLength best_coverage2 = 0;
  RNLength best_rmsd = FLT_MAX;
  int best_ncorrespondences = 0;
  for (int i = 0; i < models.NEntries(); i++) {
    Model *current_model = models.Kth(i);
    R3Affine current_affine21;
    RNScalar current_coverage1, current_coverage2, current_rmsd;
    int current_ncorrespondences;
    if (FitModel(model1, current_model, &current_affine21, 
      &current_coverage1, &current_coverage2, 
      &current_rmsd, &current_ncorrespondences, 
      initial_guess_iterations, 
      icp_iterations_per_guess, icp_start_distance, icp_end_distance,
      coverage_sigma, max_translation, max_rotation)) {
      if (current_coverage2 > best_coverage2) {
        best_index = i;
        best_model = current_model;
        best_affine21 = current_affine21;
        best_coverage1 = current_coverage1;
        best_coverage2 = current_coverage2;
        best_rmsd = current_rmsd;
        best_ncorrespondences = current_ncorrespondences;
      }
    }
  }

  // Print debug message
  if (0) {
    R4Matrix m = best_affine21.Matrix();
    printf("%g %g %g %g\n", m[0][0], m[0][1], m[0][2], m[0][3]);
    printf("%g %g %g %g\n", m[1][0], m[1][1], m[1][2], m[1][3]);
    printf("%g %g %g %g\n", m[2][0], m[2][1], m[2][2], m[2][3]);
    printf("%g %g %g %g\n", m[3][0], m[3][1], m[3][2], m[3][3]);
    printf("Score: %d %g %g %g\n", best_index, best_coverage1, best_coverage2, best_rmsd);
    printf("----\n");
  }

  // Fill in result
  if (result_ncorrespondences) *result_ncorrespondences = best_ncorrespondences;
  if (result_affine21) *result_affine21 = best_affine21;
  if (result_coverage1) *result_coverage1 = best_coverage1;
  if (result_coverage2) *result_coverage2 = best_coverage2;
  if (result_rmsd) *result_rmsd = best_rmsd;

  // Return best model
  return best_model;
}



