// Source file for the mesh alignment program



// Include files 

#include "R3Shapes/R3Shapes.h"



// Program arguments

static char *input1_name = NULL;
static char *input2_name = NULL;
static char *output1_name = NULL;
static char *weights1_name = NULL;
static char *weights2_name = NULL;
static RNScalar min_weight = 0;
static int pca_translation = 0;
static int pca_scale = 0;
static int pca_rotation = 0; // 2 = only 180s
static int ransac_rotation = 0;
static int ransac_translation = 0;
static int ransac_scale = 0;
static int icp_translation = 0;
static int icp_scale = 0;
static int icp_rotation = 0;
static int max_points = 256;
static int sample_method = 0; // 0=surface, 1=edges, 2=vertices
static int correspondence_method = 0; // 0=points, 1=surface
static int print_verbose = 0;
static int print_debug = 0;



static RNScalar *
ReadWeights(const char *filename, RNScalar *& weights, int& nweights)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Open file
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Unable to open edge weight file %s\n", filename);
    return NULL;
  }

  // Count weights
  nweights = 0;
  RNScalar dummy;
  while (fscanf(fp, "%lf", &dummy) == 1) nweights++;
  fseek(fp, 0, SEEK_SET);

  // Check number of weights
  if (nweights == 0) {
    fprintf(stderr, "There are no weights in %s\n", filename);
    return NULL;
  }

  // Allocate array for weights
  weights = new RNScalar [ nweights ];
  if (!weights) {
    fprintf(stderr, "Unable to allocate weights for %s\n", filename);
    return NULL;
  }

  // Read weights
  for (int i = 0; i < nweights; i++) {
    if (fscanf(fp, "%lf", &weights[i]) != 1) {
      fprintf(stderr, "Unable to read weight %d from %s\n", i, filename);
      return NULL;
    }
  }

  // Close file
  fclose(fp);

  // Print message
  if (print_verbose) {
    printf("Read weights from %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Weights = %d\n", nweights);
  }

  // Return weights
  return weights;
}



static int
CreatePoints(R3Mesh *mesh, RNScalar *weights, int nweights, R3Point *points, int max_points, int sample_method)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Initialize return value
  int npoints = 0;

  // Check maximum number of points
  if (max_points <= 0) return 0;

  // Check sample method
  if (sample_method == 0) { 
    // Sample faces

    // Check weights
    if (nweights > 0) {
      if (weights == NULL) nweights = 0;
      else if (nweights != mesh->NFaces()) {
        fprintf(stderr, "Invalid number of weights (%d) for sampling faces (%d)\n", nweights, mesh->NFaces());
        return 0;
      }
    }

    // Count total weighted area of faces
    RNArea total_area = 0.0;
    for (int i = 0; i < mesh->NFaces(); i++) {
      R3MeshFace *face = mesh->Face(i);
      RNScalar face_area = mesh->FaceArea(face);
      if (nweights > i) face_area *= weights[i];
      mesh->SetFaceValue(face, face_area);
      total_area += face_area;
    }

    // Generate points with a uniform distribution over surface area
    RNSeedRandomScalar();
    for (int i = 0; i < mesh->NFaces(); i++) {
      R3MeshFace *face = mesh->Face(i);

      // Get vertex positions
      R3MeshVertex *v0 = mesh->VertexOnFace(face, 0);
      R3MeshVertex *v1 = mesh->VertexOnFace(face, 1);
      R3MeshVertex *v2 = mesh->VertexOnFace(face, 2);
      const R3Point& p0 = mesh->VertexPosition(v0);
      const R3Point& p1 = mesh->VertexPosition(v1);
      const R3Point& p2 = mesh->VertexPosition(v2);

      // Determine number of points for face 
      RNScalar ideal_face_npoints = max_points * mesh->FaceValue(face) / total_area;
      int face_npoints = (int) ideal_face_npoints;
      RNScalar remainder = ideal_face_npoints - face_npoints;
      if (remainder > RNRandomScalar()) face_npoints++;

      // Generate random points in face
      for (int j = 0; j < face_npoints; j++) {
        RNScalar r1 = sqrt(RNRandomScalar());
        RNScalar r2 = RNRandomScalar();
        RNScalar t0 = (1.0 - r1);
        RNScalar t1 = r1 * (1.0 - r2);
        RNScalar t2 = r1 * r2;
        points[npoints++] = t0*p0 + t1*p1 + t2*p2;
        if (npoints >= max_points) break;
      }

      // Check number of points created
      if (npoints >= max_points) break;
    }
  }
  else if (sample_method == 1) {
    // Sample edges

    // Check weights
    if (nweights > 0) {
      if (weights == NULL) nweights = 0;
      else if (nweights != mesh->NEdges()) {
        fprintf(stderr, "Invalid number of weights (%d) for sampling edges (%d)\n", nweights, mesh->NEdges());
        return 0;
      }
    }

    // Count total weighted length of edges
    RNLength total_length = 0.0;
    for (int i = 0; i < mesh->NEdges(); i++) {
      R3MeshEdge *edge = mesh->Edge(i);
      if ((nweights > i) && (weights[i] < min_weight)) continue;
      RNScalar edge_length = mesh->EdgeLength(edge);
      if (nweights > i) edge_length *= weights[i];
      mesh->SetEdgeValue(edge, edge_length);
      total_length += edge_length;
    }
    
    // Generate points with a uniform distribution over edge length
    RNSeedRandomScalar();
    for (int i = 0; i < mesh->NEdges(); i++) {
      R3MeshEdge *edge = mesh->Edge(i);
      if ((nweights > i) && (weights[i] < min_weight)) continue;

      // Get vertex positions
      R3MeshVertex *v0 = mesh->VertexOnEdge(edge, 0);
      R3MeshVertex *v1 = mesh->VertexOnEdge(edge, 1);
      const R3Point& p0 = mesh->VertexPosition(v0);
      const R3Point& p1 = mesh->VertexPosition(v1);

      // Determine number of points for edge 
      RNScalar ideal_edge_npoints = max_points * mesh->EdgeValue(edge) / total_length;
      int edge_npoints = (int) ideal_edge_npoints;
      RNScalar remainder = ideal_edge_npoints - edge_npoints;
      if (remainder > RNRandomScalar()) edge_npoints++;

      // Generate random points in edge
      for (int j = 0; j < edge_npoints; j++) {
        RNScalar t = RNRandomScalar();
        points[npoints++] = t*p0 + (1-t)*p1;
        if (npoints >= max_points) break;
      }

      // Check number of points created
      if (npoints >= max_points) break;
    }
  }
  else if (sample_method == 2) {
    // Sample vertices

    // Check if number of vertices is less than max_points
    if (max_points > mesh->NVertices()) {
      for (int i = 0; i < mesh->NVertices(); i++) {
        R3MeshVertex *vertex = mesh->Vertex(i);
        points[npoints++] = mesh->VertexPosition(vertex);
      }
    }
    else {
      // Check weights
      if (nweights > 0) {
        if (weights == NULL) nweights = 0;
        else if (nweights != mesh->NVertices()) {
          fprintf(stderr, "Invalid number of weights (%d) for sampling vertices (%d)\n", nweights, mesh->NVertices());
          return 0;
        }
      }

      // Count total weight 
      RNScalar total_weight = 0.0;
      for (int i = 0; i < mesh->NVertices(); i++) {
        R3MeshVertex *vertex = mesh->Vertex(i);
        RNScalar vertex_weight = (nweights > i) ? weights[i] : mesh->VertexArea(vertex);
        mesh->SetVertexValue(vertex, vertex_weight);
        total_weight += vertex_weight;
      }

      // Check total weight (could be zero if there are vertices, but no weights or faces)
      if (total_weight == 0) {
        for (int i = 0; i < mesh->NVertices(); i++) {
          R3MeshVertex *vertex = mesh->Vertex(i);
          mesh->SetVertexValue(vertex, 1);
          total_weight += 1;
        }
      }

      // Generate points with probability related to vertex weights/areas
      RNSeedRandomScalar();
      for (int i = 0; i < mesh->NVertices(); i++) {
        R3MeshVertex *vertex = mesh->Vertex(i);
        const R3Point& position = mesh->VertexPosition(vertex);

        // Determine number of points for vertex
        RNScalar ideal_vertex_npoints = max_points * mesh->VertexValue(vertex) / total_weight;
        int vertex_npoints = (int) ideal_vertex_npoints;
        RNScalar remainder = ideal_vertex_npoints - vertex_npoints;
        if (remainder > RNRandomScalar()) vertex_npoints++;

        // Generate points at vertex
        for (int j = 0; j < vertex_npoints; j++) {
          points[npoints++] = position;
          if (npoints >= max_points) break;
        }

        // Check number of points created
        if (npoints >= max_points) break;
      }
    }
  }

#if 0
  if (1) {
    static int file_count = 1;
    char filename[256];
    sprintf(filename, "points%d.pts", file_count++);
    FILE *fp = fopen(filename, "wb");
    for (int i = 0; i < npoints; i++) {
      R3Point position = points[i];
      static float coordinates[6] = { 0 };
      coordinates[0] = points[i].X();
      coordinates[1] = points[i].Y();
      coordinates[2] = points[i].Z();
      fwrite(coordinates, sizeof(float), 6, fp);
    }
    fclose(fp);
  }
#endif

  // Print message
  if (print_debug) {
    printf("Created points ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Points = %d\n", npoints);
  }

  // Return number of points actually created
  return npoints;
}



static int
CreatePointPointCorrespondences(
  R3Mesh *mesh1, const R3Point *points1, int npoints1,
  R3Mesh *mesh2, const R3Point *points2, int npoints2, 
  const R3Affine& affine12, const R3Affine& affine21,
  R3Point *correspondences1, R3Point *correspondences2, 
  int max_correspondences)
{
  // Build kdtree for points1
  static R3Kdtree<const R3Point *> *tree1 = NULL;
  if (!tree1) {
    RNArray<const R3Point *> array1;
    for (int i = 0; i < npoints1; i++) array1.Insert(&points1[i]);
    tree1 = new R3Kdtree<const R3Point *>(array1);
  }

  // Build kdtree for points2
  static R3Kdtree<const R3Point *> *tree2 = NULL;
  if (!tree2) {
    RNArray<const R3Point *> array2;
    for (int i = 0; i < npoints2; i++) array2.Insert(&points2[i]);
    tree2 = new R3Kdtree<const R3Point *>(array2);
  }

  // Initialize number of correspondences
  assert(max_correspondences == npoints1 + npoints2);
  int ncorrespondences = 0;

  // Compute correspondences for points1 -> mesh2
  for (int i = 0; i < npoints1; i++) {
    R3Point position1 = points1[i];
    position1.Transform(affine12);
    const R3Point *closest = tree2->FindClosest(position1);
    assert(ncorrespondences < max_correspondences);
    correspondences1[ncorrespondences] = points1[i];
    correspondences2[ncorrespondences] = *closest;
    ncorrespondences++;
  }

  // Compute correspondences for points2 -> mesh1
  for (int i = 0; i < npoints2; i++) {
    R3Point position2 = points2[i];
    position2.Transform(affine21);
    const R3Point *closest = tree1->FindClosest(position2);
    assert(ncorrespondences < max_correspondences);
    correspondences1[ncorrespondences] = *closest;
    correspondences2[ncorrespondences] = points2[i];
    ncorrespondences++;
  }

  // Return number of correspondences
  assert(ncorrespondences == npoints1 + npoints2);
  assert(ncorrespondences == max_correspondences);
  return ncorrespondences;
}



static int
CreatePointSurfaceCorrespondences(
  R3Mesh *mesh1, const R3Point *points1, int npoints1,
  R3Mesh *mesh2, const R3Point *points2, int npoints2, 
  const R3Affine& affine12, const R3Affine& affine21,
  R3Point *correspondences1, R3Point *correspondences2, 
  int max_correspondences)
{
  // Initialize number of correspondences
  assert(max_correspondences == npoints1 + npoints2);
  int ncorrespondences = 0;

  // Compute correspondences for points1 -> mesh2
  static R3MeshSearchTree *tree2 = NULL;
  if (!tree2) tree2 = new R3MeshSearchTree(mesh2);
  else assert(mesh2 == tree2->mesh);
  for (int i = 0; i < npoints1; i++) {
    R3Point position1 = points1[i];
    position1.Transform(affine12);
    R3MeshIntersection closest;
    tree2->FindClosest(position1, closest);
    assert(ncorrespondences < max_correspondences);
    correspondences1[ncorrespondences] = points1[i];
    correspondences2[ncorrespondences] = closest.point;
    ncorrespondences++;
  }

  // Compute correspondences for points2 -> mesh1
  static R3MeshSearchTree *tree1 = NULL;
  if (!tree1) tree1 = new R3MeshSearchTree(mesh1);
  else assert(mesh1 == tree1->mesh);
  for (int i = 0; i < npoints2; i++) {
    R3Point position2 = points2[i];
    position2.Transform(affine21);
    R3MeshIntersection closest;
    tree1->FindClosest(position2, closest);
    assert(ncorrespondences < max_correspondences);
    correspondences1[ncorrespondences] = closest.point;
    correspondences2[ncorrespondences] = points2[i];
    ncorrespondences++;
  }

  // Return number of correspondences
  assert(ncorrespondences == npoints1 + npoints2);
  assert(ncorrespondences == max_correspondences);
  return ncorrespondences;
}



static int
CreatePointCorrespondences(
  R3Mesh *mesh1, const R3Point *points1, int npoints1,
  R3Mesh *mesh2, const R3Point *points2, int npoints2, 
  const R3Affine& affine12, const R3Affine& affine21,
  R3Point *correspondences1, R3Point *correspondences2, 
  int max_correspondences)
{
  // Check method
  if (correspondence_method == 0) {
    return CreatePointPointCorrespondences(
      mesh1, points1, npoints1, 
      mesh2, points2, npoints2, 
      affine12, affine21, 
      correspondences1, correspondences2, max_correspondences);
  }
  else {
    return CreatePointSurfaceCorrespondences(
      mesh1, points1, npoints1, 
      mesh2, points2, npoints2, 
      affine12, affine21, 
      correspondences1, correspondences2, max_correspondences);
  }
}


static RNScalar 
RMSD(R3Mesh *mesh1, const R3Point *points1, int npoints1,
     R3Mesh *mesh2, const R3Point *points2, int npoints2, 
     const R3Affine& affine12, const R3Affine& affine21)
{
  // Check number of points
  if (npoints1 == 0) return RN_INFINITY;
  if (npoints2 == 0) return RN_INFINITY;

  // Create array of correspondences
  int max_correspondences = npoints1 + npoints2;
  R3Point *correspondences1 = new R3Point [ max_correspondences ];
  R3Point *correspondences2 = new R3Point [ max_correspondences ];
  int ncorrespondences = CreatePointCorrespondences(
    mesh1, points1, npoints1, 
    mesh2, points2, npoints2, 
    affine12, affine21,
    correspondences1, correspondences2, 
    max_correspondences);

  // Add SSD of correspondences
  RNScalar ssd = 0;
  for (int i = 0; i < ncorrespondences; i++) {
    R3Point position1 = correspondences1[i];
    position1.Transform(affine12);
    R3Point& position2 = correspondences2[i];
    RNScalar d = R3Distance(position1, position2);
    ssd += d * d;
  }

  // Compute the RMSD
  RNScalar rmsd = sqrt(ssd / ncorrespondences);

  // Delete arrays of correspondences
  delete [] correspondences1;
  delete [] correspondences2;

  // Return RMSD
  return rmsd;
}



static int
ICPAlignmentTransformation(
  R3Mesh *mesh1, const R3Point *points1, int npoints1,
  R3Mesh *mesh2, const R3Point *points2, int npoints2, 
  R3Affine& affine12, R3Affine& affine21,
  int translation, int rotation, int scale)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate arrays of correspondences
  int ncorrespondences = npoints1 + npoints2;
  R3Point *correspondences_buffer1 = new R3Point [2 * ncorrespondences];
  R3Point *correspondences_buffer2 = new R3Point [2 * ncorrespondences];
  assert(correspondences_buffer1 && correspondences_buffer2);

  // Iterate until max_iterations or converged
  RNBoolean converged = FALSE;
  const int max_iterations = 128;
  for (int iteration = 0; iteration < max_iterations; iteration++) {
    // Get arrays of correspondences
    R3Point *correspondences1 = &correspondences_buffer1[iteration%2 * ncorrespondences];
    R3Point *correspondences2 = &correspondences_buffer2[iteration%2 * ncorrespondences];

    // Update correspondences for aligning transformation
    CreatePointCorrespondences(
      mesh1, points1, npoints1, 
      mesh2, points2, npoints2, 
      affine12, affine21, 
      correspondences1, correspondences2, 
      ncorrespondences);

    // Update aligning transformation for correspondences
    R4Matrix matrix = R3AlignPoints(ncorrespondences, correspondences2, correspondences1, 
      NULL, translation, rotation, scale);
    affine12.Reset(matrix);
    affine21 = affine12.Inverse();

    // Check for convergence
    converged = FALSE;
    if (iteration > 0) {
      converged = TRUE;
      R3Point *prev_correspondences1 = &correspondences_buffer1[(1-(iteration%2)) * ncorrespondences];
      R3Point *prev_correspondences2 = &correspondences_buffer2[(1-(iteration%2)) * ncorrespondences];
      for (int i = 0; i < ncorrespondences; i++) {
        if (!R3Contains(correspondences1[i], prev_correspondences1[i])) { converged = FALSE; break; }
        if (!R3Contains(correspondences2[i], prev_correspondences2[i])) { converged = FALSE; break; }
      }
    }
    if (converged) break;
  }

  // Return whether converged
  return converged;
}



static int
RansacAlignmentTransformation(
  R3Mesh *mesh1, const R3Point *points1, int npoints1,
  R3Mesh *mesh2, const R3Point *points2, int npoints2, 
  R3Affine& affine12, R3Affine& affine21,
  int translation, int rotation, int scale,
  RNLength support_sigma = 0, int max_iterations = 0)
{
  // Check number of points
  if (npoints1 < 3) return 0;
  if (npoints2 < 3) return 0;

  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Initialize result
  affine12 = R3identity_affine;
  affine21 = R3identity_affine;

  // Create temporary memory
  int max_correspondences = npoints1 + npoints2;
  R3Point *correspondences1 = new R3Point [ max_correspondences ];
  R3Point *correspondences2 = new R3Point [ max_correspondences ];

  // Compute useful variables
  if (max_iterations == 0) max_iterations = 0.1 * npoints1 * npoints2; // should be n^3
  if (support_sigma <= 0) support_sigma = 0.5 * mesh1->AverageRadius();
  RNScalar support_factor = -1.0 / (2 * support_sigma * support_sigma);

  // Generate random alignments and keep the one with best support
  RNScalar best_support = 0;
  R3Affine best_affine21 = R3identity_affine;
  for (int iteration = 0; iteration < max_iterations; iteration++) {
    int a1, b1, c1, a2, b2, c2;
    a1 = (int) (RNRandomScalar() * npoints1); 
    do { b1 = (int) (RNRandomScalar() * npoints1); } while (b1 == a1);
    do { c1 = (int) (RNRandomScalar() * npoints1); } while ((c1 == a1) || (c1 == b1));
    a2 = (int) (RNRandomScalar() * npoints2); 
    do { b2 = (int) (RNRandomScalar() * npoints2); } while (b2 == a2);
    do { c2 = (int) (RNRandomScalar() * npoints2); } while ((c2 == a2) || (c2 == b2));

    // Create triplets of points
    static RNArray<R3Point *> triplet1; 
    triplet1.Empty();
    triplet1.Insert((R3Point *) &points1[a1]);
    triplet1.Insert((R3Point *) &points1[b1]);
    triplet1.Insert((R3Point *) &points1[c1]);
    static RNArray<R3Point *> triplet2;
    triplet2.Empty();
    triplet2.Insert((R3Point *) &points2[a2]);
    triplet2.Insert((R3Point *) &points2[b2]);
    triplet2.Insert((R3Point *) &points2[c2]);

    // Compute transformation aligning triplet2 to triplet1
    R4Matrix matrix21 = R3AlignPoints(triplet1, triplet2, NULL, translation, rotation, scale);
    R3Affine aff21(matrix21);
    R3Affine aff12 = aff21.Inverse();

    // Compute correspondences
    int ncorrespondences = CreatePointCorrespondences(mesh1, points1, npoints1, mesh2, points2, npoints2, 
      aff12, aff21, correspondences1, correspondences2, max_correspondences);

    // Compute support
    RNScalar support = 0;
    for (int i = 0; i < ncorrespondences; i++) {
      R3Point& point1 = correspondences1[i];
      R3Point point2 = correspondences2[i];
      point2.Transform(aff21);
      RNScalar dd = R3SquaredDistance(point1, point2);
      RNScalar s = exp(support_factor * dd);
      support += s;
    }

    // Check if best support so far
    if (support > best_support) {
      best_affine21 = aff21;
      best_support = support;
    }
  }

  // Assign results
  affine21 = best_affine21;
  affine12 = best_affine21.Inverse();
  
  // Delete temporary data
  delete [] correspondences1;
  delete [] correspondences2;

  // Return success
  return 1;
}



static int
Align(R3Mesh *mesh1, R3Mesh *mesh2, 
  RNScalar *weights1, int nweights1, RNScalar *weights2, int nweights2,
  int pca_translation, int pca_rotation, int pca_scale, 
  int ransac_translation, int ransac_rotation, int ransac_scale, 
  int icp_translation, int icp_rotation, int icp_scale,
  int sample_method)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Initialize transformation
  R3Affine affine12(R3identity_affine);
  R3Affine affine21(R3identity_affine);
  RNBoolean converged = TRUE;
  RNScalar rmsd = FLT_MAX;

  // Compute info for mesh1
  R3Point centroid1 = mesh1->Centroid();
  RNScalar radius1 = (pca_scale) ? mesh1->AverageRadius(&centroid1) : 1;
  RNScalar scale1 = (pca_scale && (radius1 > 0)) ? 1.0 / radius1 : 1;
  R3Triad axes1 = (pca_rotation==1) ? mesh1->PrincipleAxes(&centroid1) : R3xyz_triad;
  R3Point *points1 = new R3Point [ max_points ];
  int npoints1 = CreatePoints(mesh1, weights1, nweights1, points1, max_points, sample_method);
  if ((scale1 == 0) || (npoints1 == 0)) {
    fprintf(stderr, "Unable to process first mesh\n");
    return 0;
  }

  // Compute info for mesh2
  R3Point centroid2 = mesh2->Centroid();
  RNScalar scale2 = (pca_scale) ? mesh2->AverageRadius(&centroid2) : 1;
  R3Triad axes2 = (pca_rotation == 1) ? mesh2->PrincipleAxes(&centroid2) : R3xyz_triad;
  R3Affine affine02 = R3identity_affine;
  if (pca_translation) affine02.Translate(centroid2.Vector());
  if (pca_rotation==1) affine02.Transform(axes2.Matrix());
  if (pca_scale) affine02.Scale(scale2);
  R3Point *points2 = new R3Point [ max_points ];
  int npoints2 = CreatePoints(mesh2, weights2, nweights2, points2, max_points, sample_method);
  if ((scale2 == 0) || (npoints2 == 0)) {
    fprintf(stderr, "Unable to process second mesh\n");
    return 0;
  }

  // Compute RMSD for alignment with all flips of principle axes
  if (pca_rotation) {
    for (int dim1 = 0; dim1 < 3; dim1++) {
      for (int dim2 = 0; dim2 < 3; dim2++) {
        if (dim1 == dim2) continue;
        for (int dir1 = 0; dir1 < 2; dir1++) {
          for (int dir2 = 0; dir2 < 2; dir2++) {
            // Create triad of axes for flip
            R3Vector axis1a = (dir1 == 1) ? axes1[dim1] : -axes1[dim1];
            R3Vector axis1b = (dir2 == 1) ? axes1[dim2] : -axes1[dim2];
            R3Vector axis1c = axis1a % axis1b;
            R3Triad triad1(axis1a, axis1b, axis1c);

            // Compute transformation for mesh1 (with flip)
            R3Affine affine10 = R3identity_affine;
            if (pca_scale) affine10.Scale(scale1);
            affine10.Transform(triad1.InverseMatrix());
            if (pca_translation) affine10.Translate(-centroid1.Vector());

            // Compute composite transformation and its inverse
            R3Affine flipped_affine12 = affine02;
            flipped_affine12.Transform(affine10);
            R3Affine flipped_affine21 = flipped_affine12.Inverse();

            // Refine alignment with ICP
            RNBoolean flipped_converged = FALSE;
            if (icp_translation || icp_rotation || icp_scale) {
              flipped_converged = ICPAlignmentTransformation(
                mesh1, points1, npoints1, 
                mesh2, points2, npoints2, 
                flipped_affine12, flipped_affine21, 
                icp_translation, icp_rotation, icp_scale);
            }

            // Compute RMSD for transformation
            RNScalar flipped_rmsd = RMSD(
              mesh1, points1, npoints1, 
              mesh2, points2, npoints2, 
              flipped_affine12, flipped_affine21);

            // Check if best so far -- if so, save
            if (flipped_rmsd < rmsd) {
              affine12 = flipped_affine12;
              affine21 = flipped_affine21;
              converged = flipped_converged;
              rmsd = flipped_rmsd;
            }

            // Print statistics
            if (print_debug) {
              const R4Matrix& m = flipped_affine12.Matrix();
              printf("Computed alignment transformation for flip %d %d %d %d ...\n", dim1, dim2, dir1, dir2);
              printf("  Time = %.2f seconds\n", start_time.Elapsed());
              printf("  Max Points = %d\n", max_points);
              printf("  Matrix[0][0-3] = %g %g %g %g\n", m[0][0], m[0][1], m[0][2], m[0][3]);
              printf("  Matrix[1][0-3] = %g %g %g %g\n", m[1][0], m[1][1], m[1][2], m[1][3]);
              printf("  Matrix[2][0-3] = %g %g %g %g\n", m[2][0], m[2][1], m[2][2], m[2][3]);
              printf("  Matrix[3][0-3] = %g %g %g %g\n", m[3][0], m[3][1], m[3][2], m[3][3]);
              printf("  Scale = %g\n", flipped_affine12.ScaleFactor());
              printf("  Converged = %d\n", flipped_converged);
              printf("  RMSD = %g\n", flipped_rmsd);
              fflush(stdout);
            }
          }
        }
      }
    }
  }
  else {
    // Compute transformation for mesh1
    R3Affine affine10 = R3identity_affine;
    if (pca_scale) affine10.Scale(scale1);
    if (pca_translation) affine10.Translate(-centroid1.Vector());

    // Compute composite transformation
    affine12 = affine02;
    affine12.Transform(affine10);
    affine21 = affine12.Inverse();

    // Refine alignment with ICP
    if (icp_translation || icp_rotation || icp_scale) {
      converged = ICPAlignmentTransformation(
        mesh1, points1, npoints1, 
        mesh2, points2, npoints2, 
        affine12, affine21,
        icp_translation, icp_rotation, icp_scale);
    }

    // Compute RMSD
    rmsd = RMSD(
      mesh1, points1, npoints1, 
      mesh2, points2, npoints2, 
      affine12, affine21);
  }

  // Consider ransac alignment
  if (ransac_translation || ransac_rotation || ransac_scale) {
    // Compute alignment with ransac 
    R3Affine ransac_affine12(R3identity_affine);
    R3Affine ransac_affine21(R3identity_affine);
    if (!RansacAlignmentTransformation(
      mesh1, points1, npoints1, 
      mesh2, points2, npoints2, 
      ransac_affine12, ransac_affine21,
      ransac_translation, ransac_rotation, ransac_scale)) {
      return 0;
    }

    // Refine alignment with ICP
    if (icp_translation || icp_rotation || icp_scale) {
      converged = ICPAlignmentTransformation(
        mesh1, points1, npoints1, 
        mesh2, points2, npoints2, 
        ransac_affine12, ransac_affine21,
        icp_translation, icp_rotation, icp_scale);
    }

    // Compute RMSD
    RNScalar ransac_rmsd = RMSD(
      mesh1, points1, npoints1, 
      mesh2, points2, npoints2, 
      ransac_affine12, ransac_affine21);

    // Check if best so far
    if (ransac_rmsd < rmsd) {
      affine12 = ransac_affine12;
      affine21 = ransac_affine21;
      rmsd = ransac_rmsd;
    }
  }

  // Apply transformation
  mesh1->Transform(affine12);

  // Print statistics
  if (print_verbose) {
    const R4Matrix& m = affine12.Matrix();
    printf("Computed alignment transformation ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Matrix[0][0-3] = %g %g %g %g\n", m[0][0], m[0][1], m[0][2], m[0][3]);
    printf("  Matrix[1][0-3] = %g %g %g %g\n", m[1][0], m[1][1], m[1][2], m[1][3]);
    printf("  Matrix[2][0-3] = %g %g %g %g\n", m[2][0], m[2][1], m[2][2], m[2][3]);
    printf("  Matrix[3][0-3] = %g %g %g %g\n", m[3][0], m[3][1], m[3][2], m[3][3]);
    printf("  Scale = %g\n", affine12.ScaleFactor());
    printf("  Converged = %d\n", converged);
    printf("  RMSD = %g\n", rmsd);
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
Align(R3Mesh *mesh1, int translation, int rotation, int scale)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Compute transform to align mesh1 to canonical coordinate system
  R3Affine affine = mesh1->PCANormalizationTransformation(translation, (rotation==1), scale);

  // Apply transformation
  mesh1->Transform(affine);

  // Print statistics
  if (print_verbose) {
    const R4Matrix& m = affine.Matrix();
    printf("Computed alignment transformation with PCA ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Matrix[0][0-3] = %g %g %g %g\n", m[0][0], m[0][1], m[0][2], m[0][3]);
    printf("  Matrix[1][0-3] = %g %g %g %g\n", m[1][0], m[1][1], m[1][2], m[1][3]);
    printf("  Matrix[2][0-3] = %g %g %g %g\n", m[2][0], m[2][1], m[2][2], m[2][3]);
    printf("  Matrix[3][0-3] = %g %g %g %g\n", m[3][0], m[3][1], m[3][2], m[3][3]);
    printf("  Scale = %g\n", affine.ScaleFactor());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static R3Mesh *
ReadMesh(const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate mesh
  R3Mesh *mesh = new R3Mesh();
  assert(mesh);

  // Read mesh from file
  if (!mesh->ReadFile(filename)) {
    delete mesh;
    return NULL;
  }

  // Check if mesh is valid
  assert(mesh->IsValid());

  // Print statistics
  if (print_verbose) {
    printf("Read mesh from %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Faces = %d\n", mesh->NFaces());
    printf("  # Edges = %d\n", mesh->NEdges());
    printf("  # Vertices = %d\n", mesh->NVertices());
    fflush(stdout);
  }

  // Return mesh
  return mesh;
}


static int
WriteMesh(R3Mesh *mesh, const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Write mesh to file
  if (!mesh->WriteFile(filename)) {
    return 0;
  }

  // Print statistics
  if (print_verbose) {
    printf("Wrote mesh to %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Faces = %d\n", mesh->NFaces());
    printf("  # Edges = %d\n", mesh->NEdges());
    printf("  # Vertices = %d\n", mesh->NVertices());
    fflush(stdout);
  }

  // Return succes
  return 1;
}


static int 
ParseArgs(int argc, char **argv)
{
  // Check number of arguments
  if (argc == 1) {
    printf("Usage: mshalign input1 [input2] output1 [options]\n");
    exit(0);
  }

  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) print_verbose = 1;
      else if (!strcmp(*argv, "-debug")) print_debug = 1;
      else if (!strcmp(*argv, "-weights1")) { argc--; argv++; weights1_name = *argv; }
      else if (!strcmp(*argv, "-weights2")) { argc--; argv++; weights2_name = *argv; }
      else if (!strcmp(*argv, "-pca")) { pca_translation = pca_rotation = pca_scale = 1; }
      else if (!strcmp(*argv, "-pca_translation")) { pca_translation = 1; }
      else if (!strcmp(*argv, "-pca_rotation")) { pca_rotation = 1; }
      else if (!strcmp(*argv, "-pca_scale")) { pca_scale = 1; }
      else if (!strcmp(*argv, "-ransac")) { ransac_translation = ransac_rotation = ransac_scale = 1; }
      else if (!strcmp(*argv, "-ransac_translation")) { ransac_translation = 1; }
      else if (!strcmp(*argv, "-ransac_rotation")) { ransac_rotation = 1; }
      else if (!strcmp(*argv, "-ransac_scale")) { ransac_scale = 1; }
      else if (!strcmp(*argv, "-icp")) { icp_translation = icp_rotation = icp_scale = 1; }
      else if (!strcmp(*argv, "-icp_translation")) { icp_translation = 1; }
      else if (!strcmp(*argv, "-icp_rotation")) { icp_rotation = 1; }
      else if (!strcmp(*argv, "-icp_scale")) { icp_scale = 1; }
      else if (!strcmp(*argv, "-no_pca")) { pca_translation = pca_rotation = pca_scale = 0; }
      else if (!strcmp(*argv, "-no_icp")) { icp_translation = icp_rotation = icp_scale = 0; }
      else if (!strcmp(*argv, "-no_ransac")) { ransac_translation = ransac_rotation = ransac_scale = 0; }
      else if (!strcmp(*argv, "-no_translation")) { pca_translation = 0; icp_translation = 0; ransac_translation = 0; }
      else if (!strcmp(*argv, "-no_rotation")) { pca_rotation = 0; icp_rotation = 0; ransac_rotation = 0; }
      else if (!strcmp(*argv, "-no_scale")) { pca_scale = 0; icp_scale = 0; ransac_scale = 0; }
      else if (!strcmp(*argv, "-axial_rotation")) { pca_rotation = 2; icp_rotation = 0; ransac_rotation = 0; }
      else if (!strcmp(*argv, "-max_points")) { argc--; argv++; max_points = atoi(*argv); }
      else if (!strcmp(*argv, "-min_weight")) { argc--; argv++; min_weight = atof(*argv); } 
      else if (!strcmp(*argv, "-correspondence_method")) { argc--; argv++; correspondence_method = atoi(*argv); }
      else if (!strcmp(*argv, "-sample_method")) { argc--; argv++; sample_method = atoi(*argv); }
      else if (!strcmp(*argv, "-sample_faces")) sample_method = 0; 
      else if (!strcmp(*argv, "-sample_edges")) sample_method = 1; 
      else if (!strcmp(*argv, "-sample_vertices")) sample_method = 2; 
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
    else {
      if (!input1_name) input1_name = *argv;
      else if (!input2_name) input2_name = *argv;
      else if (!output1_name) output1_name = *argv;
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
  }

  // Fix alignment method (default is pca + icp)
  if (!pca_translation && !pca_rotation && !pca_scale && 
      !ransac_translation && !ransac_rotation && !ransac_scale && 
      !icp_translation && !icp_rotation && !icp_scale) {
    pca_translation = pca_rotation = pca_scale = 1;
    icp_translation = icp_rotation = icp_scale = 1;
  }

  // Fix output filename
  if (input2_name && !output1_name) {
    output1_name = input2_name;
    input2_name = NULL;
  }

  // Check input filename
  if (!input1_name || !output1_name) {
    printf("Usage: mshalign input1 [input2] output1 [options]\n");
    return 0;
  }

  // Return OK status 
  return 1;
}



int main(int argc, char **argv)
{
  // Check number of arguments
  if (!ParseArgs(argc, argv)) exit(1);

  // Read mesh1
  R3Mesh *mesh1 = ReadMesh(input1_name);
  if (!mesh1) exit(-1);

  // Read weights1
  int nweights1 = 0;
  RNScalar *weights1 = NULL;
  if (weights1_name) {
    if (!ReadWeights(weights1_name, weights1, nweights1)) exit(-1);
  }

  // Check if aligning to second model
  if (input2_name) {
    // Read mesh2
    R3Mesh *mesh2 = ReadMesh(input2_name);
    if (!mesh2) exit(-1);

    // Read weights2
    int nweights2 = 0;
    RNScalar *weights2 = NULL;
    if (weights2_name) {
      if (!ReadWeights(weights2_name, weights2, nweights2)) exit(-1);
    }

    // Align mesh1 to mesh2
    if (!Align(mesh1, mesh2, 
      weights1, nweights1, weights2, nweights2,
      pca_translation, pca_rotation, pca_scale, 
      ransac_translation, ransac_rotation, ransac_scale, 
      icp_translation, icp_rotation, icp_scale,
      sample_method)) 
      exit(-1);
  }
  else {
    // Align mesh1 to canonical coordinate system
    if (!Align(mesh1, pca_translation, pca_rotation, pca_scale)) exit(-1);
  }

  // Output mesh1
  if (!WriteMesh(mesh1, output1_name)) exit(-1);

  // Return success 
  return 0;
}

















