// Program to generate points on surface of mesh



// Include files

#include "R3Shapes/R3Shapes.h"



// Constant definitions

enum {
  ITERATIVE_FURTHEST_VERTEX,  
  RANDOM_SURFACE_POINTS,
  RANDOM_VERTICES,
  SCALE_SPACE_MAXIMA,
  SCALE_SPACE_MINIMA,
  SCALE_SPACE_EXTREMA,
  PROPERTY_MAXIMA,
  PROPERTY_MINIMA,
  PROPERTY_EXTREMA,
  CENTER_OF_MASS,
  VERTEX_CLOSEST_TO_CENTER_OF_MASS,
  NUM_SELECTION_METHODS
};



// Program arguments

static char *mesh_name  = NULL;
static char *points_name = NULL;
static int selection_method = RANDOM_SURFACE_POINTS;
static int min_points = -1;
static int max_points = -1;
static int npoints = -1;
static double min_spacing = -1;
static double min_normalized_spacing = -1;
static double min_relative_spacing = -1;
static char *property_name  = NULL;
static RNBoolean print_verbose = FALSE;
static RNBoolean print_debug = FALSE;



// Type definitions

struct Point {
  Point(void) 
    : vertex(NULL), position(0,0,0), normal(0,0,0) {};
  Point(const R3Point& position, const R3Vector& normal) 
    : vertex(NULL), position(position), normal(normal) {};
  Point(R3Mesh *mesh, R3MeshVertex *vertex)
    : vertex(vertex), position(mesh->VertexPosition(vertex)), normal(mesh->VertexNormal(vertex)) {};
  R3MeshVertex *vertex;
  R3Point position;
  R3Vector normal;
};



////////////////////////////////////////////////////////////////////////

static R3Mesh *
ReadMesh(char *mesh_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate mesh
  R3Mesh *mesh = new R3Mesh();
  assert(mesh);

  // Read mesh from file
  if (!mesh->ReadFile(mesh_name)) {
    delete mesh;
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("Read mesh ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Faces = %d\n", mesh->NFaces());
    printf("  # Edges = %d\n", mesh->NEdges());
    printf("  # Vertices = %d\n", mesh->NVertices());
    fflush(stdout);
  }

  // Return success
  return mesh;
}



////////////////////////////////////////////////////////////////////////

static RNBoolean 
IsLocalExtremum(R3Mesh *mesh, R3MeshVertex *vertex, R3MeshProperty *property, int extrema_type, RNScalar epsilon = 0)
{
  // Return whether value at vertex is local extremum
  RNBoolean minimum = TRUE;
  RNBoolean maximum = TRUE;
  RNScalar value = property->VertexValue(mesh->VertexID(vertex));
  if (value == RN_UNKNOWN) return FALSE;
  for (int i = 0; i < mesh->VertexValence(vertex); i++) {
    R3MeshEdge *edge = mesh->EdgeOnVertex(vertex, i);
    R3MeshVertex *neighbor_vertex = mesh->VertexAcrossEdge(edge, vertex);
    RNScalar neighbor_value = property->VertexValue(mesh->VertexID(neighbor_vertex));
    if (neighbor_value == RN_UNKNOWN) continue;
    minimum &= (value + epsilon < neighbor_value) ? TRUE : FALSE;
    if ((extrema_type == PROPERTY_MINIMA) && !minimum) return FALSE;
    maximum &= (value - epsilon > neighbor_value) ? TRUE : FALSE;
    if ((extrema_type == PROPERTY_MAXIMA) && !maximum) return FALSE;
  }

  // Vertex value is extremum
  if (extrema_type == PROPERTY_MINIMA) return minimum;
  else if (extrema_type == PROPERTY_MAXIMA) return maximum;
  return minimum || maximum;
}



static RNArray<Point *> *
SelectPropertyExtrema(R3Mesh *mesh, int npoints, double min_spacing, const char *property_name, int extrema_type)
{
  // Read property 
  R3MeshProperty property(mesh);
  if (!property.Read(property_name)) return NULL;

  // Normalize property
  property.Normalize();

  // Select points from successively blurred property until number of extrema is npoints
  int max_blurs = 0;
  RNScalar epsilon = 0;
  RNScalar sigma = 0.01 * sqrt(mesh->Area());
  RNArray<R3MeshVertex *> selected_vertices;
  for (int blur = 0; blur <= max_blurs; blur++) { 
    // Find extremum vertices
    RNArray<R3MeshVertex *> extremum_vertices;
    for (int i = 0; i < mesh->NVertices(); i++) {
      R3MeshVertex *vertex = mesh->Vertex(i);
      if (!IsLocalExtremum(mesh, vertex, &property, extrema_type, epsilon)) continue; 
      extremum_vertices.Insert(vertex);
    }

    // Sort extremum vertices by property value
    for (int i = 0; i < extremum_vertices.NEntries(); i++) {
      for (int j = i+1; j < extremum_vertices.NEntries(); j++) {
        RNScalar value_i = property.VertexValue(extremum_vertices[i]);
        RNScalar value_j = property.VertexValue(extremum_vertices[j]);
        if (extrema_type == PROPERTY_MINIMA) { if (value_i > value_j) extremum_vertices.Swap(i, j); }
        else if (extrema_type == PROPERTY_MAXIMA) { if (value_i < value_j) extremum_vertices.Swap(i, j); }
        else { if (fabs(value_i) < fabs(value_j)) extremum_vertices.Swap(i, j); }
      }
    }

    // Select amongst extremum vertices taking into account min_spacing
    R3mesh_mark++;
    selected_vertices.Empty();
    for (int i = 0; i < extremum_vertices.NEntries(); i++) {
      R3MeshVertex *vertex = extremum_vertices.Kth(i);

      // Check if vertex is marked (i.e., too close to vertex previously selected)
      if (mesh->VertexMark(vertex) == R3mesh_mark) continue;

      // Check if vertex is disconnected completely
      if (mesh->VertexValence(vertex) == 0) continue;
    
      // Select vertex
      selected_vertices.Insert(vertex);

      // Mark nearby vertices
      mesh->SetVertexMark(vertex, R3mesh_mark);
      if (min_spacing > 0) {
        RNLength *distances = mesh->DijkstraDistances(vertex, min_spacing);
        for (int j = 0; j < mesh->NVertices(); j++) {
          R3MeshVertex *nearby_vertex = mesh->Vertex(j);
          if (mesh->VertexMark(nearby_vertex) != R3mesh_mark) {
            if (distances[j] <= min_spacing) {
              mesh->SetVertexMark(nearby_vertex, R3mesh_mark);
            }
          }
        } 
        delete [] distances;
      }
    }

    // Print debug statement
    if (print_debug) printf("  %d : %d %g %g\n", blur, selected_vertices.NEntries(), sigma, property.StandardDeviation());

    // Check if found correct number of points
    if ((npoints <= 0) || (selected_vertices.NEntries() <= npoints)) break;

    // Blur the property and iterate
    if (blur < max_blurs) {
      property.Blur(sigma);
      sigma *= 1.1;
    }
  }

  // For debugging purposes
  if (print_debug) property.Write("foo.val");

  // Allocate array of points
  RNArray<Point *> *points = new RNArray<Point *>();
  if (!points) {
    fprintf(stderr, "Unable to allocate array of points\n");
    return NULL;
  }

  // Create points
  for (int i = 0; (i < selected_vertices.NEntries()) && (i < npoints); i++) {
    R3MeshVertex *vertex = selected_vertices[i];
    Point *point = new Point(mesh, vertex);
    points->Insert(point);
  }

  // Return points
  return points;
}



////////////////////////////////////////////////////////////////////////

struct ScaleSpaceScore {
  int vertex_index;
  double score;
};
  

static int
CompareScaleSpaceScores(const void *s1, const void *s2)
{
  const ScaleSpaceScore *score1 = (const ScaleSpaceScore *) s1;
  const ScaleSpaceScore *score2 = (const ScaleSpaceScore *) s2;
  RNScalar delta = score1->score - score2->score;
  if (delta > 0) return -1;
  else if (delta < 0) return 1;
  else return 0;
}



static RNScalar 
VoteValue(R3Mesh *mesh, R3MeshVertex *vertex, R3MeshProperty *property, int extrema_type)
{
  // Get vertex value
  RNScalar value = property->VertexValue(vertex);
  if (value == RN_UNKNOWN) return 0;

  // Compute statistics of neighbors
  int count = 0;
  RNScalar sum_value = 0;
  RNScalar minimum_value = FLT_MAX;
  RNScalar maximum_value = -FLT_MAX;
  for (int i = 0; i < mesh->VertexValence(vertex); i++) {
    R3MeshEdge *edge = mesh->EdgeOnVertex(vertex, i);
    R3MeshVertex *neighbor_vertex = mesh->VertexAcrossEdge(edge, vertex);
    RNScalar neighbor_value = property->VertexValue(mesh->VertexID(neighbor_vertex));
    if (neighbor_value == RN_UNKNOWN) continue;
    if (neighbor_value < minimum_value) minimum_value = neighbor_value;
    if (neighbor_value > maximum_value) maximum_value = neighbor_value;
    sum_value += neighbor_value;
    count++;
  }

  // Just checking
  if (count == 0) return 0;

  // Return vote value
  RNScalar mean_value = sum_value / count;
  RNScalar laplacian = value - mean_value;
  if ((extrema_type == PROPERTY_MINIMA) && (value >= minimum_value)) return 0;
  if ((extrema_type == PROPERTY_MAXIMA) && (value <= maximum_value)) return 0;
  return fabs(laplacian);
}



static RNArray<Point *> *
SelectScaleSpaceExtrema(R3Mesh *mesh, int npoints, double min_spacing, const char *property_name, int extrema_type)
{
  // Read property 
  R3MeshProperty property(mesh);
  if (!property.Read(property_name)) return NULL;

  // Normalize property
  property.Normalize();

  // Allocate scores
  ScaleSpaceScore *scores = new ScaleSpaceScore [ mesh->NVertices() ];
  for (int i = 0; i < mesh->NVertices(); i++) {
    scores[i].vertex_index = i;
    scores[i].score = 0;
  }

  // Compute scale space scores
  int nblurs = 4;
  RNScalar weight = 1;
  RNScalar sigma = 0.01 * sqrt(mesh->Area());
  for (int blur = 0; blur < nblurs; blur++) { 
    // Compute strength of every vertex
    R3MeshProperty vote(property);
    for (int i = 0; i < mesh->NVertices(); i++) {
      R3MeshVertex *vertex = mesh->Vertex(i);
      RNScalar value = VoteValue(mesh, vertex, &property, extrema_type);
      vote.SetVertexValue(i, value); 
    }

    // Vote for vertex with its normalized strength value (augmented if local extremum)
    for (int i = 0; i < mesh->NVertices(); i++) {
      scores[i].score += weight * vote.VertexValue(i);
    }

    // Blur the property and iterate
    property.Blur(sigma);
    // weight *= 2;
    // sigma *= 2;
  }


  if (print_debug) {
    property.Write("property.val");
    R3MeshProperty foo(mesh, "ScaleSpaceScore");
    for (int i = 0; i < mesh->NVertices(); i++) foo.SetVertexValue(i, scores[i].score);
    foo.Write("scores.val");
  }

  // Sort vertices by score
  qsort(scores, mesh->NVertices(), sizeof(ScaleSpaceScore), CompareScaleSpaceScores);

  // Select vertices taking into account min_spacing
  R3mesh_mark++;
  RNArray<R3MeshVertex *> selected_vertices;
  for (int i = 0; i < mesh->NVertices(); i++) {
    R3MeshVertex *vertex = mesh->Vertex(scores[i].vertex_index);
    
    // Check if vertex is marked (i.e., too close to vertex previously selected)
    if (mesh->VertexMark(vertex) == R3mesh_mark) continue;

    // Check if vertex is disconnected completely
    if (mesh->VertexValence(vertex) == 0) continue;
    
    // Select vertex
    selected_vertices.Insert(vertex);
    
    // Mark nearby vertices
    mesh->SetVertexMark(vertex, R3mesh_mark);
    if (min_spacing > 0) {
      RNLength *distances = mesh->DijkstraDistances(vertex, min_spacing);
      for (int j = 0; j < mesh->NVertices(); j++) {
        R3MeshVertex *nearby_vertex = mesh->Vertex(j);
        if (mesh->VertexMark(nearby_vertex) != R3mesh_mark) {
          if (distances[j] <= min_spacing) {
            mesh->SetVertexMark(nearby_vertex, R3mesh_mark);
          }
        }
      } 
      delete [] distances;
    }

    // Check if found all points
    if ((npoints > 0) && (selected_vertices.NEntries() >= npoints)) break;
  }

  // Delete scores
  delete [] scores;

  // Allocate array of points
  RNArray<Point *> *points = new RNArray<Point *>();
  if (!points) {
    fprintf(stderr, "Unable to allocate array of points\n");
    return NULL;
  }

  // Create points
  for (int i = 0; i < selected_vertices.NEntries(); i++) {
    R3MeshVertex *vertex = selected_vertices[i];
    Point *point = new Point(mesh, vertex);
    points->Insert(point);
  }

  // Return points
  return points;
}



////////////////////////////////////////////////////////////////////////

static RNArray<Point *> *
SelectSurfacePoints(R3Mesh *mesh, int npoints, double min_spacing)
{
  // Allocate array of points
  RNArray<Point *> *points = new RNArray<Point *>();
  if (!points) {
    fprintf(stderr, "Unable to allocate array of points\n");
    return NULL;
  }

  // XXX THIS IGNORES MIN_SPACING XXX

  // Count total value of faces
  RNArea total_value = 0.0;
  for (int i = 0; i < mesh->NFaces(); i++) {
    R3MeshFace *face = mesh->Face(i);
    RNScalar face_value = 0;
    face_value = mesh->FaceArea(face);
    mesh->SetFaceValue(face, face_value);
    total_value += face_value;
  }
    
  // Generate points
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
    const R3Vector& n0 = mesh->VertexNormal(v0);
    const R3Vector& n1 = mesh->VertexNormal(v1);
    const R3Vector& n2 = mesh->VertexNormal(v2);

    // Determine number of points for face 
    RNScalar ideal_face_npoints = npoints * mesh->FaceValue(face) / total_value;
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
      R3Point position = t0*p0 + t1*p1 + t2*p2;
      R3Vector normal = t0*n0 + t1*n1 + t2*n2; normal.Normalize();
      Point *point = new Point(position, normal);
      points->Insert(point);
    }
  }

  // Return points
  return points;
}



////////////////////////////////////////////////////////////////////////

// Type definitions

struct VertexData {
  R3MeshVertex *vertex;
  double distance;
  R3MeshVertex *closest_seed;
  VertexData **heappointer;
};



static R3MeshVertex *
FindFurthestVertex(R3Mesh *mesh, const RNArray<R3MeshVertex *>& seeds, VertexData *vertex_data)
{
  // Initialize vertex data
  for (int i = 0; i < mesh->NVertices(); i++) {
    VertexData *data = &vertex_data[i];
    data->distance = FLT_MAX;
    data->closest_seed = NULL;
    data->heappointer = NULL;
  }

  // Initialize heap
  VertexData tmp;
  RNHeap<VertexData *> heap(&tmp, &(tmp.distance), &(tmp.heappointer));
  for (int i = 0; i < seeds.NEntries(); i++) {
    R3MeshVertex *vertex = seeds.Kth(i);
    VertexData *data = &vertex_data[mesh->VertexID(vertex)];
    data->closest_seed = vertex;
    data->distance = 0;
    heap.Push(data);
  }

  // Iteratively pop off heap until find furthest vertex
  R3MeshVertex *vertex = NULL;
  while (!heap.IsEmpty()) {
    VertexData *data = heap.Pop();
    vertex = data->vertex;
    for (int j = 0; j < mesh->VertexValence(vertex); j++) {
      R3MeshEdge *edge = mesh->EdgeOnVertex(vertex, j);
      R3MeshVertex *neighbor_vertex = mesh->VertexAcrossEdge(edge, vertex);
      VertexData *neighbor_data = (VertexData *) mesh->VertexData(neighbor_vertex);
      RNScalar old_distance = neighbor_data->distance;
      RNScalar new_distance = mesh->EdgeLength(edge) + data->distance;
      if (new_distance < old_distance) {
        neighbor_data->distance = new_distance;
        neighbor_data->closest_seed = data->closest_seed;
        if (old_distance < FLT_MAX) heap.Update(neighbor_data);
        else heap.Push(neighbor_data);
      }
    }
  }

  // Return furthest vertex
  return vertex;
}



static RNArray<Point *> *
SelectFurthestPoints(R3Mesh *mesh, const RNArray<R3MeshVertex *>& seeds, int npoints, double min_spacing)
{
  // Check/adjust number of points
  if (npoints > mesh->NVertices()) npoints = mesh->NVertices();

  // Allocate array of points
  RNArray<Point *> *points = new RNArray<Point *>();
  if (!points) {
    fprintf(stderr, "Unable to allocate array of points\n");
    return NULL;
  }

  // Allocate vertex data
  VertexData *vertex_data = new VertexData [ mesh->NVertices() ];
  if (!vertex_data) {
    fprintf(stderr, "Unable to allocate vertex data\n");
    return NULL;
  }

  // Initialize vertex data
  for (int i = 0; i < mesh->NVertices(); i++) {
    R3MeshVertex *vertex = mesh->Vertex(i);
    VertexData *data = &vertex_data[i];
    mesh->SetVertexData(vertex, data);
    data->vertex = vertex;
    data->distance = FLT_MAX;
    data->closest_seed = NULL;
    data->heappointer = NULL;
  }

  // Copy seeds 
  RNArray<R3MeshVertex *> vertices;
  for (int i = 0; i < seeds.NEntries(); i++) {
    R3MeshVertex *vertex = seeds.Kth(i);
    Point *point = new Point(mesh, vertex);
    points->Insert(point);
    vertices.Insert(vertex);
  }

  // Create random starting vertex in each connected component, if none provided
  RNArray<R3MeshVertex *> tmp;
  if (seeds.IsEmpty()) {
    int *marks = new int [ mesh->NVertices() ];
    for (int i = 0; i < mesh->NVertices(); i++) marks[i] = 0;
    for (int i = 0; i < mesh->NVertices(); i++) {
      R3MeshVertex *vertex = mesh->Vertex(i);

      // Check if already marked
      if (marks[i] != 0) continue;
      marks[i] = 1;

      // Check if completely disconnected
      if (mesh->VertexValence(vertex) == 0) continue;

      // Select vertex as temporary seed
      vertices.Insert(vertex);
      tmp.Insert(vertex);

      // Mark connected component
      RNArray<R3MeshVertex *> connected_component;
      mesh->FindConnectedVertices(vertex, connected_component);
      for (int j = 0; j < connected_component.NEntries(); j++) {
        R3MeshVertex *connected_vertex = connected_component.Kth(j);
        marks[mesh->VertexID(connected_vertex)] = 1;
      }
    }
  }

  // Iteratively find furthest vertex
  while (vertices.NEntries() - tmp.NEntries() < npoints) {
    R3MeshVertex *vertex = FindFurthestVertex(mesh, vertices, vertex_data);
    if (!vertex) break;
    VertexData *data = &vertex_data[mesh->VertexID(vertex)];
    if ((min_spacing > 0) && (data->distance < min_spacing)) break;
    if (tmp.FindEntry(data->closest_seed)) { 
      tmp.Remove(data->closest_seed); 
      vertices.Remove(data->closest_seed);
    }
    Point *point = new Point(mesh, vertex);
    points->Insert(point);
    vertices.Insert(vertex);
  }

  // Delete vertex data
  delete [] vertex_data;

  // Return points
  return points;
}



static RNArray<Point *> *
SelectFurthestPoints(R3Mesh *mesh, const RNArray<Point *>& seeds, int npoints, double min_spacing)
{
  // Seeed with vertices closest to points
  RNArray<R3MeshVertex *> seed_vertices;
  for (int i = 0; i < seeds.NEntries(); i++) {
    R3MeshVertex *vertex = seeds[i]->vertex;
    if (vertex) seed_vertices.Insert(vertex);
    else { fprintf(stderr, "Not implemented\n"); return NULL; }
  }

  // Select furthest points
  return SelectFurthestPoints(mesh, seed_vertices, npoints, min_spacing);
} 



static RNArray<Point *> *
SelectFurthestPoints(R3Mesh *mesh, int npoints, double min_spacing)
{
  // Select furthest points from scratch
  RNArray<R3MeshVertex *> seeds;
  return SelectFurthestPoints(mesh, seeds, npoints, min_spacing);
}



////////////////////////////////////////////////////////////////////////

static RNArray<Point *> *
SelectVertexPoints(R3Mesh *mesh, int npoints, double min_spacing)
{
  // Check number of points
  if (npoints > mesh->NVertices()) npoints = mesh->NVertices();

  // Allocate array of points
  RNArray<Point *> *points = new RNArray<Point *>();
  if (!points) {
    fprintf(stderr, "Unable to allocate array of points\n");
    return NULL;
  }

  // Set vertex values = surface area associated with vertex
  RNScalar total_value = 0;
  for (int i = 0; i < mesh->NVertices(); i++) {
    R3MeshVertex *vertex = mesh->Vertex(i);
    RNScalar value = mesh->VertexArea(vertex);
    mesh->SetVertexValue(vertex, value);
    total_value += value;
  }

  // Select vertices weighted by value
  R3mesh_mark++;
  int max_iterations = 100 * npoints;
  for (int i = 0; (npoints > 0) && (i < max_iterations); i++) {
    RNScalar cumulative_value = 0;
    RNScalar target_value = RNRandomScalar() * total_value;
    for (int j = 0; j < mesh->NVertices(); j++) {
      R3MeshVertex *vertex = mesh->Vertex(j);
      if (mesh->VertexMark(vertex) != R3mesh_mark) {
        RNScalar vertex_value = mesh->VertexValue(vertex);
        cumulative_value += vertex_value;
        if ((cumulative_value >= target_value) || (j == mesh->NVertices()-1)) {
          // Insert point
          Point *point = new Point(mesh, vertex);
          points->Insert(point);
          npoints--;

          // Mark point 
          mesh->SetVertexMark(vertex, R3mesh_mark);
          total_value -= vertex_value;

          // Mark nearby points
          if (min_spacing > 0) {
            RNLength *distances = mesh->DijkstraDistances(vertex, min_spacing);
            for (int k = 0; k < mesh->NVertices(); k++) {
              R3MeshVertex *nearby_vertex = mesh->Vertex(k);
              if (mesh->VertexMark(nearby_vertex) != R3mesh_mark) {
                if (distances[k] <= min_spacing) {
                  // Mark point within min_spacing
                  mesh->SetVertexMark(nearby_vertex, R3mesh_mark);
                  total_value -= vertex_value;
                }
              }
            }
            delete [] distances;
          }

          // Break
          break;
        }
      }
    }
  }

  // Return points
  return points;
}



////////////////////////////////////////////////////////////////////////

static RNArray<Point *> *
SelectCenterOfMass(R3Mesh *mesh)
{
  // Allocate array of points
  RNArray<Point *> *points = new RNArray<Point *>();
  if (!points) {
    fprintf(stderr, "Unable to allocate array of points\n");
    return NULL;
  }

  // Compute mesh center of mass
  R3Point mesh_centroid = mesh->Centroid();

  // Insert mesh centroid
  Point *point = new Point(mesh_centroid, R3zero_vector);
  points->Insert(point);

  // Return points
  return points;
}



////////////////////////////////////////////////////////////////////////

static RNArray<Point *> *
SelectVertexClosestToCenterOfMass(R3Mesh *mesh)
{
  // Allocate array of points
  RNArray<Point *> *points = new RNArray<Point *>();
  if (!points) {
    fprintf(stderr, "Unable to allocate array of points\n");
    return NULL;
  }

  // Compute mesh center of mass
  R3Point mesh_centroid = mesh->Centroid();

  // Find closest vertex 
  RNScalar closest_d = FLT_MAX;
  R3MeshVertex *closest_vertex = NULL;
  for (int i = 0; i < mesh->NVertices(); i++) {
    R3MeshVertex *vertex = mesh->Vertex(i);
    const R3Point& position = mesh->VertexPosition(vertex);
    RNScalar d = R3Distance(position, mesh_centroid);
    if (d < closest_d) {
      closest_vertex = vertex;
      closest_d = d;
    }
  }

  // Insert closest vertex
  assert(closest_vertex);
  Point *point = new Point(mesh, closest_vertex);
  points->Insert(point);

  // Return points
  return points;
}



////////////////////////////////////////////////////////////////////////

static RNArray<Point *> *
SelectPoints(R3Mesh *mesh, int selection_method)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Update min spacing 
  if ((min_spacing < 0) && (min_normalized_spacing > 0)) {
    min_spacing = min_normalized_spacing * sqrt(mesh->Area());
  }

  // Update number of points
  if (npoints <= 0) {
    if (min_spacing > 0) npoints = mesh->NVertices();
    else npoints = 16;
  }

  // Update min spacing 
  if ((npoints > 0) && (min_spacing <= 0) && (min_relative_spacing > 0)) {
    min_spacing = min_relative_spacing * 2 * sqrt(mesh->Area() / (RN_PI * npoints));
  }

  // Select points using requested selection method 
  RNArray<Point *> *points = NULL;
  if (selection_method == RANDOM_VERTICES) 
    points = SelectVertexPoints(mesh, npoints, min_spacing);
  else if (selection_method == RANDOM_SURFACE_POINTS) 
    points= SelectSurfacePoints(mesh, npoints, min_spacing);
  else if (selection_method == ITERATIVE_FURTHEST_VERTEX) 
    points= SelectFurthestPoints(mesh, npoints, min_spacing);
  else if (selection_method == PROPERTY_MINIMA) 
    points= SelectPropertyExtrema(mesh, npoints, min_spacing, property_name, PROPERTY_MINIMA);
  else if (selection_method == PROPERTY_MAXIMA) 
    points= SelectPropertyExtrema(mesh, npoints, min_spacing, property_name, PROPERTY_MAXIMA);
  else if (selection_method == PROPERTY_EXTREMA)
    points= SelectPropertyExtrema(mesh, npoints, min_spacing, property_name, PROPERTY_EXTREMA);
  else if (selection_method == SCALE_SPACE_MINIMA) 
    points= SelectScaleSpaceExtrema(mesh, npoints, min_spacing, property_name, PROPERTY_MINIMA);
  else if (selection_method == SCALE_SPACE_MAXIMA) 
    points= SelectScaleSpaceExtrema(mesh, npoints, min_spacing, property_name, PROPERTY_MAXIMA);
  else if (selection_method == SCALE_SPACE_EXTREMA)
    points= SelectScaleSpaceExtrema(mesh, npoints, min_spacing, property_name, PROPERTY_EXTREMA);
  else if (selection_method == CENTER_OF_MASS) 
    points= SelectCenterOfMass(mesh);
  else if (selection_method == VERTEX_CLOSEST_TO_CENTER_OF_MASS) 
    points= SelectVertexClosestToCenterOfMass(mesh);

  // Check points
  if (!points) {
    fprintf(stderr, "Error selecting points\n");
    return 0;
  }

  // Check if too few points
  if ((min_points > 0) && (points->NEntries() < min_points)) {
    points = SelectFurthestPoints(mesh, *points, npoints, min_spacing);
  }

  // Check if too many points
  if ((max_points > 0) && (points->NEntries() > max_points)) {
    points->Truncate(max_points);
  }

  // Print message
  if (print_verbose) {
    fprintf(stdout, "Generated points ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Points = %d\n", points->NEntries());
    printf("  Selection method = %d\n", selection_method);
    if (min_points > 0) printf("  Minimum # points = %d\n", min_points);
    if (max_points > 0) printf("  Maximum # points = %d\n", max_points);
    printf("  Requested # points = %d\n", npoints);
    printf("  Requested min spacing = %g\n", min_spacing);
    fflush(stdout);
  }

  // Return array of points
  return points;
}



////////////////////////////////////////////////////////////////////////

RNBoolean 
WritePoints(R3Mesh *mesh, const RNArray<Point *>& points, const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Parse input filename extension
  const char *extension;
  if (!(extension = strrchr(filename, '.'))) {
    printf("Filename %s has no extension (e.g., .ply)\n", filename);
    return 0;
  }

  // Check file type
  if (!strcmp(extension, ".pid")) {
    // Open file
    FILE *fp = stdout;
    if (filename) {
      fp = fopen(filename, "w");
      if (!fp) {
        fprintf(stderr, "Unable to open output file %s\n", filename);
        return FALSE;
      }
    }

    // Write vertex ids
    for (int i = 0; i < points.NEntries(); i++) {
      Point *point = points[i];
      if (!point->vertex) { fprintf(stderr, "Incompatible output type: pid\n"); return 0; }
      int vertex_id = (point->vertex) ? mesh->VertexID(point->vertex) : -1;
      fprintf(fp, "%d\n", vertex_id);
    }

    // Close file
    if (filename) fclose(fp);
  }
  else if (!strcmp(extension, ".vts")) {
    // Open file
    FILE *fp = stdout;
    if (filename) {
      fp = fopen(filename, "w");
      if (!fp) {
        fprintf(stderr, "Unable to open output file %s\n", filename);
        return FALSE;
      }
    }

    // Write points
    for (int i = 0; i < points.NEntries(); i++) {
      Point *point = points[i];
      int vertex_id = (point->vertex) ? mesh->VertexID(point->vertex) : -1;
      const R3Point& position = point->position;
      fprintf(fp, "%d %g %g %g\n", vertex_id, position.X(), position.Y(), position.Z());
    }

    // Close file
    if (filename) fclose(fp);
  }
  else if (!strcmp(extension, ".xyzn")) {
    // Open file
    FILE *fp = stdout;
    if (filename) {
      fp = fopen(filename, "w");
      if (!fp) {
        fprintf(stderr, "Unable to open output file %s\n", filename);
        return FALSE;
      }
    }

    // Write points
    for (int i = 0; i < points.NEntries(); i++) {
      Point *point = points[i];
      fprintf(fp, "%g %g %g   %g %g %g\n", 
        point->position.X(), point->position.Y(), point->position.Z(),
        point->normal.X(), point->normal.Y(), point->normal.Z());
    }

    // Close file
    if (filename) fclose(fp);
  }
  else if (!strcmp(extension, ".xyz")) {
    // Open file
    FILE *fp = stdout;
    if (filename) {
      fp = fopen(filename, "w");
      if (!fp) {
        fprintf(stderr, "Unable to open output file %s\n", filename);
        return FALSE;
      }
    }

    // Write points
    for (int i = 0; i < points.NEntries(); i++) {
      Point *point = points[i];
      fprintf(fp, "%g %g %g\n", point->position.X(), point->position.Y(), point->position.Z());
    }

    // Close file
    if (filename) fclose(fp);
  }
  else {
    // Open file
    FILE *fp = stdout;
    if (filename) {
      fp = fopen(filename, "wb");
      if (!fp) {
        fprintf(stderr, "Unable to open output file %s\n", filename);
        return FALSE;
      }
    }

    // Write points
    float coordinates[6];
    for (int i = 0; i < points.NEntries(); i++) {
      Point *point = points[i];
      coordinates[0] = point->position.X();
      coordinates[1] = point->position.Y();
      coordinates[2] = point->position.Z();
      coordinates[3] = point->normal.X();
      coordinates[4] = point->normal.Y();
      coordinates[5] = point->normal.Z();
      if (fwrite(coordinates, sizeof(float), 6, fp) != (unsigned int) 6) {
        fprintf(stderr, "Unable to write point to output file %s\n", filename);
        return FALSE;
      }
    }

    // Close file
    if (filename) fclose(fp);
  }

  // Print message
  if (print_verbose) {
    fprintf(stdout, "Wrote points to %s ...\n", (filename) ? filename : "stdout");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    fflush(stdout);
  }

  // Return success
  return TRUE;
}



////////////////////////////////////////////////////////////////////////

int
ParseArgs (int argc, char **argv)
{
  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-npoints")) { argv++; argc--; npoints = atoi(*argv); }
      else if (!strcmp(*argv, "-min_points")) { argv++; argc--; min_points = atoi(*argv); }
      else if (!strcmp(*argv, "-max_points")) { argv++; argc--; max_points = atoi(*argv); }
      else if (!strcmp(*argv, "-min_spacing")) { argc--; argv++; min_spacing = atof(*argv); }
      else if (!strcmp(*argv, "-min_normalized_spacing")) { argc--; argv++; min_normalized_spacing = atof(*argv); }
      else if (!strcmp(*argv, "-min_relative_spacing")) { argc--; argv++; min_relative_spacing = atof(*argv); }
      else if (!strcmp(*argv, "-selection_method")) { argc--; argv++; selection_method = atoi(*argv); }
      else if (!strcmp(*argv, "-random_surface_points")) { selection_method = RANDOM_SURFACE_POINTS; }
      else if (!strcmp(*argv, "-random_vertices")) { selection_method = RANDOM_VERTICES; }
      else if (!strcmp(*argv, "-iterative_furthest_vertex")) { selection_method = ITERATIVE_FURTHEST_VERTEX; }
      else if (!strcmp(*argv, "-center_of_mass")) { selection_method = CENTER_OF_MASS; }
      else if (!strcmp(*argv, "-vertex_closest_to_center_of_mass")) { selection_method = VERTEX_CLOSEST_TO_CENTER_OF_MASS; }
      else if (!strcmp(*argv, "-property_minima")) { argc--; argv++;  property_name = *argv; selection_method = PROPERTY_MINIMA; }
      else if (!strcmp(*argv, "-property_maxima")) { argc--; argv++;  property_name = *argv; selection_method = PROPERTY_MAXIMA; }
      else if (!strcmp(*argv, "-property_extrema")) { argc--; argv++;  property_name = *argv; selection_method = PROPERTY_EXTREMA; }
      else if (!strcmp(*argv, "-scale_space_minima")) { argc--; argv++;  property_name = *argv; selection_method = SCALE_SPACE_MINIMA; }
      else if (!strcmp(*argv, "-scale_space_maxima")) { argc--; argv++;  property_name = *argv; selection_method = SCALE_SPACE_MAXIMA; }
      else if (!strcmp(*argv, "-scale_space_extrema")) { argc--; argv++;  property_name = *argv; selection_method = SCALE_SPACE_EXTREMA; }
      else if (!strcmp(*argv, "-v")) { print_verbose = TRUE; }
      else if (!strcmp(*argv, "-debug")) { print_debug = TRUE; }
      else { fprintf(stderr, "Invalid program argument: %s\n", *argv); return 0; }
      argv++; argc--;
    }
    else {
      if (!mesh_name) mesh_name = *argv;
      else if (!points_name) points_name = *argv;
      else RNAbort("Invalid program argument: %s", *argv);
      argv++; argc--;
    }
  }

  // Check mesh name
  if (!mesh_name) {
    // Print usage statement
    fprintf(stderr, 
      "Usage: off2pts <options> input_mesh output_points\n\n"
      "options:\n"
      " -npoints #\n"      
      " -min_spacing #\n"      
      "\n");
    return FALSE;
  }

  // Return success
  return TRUE;
}



////////////////////////////////////////////////////////////////////////

int
main(int argc, char **argv)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Parse args
  if(!ParseArgs(argc, argv)) exit(-1);

  // Read the mesh
  R3Mesh *mesh = ReadMesh(mesh_name);
  if (!mesh) exit(-1);

  // Select points
  RNArray<Point *> *points = SelectPoints(mesh, selection_method);
  if (!points) exit(-1);

  // Write points
  if (!WritePoints(mesh, *points, points_name)) exit(-1);

  // Print message
  if (print_verbose) {
    fprintf(stdout, "Finished in %6.3f seconds.\n", start_time.Elapsed());
    fflush(stdout);
  }

  // Return success
  return 0;
}








