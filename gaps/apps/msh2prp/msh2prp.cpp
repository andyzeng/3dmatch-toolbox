// Source file for the GAPS mesh analysis program



////////////////////////////////////////////////////////////////////////
// Include files 
////////////////////////////////////////////////////////////////////////

#include "R3Shapes/R3Shapes.h"



////////////////////////////////////////////////////////////////////////
// Program arguments
////////////////////////////////////////////////////////////////////////

static char *input_mesh_name = NULL;
static char *output_properties_name = NULL;
static int compute_basic_properties = 0;
static int compute_coordinate_properties = 0;
static int compute_curvature_properties = 0;
static int compute_volume_properties = 0;
static int compute_boundary_properties = 0;
static int compute_laplacian_properties = 0;
static int compute_dijkstra_distance_properties = 0;
static int compute_dijkstra_histogram_properties = 0;
static int compute_raytrace_properties = 0;
static int compute_solidtexture_properties = 0;
static char *input_mturk_segmentation_name = NULL;
static char *input_mturk_annotation_name = NULL;
static char *input_mturk_label_mapping_name = NULL;
static char *input_map_name = NULL;
static int print_verbose = 0;
static int print_debug = 0;



////////////////////////////////////////////////////////////////////////
// Input/output functions
////////////////////////////////////////////////////////////////////////

static R3Mesh *
ReadMesh(char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate mesh
  R3Mesh *mesh = new R3Mesh();
  assert(mesh);

  // Read mesh from file
  if (!mesh->ReadFile(filename)) {
    fprintf(stderr, "Unable to read mesh from %s\n", filename);
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("Read mesh from %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Faces = %d\n", mesh->NFaces());
    printf("  # Edges = %d\n", mesh->NEdges());
    printf("  # Vertices = %d\n", mesh->NVertices());
    fflush(stdout);
  }

  // Return success
  return mesh;
}



static int
WriteProperties(R3MeshPropertySet *properties, char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Write properties to file
  if (!properties->Write(filename)) {
    fprintf(stderr, "Unable to write properties to %s\n", filename);
    return 0;
  }

  // Print statistics
  if (print_verbose) {
    printf("Wrote properties to %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Properties = %d\n", properties->NProperties());
    printf("  # Vertices = %d\n", properties->Mesh()->NVertices());
    fflush(stdout);
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Statistics utility functions
////////////////////////////////////////////////////////////////////////

static RNScalar
Mean(RNScalar *values, int nvalues)
{
  // Return mean
  if (nvalues == 0) return 0;
  RNScalar sum = 0;
  for (int i = 0; i < nvalues; i++) 
    sum += values[i];
  return sum / nvalues;
}



static RNScalar
StandardDeviation(RNScalar *values, int nvalues)
{
  // Return standard deviation
  if (nvalues == 0) return 0;
  RNScalar mean = Mean(values, nvalues);
  RNScalar ssd = 0;
  for (int i = 0; i < nvalues; i++) {
    RNScalar delta = values[i] - mean;
    ssd += delta * delta;
  }
  return sqrt(ssd / nvalues);
}


#if 0
static RNScalar
Minimum(RNScalar *values, int nvalues)
{
  // Return minimum of values
  if (nvalues == 0) return 0;
  RNScalar minimum = FLT_MIN;
  for (int i = 0; i < nvalues; i++) {
    if (values[i] < minimum) minimum = values[i];
  }
  return minimum;
}
#endif


static RNScalar
Maximum(RNScalar *values, int nvalues)
{
  // Return maximum of values
  if (nvalues == 0) return 0;
  RNScalar maximum = -FLT_MAX;
  for (int i = 0; i < nvalues; i++) {
    if (values[i] > maximum) maximum = values[i];
  }
  return maximum;
}



static RNScalar
Percentile(RNScalar *values, int nvalues, RNScalar percentile)
{
  // Return value at given percentile (0-100)
  if (nvalues == 0) return 0;
  RNScalar *copy = new RNScalar [ nvalues ];
  for (int i = 0; i < nvalues; i++) copy[i] = values[i];
  qsort(copy, nvalues, sizeof(RNScalar), RNCompareScalars);
  int index = (int) (percentile * nvalues / 100.0);
  if (index >= nvalues) index = nvalues-1;
  RNScalar result = copy[index];
  delete [] copy;
  return result;
}



static RNScalar
Median(RNScalar *values, int nvalues)
{
  // Return median
  return Percentile(values, nvalues, 50);
}



////////////////////////////////////////////////////////////////////////
// Mesh processing utility functions
////////////////////////////////////////////////////////////////////////

RNScalar
Sigma(R3Mesh *mesh)
{
  // Compute smallest sigma
  static RNScalar sigma = 0;
  static R3Mesh *last_mesh = NULL;
  if (mesh != last_mesh) {
    sigma = 0.005 * sqrt(mesh->Area());
    last_mesh = mesh;
  }

  // Return sigma
  return sigma;
}



static R3Grid *
CreateGrid(R3Mesh *mesh)
{
  // Determine grid resolution
  int max_resolution = 256;
  R3Box bbox = mesh->BBox();
  RNScalar spacing = bbox.LongestAxisLength() / max_resolution;
  if (spacing == 0) return NULL;
  int xres = (int) (bbox.XLength() / spacing); if (xres == 0) xres = 1;
  int yres = (int) (bbox.YLength() / spacing); if (yres == 0) yres = 1;
  int zres = (int) (bbox.ZLength() / spacing); if (zres == 0) zres = 1;

  // Create grid
  R3Grid *grid = new R3Grid(xres, yres, zres, bbox);
  if (!grid) {
    fprintf(stderr, "Unable to allocate grid\n");
    exit(-1);
  }

  // Rasterize each triangle into grid
  for (int i = 0; i < mesh->NFaces(); i++) {
    R3MeshFace *face = mesh->Face(i);
    const R3Point& p0 = mesh->VertexPosition(mesh->VertexOnFace(face, 0));
    const R3Point& p1 = mesh->VertexPosition(mesh->VertexOnFace(face, 1));
    const R3Point& p2 = mesh->VertexPosition(mesh->VertexOnFace(face, 2));
    grid->RasterizeWorldTriangle(p0, p1, p2, 1.0);
  }

  // Make sure every pixel is set only once
  grid->Threshold(0.5, 0, 1);

  // Return grid
  return grid;
}


static RNArray<R3MeshVertex *> *
CreateVertexSampling(R3Mesh *mesh, int num_vertices, RNScalar *weights)
{
  // Create array of selected vertices
  RNArray<R3MeshVertex *> *selected_vertices = new RNArray<R3MeshVertex *>();
  if (!selected_vertices) {
    fprintf(stderr, "Unable to create array of sample vertices\n");
    return NULL;
  }

  // Check if mesh has more than num_vertices
  if (mesh->NVertices() <= num_vertices) {
    // Make list of all vertices
    for (int i = 0; i < mesh->NVertices(); i++) {
      selected_vertices->Insert(mesh->Vertex(i));
    }

    // Assign weights based on area
    for (int i = 0; i < selected_vertices->NEntries(); i++) {
      R3MeshVertex *vertex = selected_vertices->Kth(i);
      weights[i] = mesh->VertexArea(vertex);
    }
  }
  else {
    // Select random starting vertex
    int i = (int) (RNRandomScalar() * mesh->NVertices());
    R3MeshVertex *vertex = mesh->Vertex(i);
    selected_vertices->Insert(vertex);
    
    // Iteratively select furthest vertex
    for (int i = 0; i < num_vertices; i++) {

      // Compute distances from selected vertices to all vertices
      RNLength *distances = mesh->DijkstraDistances(*selected_vertices);
      
      // Find furthest vertex
      float furthest_distance = 0;
      R3MeshVertex *furthest_vertex = NULL;
      for (int j = 0; j < mesh->NVertices(); j++) {
        if (distances[j] > furthest_distance) {
          furthest_distance = distances[j];
          furthest_vertex = mesh->Vertex(j);
        }
      }
      
      // Check if found furthest vertex
      if (!furthest_vertex) break;
      
      // Add furthest vertex to selected set
      if (i == 0) selected_vertices->Truncate(0);
      selected_vertices->Insert(furthest_vertex);

      // Delete distances
      delete [] distances;
    }

    // Assign weights equally (hoping that FPS spread points evenly)
    if (weights) {
      for (int i = 0; i < selected_vertices->NEntries(); i++) {
        weights[i] = 1;
      }
    }
  }

  // Return array of selected vertices
  return selected_vertices;
}



////////////////////////////////////////////////////////////////////////
// Property processing utility functions
////////////////////////////////////////////////////////////////////////

int 
InsertProperty(R3MeshPropertySet *properties, R3MeshProperty *property, int num_blurs = 0)
{
  // Insert property
  properties->Insert(property);

  // Insert coies of property with increasing blur
  if (num_blurs > 0) {
    RNScalar sigma = Sigma(properties->Mesh());
    R3MeshProperty *p = property;
    for (int i = 1; i < num_blurs; i++) {
      p = new R3MeshProperty(*p);
      char buffer[1024];
      sprintf(buffer, "%s%d", property->Name(), i);
      p->SetName(buffer);
      p->Blur(sigma);
      properties->Insert(p);
      sigma *= 2;
    }
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Basic properties
////////////////////////////////////////////////////////////////////////

static R3MeshPropertySet *
ComputeBasicProperties(R3Mesh *mesh)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Computing basic properties ...\n");
    fflush(stdout);
  }

  // Allocate property set
  R3MeshPropertySet *properties = new R3MeshPropertySet(mesh);
  if (!properties) {
    fprintf(stderr, "Unable to allocate property set.\n");
    return NULL;
  }

  // Allocate properites
  R3MeshProperty *index = new R3MeshProperty(mesh, "Index");
  R3MeshProperty *valence = new R3MeshProperty(mesh, "Valence");
  R3MeshProperty *length = new R3MeshProperty(mesh, "AverageEdgeLength");
  R3MeshProperty *area = new R3MeshProperty(mesh, "Area");

  // Compute properties
  for (int i = 0; i < mesh->NVertices(); i++) {
    R3MeshVertex *vertex = mesh->Vertex(i);
    index->SetVertexValue(i, mesh->VertexID(vertex));
    valence->SetVertexValue(i, mesh->VertexValence(vertex));
    length->SetVertexValue(i, mesh->VertexAverageEdgeLength(vertex));
    area->SetVertexValue(i, mesh->VertexArea(vertex));
  }

  // Insert properties at multiple scales
  InsertProperty(properties, index);
  InsertProperty(properties, valence);
  InsertProperty(properties, length);
  InsertProperty(properties, area);

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Properties = %d\n", properties->NProperties());
    fflush(stdout);
  }

  // Return property set
  return properties;
}



////////////////////////////////////////////////////////////////////////
// Basic properties
////////////////////////////////////////////////////////////////////////

static R3MeshPropertySet *
ComputeCoordinateProperties(R3Mesh *mesh)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Computing coordinate properties ...\n");
    fflush(stdout);
  }

  // Allocate property set
  R3MeshPropertySet *properties = new R3MeshPropertySet(mesh);
  if (!properties) {
    fprintf(stderr, "Unable to allocate property set.\n");
    return NULL;
  }

  // Allocate properites
  R3MeshProperty *xposition = new R3MeshProperty(mesh, "XPosition");
  R3MeshProperty *yposition = new R3MeshProperty(mesh, "YPosition");
  R3MeshProperty *zposition = new R3MeshProperty(mesh, "ZPosition");
  R3MeshProperty *xnormal = new R3MeshProperty(mesh, "XNormal");
  R3MeshProperty *ynormal = new R3MeshProperty(mesh, "YNormal");
  R3MeshProperty *znormal = new R3MeshProperty(mesh, "ZNormal");

  // Compute properties
  for (int i = 0; i < mesh->NVertices(); i++) {
    R3MeshVertex *vertex = mesh->Vertex(i);
    const R3Point& position = mesh->VertexPosition(vertex);
    const R3Vector& normal = mesh->VertexNormal(vertex);
    xposition->SetVertexValue(i, position.X());
    yposition->SetVertexValue(i, position.Y());
    zposition->SetVertexValue(i, position.Z());
    xnormal->SetVertexValue(i, normal.X());
    ynormal->SetVertexValue(i, normal.Y());
    znormal->SetVertexValue(i, normal.Z());
  }

  // Insert properties at multiple scales
  InsertProperty(properties, xposition);
  InsertProperty(properties, yposition);
  InsertProperty(properties, zposition);
  InsertProperty(properties, xnormal);
  InsertProperty(properties, ynormal);
  InsertProperty(properties, znormal);

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Properties = %d\n", properties->NProperties());
    fflush(stdout);
  }

  // Return property set
  return properties;
}



////////////////////////////////////////////////////////////////////////
// Curvature properties
////////////////////////////////////////////////////////////////////////

// Rotate a coordinate system to be perpendicular to the given normal
static void rot_coord_sys(const R3Vector &old_u, const R3Vector &old_v,
			  const R3Vector &new_norm,
			  R3Vector &new_u, R3Vector &new_v)
{
  new_u = old_u;
  new_v = old_v;
  R3Vector old_norm = old_u % old_v;
  RNScalar ndot = old_norm.Dot(new_norm);
  if (ndot <= -1.0) {
    new_u = -new_u;
    new_v = -new_v;
    return;
  }
  R3Vector perp_old = new_norm - ndot * old_norm;
  R3Vector dperp = 1.0f / (1 + ndot) * (old_norm + new_norm);
  new_u -= dperp * new_u.Dot(perp_old);
  new_v -= dperp * new_v.Dot(perp_old);
}



// Reproject a curvature tensor from the basis spanned by old_u and old_v
// (which are assumed to be unit-length and perpendicular) to the
// new_u, new_v basis.
static void proj_curv(const R3Vector &old_u, const R3Vector &old_v,
                      RNScalar old_ku, RNScalar old_kuv, RNScalar old_kv,
                      const R3Vector &new_u, const R3Vector &new_v,
                      RNScalar &new_ku, RNScalar &new_kuv, RNScalar &new_kv)
{
  R3Vector r_new_u, r_new_v;
  rot_coord_sys(new_u, new_v, old_u % old_v, r_new_u, r_new_v);

  RNScalar u1 = r_new_u.Dot(old_u);
  RNScalar v1 = r_new_u.Dot(old_v);
  RNScalar u2 = r_new_v.Dot(old_u);
  RNScalar v2 = r_new_v.Dot(old_v);
  new_ku  = old_ku * u1*u1 + old_kuv * (2.0  * u1*v1) + old_kv * v1*v1;
  new_kuv = old_ku * u1*u2 + old_kuv * (u1*v2 + u2*v1) + old_kv * v1*v2;
  new_kv  = old_ku * u2*u2 + old_kuv * (2.0  * u2*v2) + old_kv * v2*v2;
}



static void diagonalize_curv(const R3Vector &old_u, const R3Vector &old_v,
                             RNScalar ku, RNScalar kuv, RNScalar kv,
                             const R3Vector &new_norm,
                             R3Vector &pdir1, R3Vector &pdir2, RNScalar &k1, RNScalar &k2)
{
  R3Vector r_old_u, r_old_v;
  rot_coord_sys(old_u, old_v, new_norm, r_old_u, r_old_v);

  RNScalar c = 1.0, s = 0.0, tt = 0.0;
  if (kuv != 0.0) {
    // Jacobi rotation to diagonalize
    RNScalar h = 0.5 * (kv - ku) / kuv;
    tt = (h < 0.0) ?
      1.0 / (h - sqrt(1.0 + h*h)) :
      1.0 / (h + sqrt(1.0 + h*h));
    c = 1.0 / sqrt(1.0 + tt*tt);
    s = tt * c;
  }

  k1 = ku - tt * kuv;
  k2 = kv + tt * kuv;

  if (fabs(k1) >= fabs(k2)) {
    pdir1 = c*r_old_u - s*r_old_v;
  } 
  else {
    RNScalar swap = k1;
    k1 = k2;
    k2 = swap;
    pdir1 = s*r_old_u + c*r_old_v;
  }
  pdir2 = new_norm % pdir1;
}



static R3MeshPropertySet *
ComputeCurvatureProperties(R3Mesh *mesh)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Computing curvature properties ...\n");
    fflush(stdout);
  }

  // Allocate property set
  R3MeshPropertySet *properties = new R3MeshPropertySet(mesh);
  if (!properties) {
    fprintf(stderr, "Unable to allocate property set.\n");
    return NULL;
  }

  // Allocate properites
  R3MeshProperty *gauss = new R3MeshProperty(mesh, "GaussCurvature");
  R3MeshProperty *mean = new R3MeshProperty(mesh, "MeanCurvature");
  R3MeshProperty *min = new R3MeshProperty(mesh, "MinCurvature");
  R3MeshProperty *max = new R3MeshProperty(mesh, "MaxCurvature");

  // Get convenient variables
  int nf = mesh->NFaces();
  int nv = mesh->NVertices();

  // Compute Vertex Area
  double *pointareas = new double [ nv ];
  R3Vector *cornerareas = new R3Vector [ nf ];
  for (int i = 0; i < nf; i++) {
    // Edges
    R3MeshFace * face = mesh->Face(i);
    R3MeshVertex * vertex[3];
    vertex[0] = mesh->VertexOnFace(face, 0);
    vertex[1] = mesh->VertexOnFace(face, 1);
    vertex[2] = mesh->VertexOnFace(face, 2);
    R3Vector e[3];
    e[0] = mesh->VertexPosition(vertex[2]) - mesh->VertexPosition(vertex[1]);
    e[1] = mesh->VertexPosition(vertex[0]) - mesh->VertexPosition(vertex[2]);
    e[2] = mesh->VertexPosition(vertex[1]) - mesh->VertexPosition(vertex[0]);
    // Compute corner weights
    R3Vector rcross = e[0] % e[1];
    RNScalar area = 0.5 * rcross.Length();
    RNScalar l2[3] = { e[0].Dot(e[0]), e[1].Dot(e[1]), e[2].Dot(e[2]) };
    RNScalar ew[3] = { l2[0] * (l2[1] + l2[2] - l2[0]),
                       l2[1] * (l2[2] + l2[0] - l2[1]),
                       l2[2] * (l2[0] + l2[1] - l2[2]) };
    if (ew[0] <= 0.0) {
      cornerareas[i][1] = -0.25 * l2[2] * area /
        (e[0].Dot(e[2]));
      cornerareas[i][2] = -0.25 * l2[1] * area /
        (e[0].Dot(e[1]));
      cornerareas[i][0] = area - cornerareas[i][1] -
        cornerareas[i][2];
    } 
    else if (ew[1] <= 0.0) {
      cornerareas[i][2] = -0.25 * l2[0] * area /
        (e[1].Dot(e[0]));
      cornerareas[i][0] = -0.25 * l2[2] * area /
        (e[1].Dot(e[2]));
      cornerareas[i][1] = area - cornerareas[i][2] -
        cornerareas[i][0];
    } 
    else if (ew[2] <= 0.0) {
      cornerareas[i][0] = -0.25 * l2[1] * area /
        (e[2].Dot(e[1]));
      cornerareas[i][1] = -0.25 * l2[0] * area /
        (e[2].Dot(e[0]));
      cornerareas[i][2] = area - cornerareas[i][0] -
        cornerareas[i][1];
    } 
    else {
      double ewscale = 0.5 * area / (ew[0] + ew[1] + ew[2]);
      for (int j = 0; j < 3; j++)
        cornerareas[i][j] = ewscale * (ew[(j+1)%3] +
                                       ew[(j+2)%3]);
    }
    pointareas[mesh->VertexID(vertex[0])] += cornerareas[i][0];
    pointareas[mesh->VertexID(vertex[1])] += cornerareas[i][1];
    pointareas[mesh->VertexID(vertex[2])] += cornerareas[i][2];
  }

  RNScalar *curv1 = new RNScalar [nv];
  RNScalar *curv2 = new RNScalar [nv];
  RNScalar *curv12 = new RNScalar [nv];
  R3Vector *pdir1 = new R3Vector [nv];
  R3Vector *pdir2 = new R3Vector [nv];

  // Set up an initial coordinate system per vertex
  for (int i = 0; i < nf; i++) {
    R3MeshFace * face = mesh->Face(i);
    R3MeshVertex * vertex[3];
    vertex[0] = mesh->VertexOnFace(face, 0);
    vertex[1] = mesh->VertexOnFace(face, 1);
    vertex[2] = mesh->VertexOnFace(face, 2);
    pdir1[mesh->VertexID(vertex[0])] = mesh->VertexPosition(vertex[1]) -
      mesh->VertexPosition(vertex[0]);
    pdir1[mesh->VertexID(vertex[1])] = mesh->VertexPosition(vertex[2]) -
      mesh->VertexPosition(vertex[1]);
    pdir1[mesh->VertexID(vertex[2])] = mesh->VertexPosition(vertex[0]) -
      mesh->VertexPosition(vertex[2]);
  }

  for (int i = 0; i < nv; i++) {
    R3MeshVertex * vertex = mesh->Vertex(i);
    pdir1[i] = pdir1[i] % mesh->VertexNormal(vertex);
    pdir1[i].Normalize();
    pdir2[i] = mesh->VertexNormal(vertex) % pdir1[i];
  }

  // Compute curvature per-face
  for (int i = 0; i < nf; i++) {
    R3MeshFace * face = mesh->Face(i);
    R3MeshVertex * vertex[3];
    vertex[0] = mesh->VertexOnFace(face, 0);
    vertex[1] = mesh->VertexOnFace(face, 1);
    vertex[2] = mesh->VertexOnFace(face, 2);
    // Edges
    R3Vector e[3];
    e[0] = mesh->VertexPosition(vertex[2]) - mesh->VertexPosition(vertex[1]);
    e[1] = mesh->VertexPosition(vertex[0]) - mesh->VertexPosition(vertex[2]);
    e[2] = mesh->VertexPosition(vertex[1]) - mesh->VertexPosition(vertex[0]);
    // N-T-B coordinate system per face
    R3Vector t = e[0];
    t.Normalize();
    R3Vector n = e[0] % e[1];
    R3Vector b = n % t;
    b.Normalize();

    // Estimate curvature based on variation of normals along edges
    RNScalar m[3] = { 0.0, 0.0, 0.0 };
    RNScalar w[3][3] = { {0,0,0}, {0,0,0}, {0,0,0} };
    for (int j = 0; j < 3; j++) {
      RNScalar u = e[j].Dot(t);
      RNScalar v = e[j].Dot(b);
      w[0][0] += u*u;
      w[0][1] += u*v;
      w[2][2] += v*v;
      R3Vector dn = mesh->VertexNormal(vertex[(j+2)%3]) -
        mesh->VertexNormal(vertex[(j+1)%3]);
      RNScalar dnu = dn.Dot(t);
      RNScalar dnv = dn.Dot(b);
      m[0] += dnu*u;
      m[1] += dnu*v + dnv*u;
      m[2] += dnv*v;
    }
    w[1][1] = w[0][0] + w[2][2];
    w[1][2] = w[0][1];
    w[1][0] = w[0][1];
    w[2][1] = w[1][2];
    RNScalar * matrix_a = new RNScalar[9];
    for (int kk=0; kk<9; kk++) {
      matrix_a[kk] = w[kk/3][kk%3];
    }
    RNScalar * bp = new RNScalar[3];
    bp[0]=m[0];bp[1]=m[1];bp[2]=m[2];
    RNSvdSolve(3, 3, matrix_a, bp, m, 0.0);
    delete[] matrix_a;
    delete[] bp;
    // Push it back out to the vertices
    for (int j = 0; j < 3; j++) {
      int vj = mesh->VertexID(vertex[j]);
      RNScalar c1, c12, c2;
      proj_curv(t, b, m[0], m[1], m[2], pdir1[vj], pdir2[vj], c1, c12, c2);
      RNScalar wt = cornerareas[i][j] / pointareas[vj];
      curv1[vj]  += wt * c1;
      curv12[vj] += wt * c12;
      curv2[vj]  += wt * c2;
    }
  }

  for (int i = 0; i < nv; i++) {
    R3MeshVertex * vertex = mesh->Vertex(i);
    R3Vector normal = mesh->VertexNormal(vertex);
    diagonalize_curv(pdir1[i], pdir2[i],
                     curv1[i], curv12[i], curv2[i],
                     normal, pdir1[i], pdir2[i],
                     curv1[i], curv2[i]);
    gauss->SetVertexValue(i, curv1[i] * curv2[i]);
    mean->SetVertexValue(i, (curv1[i] + curv2[i])/2);
    max->SetVertexValue(i, curv1[i]);
    min->SetVertexValue(i, curv2[i]);
  }

  // Insert properties at multiple scales
  InsertProperty(properties, gauss);
  InsertProperty(properties, mean);
  InsertProperty(properties, max);
  InsertProperty(properties, min);

  // Delete temporary memory
  delete [] pointareas;
  delete [] cornerareas;
  delete [] curv1;
  delete [] curv2;
  delete [] curv12;
  delete [] pdir1;
  delete [] pdir2;

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Properties = %d\n", properties->NProperties());
    fflush(stdout);
  }

  // Return property set
  return properties;
}



////////////////////////////////////////////////////////////////////////
// Laplacian properties
////////////////////////////////////////////////////////////////////////

static R3MeshPropertySet *
ComputeLaplacianProperties(R3Mesh *mesh)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Computing laplacian properties ...\n");
    fflush(stdout);
  }

  // Allocate property set
  R3MeshPropertySet *properties = new R3MeshPropertySet(mesh);
  if (!properties) {
    fprintf(stderr, "Unable to allocate property set.\n");
    return NULL;
  }

  // Allocate/initialize laplacian matrix
  int n = mesh->NVertices();
  RNScalar *laplacian_matrix = new RNScalar [ n * n ];
  for (int i = 0; i < n * n; i++) laplacian_matrix[i] = 0;
  for (int i = 0; i < n; i++) laplacian_matrix[i*n+i] = -1;

  // Compute laplacian matrix entries
  for (int i1 = 0; i1 < n; i1++) {
    RNScalar total_weight = 0;
    R3MeshVertex *v1 = mesh->Vertex(i1);
    const R3Point& p1 = mesh->VertexPosition(v1);
    for (int j = 0; j < mesh->VertexValence(v1); j++) {
      R3MeshEdge *e = mesh->EdgeOnVertex(v1, j);
      R3MeshVertex *v2 = mesh->VertexAcrossEdge(e, v1);
      const R3Point& p2 = mesh->VertexPosition(v2);
      int i2 = mesh->VertexID(v2);

      // Compute cotan weight
      double weight = 0;
      for (int k = 0; k < 2; k++) {
        R3MeshFace *f = mesh->FaceOnEdge(e, k);
        if (!f) continue;
        R3MeshVertex *v3 = mesh->VertexAcrossFace(f, e);
        const R3Point& p3 = mesh->VertexPosition(v3);
        R3Vector vec1 = p1 - p3; vec1.Normalize();
        R3Vector vec2 = p2 - p3; vec2.Normalize();
        RNAngle angle = R3InteriorAngle(vec1, vec2);
        if (angle == 0) continue;
        double tan_angle = tan(angle);
        if (tan_angle == 0) continue;
        weight += 1.0 / tan_angle;
      }

      // Add weighted position
      laplacian_matrix[i1*n + i2] = weight;
      total_weight += weight;
    }

    // Normalize weights
    if (total_weight > 0) {
      for (int j = 0; j < mesh->VertexValence(v1); j++) {
        R3MeshEdge *e = mesh->EdgeOnVertex(v1, j);
        R3MeshVertex *v2 = mesh->VertexAcrossEdge(e, v1);
        int i2 = mesh->VertexID(v2);
        laplacian_matrix[i1*n + i2] /= total_weight;
      }
    }
  }

  // Compute eigenvectors of laplacian
  RNScalar *u = new RNScalar [ n * n ];
  RNScalar *eigenvalues = new RNScalar [ n  ];
  RNScalar *eigenvectors = new RNScalar [ n * n ];
  RNSvdDecompose(n, n, laplacian_matrix, u, eigenvalues, eigenvectors);

  // Determine time scale factor -- from [de Goes 2008] and [Rustimov 2010]
  RNScalar time_scale = 1;
  RNScalar lambda1 = eigenvalues[n-2];
  if (lambda1 > 0) time_scale = 1 / (2 * lambda1);

  // Compute HKS at several times
  RNScalar *hks = new RNScalar [ n ];
  RNScalar t = 0.01 * time_scale;
  for (int k = 0; k < 8; k++) {
    char name[256];
    sprintf(name, "HeatKernelSignature%d", k+1);
    R3MeshProperty *property = new R3MeshProperty(mesh, name);
    for (int i = 0; i < n; i++) {
      RNScalar hks = 0;
      for (int j = 0; j < n; j++) {
        RNScalar lambda = eigenvalues[j];
        RNScalar phi = eigenvectors[j*n+i];
        hks += exp(-t * lambda) * phi * phi;
      }
      property->SetVertexValue(i, hks);
    }
    InsertProperty(properties, property);
    t *= 2;
  }
  delete [] hks;

  // Create/insert properties
  int num_eigenvalues = 10;
  for (int i = 0; i < num_eigenvalues; i++) {
    char name[256];
    sprintf(name, "LaplacianEigenvector%d", i+1);
    R3MeshProperty *property = new R3MeshProperty(mesh, name);
    for (int j = 0; j < n; j++) property->SetVertexValue(j, eigenvectors[(n-i-1)*n+j]);
    InsertProperty(properties, property);
  }

  // Delete stuff
  delete [] laplacian_matrix;
  delete [] u;
  delete [] eigenvalues;
  delete [] eigenvectors;

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Properties = %d\n", properties->NProperties());
    fflush(stdout);
  }

  // Return property set
  return properties;
}



////////////////////////////////////////////////////////////////////////
// Volume properties
////////////////////////////////////////////////////////////////////////

static R3MeshPropertySet *
ComputeVolumeProperties(R3Mesh *mesh)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Computing volume properties ...\n");
    fflush(stdout);
  }

  // Allocate property set
  R3MeshPropertySet *properties = new R3MeshPropertySet(mesh);
  if (!properties) {
    fprintf(stderr, "Unable to allocate property set.\n");
    return NULL;
  }

  // Rasterize mesh surface into a grid
  R3Grid *grid = CreateGrid(mesh);
  if (!grid) {
    fprintf(stderr, "Unable to create grid from mesh\n");
    return NULL;
  }

  // Consider density blurred in 3D at multiple scales
  int num_scales = 6;
  RNScalar sigma = Sigma(mesh) * grid->WorldToGridScaleFactor();
  for (int i = 0; i < num_scales; i++) {
    char name[1024];
    sprintf(name, "VolumeDensity%d", i);
    R3MeshProperty *property = new R3MeshProperty(mesh, name);

    // Compute vertex values
    grid->Blur(sigma);
    for (int i = 0; i < mesh->NVertices(); i++) {
      R3MeshVertex *vertex = mesh->Vertex(i);
      const R3Point& position = mesh->VertexPosition(vertex);
      RNScalar value = grid->WorldValue(position);
      property->SetVertexValue(i, value);
    }

    // Insert property
    InsertProperty(properties, property);

    // Increate sigma
    sigma *= 2;
  }

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Properties = %d\n", properties->NProperties());
    printf("  Grid resolution = %d %d %d\n", grid->XResolution(), grid->YResolution(), grid->ZResolution());
    fflush(stdout);
  }

  // Delete grid
  delete grid;

  // Return property set
  return properties;
}



////////////////////////////////////////////////////////////////////////
// Boundary properties
////////////////////////////////////////////////////////////////////////

static R3MeshPropertySet *
ComputeBoundaryProperties(R3Mesh *mesh)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Computing boundary properties ...\n");
    fflush(stdout);
  }

  // Allocate property set
  R3MeshPropertySet *properties = new R3MeshPropertySet(mesh);
  if (!properties) {
    fprintf(stderr, "Unable to allocate property set.\n");
    return NULL;
  }

  // Create array of boundary vertices
  RNArray<R3MeshVertex *> boundary_vertices;
  for (int i = 0; i < mesh->NVertices(); i++) {
    R3MeshVertex *vertex = mesh->Vertex(i);
    if (!mesh->IsVertexOnBoundary(vertex)) continue;
    boundary_vertices.Insert(vertex);
  }

  // Compute dijkstra distances to closest boundary vertex
  RNScalar *distances = mesh->DijkstraDistances(boundary_vertices);
  if (!distances) return 0;

  // Fill property
  R3MeshProperty *property = new R3MeshProperty(mesh, "BoundaryDijkstraDistance");
  for (int i = 0; i < mesh->NVertices(); i++) {
    R3MeshVertex *vertex = mesh->Vertex(i);
    if (mesh->VertexValence(vertex) == 0) continue;
    property->SetVertexValue(i, distances[i]);
  }

  // Insert property
  InsertProperty(properties, property);

  // Delete distances
  delete [] distances;

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Properties = %d\n", properties->NProperties());
    fflush(stdout);
  }

  // Return property set
  return properties;
}



////////////////////////////////////////////////////////////////////////
// Dijkstra distance properties
////////////////////////////////////////////////////////////////////////

static R3MeshPropertySet *
ComputeDijkstraDistanceProperties(R3Mesh *mesh)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Computing dijkstra distance properties ...\n");
    fflush(stdout);
  }

  // Allocate property set
  R3MeshPropertySet *properties = new R3MeshPropertySet(mesh);
  if (!properties) {
    fprintf(stderr, "Unable to allocate property set.\n");
    return NULL;
  }

  // Allocate/insert properties
  R3MeshProperty *mean_property = new R3MeshProperty(mesh, "DijkstraDistanceMean");
  R3MeshProperty *stddev_property = new R3MeshProperty(mesh, "DijkstraDistanceStddev");
  R3MeshProperty *median_property = new R3MeshProperty(mesh, "DijkstraDistanceMedian");
  R3MeshProperty *ten_property = new R3MeshProperty(mesh, "DijkstraDistanceTen");
  R3MeshProperty *ninety_property = new R3MeshProperty(mesh, "DijkstraDistanceNinety");
  R3MeshProperty *maximum_property = new R3MeshProperty(mesh, "DijkstraDistanceMaximum");
 
  // Compute properties
  for (int i = 0; i < mesh->NVertices(); i++) {
    R3MeshVertex *vertex = mesh->Vertex(i);
    RNLength *distances = mesh->DijkstraDistances(vertex);
    mean_property->SetVertexValue(i, Mean(distances, mesh->NVertices()));
    stddev_property->SetVertexValue(i, StandardDeviation(distances, mesh->NVertices()));
    median_property->SetVertexValue(i, Median(distances, mesh->NVertices()));
    ten_property->SetVertexValue(i, Percentile(distances, mesh->NVertices(), 10));
    ninety_property->SetVertexValue(i, Percentile(distances, mesh->NVertices(), 90));
    maximum_property->SetVertexValue(i, Maximum(distances, mesh->NVertices()));
    delete [] distances;
  }

  // Insert properties
  InsertProperty(properties, mean_property);
  InsertProperty(properties, stddev_property);
  InsertProperty(properties, median_property);
  InsertProperty(properties, ten_property);
  InsertProperty(properties, ninety_property);
  InsertProperty(properties, maximum_property);

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Properties = %d\n", properties->NProperties());
    fflush(stdout);
  }

  // Return property set
  return properties;
}



static R3MeshPropertySet *
ComputeDijkstraHistogramProperties(R3Mesh *mesh)
{
  // Parameters
  int nbins = 32;
  int nsamples_in_smallest_bin = 32;
  int nsamples = nsamples_in_smallest_bin * (nbins * nbins) / 2;
  if (nsamples < 3*mesh->NVertices()/2) nsamples = mesh->NVertices();
  if (mesh->NVertices() < 60000) nsamples = mesh->NVertices();

  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Computing dijkstra histogram properties ...\n");
    fflush(stdout);
  }

  // Allocate property set
  R3MeshPropertySet *properties = new R3MeshPropertySet(mesh);
  if (!properties) {
    fprintf(stderr, "Unable to allocate property set.\n");
    return NULL;
  }

  // Allocate/insert properties
  for (int i = 0; i < nbins; i++) {
    char name[1024];
    sprintf(name, "DijkstraHistogramBin%d", i);
    R3MeshProperty *property = new R3MeshProperty(mesh, name);
    properties->Insert(property);
  }
 
  // Create a sampled set of vertices
  RNScalar *weights = new RNScalar [ mesh->NVertices() ];
  RNArray<R3MeshVertex *> *samples = CreateVertexSampling(mesh, nsamples, weights);
  if (!samples) {
    fprintf(stderr, "Unable to sample vertices\n");
    return NULL;
  }

  // Compute normalization factors
  RNScalar area = mesh->Area();
  RNScalar normalization = (area > 0) ? nbins / (1.5 * sqrt(area)) : 1;

  // Compute histogram of distances
  RNScalar total_vote = 0;
  for (int i = 0; i < samples->NEntries(); i++) {
    R3MeshVertex *vertex = samples->Kth(i);
    RNScalar vote = weights[i];

    // Compute distances to other vertices
    RNLength *distances = mesh->DijkstraDistances(vertex);

    // Add values to histogram
    for (int j = 0; j < mesh->NVertices(); j++) {
      RNScalar bin = normalization * distances[j];
      int bin1 = (int) bin;
      int bin2 = bin1 + 1;
      RNScalar t = bin - bin1;
      if (bin1 >= nbins) bin1 = nbins-1;
      if (bin2 >= nbins) bin2 = nbins-1;
      properties->Property(bin1)->AddVertexValue(j, (1-t) * vote);
      properties->Property(bin2)->AddVertexValue(j, t * vote);
    }

    // Delete distances
    delete [] distances;
  }

  // Normalize distribution by total_vote
  if (total_vote > 0) {
    for (int i = 1; i < nbins; i++) {
      R3MeshProperty *property = properties->Property(i);
      property->Divide(total_vote);
    }
  }

  // Make a cumulative distribution 
  for (int i = 1; i < nbins; i++) {
    R3MeshProperty *prev_property = properties->Property(i-1);
    R3MeshProperty *property = properties->Property(i);
    for (int j = 0; j < mesh->NVertices(); j++) {
      property->AddVertexValue(j, prev_property->VertexValue(j));
    }
  }

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Properties = %d\n", properties->NProperties());
    printf("  # Samples = %d\n", samples->NEntries());
    fflush(stdout);
  }

  // Delete sample points
  delete samples;
  delete [] weights;

  // Return property set
  return properties;
}



////////////////////////////////////////////////////////////////////////
// Ray tracing properties
////////////////////////////////////////////////////////////////////////

static R3MeshPropertySet *
ComputeRayTracingProperties(R3Mesh *mesh)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Computing ray tracing properties ...\n");
    fflush(stdout);
  }

  // Allocate property set
  R3MeshPropertySet *properties = new R3MeshPropertySet(mesh);
  if (!properties) {
    fprintf(stderr, "Unable to allocate property set.\n");
    return NULL;
  }

  // Allocate/insert properties
  R3MeshProperty *median_property = new R3MeshProperty(mesh, "RayLengthMedian");
  R3MeshProperty *ten_property = new R3MeshProperty(mesh, "RayLengthTen");
  R3MeshProperty *ninety_property = new R3MeshProperty(mesh, "RayLengthNinety");
  R3MeshProperty *coverage_property = new R3MeshProperty(mesh, "RayCoverage");
    
  // Compute properties based on intersections of random rays
  const int nphis = 8;
  const int nthetas = 8;
  const int num_rays = nphis * nthetas;
  double interior_distances[num_rays];
  for (int i = 0; i < mesh->NVertices(); i++) {
    // Get vertex info
    R3MeshVertex *vertex = mesh->Vertex(i);
    const R3Point& vertex_position = mesh->VertexPosition(vertex);
    const R3Vector& vertex_normal = mesh->VertexNormal(vertex);
    R3Vector phi_rotation_axis = vertex_normal % R3xyz_triad.Axis(vertex_normal.MinDimension());
    R3Vector theta_rotation_axis = vertex_normal;

    // Compute intersections of mesh with random rays from vertex 
    int num_intersections = 0;
    int num_interior_distances = 0;
    for (int j = 0; j < nthetas; j++) {
      RNAngle theta = (j+RNRandomScalar()) * RN_TWO_PI / nthetas;
      for (int k = 0; k < nphis; k++) {
        RNAngle phi = (k+RNRandomScalar()) * RN_PI / nphis;

        // Compute ray
        R3Vector ray_direction = vertex_normal;
        ray_direction.Rotate(phi_rotation_axis, phi);
        ray_direction.Rotate(theta_rotation_axis, theta);
        R3Point ray_source_position = vertex_position + 1000 * RN_EPSILON * ray_direction;
        R3Ray ray(ray_source_position, ray_direction);

        // Compute ray intersection
        R3MeshIntersection intersection;
        if (mesh->Intersection(ray, &intersection)) {
          num_intersections++;
          const R3Vector& face_normal = mesh->FaceNormal(intersection.face);
          if (ray_direction.Dot(face_normal) > 0) {
            interior_distances[num_interior_distances] = intersection.t;
            num_interior_distances++;
          }
        }
      }
    }

    // Compute properties
    median_property->SetVertexValue(i, Median(interior_distances, num_interior_distances));
    ten_property->SetVertexValue(i, Percentile(interior_distances, num_interior_distances, 10));
    ninety_property->SetVertexValue(i, Percentile(interior_distances, num_interior_distances, 90));
    coverage_property->SetVertexValue(i, (RNScalar) num_intersections / (RNScalar) num_rays);
  }

  // Blur the properties to reduce effects of undersampling
  RNScalar sigma = Sigma(mesh);
  median_property->Blur(sigma);
  ten_property->Blur(sigma);
  ninety_property->Blur(sigma);
  coverage_property->Blur(sigma);

  // Insert the properties
  InsertProperty(properties, median_property);
  InsertProperty(properties, ten_property);
  InsertProperty(properties, ninety_property);
  InsertProperty(properties, coverage_property);

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Properties = %d\n", properties->NProperties());
    fflush(stdout);
  }

  // Return property set
  return properties;
}



////////////////////////////////////////////////////////////////////////
// MTurk Segmentation properties
////////////////////////////////////////////////////////////////////////

struct MTurkLabel {
  int id;
  char *name;
  int nyuId;
  int nyu40id;
};

static R3MeshPropertySet *
ReadMTurkSegmentationProperties(R3Mesh *mesh, const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Reading mturk segmentation properties ...\n");
    fflush(stdout);
  }

  // Open file
  FILE* fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open json file %s\n", filename);
    return 0;
  }

  // Read file 
  std::string text;
  fseek(fp, 0, SEEK_END);
  long const size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  char* buffer = new char[size + 1];
  unsigned long const usize = static_cast<unsigned long const>(size);
  if (fread(buffer, 1, usize, fp) != usize) { fprintf(stderr, "Unable to read %s\n", filename); return 0; }
  else { buffer[size] = 0; text = buffer; }
  delete[] buffer;

  // Close file
  fclose(fp);

  // Parse file
  Json::Value json_root;
  Json::Reader json_reader;
  if (!json_reader.parse(text, json_root, false)) {
    fprintf(stderr, "Unable to parse %s\n", filename);
    return 0;
  }

  // Allocate property set
  R3MeshPropertySet *properties = new R3MeshPropertySet(mesh);
  if (!properties) {
    fprintf(stderr, "Unable to allocate property set.\n");
    return NULL;
  }

  // Allocate property
  R3MeshProperty *segmentation = new R3MeshProperty(mesh, "MTurkSegment");

  // Parse segment identifiers
  if (json_root.isMember("segIndices")) {
    Json::Value json_segments = json_root["segIndices"];
    for (Json::ArrayIndex index = 0; index < json_segments.size(); index++) {
      Json::Value json_segment = json_segments[index];
      segmentation->SetVertexValue(index, json_segment.asInt());
    }
  }

  // Insert property
  InsertProperty(properties, segmentation);

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Properties = %d\n", properties->NProperties());
    fflush(stdout);
  }

  // Return property set
  return properties;
}



static int
ReadMTurkLabelMapping(RNSymbolTable<MTurkLabel *>& labels, const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Reading mturk label mapping ...\n");
    fflush(stdout);
  }

  // Open file
  FILE* fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Unable to open label mapping file %s\n", filename);
    return 0;
  }

  // Read keys from first line
  char key_buffer[4096];
  RNArray<char *> keys;
  if (fgets(key_buffer, 4096, fp)) {
    char *token = strtok(key_buffer, ",\t\n");
    while (token) {
      keys.Insert(token);
      token = strtok(NULL, ",\t\n");
    }
  }

  // Extract index of model_id
  int id_index = -1;
  int name_index = -1;
  int nyuId_index = -1;
  int nyu40id_index = -1;
  for (int i = 0; i < keys.NEntries(); i++) {
    if (!strcmp(keys[i], "index")) id_index = i;
    else if (!strcmp(keys[i], "category")) name_index = i; 
    else if (!strcmp(keys[i], "nyuId")) nyuId_index = i; 
    else if (!strcmp(keys[i], "nyu40id")) nyu40id_index = i; 
  }

  // Check if found key fields in header
  if ((id_index < 0) || (name_index < 0) || (nyuId_index < 0) || (nyu40id_index < 0)) {
    fprintf(stderr, "Did not find index, category, nyuId, and nyu40id in header of %s\n", filename);
    return 0;
  }

  // Read subsequent lines of file
  char value_buffer[4096];
  while (fgets(value_buffer, 4096, fp)) {
    // Read values
    RNArray<char *> values;
    char *token = strtok(value_buffer, ",\t\n");
    while (token) {
      values.Insert(token);
      token = strtok(NULL, ",\t\n");
    }

    // Create label
    MTurkLabel *label = new MTurkLabel();
    label->id = atoi(values[id_index]);
    label->name = strdup(values[name_index]);
    label->nyuId = atoi(values[nyuId_index]);
    label->nyu40id = atoi(values[nyu40id_index]);
    labels.Insert(label->name, label);
  }
  
  // Close file
  fclose(fp);

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Labels = %d\n", labels.NEntries());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static R3MeshPropertySet *
ReadMTurkAnnotationProperties(R3Mesh *mesh,
  R3MeshProperty *segmentation_property, RNSymbolTable<MTurkLabel *> *label_mapping,
  const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Reading mturk annotation properties ...\n");
    fflush(stdout);
  }

  // Open file
  FILE* fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open json file %s\n", filename);
    return 0;
  }

  // Read file 
  std::string text;
  fseek(fp, 0, SEEK_END);
  long const size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  char* buffer = new char[size + 1];
  unsigned long const usize = static_cast<unsigned long const>(size);
  if (fread(buffer, 1, usize, fp) != usize) { fprintf(stderr, "Unable to read %s\n", filename); return 0; }
  else { buffer[size] = 0; text = buffer; }
  delete[] buffer;

  // Close file
  fclose(fp);

  // Parse file
  Json::Value json_root;
  Json::Reader json_reader;
  if (!json_reader.parse(text, json_root, false)) {
    fprintf(stderr, "Unable to parse %s\n", filename);
    return 0;
  }

  // Allocate property set
  R3MeshPropertySet *properties = new R3MeshPropertySet(mesh);
  if (!properties) {
    fprintf(stderr, "Unable to allocate property set.\n");
    return NULL;
  }

  // Allocate properites
  R3MeshProperty *instance = new R3MeshProperty(mesh, "MTurkInstance");
  R3MeshProperty *nyuId = new R3MeshProperty(mesh, "MTurkNNYUId");
  R3MeshProperty *nyu40id = new R3MeshProperty(mesh, "MTurkNYU40Id");

  // Parse instance identifiers
  if (json_root.isMember("segGroups")) {
    MTurkLabel *label = NULL;
    Json::Value json_groups = json_root["segGroups"];
    for (Json::ArrayIndex group_index = 0; group_index < json_groups.size(); group_index++) {
      Json::Value json_group = json_groups[group_index];
      if (!json_group.isMember("segments")) continue;
      if (!json_group.isMember("label")) continue;
      Json::Value json_segments = json_group["segments"];
      Json::Value json_label = json_group["label"];
      label_mapping->Find(json_label.asString(), &label);
      for (Json::ArrayIndex segment_index = 0; segment_index < json_segments.size(); segment_index++) {
        Json::Value json_segment = json_segments[segment_index];
        int segment_id = json_segment.asInt();
        for (int i = 0; i < mesh->NVertices(); i++) {
          if (RNIsEqual(segmentation_property->VertexValue(i), segment_id)) {
            instance->SetVertexValue(i, group_index);
            if (label) {
              nyuId->SetVertexValue(i, label->nyuId);
              nyu40id->SetVertexValue(i, label->nyu40id);
            }
          }
        }
      }
    }
  }

  // Insert properties
  InsertProperty(properties, instance);
  InsertProperty(properties, nyuId);
  InsertProperty(properties, nyu40id);

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Properties = %d\n", properties->NProperties());
    fflush(stdout);
  }

  // Return property set
  return properties;
}



////////////////////////////////////////////////////////////////////////
// Symmetry map properties
////////////////////////////////////////////////////////////////////////

static R3MeshPropertySet *
ComputeMapProperties(R3Mesh *mesh, const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Computing map properties ...\n");
    fflush(stdout);
  }

  // Allocate property set
  R3MeshPropertySet *properties = new R3MeshPropertySet(mesh);
  if (!properties) {
    fprintf(stderr, "Unable to allocate property set.\n");
    return NULL;
  }

  // Allocate properites
  R3MeshProperty *distance_to_correspondence = new R3MeshProperty(mesh, "DistanceToCorrespondence");

  // Compute distance normalization scale factor
  RNScalar sqrt_area = sqrt(mesh->Area());
  if (sqrt_area == 0) return NULL;
  RNScalar scale = 1 / sqrt_area;

  // Open map file
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Unable to open map file: %s\n", filename);
    return NULL;
  }

  // Read map file and compute distance to each correspondence
  for (int i = 0; i < mesh->NVertices(); i++) {
    R3MeshVertex *vertex = mesh->Vertex(i);

    // Read corresponding vertex ID
    int id;
    if (fscanf(fp, "%d", &id) != (unsigned int) 1) { 
      fprintf(stderr, "Unable to read %s\n", filename); 
      return NULL; 
    }

    // Compute distance from vertex to its correspondence
    RNScalar distance = 10;
    if ((id >= 0) && (id < mesh->NVertices())) {
      R3MeshVertex *corresponding_vertex = mesh->Vertex(id);
      distance = scale * mesh->DijkstraDistance(vertex, corresponding_vertex);
    }

    // Set property value
    distance_to_correspondence->SetVertexValue(i, distance);
  }

  // Close map file
  fclose(fp);

  // Insert property
  InsertProperty(properties, distance_to_correspondence);

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Properties = %d\n", properties->NProperties());
    fflush(stdout);
  }

  // Return property set
  return properties;
}



////////////////////////////////////////////////////////////////////////
// Solid texture properties
////////////////////////////////////////////////////////////////////////

static R3MeshPropertySet *
ComputeSolidTextureProperties(R3Mesh *mesh)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Computing solid texture properties ...\n");
    fflush(stdout);
  }

  // Allocate property set
  R3MeshPropertySet *properties = new R3MeshPropertySet(mesh);
  if (!properties) {
    fprintf(stderr, "Unable to allocate property set.\n");
    return NULL;
  }

  // Allocate properites
  R3MeshProperty *sine1 = new R3MeshProperty(mesh, "Sine1");
  R3MeshProperty *sine2 = new R3MeshProperty(mesh, "Sine2");
  R3MeshProperty *sine4 = new R3MeshProperty(mesh, "Sine4");
  R3MeshProperty *sine8 = new R3MeshProperty(mesh, "Sine8");
  R3MeshProperty *sine16 = new R3MeshProperty(mesh, "Sine16");
  R3MeshProperty *sine32 = new R3MeshProperty(mesh, "Sine32");
  R3MeshProperty *sine64 = new R3MeshProperty(mesh, "Sine64");

  // Read map file and compute distance to each correspondence
  for (int i = 0; i < mesh->NVertices(); i++) {
    R3MeshVertex *vertex = mesh->Vertex(i);
    const R3Point& position = mesh->VertexPosition(vertex);
    const R3Vector& normal = mesh->VertexNormal(vertex);
    int dim = normal.MaxDimension();

    // Compute sine properties from vertex coordinates 
    sine1->SetVertexValue(i, sin(1*position[(dim+1)%3]) * sin(1*position[(dim+2)%3]));
    sine2->SetVertexValue(i, sin(2*position[(dim+1)%3]) * sin(2*position[(dim+2)%3]));
    sine4->SetVertexValue(i, sin(4*position[(dim+1)%3]) * sin(4*position[(dim+2)%3]));
    sine8->SetVertexValue(i, sin(8*position[(dim+1)%3]) * sin(8*position[(dim+2)%3]));
    sine16->SetVertexValue(i, sin(16*position[(dim+1)%3]) * sin(16*position[(dim+2)%3]));
    sine32->SetVertexValue(i, sin(32*position[(dim+1)%3]) * sin(32*position[(dim+2)%3]));
    sine64->SetVertexValue(i, sin(64*position[(dim+1)%3]) * sin(64*position[(dim+2)%3]));
  }

  // Insert properties
  InsertProperty(properties, sine1);
  InsertProperty(properties, sine2);
  InsertProperty(properties, sine4);
  InsertProperty(properties, sine8);
  InsertProperty(properties, sine16);
  InsertProperty(properties, sine32);
  InsertProperty(properties, sine64);

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Properties = %d\n", properties->NProperties());
    fflush(stdout);
  }

  // Return property set
  return properties;
}



////////////////////////////////////////////////////////////////////////
// Property set composition 
////////////////////////////////////////////////////////////////////////

static R3MeshPropertySet *
ComputeProperties(R3Mesh *mesh)
{
  // Allocate property set
  R3MeshPropertySet *properties = new R3MeshPropertySet(mesh);
  if (!properties) {
    fprintf(stderr, "Unable to allocate property set.\n");
    return NULL;
  }

  // Compute basic properties
  if (compute_basic_properties) {
    R3MeshPropertySet *basic_properties = ComputeBasicProperties(mesh);
    if (!basic_properties) return NULL;
    properties->Insert(basic_properties);
    delete basic_properties;
  }

  // Compute coordinate properties
  if (compute_coordinate_properties) {
    R3MeshPropertySet *coordinate_properties = ComputeCoordinateProperties(mesh);
    if (!coordinate_properties) return NULL;
    properties->Insert(coordinate_properties);
    delete coordinate_properties;
  }

  // Compute curvature properties
  if (compute_curvature_properties) {
    R3MeshPropertySet *curvature_properties = ComputeCurvatureProperties(mesh);
    if (!curvature_properties) return NULL;
    properties->Insert(curvature_properties);
    delete curvature_properties;
  }

  // Compute laplacian properties
  if (compute_laplacian_properties) {
    R3MeshPropertySet *laplacian_properties = ComputeLaplacianProperties(mesh);
    if (!laplacian_properties) return NULL;
    properties->Insert(laplacian_properties);
    delete laplacian_properties;
  }

  // Compute volume properties
  if (compute_volume_properties) {
    R3MeshPropertySet *volume_properties = ComputeVolumeProperties(mesh);
    if (!volume_properties) return NULL;
    properties->Insert(volume_properties);
    delete volume_properties;
  }

  // Compute boundary properties
  if (compute_boundary_properties) {
    R3MeshPropertySet *boundary_properties = ComputeBoundaryProperties(mesh);
    if (!boundary_properties) return NULL;
    properties->Insert(boundary_properties);
    delete boundary_properties;
  }

  // Compute dijkstra distance properties
  if (compute_dijkstra_distance_properties) {
    R3MeshPropertySet *dijkstra_distance_properties = ComputeDijkstraDistanceProperties(mesh);
    if (!dijkstra_distance_properties) return NULL;
    properties->Insert(dijkstra_distance_properties);
    delete dijkstra_distance_properties;
  }

  // Compute dijkstra histogram properties
  if (compute_dijkstra_histogram_properties) {
    R3MeshPropertySet *dijkstra_histogram_properties = ComputeDijkstraHistogramProperties(mesh);
    if (!dijkstra_histogram_properties) return NULL;
    properties->Insert(dijkstra_histogram_properties);
    delete dijkstra_histogram_properties;
  }

  // Compute ray tracing properties
  if (compute_raytrace_properties) {
    R3MeshPropertySet *raytrace_properties = ComputeRayTracingProperties(mesh);
    if (!raytrace_properties) return NULL;
    properties->Insert(raytrace_properties);
    delete raytrace_properties;
  }

  // Read mturk segmentation properties
  if (input_mturk_segmentation_name) {
    R3MeshPropertySet *segmentation_properties = ReadMTurkSegmentationProperties(mesh, input_mturk_segmentation_name);
    if (!segmentation_properties) return NULL;
    properties->Insert(segmentation_properties);
    delete segmentation_properties;
    if (input_mturk_annotation_name && input_mturk_label_mapping_name) {
      R3MeshProperty *segmentation_property = properties->Property("MTurkSegment");
      if (segmentation_property) {
        RNSymbolTable<MTurkLabel *> label_mapping;
        if (ReadMTurkLabelMapping(label_mapping, input_mturk_label_mapping_name)) {
          R3MeshPropertySet *annotation_properties = ReadMTurkAnnotationProperties(mesh, segmentation_property, &label_mapping, input_mturk_annotation_name);
          if (!annotation_properties) return NULL;
          properties->Insert(annotation_properties);
          delete annotation_properties;
        }
      }
    }
  }

  // Compute map properties
  if (input_map_name) {
    R3MeshPropertySet *map_properties = ComputeMapProperties(mesh, input_map_name);
    if (!map_properties) return NULL;
    properties->Insert(map_properties);
    delete map_properties;
  }

  // Compute solidtexture properties
  if (compute_solidtexture_properties) {
    R3MeshPropertySet *solidtexture_properties = ComputeSolidTextureProperties(mesh);
    if (!solidtexture_properties) return NULL;
    properties->Insert(solidtexture_properties);
    delete solidtexture_properties;
  }

  // Return property set
  return properties;
}



////////////////////////////////////////////////////////////////////////
// Program argument parsing
////////////////////////////////////////////////////////////////////////

static int 
ParseArgs(int argc, char **argv)
{
  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-v")) print_verbose = 1;
      else if (!strcmp(*argv, "-debug")) print_debug = 1;
      else if (!strcmp(*argv, "-basic")) { compute_basic_properties = 1; }
      else if (!strcmp(*argv, "-coordinate")) { compute_coordinate_properties = 1; }
      else if (!strcmp(*argv, "-curvature")) { compute_curvature_properties = 1; }
      else if (!strcmp(*argv, "-volume")) { compute_volume_properties = 1; }
      else if (!strcmp(*argv, "-boundary")) { compute_boundary_properties = 1; }
      else if (!strcmp(*argv, "-laplacian")) { compute_laplacian_properties = 1; }
      else if (!strcmp(*argv, "-dijkstra")) { compute_dijkstra_distance_properties = 1; }
      else if (!strcmp(*argv, "-dijkstra_statistics")) { compute_dijkstra_distance_properties = 1; }
      else if (!strcmp(*argv, "-dijkstra_histogram")) { compute_dijkstra_histogram_properties = 1; }
      else if (!strcmp(*argv, "-raytrace")) { compute_raytrace_properties = 1; }
      else if (!strcmp(*argv, "-solidtexture")) { compute_solidtexture_properties = 1; }
      else if (!strcmp(*argv, "-map")) { argv++; argc--; input_map_name = *argv; }
      else if (!strcmp(*argv, "-mturk_semantic_segmentation")) {
        argv++; argc--; input_mturk_segmentation_name = *argv; 
        argv++; argc--; input_mturk_annotation_name = *argv;         
        argv++; argc--; input_mturk_label_mapping_name = *argv;         
      }
      else if (!strcmp(*argv, "-all")) { 
        compute_basic_properties = 1;
        compute_coordinate_properties = 1;
        compute_curvature_properties = 1;
        compute_volume_properties = 1;
        compute_boundary_properties = 1;
        // compute_laplacian_properties = 1;
        compute_dijkstra_distance_properties = 1;
        compute_dijkstra_histogram_properties = 1;
        compute_raytrace_properties = 1;
      }
      else { 
        fprintf(stderr, "Invalid program argument: %s", *argv); 
        exit(1); 
      }
      argv++; argc--;
    }
    else {
      if (!input_mesh_name) input_mesh_name = *argv;
      else if (!output_properties_name) output_properties_name = *argv;
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
  }

  // Check input filename
  if (!input_mesh_name || !output_properties_name) {
    fprintf(stderr, "Usage: msh2prp meshfile propertiesfile [options]\n");
    return 0;
  }

  // Return OK status 
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
  // Parse program arguments
  if (!ParseArgs(argc, argv)) exit(1);

  // Read mesh file
  R3Mesh *mesh = ReadMesh(input_mesh_name);
  if (!mesh) exit(-1);

  // Compute property set
  R3MeshPropertySet *properties = ComputeProperties(mesh);
  if (!properties) exit(-1);

  // Write properties file
  if (!WriteProperties(properties, output_properties_name)) exit(-1);

  // Return success 
  return 0;
}




