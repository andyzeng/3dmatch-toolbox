// Source file for the mesh converter program



// Include files 

#include "R3Shapes/R3Shapes.h"



// Program arguments

const char *input_name = NULL;
const char *output_name = NULL;
const char *color_name = NULL;
int flip_faces = 0;
int clean = 0;
int smooth = 0;
R3Affine xform(R4Matrix(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1));
RNLength min_edge_length = 0;
RNLength max_edge_length = 0;
char *xform_name = NULL;
int scale_by_area = 0;
int align_by_pca = 0;
int print_verbose = 0;



////////////////////////////////////////////////////////////////////////
// I/O STUFF
////////////////////////////////////////////////////////////////////////

static R3Mesh *
ReadMesh(const char *mesh_name)
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

  // Check if mesh is valid
  assert(mesh->IsValid());

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



static int
WriteMesh(R3Mesh *mesh, const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Write mesh to file
  if (!mesh->WriteFile(filename)) return 0;

  // Print statistics
  if (print_verbose) {
    printf("Wrote mesh to %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Faces = %d\n", mesh->NFaces());
    printf("  # Edges = %d\n", mesh->NEdges());
    printf("  # Vertices = %d\n", mesh->NVertices());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
ReadMatrix(R4Matrix& m, const char *filename)
{
  // Open file
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Unable to open matrix file: %s\n", filename);
    return 0;
  }

  // Read matrix from file
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      double value;
      fscanf(fp, "%lf", &value);
      m[i][j] = value;
    }
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// PROCESSING STUFF
////////////////////////////////////////////////////////////////////////

static int
CopyColors(R3Mesh *mesh, const char *source_mesh_name)
{
  // Read source mesh
  R3Mesh source_mesh;
  if (!source_mesh.ReadFile(source_mesh_name)) return 0;

  // Create kdtree
  R3MeshSearchTree kdtree(&source_mesh);
  
  // Copy colors
  for (int i = 0; i < mesh->NVertices(); i++) {
    R3MeshVertex *vertex = mesh->Vertex(i);
    mesh->SetVertexColor(vertex, RNblack_rgb);
    const R3Point& position = mesh->VertexPosition(vertex);

    // Search kdtree
    R3MeshIntersection closest;
    kdtree.FindClosest(position, closest);
    if (closest.type == R3_MESH_VERTEX_TYPE) {
      mesh->SetVertexColor(vertex, source_mesh.VertexColor(closest.vertex));
    }
    else if (closest.type == R3_MESH_EDGE_TYPE) {
      R3Span span = mesh->EdgeSpan(closest.edge);
      RNScalar t = span.T(position);
      int k = (t < 0.5 * span.Length()) ?  0 : 1;
      R3MeshVertex *source_vertex = source_mesh.VertexOnEdge(closest.edge, k);
      mesh->SetVertexColor(vertex, source_mesh.VertexColor(source_vertex));
    }
    else if (closest.type == R3_MESH_FACE_TYPE) {
      R3Point b = source_mesh.FaceBarycentric(closest.face, position);
      int k = b.Vector().MaxDimension();
      R3MeshVertex *source_vertex = source_mesh.VertexOnFace(closest.face, k);
      mesh->SetVertexColor(vertex, source_mesh.VertexColor(source_vertex));
    }
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// PROGRAM ARGUMENT PARSING
////////////////////////////////////////////////////////////////////////

int ParseArgs(int argc, char **argv)
{
  // Check number of arguments
  if (argc == 1) {
    printf("Usage: mesh2mesh inputname outputname [options]\n");
    exit(0);
  }

  // Parse arguments
  argc--; argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      R3Affine prev_xform = xform;
      if (!strcmp(*argv, "-v")) print_verbose = 1;
      else if (!strcmp(*argv, "-flip")) flip_faces = 1;
      else if (!strcmp(*argv, "-clean")) clean = 1;
      else if (!strcmp(*argv, "-smooth")) smooth = 1;
      else if (!strcmp(*argv, "-align_by_pca")) align_by_pca = 1;
      else if (!strcmp(*argv, "-scale_by_area")) scale_by_area = 1;
      else if (!strcmp(*argv, "-scale")) { argv++; argc--; xform = R3identity_affine; xform.Scale(atof(*argv)); xform.Transform(prev_xform); }
      else if (!strcmp(*argv, "-tx")) { argv++; argc--; xform = R3identity_affine; xform.XTranslate(atof(*argv)); xform.Transform(prev_xform); }
      else if (!strcmp(*argv, "-ty")) { argv++; argc--; xform = R3identity_affine; xform.YTranslate(atof(*argv)); xform.Transform(prev_xform);}
      else if (!strcmp(*argv, "-tz")) { argv++; argc--; xform = R3identity_affine; xform.ZTranslate(atof(*argv)); xform.Transform(prev_xform);}
      else if (!strcmp(*argv, "-sx")) { argv++; argc--; xform = R3identity_affine; xform.XScale(atof(*argv)); xform.Transform(prev_xform);}
      else if (!strcmp(*argv, "-sy")) { argv++; argc--; xform = R3identity_affine; xform.YScale(atof(*argv)); xform.Transform(prev_xform);}
      else if (!strcmp(*argv, "-sz")) { argv++; argc--; xform = R3identity_affine; xform.ZScale(atof(*argv)); xform.Transform(prev_xform);}
      else if (!strcmp(*argv, "-rx")) { argv++; argc--; xform = R3identity_affine; xform.XRotate(RN_PI*atof(*argv)/180.0); xform.Transform(prev_xform);}
      else if (!strcmp(*argv, "-ry")) { argv++; argc--; xform = R3identity_affine; xform.YRotate(RN_PI*atof(*argv)/180.0); xform.Transform(prev_xform);}
      else if (!strcmp(*argv, "-rz")) { argv++; argc--; xform = R3identity_affine; xform.ZRotate(RN_PI*atof(*argv)/180.0); xform.Transform(prev_xform);}
      else if (!strcmp(*argv, "-xform")) { argv++; argc--; R4Matrix m;  if (ReadMatrix(m, *argv)) { xform = R3identity_affine; xform.Transform(R3Affine(m)); xform.Transform(prev_xform);} } 
      else if (!strcmp(*argv, "-min_edge_length")) { argv++; argc--; min_edge_length = atof(*argv); }
      else if (!strcmp(*argv, "-max_edge_length")) { argv++; argc--; max_edge_length = atof(*argv); }
      else if (!strcmp(*argv, "-color")) { argv++; argc--; color_name = *argv; }
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
    else {
      if (!input_name) input_name = *argv;
      else if (!output_name) output_name = *argv;
      else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
      argv++; argc--;
    }
  }

  // Check input filename
  if (!input_name) {
    fprintf(stderr, "You did not specify an input file name.\n");
    return 0;
  }

  // Check output filename
  if (!output_name) {
    fprintf(stderr, "You did not specify an output file name.\n");
    return 0;
  }

  // Return OK status 
  return 1;
}



////////////////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
  // Check number of arguments
  if (!ParseArgs(argc, argv)) exit(1);

  // Read mesh
  R3Mesh *mesh = ReadMesh(input_name);
  if (!mesh) exit(-1);

  // Clean 
  if (clean) {
    mesh->DeleteUnusedEdges();
    mesh->DeleteUnusedVertices();
  }

  // Flip every face
  if (flip_faces) {
    for (int i = 0; i < mesh->NFaces(); i++) {
      mesh->FlipFace(mesh->Face(i));
    }
  }

  // Smooth
  if (smooth) {
    mesh->Smooth();
  }

  // Subdivide edges that are too long
  if (max_edge_length > 0) {
    mesh->SubdivideLongEdges(max_edge_length);
  }

  // Split edges that are too long
  if (min_edge_length > 0) {
    mesh->CollapseShortEdges(min_edge_length);
  }

  // Transform 
  if (!xform.IsIdentity()) {
    mesh->Transform(xform);
  }

  // Normalize translation, rotation, and scale
  if (align_by_pca) {
    R3Affine xf = mesh->PCANormalizationTransformation();
    mesh->Transform(xf);
  }

  // Scale based on mesh area
  if (scale_by_area) {
    RNArea area = mesh->Area();
    if (area > 0) xform.Scale(1 / sqrt(area));
  }

  // Transfer colors
  if (color_name) {
    CopyColors(mesh, color_name);
  }
  
  // Write mesh
  if (!WriteMesh(mesh, output_name)) exit(-1);

  // Return success 
  return 0;
}

















