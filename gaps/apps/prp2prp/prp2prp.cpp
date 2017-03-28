// Source file for the GAPS mesh analysis program



////////////////////////////////////////////////////////////////////////
// Include files 
////////////////////////////////////////////////////////////////////////

#include "R3Shapes/R3Shapes.h"



////////////////////////////////////////////////////////////////////////
// Program arguments
////////////////////////////////////////////////////////////////////////

static const char *input_mesh_name = NULL;
static const char *input_properties_name = NULL;
static const char *output_properties_name = NULL;
static int parameters_in_relative_units = 1;
static int print_verbose = 0;
static int print_debug = 0;



////////////////////////////////////////////////////////////////////////
// Input/output functions
////////////////////////////////////////////////////////////////////////

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



static R3MeshPropertySet *
ReadProperties(R3Mesh *mesh, const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate properties
  R3MeshPropertySet *properties = new R3MeshPropertySet(mesh);
  if (!properties) {
    fprintf(stderr, "Unable to allocate properties for %s\n", filename);
    return NULL;
  }

  // Read properties from file
  if (!properties->Read(filename)) {
    delete properties;
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("Read properties from %s ...\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Properties = %d\n", properties->NProperties());
    printf("  # Vertices = %d\n", properties->Mesh()->NVertices());
    fflush(stdout);
  }

  // Print more statistics
  if (print_debug) {
    printf("\n");
    for (int i = 0; i < properties->NProperties(); i++) {
      R3MeshProperty *property = properties->Property(i);
      printf("  Property %d %s\n", i, property->Name());
      printf("    Minimum = %g\n", property->Minimum());
      printf("    Maximum = %g\n", property->Maximum());
      printf("    Median = %g\n", property->Median());
      printf("    Mean = %g\n", property->Mean());
      printf("    # Minima = %d\n", property->LocalMinimumCount());
      printf("    # Maxima = %d\n", property->LocalMaximumCount());
      printf("    L1Norm = %g\n", property->L1Norm());
      printf("    L2Norm = %g\n", property->L2Norm());
      printf("\n");
    }
  }

  // Return property set
  return properties;
}



static int
WriteProperties(R3MeshPropertySet *properties, const char *filename)
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

  // Print more statistics
  if (print_debug) {
    printf("\n");
    for (int i = 0; i < properties->NProperties(); i++) {
      R3MeshProperty *property = properties->Property(i);
      printf("  Property %d %s\n", i, property->Name());
      printf("    Minimum = %g\n", property->Minimum());
      printf("    Maximum = %g\n", property->Maximum());
      printf("    Median = %g\n", property->Median());
      printf("    Mean = %g\n", property->Mean());
      printf("    # Minima = %d\n", property->LocalMinimumCount());
      printf("    # Maxima = %d\n", property->LocalMaximumCount());
      printf("    L1Norm = %g\n", property->L1Norm());
      printf("    L2Norm = %g\n", property->L2Norm());
      printf("\n");
    }
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////////

static int
IsNumber(const char *s)
{
  if (!strcmp(s, "keep") || !strcmp(s, "Keep")) return 1;
  if (*s == '\0') return 0;
  if (*s == '+') s++;
  if (*s == '\0') return 0;
  if (*s == '-') s++;
  if (*s == '\0') return 0;
  while (*s) {
    if (!isdigit(*s) && (*s != '.') && (*s != 'E')) return 0;
    s++;
  }
  return 1;
}



static RNScalar
Scale(R3Mesh *mesh)
{
  // Check if absolute scale
  if (!parameters_in_relative_units) return 1.0;

  // Compute smallest sigma
  static RNScalar scale = 1;
  static R3Mesh *last_mesh = NULL;
  if (mesh != last_mesh) {
    RNLength length = sqrt(mesh->Area());
    if (length > 0) scale = 1.0 / length;
    last_mesh = mesh;
  }

  // Return scale
  return scale;
}



static void 
Test(R3MeshProperty *property)
{
  // For debugging, etc.
  R3Mesh *mesh = property->Mesh();
  for (int i = 0; i < mesh->NVertices(); i++) {
    RNScalar value = property->VertexValue(i);
    value = 0.5 + 0.5 * sin(10 * value);
    property->SetVertexValue(i, value);
  }
}



static RNArray<R3MeshProperty *>
FindProperties(R3MeshPropertySet *properties, const char *property_name)
{
  // Make an array of matching properties
  RNArray<R3MeshProperty *> result;

  // Check if "all"
  if (!strcmp(property_name, "all")) {
    // Insert all properties
    for (int i = 0; i < properties->NProperties(); i++) {
      R3MeshProperty *property = properties->Property(i);
      result.Insert(property);
    }
  }
  else if (properties->Property(property_name)) { 
    // Insert single property
    R3MeshProperty *property = properties->Property(property_name);
    result.Insert(property);
  }
  else { 
    // Insert properties names read from a file
    char buffer[1024];
    FILE *fp = fopen(property_name, "r");
    if (!fp) { fprintf(stderr, "Invalid property name: %s\n", property_name); exit(-1); }
    while (fscanf(fp, "%s", buffer) == (unsigned int) 1) {
      R3MeshProperty *property = properties->Property(buffer);
      if (!property) { fprintf(stderr, "Unrecognized property %s in file %s\n", buffer, property_name); exit(-1); }
      result.Insert(property);
    }
    fclose(fp);
  }

  // Check if found any properties
  if (result.IsEmpty()) {
    fprintf(stderr, "No properties found for %s\n", property_name);
    exit(-1);
  }

  // Return array of properties
  return result;
}



R3MeshProperty
GetProperty(R3MeshPropertySet *properties, const char *name)
{
  // Initialize property
  R3MeshProperty property(properties->Mesh());

  // Find property in properties
  R3MeshProperty *existing_property = properties->Property(name);
  if (existing_property) return R3MeshProperty(*existing_property);

  // Read property from file
  R3MeshProperty new_property(properties->Mesh());
  if (new_property.Read(name)) return new_property;

  // Property not found
  fprintf(stderr, "Unable to get property %s\n", name); 
  exit(-1); 
}



////////////////////////////////////////////////////////////////////////
// Operation functions
////////////////////////////////////////////////////////////////////////

static void
ApplyOperation(R3MeshPropertySet *properties, R3MeshProperty *property, const char *operation_name, 
  int& argc, char **& argv, RNBoolean update_name = TRUE)
{
  // Get convenient variables
  R3Mesh *mesh = property->Mesh();

  // Apply operation
  if (!strcmp(operation_name, "Test")) Test(property);
  else if (!strcmp(operation_name, "Abs")) property->Abs();
  else if (!strcmp(operation_name, "Sqrt")) property->Sqrt();
  else if (!strcmp(operation_name, "Square")) property->Square();
  else if (!strcmp(operation_name, "Negate")) property->Negate();
  else if (!strcmp(operation_name, "Invert")) property->Invert();
  else if (!strcmp(operation_name, "Clear")) property->Clear();
  else if (!strcmp(operation_name, "Normalize")) property->Normalize();
  else if (!strcmp(operation_name, "Percentile")) property->Percentilize();
  else if (!strcmp(operation_name, "Laplace")) property->Laplace();
  else if (!strcmp(operation_name, "Substitute")) {
    argc--; argv++; RNScalar current = atof(*argv);
    if (!strcmp(*argv, "unknown")) current = RN_UNKNOWN;
    else if (!strcmp(*argv, "infinity")) current = RN_INFINITY;
    argc--; argv++; RNScalar replacement = atof(*argv);
    if (!strcmp(*argv, "unknown")) replacement = RN_UNKNOWN;
    else if (!strcmp(*argv, "infinity")) replacement = RN_INFINITY;
    property->Substitute(current, replacement);
  }
  else if (!strcmp(operation_name, "Add")) {
    argc--; argv++; const char *operand = *argv;
    if (IsNumber(operand)) property->Add(atof(operand));
    else property->Add(GetProperty(properties, operand));
  }
  else if (!strcmp(operation_name, "Subtract")) {
    argc--; argv++; const char *operand = *argv;
    if (IsNumber(operand)) property->Subtract(atof(operand));
    else property->Subtract(GetProperty(properties, operand));
  }
  else if (!strcmp(operation_name, "Multiply")) {
    argc--; argv++; const char *operand = *argv;
    if (IsNumber(operand)) property->Multiply(atof(operand));
    else property->Multiply(GetProperty(properties, operand));
  }
  else if (!strcmp(operation_name, "Divide")) {
    argc--; argv++; const char *operand = *argv;
    if (IsNumber(operand)) property->Divide(atof(operand));
    else property->Divide(GetProperty(properties, operand));
  }
  else if (!strcmp(operation_name, "Pow")) {
    argc--; argv++; const char *operand = *argv;
    if (IsNumber(operand)) property->Pow(atof(operand));
    else property->Pow(GetProperty(properties, operand));
  }
  else if (!strcmp(operation_name, "Mask")) {
    argc--; argv++; const char *operand = *argv;
    property->Mask(GetProperty(properties, operand));
  }
  else if (!strcmp(operation_name, "Dilate")) {
    argc--; argv++; RNScalar radius = atof(*argv);
    property->Dilate(radius * Scale(mesh));
  }
  else if (!strcmp(operation_name, "Erode")) {
    argc--; argv++; RNScalar radius = atof(*argv);
    property->Erode(radius * Scale(mesh));
  }
  else if (!strcmp(operation_name, "NonExtremumSuppression")) {
    argc--; argv++; RNScalar radius = atof(*argv);
    property->NonExtremumSuppression(radius * Scale(mesh));
  }
  else if (!strcmp(operation_name, "NonMaximumSuppression")) {
    argc--; argv++; RNScalar radius = atof(*argv);
    property->NonMaximumSuppression(radius * Scale(mesh));
  }
  else if (!strcmp(operation_name, "NonMinimumSuppression")) {
    argc--; argv++; RNScalar radius = atof(*argv);
    property->NonMinimumSuppression(radius * Scale(mesh));
  }
  else if (!strcmp(operation_name, "Blur")) {
    argc--; argv++; RNScalar sigma = atof(*argv);
    property->Blur(sigma * Scale(mesh));
  }
  else if (!strcmp(operation_name, "DoG")) {
    argc--; argv++; RNScalar sigma = atof(*argv);
    property->DoG(sigma * Scale(mesh));
  }
  else if (!strcmp(operation_name, "Strength")) {
    argc--; argv++; RNScalar sigma = atof(*argv);
    property->Strength(sigma * Scale(mesh));
  }
  else if (!strcmp(operation_name, "Threshold")) {
    argc--; argv++; RNScalar threshold = atof(*argv);
    argc--; argv++; const char *below_str = *argv;
    argc--; argv++; const char *above_str = *argv;
    RNScalar below_value = RN_UNKNOWN;
    if (!strcmp(below_str, "keep")) below_value = R3_MESH_PROPERTY_KEEP_VALUE;
    else if (strcmp(below_str, "unknown")) below_value = atof(below_str);
    RNScalar above_value = RN_UNKNOWN;
    if (!strcmp(above_str, "keep")) above_value = R3_MESH_PROPERTY_KEEP_VALUE;
    else if (strcmp(above_str, "unknown")) above_value = atof(above_str);
    property->Threshold(threshold, below_value, above_value);
  }
  else { 
    fprintf(stderr, "Unrecognized operation: %s\n", operation_name); 
    exit(-1); 
  }

  // Set new property name
  if (update_name) {
    char name[4096] = { '\0' };
    strcpy(name, property->Name());
    strcat(name, "_"); strcat(name, operation_name);
    property->SetName(name);
  }

  // Print message
  if (print_debug) {
    printf("  %s\n", property->Name());
    fflush(stdout);
  }
}



static void
Merge(R3MeshPropertySet *properties, const char *filename, const char *property_name)
{
  // Read new property set
  R3MeshPropertySet *new_properties = ReadProperties(properties->Mesh(), filename);
  if (!new_properties) exit(-1);

  // Find properties matching property_name
  RNArray<R3MeshProperty *> subset = FindProperties(new_properties, property_name);

  // Apply operation to properties
  for (int i = 0; i < subset.NEntries(); i++) {
    R3MeshProperty *property = subset[i];
    properties->Insert(property);
  }

  // Delete new property set (does not delete properties)
  delete new_properties;

  // Print message
  if (print_debug) {
    printf("  Insert %s\n", filename);
    fflush(stdout);
  }
}



static void
Replace(R3MeshPropertySet *properties, const char *property_name, const char *operation_name, int& argc, char **& argv) 
{
  // What a hack
  int save_argc = argc;
  char **save_argv = argv;

  // Find properties matching property_name
  RNArray<R3MeshProperty *> subset = FindProperties(properties, property_name);

  // Apply operation to properties
  for (int i = 0; i < subset.NEntries(); i++) {
    R3MeshProperty *property = subset[i];
    argc = save_argc; argv = save_argv;
    ApplyOperation(properties, property, operation_name, argc, argv, FALSE);
  }
}



static void
Insert(R3MeshPropertySet *properties, const char *property_name, const char *operation_name, int& argc, char **& argv)
{
  // What a hack
  int save_argc = argc;
  char **save_argv = argv;

  // Find properties matching property_name
  RNArray<R3MeshProperty *> subset = FindProperties(properties, property_name);

  // Insert copy and apply operation to properties
  for (int i = 0; i < subset.NEntries(); i++) {
    R3MeshProperty *property = subset[i];
    R3MeshProperty *copy = new R3MeshProperty(*property);
    argc = save_argc; argv = save_argv;
    ApplyOperation(properties, copy, operation_name, argc, argv);
    properties->Insert(copy);
  }
}



static void
Remove(R3MeshPropertySet *properties, const char *property_name)
{
  // Find properties matching property_name
  RNArray<R3MeshProperty *> subset = FindProperties(properties, property_name);

  // Remove properties
  for (int i = 0; i < subset.NEntries(); i++) {
    R3MeshProperty *property = subset[i];
    properties->Remove(property);
  }
}



static void
Select(R3MeshPropertySet *properties, const char *property_name)
{
  // Find properties matching property_name
  RNArray<R3MeshProperty *> subset = FindProperties(properties, property_name);

  // Reset properties to include only subset
  properties->Empty();
  for (int i = 0; i < subset.NEntries(); i++) {
    R3MeshProperty *property = subset[i];
    properties->Insert(property);
  }
}



static void
Rename(R3MeshPropertySet *properties, const char *property_name, const char *new_property_name)
{
  // Find properties matching property_name
  RNArray<R3MeshProperty *> subset = FindProperties(properties, property_name);
  if (subset.NEntries() != 1) { fprintf(stderr, "Unable to find unambiguous property %s\n", property_name); abort(); }
  subset[0]->SetName((char *) new_property_name);
}



static void
Multiscale(R3MeshPropertySet *properties, const char *property_name, int& argc, char **& argv)
{
  // Get convenient variables
  R3Mesh *mesh = properties->Mesh();
  RNScalar sigma0 = atof(argv[0]) * Scale(mesh);
  RNScalar scale_factor = atof(argv[1]);
  int nscales = atoi(argv[2]);
  argc -= 3; argv += 3;
  if (sigma0 == 0) return;
  if (scale_factor == 0) return;
  if (nscales == 0) return;;

  // Find properties matching property_name
  RNArray<R3MeshProperty *> subset = FindProperties(properties, property_name);
  if (subset.NEntries() == 0) return;;

  // Allocate multiscale properties
  R3MeshProperty ***blurs = new R3MeshProperty **[ subset.NEntries() ];
  for (int i = 0; i < subset.NEntries(); i++) {
    R3MeshProperty *property = subset.Kth(i);
    blurs[i] = new R3MeshProperty *[nscales];
    for (int j = 0; j < nscales; j++) {
      char name[256];
      sprintf(name, "%s_Multiscale_%d", property->Name(), j+1);
      blurs[i][j] = new R3MeshProperty(mesh, name);
    }
  }

  // Compute multiscale properties
  for (int k = 0; k < mesh->NVertices(); k++) {
    R3MeshVertex *vertex = mesh->Vertex(k);

    // Compute distances
    RNLength *distances = mesh->DijkstraDistances(vertex);

    // Consider all properties
    for (int i = 0; i < subset.NEntries(); i++) {
      RNScalar sigma = sigma0;
      for (int j = 0; j < nscales; j++) {
        // Compute blurred values
        RNScalar total_value = 0;
        RNScalar total_weight = 0;
        RNScalar denom = -2 * sigma * sigma;
        for (int m = 0; m < mesh->NVertices(); m++) {
          RNLength distance = distances[m];
          if (distance > 3 * sigma) continue;
          RNScalar value = subset[i]->VertexValue(m);;
          RNScalar weight = exp(distance * distance / denom); 
          total_value += weight * value;
          total_weight += weight;
        }

        // Assign blurred value (normalized by total weight)
        if (total_weight > 0) {
          blurs[i][j]->SetVertexValue(k, total_value / total_weight);
        }

        // Update sigma
        sigma *= scale_factor;
      }
    }

    // Delete distances
    delete [] distances;
  }

  // Insert multiscale properties
  for (int i = 0; i < subset.NEntries(); i++) {
    for (int j = 0; j < nscales; j++) {
      properties->Insert(blurs[i][j]);
    }
  }
}



static int
ApplyOperations(R3Mesh *mesh, R3MeshPropertySet *properties, int argc, char **argv)
{
  // Check if any operations required
  if (argc == 0) return 1;

  // Start statistics
  RNTime start_time;
  start_time.Read();
  int count = 0;

  // Print message
  if (print_verbose) {
    printf("Applying operations ...\n");
    fflush(stdout);
  }

  // Apply operations
  while (argc > 0) {
    if (!strcmp(*argv, "-merge")) {
      argv++; argc--; const char *filename = *argv;
      argv++; argc--; const char *property_name = *argv;
      Merge(properties, filename, property_name);
      count++;
    }
    else if (!strcmp(*argv, "-replace")) {
      argv++; argc--; const char *property_name = *argv;
      argv++; argc--; const char *operation_name = *argv;
      Replace(properties, property_name, operation_name, argc, argv);
      count++;
    }
    else if (!strcmp(*argv, "-insert")) {
      argv++; argc--; const char *property_name = *argv;
      argv++; argc--; const char *operation_name = *argv;
      Insert(properties, property_name, operation_name, argc, argv);
      count++;
    }
    else if (!strcmp(*argv, "-multiscale")) {
      argv++; argc--; const char *property_name = *argv;
      Multiscale(properties, property_name, argc, argv);
      count++;
    }
    else if (!strcmp(*argv, "-remove")) {
      argv++; argc--; const char *property_name = *argv;
      Remove(properties, property_name);
      count++;
    }
    else if (!strcmp(*argv, "-select")) {
      argv++; argc--; const char *property_name = *argv;
      Select(properties, property_name);
      count++;
    }
    else if (!strcmp(*argv, "-rename")) {
      argv++; argc--; const char *property_name = *argv;
      argv++; argc--; const char *new_property_name = *argv;
      Rename(properties, property_name, new_property_name);
      count++;
    }
    else if (!strcmp(*argv, "-sort")) {
      properties->SortByName();
      count++;
    }
    else if (!strcmp(*argv, "-parameters_in_absolute_units")) {
      parameters_in_relative_units = 0;
    }
    else if (!strcmp(*argv, "-parameters_in_relative_units")) {
      parameters_in_relative_units = 1;
    }
    else if (!strcmp(*argv, "-debug")) {
      print_debug = 1;
    }
    else if (!strcmp(*argv, "-v")) {
      print_verbose = 1;
    }
    else {
      fprintf(stderr, "Invalid program argument: %s\n", *argv);
      return 0;
    }

    // Update counters
    argv++; argc--; 
  }

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Operations  %d\n", count);
    fflush(stdout);
  }

  // Return success
  return 1;
}



int ParseArgs(int argc, char **argv)
{
  // Parse arguments
  for (int i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "-v")) print_verbose = 1; 
    else if (!strcmp(argv[i], "-debug")) print_debug = 1; 
    else if (!strcmp(argv[i], "-parameters_in_absolute_units")) parameters_in_relative_units = 0; 
    else if (!strcmp(argv[i], "-parameters_in_relative_units")) parameters_in_relative_units = 1; 
  }

  // Return OK status 
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
  // Check program arguments
  if (argc < 4) { printf("Usage: prp2prp inputmesh inputprops outputprops [options]\n"); exit(-1); }
  input_mesh_name = argv[1];
  input_properties_name = argv[2];
  output_properties_name = argv[3];
  argc -=4; argv+=4;

  // Parse program arguments
  if (!ParseArgs(argc, argv)) exit(-1);

  // Read mesh file
  R3Mesh *mesh = ReadMesh(input_mesh_name);
  if (!mesh) exit(-1);

  // Read properties file
  R3MeshPropertySet *properties = ReadProperties(mesh, input_properties_name);
  if (!properties) exit(-1);

  // Apply property processing operations
  if (!ApplyOperations(mesh, properties, argc, argv)) exit(-1);

  // Write properties file
  if (!WriteProperties(properties, output_properties_name)) exit(-1);

  // Return success 
  return 0;
}



