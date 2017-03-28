// Source file for the surfel info program



////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// Program arguments
////////////////////////////////////////////////////////////////////////

static char *input_scene_name = NULL;
static char *input_database_name = NULL;
static int print_features = 0;
static int print_scans = 0;
static int print_objects = 0;
static int print_labels = 0;
static int print_object_properties = 0;
static int print_label_properties = 0;
static int print_tree = 0;
static int print_nodes = 0;
static int print_database = 0;
static int print_blocks = 0;
static int print_surfels = 0;
static char *accuracy_arff_name = NULL;



////////////////////////////////////////////////////////////////////////
// Surfel database I/O Functions
////////////////////////////////////////////////////////////////////////

static R3SurfelScene *
OpenScene(const char *input_scene_name, const char *input_database_name)
{
  // Allocate surfel scene
  R3SurfelScene *scene = new R3SurfelScene();
  if (!scene) {
    fprintf(stderr, "Unable to allocate scene\n");
    return NULL;
  }

  // Open surfel scene files
  if (!scene->OpenFile(input_scene_name, input_database_name, "r", "r")) {
    delete scene;
    return NULL;
  }

  // Return scene
  return scene;
}



static int
CloseScene(R3SurfelScene *scene)
{
  // Close surfel scene
  if (!scene->CloseFile()) return 0;

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Print accuracy vs. arff file
////////////////////////////////////////////////////////////////////////

static int 
PrintAccuracy(R3SurfelScene *scene, const char *arff_filename)
{
#if 0
  // Open file
  FILE *fp = fopen(arff_filename, "r");
  if (!fp) {
    fprintf(stderr, "Unable to open ARFF file %s\n", arff_filename);
    return 0;
  }

  // Initialize statistics
  int ncorrect = 0;
  int nincorrect = 0;
  int nunlabeled = 0;
  int nobjects = 0;

  // Parse file
  static char buffer [ 1024 * 1024 ];
  double position[3], axis[3][3], plane[4], color[3];
  while (fgets(buffer, 1024 * 1024, fp)) {
    if (!strncmp(buffer, "% Object", 8)) {
      // Parse basic stuff in comment
      /* char *percent_keyword = */ strtok(buffer, " \t\n");
      /* char *object_keyword = */ strtok(NULL, " \t\n");
      /* int id = */ atoi(strtok(NULL, " \t\n"));
      position[0] = atof(strtok(NULL, " \t\n"));
      position[1] = atof(strtok(NULL, " \t\n"));
      position[2] = atof(strtok(NULL, " \t\n"));
      axis[0][0] = atof(strtok(NULL, " \t\n"));
      axis[0][1] = atof(strtok(NULL, " \t\n"));
      axis[0][2] = atof(strtok(NULL, " \t\n"));
      axis[1][0] = atof(strtok(NULL, " \t\n"));
      axis[1][1] = atof(strtok(NULL, " \t\n"));
      axis[1][2] = atof(strtok(NULL, " \t\n"));
      axis[2][0] = atof(strtok(NULL, " \t\n"));
      axis[2][1] = atof(strtok(NULL, " \t\n"));
      axis[2][2] = atof(strtok(NULL, " \t\n"));
      plane[0] = atof(strtok(NULL, " \t\n"));
      plane[1] = atof(strtok(NULL, " \t\n"));
      plane[2] = atof(strtok(NULL, " \t\n"));
      plane[3] = atof(strtok(NULL, " \t\n"));
      color[0] = atof(strtok(NULL, " \t\n"));
      color[1] = atof(strtok(NULL, " \t\n"));
      color[2] = atof(strtok(NULL, " \t\n"));

      // Parse feature info in comment
      int nfeatures = atoi(strtok(NULL, " \t\n"));
      for (int i = 0; i < nfeatures; i++) {
        int feature_type = atoi(strtok(NULL, " \t\n"));
        if (feature_type == 1) {
          // Spin image
          /* double radius = */ atof(strtok(NULL, " \t\n"));
          /* double height = */ atof(strtok(NULL, " \t\n"));
          /* int nshells = */ atoi(strtok(NULL, " \t\n"));
          /* int nslices = */ atoi(strtok(NULL, " \t\n"));
        }
        else if (feature_type == 2) {
          // Template
          /* double radius = */ atof(strtok(NULL, " \t\n"));
          /* double height = */ atof(strtok(NULL, " \t\n"));
          /* double spacing = */ atof(strtok(NULL, " \t\n"));
          /* int ntemplates = */ atoi(strtok(NULL, " \t\n"));
        }
        else if (feature_type == 3) {
          // Oracle
          /* int parent1 = */ atoi(strtok(NULL, " \t\n"));
          /* int parent2 = */ atoi(strtok(NULL, " \t\n"));
          /* int parent3 = */ atoi(strtok(NULL, " \t\n"));
        }
        else if (feature_type == 4) {
          // Grid
          /* int nvalues = */ atoi(strtok(NULL, " \t\n"));
        }
        else if (feature_type == 5) {
          // Point set
          // Nothing
        }
        else if (feature_type == 6) {
          // Shape context
          /* double radius = */ atof(strtok(NULL, " \t\n"));
          /* double height = */ atof(strtok(NULL, " \t\n"));
          /* int nshells = */ atoi(strtok(NULL, " \t\n"));
          /* int nslices = */ atoi(strtok(NULL, " \t\n"));
          /* int nsectors = */ atoi(strtok(NULL, " \t\n"));
        }
        else if (feature_type == 7) {
          // Segmentation info
          // Nothing
        }
       else if (feature_type == 8) {
          // Alignments
          // Nothing
        }
      }

      // Parse assignments
      int label_id = -1;
      double confidence = 0;
      int nassignments = atoi(strtok(NULL, " \t\n"));
      for (int i = 0; i < nassignments; i++) {
        label_id = atoi(strtok(NULL, " \t\n"));
        /* char *label_name = */ strtok(NULL, " \t\n");
        confidence = atof(strtok(NULL, " \t\n"));
      }

      // Get object
      char object_name[1024];
      sprintf(object_name, "%d_%.3f_%.3f_%.3f", label_id, position[0], position[1], position[2]);
      R3SurfelObject *object = scene->FindObjectByName(object_name);
      if (!object) {
        fprintf(stderr, "Unable to find object %s\n", object_name);
        return 0;
      }

      // Update statistics
      if (label_id >= 0) {
        if (object->NLabelAssignments() > 0) {
          // Check if label matches
          R3SurfelLabelAssignment *assignment = object->LabelAssignment(0);
          R3SurfelLabel *scene_label = assignment->Label();

          // Search for match to arff label
          int correct = 0;
          LPLabelSet *label_set = LPCurrentLabelSDatabase()->NSurfels()et();
          while (1) {
            if (label_id == scene_label->Identifier()) { correct = 1; break; }
            if (!label_set->Label(label_id)->Parent()) break;
            if (label_set->Label(label_id)->Parent() == label_set->Label(label_id)) break;
            label_id = label_set->Label(label_id)->Parent()->ID();
          } 
          printf("%40s %d\n", scene_label->Name(), correct);
          if (correct) ncorrect++;
          else nincorrect++;
        }
        else {
          nunlabeled++;
        }
        nobjects++;
      }
    }
  }

  // Close file
  fclose(fp);

  // Print stats
  printf("Label Accuracy:\n");
  printf("  # Correct = %d ( %.1f %%)\n", ncorrect, 100.0 * ncorrect / nobjects);
  printf("  # Incorrect = %d ( %.1f %%)\n", nincorrect, 100.0 * nincorrect / nobjects);
  printf("  # Unlabeled = %d ( %.1f %%)\n", nunlabeled, 100.0 * nunlabeled / nobjects);
  printf("  # Total = %d ( %.1f %%)\n", nobjects, 100.0 * ncorrect / nobjects);
#endif

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Printing 
////////////////////////////////////////////////////////////////////////

static int
PrintInfo(R3SurfelScene *scene)
{
  // Get convenient variables
  R3SurfelTree *tree = scene->Tree();
  R3SurfelDatabase *database = tree->Database();

  // Print scene info
  const R3Box& bbox = scene->BBox();
  const R3Point& centroid = scene->Centroid();
  printf("Scene:\n");
  printf("  Name = %s\n", (scene->Name()) ? scene->Name() : "None");
  printf("  Centroid = ( %g %g %g )\n", centroid[0], centroid[1], centroid[2]);
  printf("  Bounding box = ( %g %g %g ) ( %g %g %g )\n", bbox[0][0], bbox[0][1], bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]);
  printf("  Axial lengths = ( %g %g %g )\n", bbox.XLength(), bbox.YLength(), bbox.ZLength());
  printf("  # Objects = %d\n", scene->NObjects());
  printf("  # Labels = %d\n", scene->NLabels());
  printf("  # Features = %d\n", scene->NFeatures());
  printf("  # Scans = %d\n", scene->NScans());
  printf("  # Object Properties = %d\n", scene->NObjectProperties());
  printf("  # Label Properties = %d\n", scene->NLabelProperties());
  printf("  # Object Relationships = %d\n", scene->NObjectRelationships());
  printf("  # Label Relationships = %d\n", scene->NLabelRelationships());
  printf("  # Label Assignments = %d\n", scene->NLabelAssignments());
  printf("  # Nodes = %d\n", tree->NNodes());
  printf("  # Blocks = %d\n", database->NBlocks());
  printf("  # Surfels = %d\n", database->NSurfels());
 
  // Print label info
  if (print_labels) {
    printf("Labels:\n");
    RNArray<R3SurfelLabel *> stack;
    for (int i = 0; i < scene->NLabels(); i++) {
      R3SurfelLabel *label = scene->Label(i);
      if (!label->Parent()) stack.Insert(label);
    }
    while(!stack.IsEmpty()) {
      R3SurfelLabel *label = stack.Tail();
      stack.RemoveTail();
      for (int i = 0; i < label->NParts(); i++) stack.Insert(label->Part(i));
      char prefix[16536];
      strncpy(prefix, " ", 16536);
      int level = label->PartHierarchyLevel();
      char assignment_keystroke = (label->AssignmentKeystroke() >= 0) ? label->AssignmentKeystroke() : ' ';
      for (int i = 0; i < level; i++) strncat(prefix, " ", 16536);
      printf("%s Label %d\n", prefix, label->SceneIndex());
      printf("%s Name = %s\n", prefix, (label->Name()) ? label->Name() : "None");
      printf("%s Assignment keystroke = %c\n", prefix, assignment_keystroke);
      printf("%s Part hierarchy level = %d\n", prefix, label->PartHierarchyLevel());
      printf("%s # Parts = %d\n", prefix, label->NParts());
      printf("%s # Label Properties = %d\n", prefix, label->NLabelProperties());
      printf("%s # Label Relationships = %d\n", prefix, label->NLabelRelationships());
      printf("%s # Assignments = %d\n", prefix, label->NLabelAssignments());
      printf("\n");
    }
    printf("\n");
  }

  // Print object info
  if (print_objects) {
    printf("Objects:\n");
    RNArray<R3SurfelObject *> stack;
    for (int i = 0; i < scene->NObjects(); i++) {
      R3SurfelObject *object = scene->Object(i);
      if (!object->Parent()) stack.Insert(object);
    }
    while(!stack.IsEmpty()) {
      R3SurfelObject *object = stack.Tail();
      stack.RemoveTail();
      for (int i = 0; i < object->NParts(); i++) stack.Insert(object->Part(i));
      char prefix[16536];
      strncpy(prefix, " ", 16536);
      int level = object->PartHierarchyLevel();
      for (int i = 0; i < level; i++) strncat(prefix, " ", 16536);
      R3Box bbox = object->BBox();
      R3Point centroid = object->Centroid();
      R3SurfelLabel *predicted_label = object->PredictedLabel();
      R3SurfelLabel *ground_truth_label = object->GroundTruthLabel();
      const R3SurfelFeatureVector& vector = object->FeatureVector();
      printf("%s Object %d\n", prefix, object->SceneIndex());
      printf("%s Name = %s\n", prefix, (object->Name()) ? object->Name() : "None");
      printf("%s Identifier = %d\n", prefix, object->Identifier());
      printf("%s Complexity = %g\n", prefix, object->Complexity());
      printf("%s Part hierarchy level = %d\n", prefix, object->PartHierarchyLevel());
      printf("%s Centroid = ( %g %g %g )\n", prefix, centroid[0], centroid[1], centroid[2]);
      printf("%s Bounding box = ( %g %g %g ) ( %g %g %g )\n", prefix, bbox[0][0], bbox[0][1], bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]);
      printf("%s # Nodes = %d\n", prefix, object->NNodes());
      printf("%s # Parts = %d\n", prefix, object->NParts());
      printf("%s # Object Properties = %d\n", prefix, object->NObjectProperties());
      printf("%s # Object Relationships = %d\n", prefix, object->NObjectRelationships());
      printf("%s # Assignments = %d\n", prefix, object->NLabelAssignments());
      printf("%s Predicted Label = %s\n", prefix, (predicted_label) ? predicted_label->Name() : "None");
      printf("%s Ground Truth Label = %s\n", prefix, (ground_truth_label) ? ground_truth_label->Name() : "None");
      printf("%s Feature Vector = ", prefix);
      for (int i = 0; i < vector.NValues(); i++) printf("%12.6f ", vector.Value(i));
      printf("\n");
      printf("\n");
    }
    printf("\n");
  }

  // Print label property info
  if (print_label_properties) {
    printf("Label Properties:\n");
    for (int i = 0; i < scene->NLabelProperties(); i++) {
      R3SurfelLabelProperty *property = scene->LabelProperty(i);
      R3SurfelLabel *label = property->Label();
      printf("  Label Property %d\n", i);
      printf("    Type = %d\n", property->Type());
      printf("    Label = %d\n", (label) ? label->SceneIndex() : -1);
      printf("    Operands = %d : ", property->NOperands());
      for (int j = 0; j < property->NOperands(); j++) printf("%12.6f ", property->Operand(j));
      printf("\n");
      printf("\n");
    }
    printf("\n");
  }

  // Print object property info
  if (print_object_properties) {
    printf("Object Properties:\n");
    for (int i = 0; i < scene->NObjectProperties(); i++) {
      R3SurfelObjectProperty *property = scene->ObjectProperty(i);
      R3SurfelObject *object = property->Object();
      printf("  Object Property %d\n", i);
      printf("    Type = %d\n", property->Type());
      printf("    Object = %d\n", (object) ? object->SceneIndex() : -1);
      printf("    Operands = %d : ", property->NOperands());
      for (int j = 0; j < property->NOperands(); j++) printf("%12.6f ", property->Operand(j));
      printf("\n");
      printf("\n");
    }
    printf("\n");
  }

  // Print tree info
  if (print_tree) {
    const R3Box& bbox = tree->BBox();
    const R3Point& centroid = tree->Centroid();
    printf("Tree:\n");
    printf("  # Nodes = %d\n", tree->NNodes());
    printf("  Centroid = ( %g %g %g )\n", centroid[0], centroid[1], centroid[2]);
    printf("  Bounding box = ( %g %g %g ) ( %g %g %g )\n", bbox[0][0], bbox[0][1], bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]);
    printf("  Axial lengths = ( %g %g %g )\n", bbox.XLength(), bbox.YLength(), bbox.ZLength());
    printf("\n");
  }

  // Print node info
  if (print_nodes) {
    printf("Nodes:\n");
    RNArray<R3SurfelNode *> stack;
    R3SurfelNode *node = tree->RootNode();
    if (node) stack.Insert(node);
    while(!stack.IsEmpty()) {
      R3SurfelNode *node = stack.Tail();
      stack.RemoveTail();
      for (int i = 0; i < node->NParts(); i++) stack.Insert(node->Part(i));
      char prefix[16536];
      strncpy(prefix, " ", 16536);
      int level = node->TreeLevel();
      for (int i = 0; i < level; i++) strncat(prefix, " ", 16536);
      R3Box bbox = node->BBox();
      R3Point centroid = node->Centroid();
      printf("%s  Node %s\n", prefix, node->Name());
      printf("%s    # Parts = %d\n", prefix, node->NParts());
      printf("%s    # Blocks = %d\n", prefix, node->NBlocks());
      printf("%s    Object = %d\n", prefix, (node->Object()) ? node->Object()->SceneIndex() : -1);
      printf("%s    Scan = %d\n", prefix, (node->Scan()) ? node->Scan()->SceneIndex() : -1);
      printf("%s    Complexity = %g\n", prefix, node->Complexity());
      printf("%s    Resolution = %g\n", prefix, node->Resolution());
      printf("%s    Average Radius = %g\n", prefix, node->AverageRadius());
      printf("%s    Centroid = ( %g %g %g )\n", prefix, centroid[0], centroid[1], centroid[2]);
      printf("%s    Bounding box = ( %g %g %g ) ( %g %g %g )\n", prefix, bbox[0][0], bbox[0][1], bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]);
      printf("\n");
    }
    printf("\n");
  }

  // Print database info
  if (print_database) {
    const R3Box& bbox = database->BBox();
    const R3Point& centroid = database->Centroid();
    printf("Database:\n");
    printf("  # Blocks = %d\n", database->NBlocks());
    printf("  # Surfels = %d\n", database->NSurfels());
    printf("  Centroid = ( %g %g %g )\n", centroid[0], centroid[1], centroid[2]);
    printf("  Bounding box = ( %g %g %g ) ( %g %g %g )\n", bbox[0][0], bbox[0][1], bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]);
    printf("  Axial lengths = ( %g %g %g )\n", bbox.XLength(), bbox.YLength(), bbox.ZLength());
    printf("\n");
  }

  // Print block info
  if (print_blocks) {
    printf("Blocks:\n");
    for (int i = 0; i < database->NBlocks(); i++) {
      R3SurfelBlock *block = database->Block(i);
      R3Point origin = block->Origin();
      R3Box bbox = block->BBox();
      R3Point centroid = block->Centroid();
      printf("  Block %d\n", i);
      printf("    # Surfels = %d\n", block->NSurfels());
      printf("    Node = %d\n", (block->Node()) ? block->Node()->TreeIndex() : -1);
      printf("    Resolution = %g\n", block->Resolution());
      printf("    Average Radius = %g\n", block->AverageRadius());
      printf("    Origin = ( %g %g %g )\n", origin[0], origin[1], origin[2]);
      printf("    Centroid = ( %g %g %g )\n", centroid[0], centroid[1], centroid[2]);
      printf("    Bounding box = ( %g %g %g ) ( %g %g %g )\n", bbox[0][0], bbox[0][1], bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]);
      printf("\n");
    }
    printf("\n");
  }

  // Print surfel info
  if (print_surfels) {
    printf("Surfels:\n");
    for (int i = 0; i < database->NBlocks(); i++) {
      R3SurfelBlock *block = database->Block(i);
      database->ReadBlock(block);
      printf("  Block %d\n", i);
      printf("    # Surfels = %d\n", block->NSurfels());
      for (int j = 0; j < block->NSurfels(); j++) {
        const R3Surfel *surfel = block->Surfel(j);
        printf("    Surfel %d\n", j);
        printf("      Position = %f %f %f\n", surfel->X(), surfel->Y(), surfel->Z());
        printf("      Normal = %f %f %f\n", surfel->NX(), surfel->NY(), surfel->NZ());
        printf("      Color = %d %d %d\n", surfel->R(), surfel->G(), surfel->B());
        printf("      Radius = %f\n", surfel->Radius());
        printf("      Flags = %d\n", surfel->Flags());
      }
      printf("\n");
    }
    printf("\n");
  }

  // Print feature info
  if (print_features) {
    printf("Features:\n");
    for (int i = 0; i < scene->NFeatures(); i++) {
      R3SurfelFeature *feature = scene->Feature(i);
      printf("  Name = %s\n", (feature->Name()) ? feature->Name() : "None");
      printf("  Weight = %g\n", feature->Weight());
      printf("  Minimum = %g\n", feature->Minimum());
      printf("  Maximum = %g\n", feature->Maximum());
      printf("\n");
    }
    printf("\n");
  }

  // Print scan info
  if (print_scans) {
    printf("Scans:\n");
    for (int i = 0; i < scene->NScans(); i++) {
      R3SurfelScan *scan = scene->Scan(i);
      printf("  Name = %s\n", (scan->Name()) ? scan->Name() : "None");
      printf("  Viewpoint = %g %g %g\n", scan->Viewpoint().X(), scan->Viewpoint().Y(), scan->Viewpoint().Z());
      printf("  Towards = %g %g %g\n", scan->Towards().X(), scan->Towards().Y(), scan->Towards().Z());
      printf("  Up = %g %g %g\n", scan->Up().X(), scan->Up().Y(), scan->Up().Z());
      printf("  Timestamp = %g\n", scan->Timestamp());
      printf("  Node = %d\n", (scan->Node()) ? scan->Node()->TreeIndex() : -1);
      printf("\n");
    }
    printf("\n");
  }

  // Print accuracy info
  if (accuracy_arff_name) {
    PrintAccuracy(scene, accuracy_arff_name);
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Argument Parsing Functions
////////////////////////////////////////////////////////////////////////

static int 
ParseArgs(int argc, char **argv)
{
  // Check number of program arguments
  if (argc < 3) {
    printf("Usage: sflinfo scenefile databasefile [options]\n");
    return 0;
  }

  // File names are first three arguments
  input_scene_name = argv[1];
  input_database_name = argv[2];

  // Parse arguments
  argc -= 3; argv += 3;
  while (argc > 0) {
    if (!strcmp(*argv, "-v")) { print_labels = 1; print_objects = 1; }
    else if (!strcmp(*argv, "-features")) { print_features = 1; }
    else if (!strcmp(*argv, "-objects")) { print_objects = 1; }
    else if (!strcmp(*argv, "-labels")) { print_labels = 1; }
    else if (!strcmp(*argv, "-properties")) { print_object_properties = print_label_properties = 1; }
    else if (!strcmp(*argv, "-object_properties")) { print_object_properties = 1; }
    else if (!strcmp(*argv, "-label_properties")) { print_label_properties = 1; }
    else if (!strcmp(*argv, "-tree")) { print_tree = 1; }
    else if (!strcmp(*argv, "-nodes")) { print_nodes = 1; }
    else if (!strcmp(*argv, "-database")) { print_database = 1; }
    else if (!strcmp(*argv, "-blocks")) { print_blocks = 1; }
    else if (!strcmp(*argv, "-surfels")) { print_surfels = 1; }
    else if (!strcmp(*argv, "-scans")) { print_scans = 1; }
    else if (!strcmp(*argv, "-accuracy")) { argc--; argv++; accuracy_arff_name = *argv; }
    else { fprintf(stderr, "Invalid program argument: %s", *argv); exit(1); }
    argv++; argc--;
  }

  // Return OK status 
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
  // Parse program arguments
  if (!ParseArgs(argc, argv)) exit(-1);

  // Open scene file
  R3SurfelScene *scene = OpenScene(input_scene_name, input_database_name);
  if (!scene) exit(-1);

  // Print info
  if (!PrintInfo(scene)) exit(-1);

  // Close scene file
  if (!CloseScene(scene)) exit(-1);;

  // Return success 
  return 0;
}


