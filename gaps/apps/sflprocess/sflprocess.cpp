// Source file for the surfel scene processing program



////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// Program arguments
////////////////////////////////////////////////////////////////////////

static char *scene_name = NULL;
static char *database_name = NULL;
static int aerial_only = 0;
static int terrestrial_only = 0;
static int print_verbose = 0;
static int print_debug = 0;



////////////////////////////////////////////////////////////////////////
// Scene I/O Functions
////////////////////////////////////////////////////////////////////////

static R3SurfelScene *
OpenScene(const char *scene_name, const char *database_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate scene
  R3SurfelScene *scene = new R3SurfelScene();
  if (!scene) {
    fprintf(stderr, "Unable to allocate scene\n");
    return NULL;
  }

  // Open scene files
  if (!scene->OpenFile(scene_name, database_name, "r+", "r+")) {
    delete scene;
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("Opened scene ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Objects = %d\n", scene->NObjects());
    printf("  # Labels = %d\n", scene->NLabels());
    printf("  # Assignments = %d\n", scene->NLabelAssignments());
    printf("  # Features = %d\n", scene->NFeatures());
    printf("  # Nodes = %d\n", scene->Tree()->NNodes());
    printf("  # Blocks = %d\n", scene->Tree()->Database()->NBlocks());
    printf("  # Surfels = %d\n", scene->Tree()->Database()->NSurfels());
    fflush(stdout);
  }

  // Return scene
  return scene;
}



static int
CloseScene(R3SurfelScene *scene)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Close scene files
  if (!scene->CloseFile()) return 0;

  // Print statistics
  if (print_verbose) {
    printf("Closed scene ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Objects = %d\n", scene->NObjects());
    printf("  # Labels = %d\n", scene->NLabels());
    printf("  # Assignments = %d\n", scene->NLabelAssignments());
    printf("  # Features = %d\n", scene->NFeatures());
    printf("  # Nodes = %d\n", scene->Tree()->NNodes());
    printf("  # Blocks = %d\n", scene->Tree()->Database()->NBlocks());
    printf("  # Surfels = %d\n", scene->Tree()->Database()->NSurfels());
    fflush(stdout);
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Other I/O Functions
////////////////////////////////////////////////////////////////////////

static R2Grid *
ReadGrid(const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate a grid
  R2Grid *grid = new R2Grid();
  if (!grid) {
    RNFail("Unable to allocate grid");
    return NULL;
  }

  // Read grid
  int status = grid->Read(filename);
  if (!status) {
    RNFail("Unable to read grid file %s", filename);
    return NULL;
  }

  // Print statistics
  if (print_verbose) {
    printf("Read grid from %s\n", filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  Resolution = %d %d\n", grid->XResolution(), grid->YResolution());
    printf("  Spacing = %g\n", grid->GridToWorldScaleFactor());
    printf("  Cardinality = %d\n", grid->Cardinality());
    RNInterval grid_range = grid->Range();
    printf("  Minimum = %g\n", grid_range.Min());
    printf("  Maximum = %g\n", grid_range.Max());
    printf("  L1Norm = %g\n", grid->L1Norm());
    printf("  L2Norm = %g\n", grid->L2Norm());
    fflush(stdout);
  }

  // Return success
  return grid;
}



////////////////////////////////////////////////////////////////////////
// CREATE FUNCTIONS (creates structure without loading surfels)
////////////////////////////////////////////////////////////////////////

static R3SurfelNode *
CreateNode(R3SurfelScene *scene, const char *node_name, const char *parent_name)
{
  // Get surfel tree
  R3SurfelTree *tree = scene->Tree();
  if (!tree) {
    fprintf(stderr, "Scene has no tree\n");
    return NULL;
  }    

  // Find parent node
  R3SurfelNode *parent_node = tree->FindNodeByName(parent_name);
  if (!parent_node) {
    fprintf(stderr, "Unable to find parent node with name %s\n", parent_name);
    return NULL;
  }

  // Create node
  R3SurfelNode *node = new R3SurfelNode(node_name);
  if (!node) {
    fprintf(stderr, "Unable to allocate node\n");
    return NULL;
  }
            
  // Insert node into tree
  tree->InsertNode(node, parent_node);

  // Return node
  return node;
}


static R3SurfelObject *
CreateObject(R3SurfelScene *scene, const char *object_name, 
  const char *parent_name, const char *node_name)
{
  // Get surfel tree
  R3SurfelTree *tree = scene->Tree();
  if (!tree) {
    fprintf(stderr, "Scene has no tree\n");
    return NULL;
  }    

  // Find parent object
  R3SurfelObject *parent_object = scene->FindObjectByName(parent_name);
  if (!parent_object) {
    fprintf(stderr, "Unable to find parent object with name %s\n", parent_name);
    return NULL;
  }

  // Find node
  R3SurfelNode *node = NULL;
  if (strcmp(node_name, "None") && strcmp(node_name, "none") && strcmp(node_name, "NONE")) {
    node = tree->FindNodeByName(node_name);
    if (!node) {
      fprintf(stderr, "Unable to find parent node with name %s\n", node_name);
      return NULL;
    }
  }

  // Create object
  R3SurfelObject *object = new R3SurfelObject(object_name);
  if (!object) {
    fprintf(stderr, "Unable to allocate object\n");
    return NULL;
  }
         
  // Insert node into object
  if (node) object->InsertNode(node);
  
  // Insert object into scene
  scene->InsertObject(object, parent_object);

  // Return object
  return object;
}



static R3SurfelLabel *
CreateLabel(R3SurfelScene *scene, const char *label_name, const char *parent_name)
{
  // Get surfel tree
  R3SurfelTree *tree = scene->Tree();
  if (!tree) {
    fprintf(stderr, "Scene has no tree\n");
    return NULL;
  }    

  // Find parent label
  R3SurfelLabel *parent_label = scene->FindLabelByName(parent_name);
  if (!parent_label) {
    fprintf(stderr, "Unable to find parent label with name %s\n", parent_name);
    return NULL;
  }

  // Create label
  R3SurfelLabel *label = new R3SurfelLabel(label_name);
  if (!label) {
    fprintf(stderr, "Unable to allocate label\n");
    return NULL;
  }
         
  // Insert label into scene
  scene->InsertLabel(label, parent_label);

  // Return label
  return label;
}



////////////////////////////////////////////////////////////////////////
// REMOVE FUNCTIONS 
////////////////////////////////////////////////////////////////////////

static int
RemoveParts(R3SurfelScene *scene, R3SurfelObject *object)
{
  // Copy array of parts
  RNArray<R3SurfelObject *> parts;
  for (int i = 0; i < object->NParts(); i++) {
    R3SurfelObject *part = object->Part(i);
    parts.Insert(part);
  }
  
  // Delete parts
  for (int i = 0; i < parts.NEntries(); i++) {
    R3SurfelObject *part = parts.Kth(i);
    RemoveParts(scene, part);
    scene->RemoveObject(part);
    delete part;
  }

  // Return success
  return 1;
}



static int
RemoveParts(R3SurfelScene *scene, R3SurfelLabel *label)
{
  // Copy array of parts
  RNArray<R3SurfelLabel *> parts;
  for (int i = 0; i < label->NParts(); i++) {
    R3SurfelLabel *part = label->Part(i);
    parts.Insert(part);
  }
  
  // Delete parts
  for (int i = 0; i < parts.NEntries(); i++) {
    R3SurfelLabel *part = parts.Kth(i);
    RemoveParts(scene, part);
    scene->RemoveLabel(part);
    delete part;
  }

  // Return success
  return 1;
}



static int
RemoveObjects(R3SurfelScene *scene)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Remove objects recursively
  int nobjects = scene->NObjects();
  if (!RemoveParts(scene, scene->RootObject())) return 0;

  // Print statistics
  if (print_verbose) {
    printf("Removed objects ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Objects Removed = %d\n", nobjects - scene->NObjects());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
RemoveLabels(R3SurfelScene *scene)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Remove labels recursively
  int nlabels = scene->NLabels();
  if (!RemoveParts(scene, scene->RootLabel())) return 0;

  // Print statistics
  if (print_verbose) {
    printf("Removed labels ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Labels Removed = %d\n", nlabels - scene->NLabels());
    fflush(stdout);
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// LOAD FUNCTIONS
////////////////////////////////////////////////////////////////////////

static R3SurfelNode *
LoadSurfels(R3SurfelScene *scene, R3SurfelNode *parent_node, 
  R3Point *points, int npoints, const char *node_name)
{
  // Get surfel tree
  R3SurfelTree *tree = scene->Tree();
  if (!tree) {
    fprintf(stderr, "Scene has no tree\n");
    return NULL;
  }    

  // Get surfel database
  R3SurfelDatabase *database = tree->Database();
  if (!database) {
    fprintf(stderr, "Scene has no database\n");
    return NULL;
  }    

  // Create node
  R3SurfelNode *node = new R3SurfelNode(node_name);
  if (!node) {
    fprintf(stderr, "Unable to allocate node\n");
    return NULL;
  }
            
  // Insert node into tree
  tree->InsertNode(node, parent_node);
          
  // Create block
  R3SurfelBlock *block = new R3SurfelBlock(points, npoints);
  if (!block) {
    fprintf(stderr, "Unable to allocate block\n");
    return NULL;
  }
          
  // Update block properties
  block->UpdateProperties();
          
  // Insert block into database
  database->InsertBlock(block);
            
  // Insert block into node
  node->InsertBlock(block);
          
  // Update node properties
  node->UpdateProperties();
          
  // Release block
  database->ReleaseBlock(block);

  // Return node
  return node;
}



static R3SurfelNode *
LoadSurfels(R3SurfelScene *scene, const char *surfels_filename, 
  const char *object_name, const char *parent_object_name,
  const char *node_name, const char *parent_node_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Get surfel tree
  R3SurfelTree *tree = scene->Tree();
  if (!tree) {
    fprintf(stderr, "Scene has no tree\n");
    return NULL;
  }    

  // Get surfel database
  R3SurfelDatabase *database = tree->Database();
  if (!database) {
    fprintf(stderr, "Scene has no database\n");
    return NULL;
  }    

  // Find parent object
  R3SurfelObject *parent_object = NULL;
  if (parent_object_name && 
      strcmp(parent_object_name, "None") && 
      strcmp(parent_object_name, "none") && 
      strcmp(parent_object_name, "NONE")) {
    parent_object = scene->FindObjectByName(parent_object_name);
    if (!parent_object) {
      fprintf(stderr, "Unable to find parent object with name %s\n", parent_object_name);
      return NULL;
    }
  }

  // Find parent node
  R3SurfelNode *parent_node = tree->FindNodeByName(parent_node_name);
  if (!parent_node) {
    fprintf(stderr, "Unable to find parent node with name %s\n", parent_node_name);
    return NULL;
  }

  // Create node
  R3SurfelNode *node = new R3SurfelNode(node_name);
  if (!node) {
    fprintf(stderr, "Unable to allocate node for %s\n", surfels_filename);
    return NULL;
  }

  // Insert node into tree
  tree->InsertNode(node, parent_node);

  // Create block
  R3SurfelBlock *block = new R3SurfelBlock();
  if (!block) {
    fprintf(stderr, "Unable to allocate block for %s\n", surfels_filename);
    return NULL;
  }

  // Read block
  if (!block->ReadFile(surfels_filename)) {
    fprintf(stderr, "Unable to read block from %s\n", surfels_filename);
    return NULL;
  }

  // Extract subset of surfels
  if (aerial_only || terrestrial_only) {
    // Create subset
    R3SurfelPointSet *subset = new R3SurfelPointSet();

    // Fill subset
    for (int i = 0; i < block->NSurfels(); i++) {
      const R3Surfel *surfel = block->Surfel(i);
      if (surfel->IsAerial() && terrestrial_only) continue;
      if (!surfel->IsAerial() && aerial_only) continue;
      R3SurfelPoint point(block, surfel);
      subset->InsertPoint(point);
    }

    // Replace block with subset
    // Note: it is important to delete subset first, since it references block
    R3SurfelBlock *subset_block = new R3SurfelBlock(subset);
    delete subset;
    delete block;
    block = subset_block;
  }

  // Update block properties
  block->UpdateProperties();

  // Insert block into database
  database->InsertBlock(block);

  // Insert block into node
  node->InsertBlock(block);

  // Update node properties
  node->UpdateProperties();

  // Create object
  if (object_name && 
      strcmp(object_name, "None") && 
      strcmp(object_name, "none") && 
      strcmp(object_name, "NONE") && 
      parent_object) {
    // Create object
    R3SurfelObject *object = new R3SurfelObject(object_name);
    if (!object) {
      fprintf(stderr, "Unable to create object\n");
      return NULL;
    }

    // Insert object into scene
    scene->InsertObject(object, parent_object);

    // Insert node into object
    object->InsertNode(node);
      
    // Update object properties
    object->UpdateProperties();
  }

  // Release block
  database->ReleaseBlock(block);

  // Print statistics
  if (print_verbose) {
    printf("Loaded surfels from %s ...\n", surfels_filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Surfels = %g\n", node->Complexity());
    fflush(stdout);
  }

  // Return node
  return node;
}



static int
LoadSurfelsList(R3SurfelScene *scene, const char *list_filename, 
  const char *parent_object_name, const char *parent_node_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  int saved_print_verbose = print_verbose;
  print_verbose = print_debug;

  // Open file
  FILE *fp = fopen(list_filename, "r");
  if (!fp) {
    fprintf(stderr, "Unable to open %s\n", list_filename);
    return 0;
  }

  // Read objects/nodes from file with list
  int count = 0;
  char buffer[4096];
  char node_filename[4096];
  while (fgets(buffer, 4096, fp)) {
    if (buffer[0] == '#') continue;
    if (sscanf(buffer, "%s", node_filename) == (unsigned int) 1) {
      char *start = strrchr(node_filename, '/');
      start = (start) ? start+1 : node_filename;
      char node_name[1024];
      strncpy(node_name, start, 1024);
      char *end = strrchr(node_name, '.');
      if (end) *end = '\0';
      if (!LoadSurfels(scene, node_filename, node_name, parent_object_name, node_name, parent_node_name)) return 0;
      count++;
    }
  }

  // Close file
  fclose(fp);

  // Print statistics
  print_verbose = saved_print_verbose;
  if (print_verbose) {
    printf("Loaded surfels from %s ...\n", list_filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Files = %d\n", count);
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
LoadSurfelsFromGoogleStreetView(R3SurfelScene *scene, const char *list_filename, 
  const char *parent_node_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Get surfel tree
  R3SurfelTree *tree = scene->Tree();
  if (!tree) {
    fprintf(stderr, "Scene has no tree\n");
    return 0;
  }    

  // Get surfel database
  R3SurfelDatabase *database = tree->Database();
  if (!database) {
    fprintf(stderr, "Scene has no database\n");
    return 0;
  }    

  // Find parent node
  R3SurfelNode *parent_node = tree->FindNodeByName(parent_node_name);
  if (!parent_node) {
    fprintf(stderr, "Unable to find parent node with name %s\n", parent_node_name);
    return 0;
  }

  // Open list file
  FILE *list_fp = fopen(list_filename, "r");
  if (!list_fp) {
    fprintf(stderr, "Unable to open %s\n", list_filename);
    return 0;
  }

  // Allocate data for scanlines
  int scanline_npoints = 0;
  const int max_points_per_scanline = 1024 * 1024;
  R3Point *scanline_points = new R3Point [max_points_per_scanline];
  char scanline_name[1024] = { '\0' };

  // Read list of run names
  int count = 0;
  char buffer[4096];
  char run_name[4096];
  while (fgets(buffer, 4096, list_fp)) {
    if (buffer[0] == '#') continue;

    // Get run name
    if (sscanf(buffer, "%s", run_name) != (unsigned int) 1) continue;
    char *start = strrchr(run_name, '/');
    start = (start) ? start+1 : run_name;
    char node_name[1024];
    strncpy(node_name, start, 1024);
    char *end = strrchr(node_name, '.');
    if (end) *end = '\0';

    // Create run node
    R3SurfelNode *run_node = new R3SurfelNode(run_name);
    if (!run_node) {
      fprintf(stderr, "Unable to allocate node\n");
      return 0;
    }
            
    // Insert run node into tree
    tree->InsertNode(run_node, parent_node);
          
     // Read all laser obj files in run directory
    const int max_laser_index = 2;
    for (int laser_index = 0; laser_index <= max_laser_index; laser_index++) {
      // Temporary ???
      if (laser_index == 1) continue;

      // Create obj filename
      char obj_filename[4096];
      sprintf(obj_filename, "raw_data/%s/laser_point_cloud_%d.obj", run_name, laser_index+1);

      // Open file
      FILE *obj_fp = fopen(obj_filename, "r");
      if (!obj_fp) {
        fprintf(stderr, "Unable to open laser pose file: %s\n", obj_filename);
        return 0;
      }

      // Create scan node
      char scan_name[2048];
      sprintf(scan_name, "scan_%d", laser_index);
      R3SurfelNode *scan_node = new R3SurfelNode(scan_name);
      if (!scan_node) {
        fprintf(stderr, "Unable to allocate node\n");
        return 0;
      }
            
      // Insert scan node into tree
      tree->InsertNode(scan_node, run_node);

      // Read points for each scanline
      int line_count = 0;
      const int buffer_size = 16*1024;
      char buffer[buffer_size];
      char keyword[1024];
      while (fgets(buffer, buffer_size, obj_fp)) {
        line_count++;
        if (buffer[0] == '\0') continue;
        if (buffer[0] == '#') continue;
        if (buffer[0] == 'v') {
          // Parse point
          double x, y, z; 
          if (sscanf(buffer, "%s%lf%lf%lf", keyword, &x, &y, &z) != 4) {
            fprintf(stderr, "Error reading line %d from %s\n", line_count, obj_filename);
            return 0;
          }
          
          // Insert point
          if (scanline_npoints < max_points_per_scanline) {
            scanline_points[scanline_npoints].Reset(x, y, z);
            scanline_npoints++;
          }
        }
        else if (buffer[0] == 'g') {
          // Process points from previous scanline
          if (scanline_npoints > 0) {
            if (!LoadSurfels(scene, scan_node, scanline_points, scanline_npoints, scanline_name)) return 0;
            scanline_name[0] = '\0';
            scanline_npoints = 0;
          }

          // Get scanline string
          char scanline_string[1024];
          if (sscanf(buffer, "%s%s", keyword, scanline_string) != 2) {
            fprintf(stderr, "Error reading line %d from %s\n", line_count, obj_filename);
            return 0;
          }

          // Parse scanline string
          int segment_index = 0;
          int scanline_index = 0;
          char *bufferp = strtok(scanline_string, "_ \n\t");
          if (bufferp) { 
            assert(!strcmp(bufferp, "seg"));
            bufferp = strtok(NULL, "_ \n\t"); 
            if (bufferp) {
              segment_index = atoi(bufferp);
              bufferp = strtok(NULL, "_ \n\t"); 
              if (bufferp) {
                assert(!strcmp(bufferp, "scanline"));
                bufferp = strtok(NULL, "_ \n\t"); 
                if (bufferp) {
                  scanline_index = atoi(bufferp);
                }
              }
            }
          }

          // Adjust scanline_index to count within segment
          static int last_segment_index = -1;
          static int segment_index_offset = 0;
          if (segment_index != last_segment_index) {
            segment_index_offset = scanline_index;
            last_segment_index = segment_index;
          }

          // Initialize everything for next scanline
          scanline_index -= segment_index_offset;
          const char *run_name_without_directory = strrchr(run_name, '/');
          if (run_name_without_directory) run_name_without_directory++;
          else run_name_without_directory = run_name;
          sprintf(scanline_name, "%s__%d__%d__%d", 
            run_name_without_directory, segment_index, laser_index, scanline_index);
          scanline_npoints = 0;
        }
      }

      // Process points from last scanline
      if (scanline_npoints > 0) {
       if (!LoadSurfels(scene, scan_node, scanline_points, scanline_npoints, scanline_name)) return 0;
       scanline_name[0] = '\0';
       scanline_npoints = 0;
      }

      // Update scan node properties
      scan_node->UpdateProperties();

      // Close file
      fclose(obj_fp);
    }
  }

  // Close file
  fclose(list_fp);

  // Delete temporary data for scan lines
  delete [] scanline_points;

  // Print statistics
  if (print_verbose) {
    printf("Loaded surfels from %s ...\n", list_filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Nodes = %d\n", count);
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
LoadLabelList(R3SurfelScene *scene, const char *list_filename, const char *root_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Open file
  FILE *fp = fopen(list_filename, "r");
  if (!fp) {
    fprintf(stderr, "Unable to open %s\n", list_filename);
    return 0;
  }

  // Find root
  R3SurfelLabel *root = NULL;
  if (root_name && strcmp(root_name, "Null")) {
    root = scene->FindLabelByName(root_name);
    if (!root) {
      fprintf(stderr, "Unable to find root label %s\n", root_name);
      return 0;
    }
  }
 
  // Read labels from file with list
  int count = 0;
  double r, g, b;
  int identifier, visibility;
  char assignment_keystroke[64];
  char label_name[4096], parent_name[4096], buffer[16384];
  while (fgets(buffer, 16384, fp)) {
    char *bufferp = buffer;
    while (*bufferp && isspace(*bufferp)) bufferp++;
    if (*bufferp == '\0') continue;
    if (*bufferp == '#') continue;
    if (sscanf(buffer, "%s%d%s%s%d%lf%lf%lf", label_name, &identifier, assignment_keystroke, parent_name, &visibility, &r, &g, &b) != (unsigned int) 8) {
      fprintf(stderr, "Invalid format for label %d in %s\n", count, list_filename);
      return 0;
    }
          
    // Check if label already exists
    if (scene->FindLabelByName(label_name)) continue;

    // Create label
    R3SurfelLabel *label = new R3SurfelLabel(label_name);
    if (assignment_keystroke[0] != '-') label->SetAssignmentKeystroke(assignment_keystroke[0]);
    label->SetIdentifier(identifier);
    label->SetColor(RNRgb(r, g, b));
    
    // Find parent
    R3SurfelLabel *parent = NULL;
    if (!strcmp(parent_name, "Null")) parent = root;
    else {
      parent = scene->FindLabelByName(parent_name);
      if (!parent) {
        fprintf(stderr, "Unable to find label's parent (%s) in label %d of %s\n", parent_name, count, list_filename);
        return 0;
      }
    }
    
    // Insert into scene
    scene->InsertLabel(label, parent);
    
    // Update stats
    count++;
  }

  // Close file
  fclose(fp);

  // Print statistics
  if (print_verbose) {
    printf("Loaded labels from %s ...\n", list_filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Labels = %d\n", count);
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
LoadAssignmentList(R3SurfelScene *scene, const char *list_filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  int count = 0;

  // Open file
  FILE *fp = fopen(list_filename, "r");
  if (!fp) {
    fprintf(stderr, "Unable to open %s\n", list_filename);
    return 0;
  }

  // Read labels from file with list
  char buffer[4096];
  char object_name[4096];
  char label_name[4096];
  char originator_str[4096];
  double confidence;
  while (fgets(buffer, 4096, fp)) {
    if (buffer[0] == '\0') continue;
    if (buffer[0] == '#') continue;
    if (sscanf(buffer, "%s%s%lf%s", object_name, label_name, &confidence, originator_str) == 4) {
      // Find object
      R3SurfelObject *object = scene->FindObjectByName(object_name);
      if (!object) {
        fprintf(stderr, "Unable to find object %s in assignments file %s\n", object_name, list_filename);
        return 0;
      }

      // Find label
      R3SurfelLabel *label = scene->FindLabelByName(label_name);
      if (!label) {
        fprintf(stderr, "Unable to find label %s in assignments file %s\n", label_name, list_filename);
        return 0;
      }

      // Create assignment
      int originator = R3_SURFEL_LABEL_ASSIGNMENT_MACHINE_ORIGINATOR;
      if (!strcmp(originator_str, "Human")) originator = R3_SURFEL_LABEL_ASSIGNMENT_HUMAN_ORIGINATOR;
      else if (!strcmp(originator_str, "GroundTruth")) originator = R3_SURFEL_LABEL_ASSIGNMENT_GROUND_TRUTH_ORIGINATOR;
      R3SurfelLabelAssignment *assignment = new R3SurfelLabelAssignment(object, label, confidence, originator);
      scene->InsertLabelAssignment(assignment);
      count++;
    }
  }

  // Close file
  fclose(fp);

  // Print statistics
  if (print_verbose) {
    printf("Loaded assignments from %s ...\n", list_filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Assignments = %d\n", count);
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
LoadFeatureList(R3SurfelScene *scene, const char *list_filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  int count = 0;

  // Open file
  FILE *fp = fopen(list_filename, "r");
  if (!fp) {
    fprintf(stderr, "Unable to open %s\n", list_filename);
    return 0;
  }

  // Read features from file with list
  char buffer[4096];
  char type[1024], name[1024], filename[1024];
  double minimum, maximum, weight;
  while (fgets(buffer, 4096, fp)) {
    if (buffer[0] == '\0') continue;
    if (buffer[0] == '#') continue;
    if (sscanf(buffer, "%s%s%lf%lf%lf%s", type, name, &minimum, &maximum, &weight, filename) == 6) {
      // Create feature of appropriate type
      if (!strcmp(type, "PointSet")) {
        R3SurfelPointSetFeature *feature = new R3SurfelPointSetFeature(name, minimum, maximum, weight);
        scene->InsertFeature(feature);
      }
      else if (!strcmp(type, "OverheadGrid")) {
        R3SurfelOverheadGridFeature *feature = new R3SurfelOverheadGridFeature(filename, name, minimum, maximum, weight);
        scene->InsertFeature(feature);
      }
      else {
        R3SurfelFeature *feature = new R3SurfelFeature(name, minimum, maximum, weight);
        scene->InsertFeature(feature);
      }
      count++;
    }
  }

  // Close file
  fclose(fp);

  // Print statistics
  if (print_verbose) {
    printf("Loaded features from %s ...\n", list_filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Features = %d\n", count);
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
LoadScene(R3SurfelScene *scene1, 
  const char *scene_filename, const char *database_filename, 
  const char *parent_object_name, const char *parent_label_name, const char *parent_node_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Find parent surfel tree node in scene1
  R3SurfelNode *parent_node = scene1->Tree()->FindNodeByName(parent_node_name);
  if (!parent_node) {
    fprintf(stderr, "Unable to find parent node with name %s\n", parent_node_name);
    return 0;
  }

  // Find parent object in scene1
  R3SurfelObject *parent_object = scene1->FindObjectByName(parent_object_name);
  if (!parent_object) {
    fprintf(stderr, "Unable to find parent object with name %s\n", parent_object_name);
    return 0;
  }

  // Find parent label in scene1
  R3SurfelLabel *parent_label = scene1->FindLabelByName(parent_label_name);
  if (!parent_label) {
    fprintf(stderr, "Unable to find parent label with name %s\n", parent_label_name);
    return 0;
  }

  // Allocate scene2
  R3SurfelScene *scene2 = new R3SurfelScene();
  if (!scene2) {
    fprintf(stderr, "Unable to allocate scene\n");
    return 0;
  }

  // Open scene2
  if (!scene2->OpenFile(scene_filename, database_filename, "r", "r")) {
    delete scene2;
    return 0;
  }

  // Insert scene2 into scene1
  scene1->InsertScene(*scene2, parent_object, parent_label, parent_node);

  // Print statistics
  if (print_verbose) {
    printf("Loaded scene from %s and %s ...\n", scene_filename, database_filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Objects = %d\n", scene2->NObjects());
    printf("  # Labels = %d\n", scene2->NLabels());
    printf("  # Assignments = %d\n", scene2->NLabelAssignments());
    printf("  # Nodes = %d\n", scene2->Tree()->NNodes());
    printf("  # Blocks = %d\n", scene2->Tree()->Database()->NBlocks());
    printf("  # Surfels = %d\n", scene2->Tree()->Database()->NSurfels());
    fflush(stdout);
  }

  // Close scene
  if (!scene2->CloseFile()) return 0;

  // Delete scene
  delete scene2;

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// ALIGNMENT FUNCTIONS
////////////////////////////////////////////////////////////////////////

#if 0

static R3CoordSystem
PrinciplePlanarCoordSystem(R3SurfelScene *scene)
{
}


static int
Transform(R3SurfelScene *scene, const R3Affine& transformation)
{
  // Transform all nodes
  R3SurfelTree *tree = scene->Tree();
  for (int i = 0; i < tree->NNodes(); i++) {
    R3SurfelNode *node = tree->Node(i);
    node->Transform(transformation);
  }

  // Return success
  return 1;
}



static int
AlignPrinciplePlanarAxes(R3SurfelScene *scene)
{
  // Determine principle planar coordinate system
  R3CoordSystem cs = PrinciplePlanarCoordSystem(scene);

  // Determine transformation
  R4Matrix matrix = cs.InverseMatrix();
  R3Affine transformation(matrix, 0);

  // Apply transformation
  return Transform(scene, transformation);
}

#endif



static int
TransformWithConfigurationFile(R3SurfelScene *scene, const char *filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  int count = 0;
  
  // Get surfel tree
  R3SurfelTree *tree = scene->Tree();
  if (!tree) {
    fprintf(stderr, "Scene has no surfel tree\n");
    return 0;
  }    

  // Open file
  FILE *fp = fopen(filename, "r");
  if (!fp) { 
    fprintf(stderr, "Unable to open extrinsics file %s\n", filename); 
    return 0; 
  }

  // Parse file
  char buffer[4096];
  int line_number = 0;
  while (fgets(buffer, 4096, fp)) {
    char cmd[4096];
    line_number++;
    if (sscanf(buffer, "%s", cmd) != (unsigned int) 1) continue;
    if (cmd[0] == '#') continue;

    // Check cmd
    if (!strcmp(cmd, "scan")) {
      // Parse image name and alignment transformation
      RNScalar m[16];
      char depth_name[4096], color_name[4096];
      if (sscanf(buffer, "%s%s%s%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf", cmd, 
         depth_name, color_name,
         &m[0], &m[1], &m[2], &m[3], &m[4], &m[5], &m[6], &m[7], 
         &m[8], &m[9], &m[10], &m[11], &m[12], &m[13], &m[14], &m[15]) != (unsigned int) 19) {
        fprintf(stderr, "Error parsing line %d of %s\n", line_number, filename);
        return 0;
      }

      // Get transformation
      R3Affine transformation(R4Matrix(m), 0);

      // Get node name
      char tmp[4096];
      strncpy(tmp, depth_name, 4096);
      char *scan_name = strrchr(tmp, '/');
      if (!scan_name) scan_name = tmp;
      char *endp = strrchr(scan_name, '.');
      if (endp) *endp = '\0';
      char node_name[4096];
      sprintf(node_name, "SCAN:%s", scan_name);

#if 1
      // Find node
      R3SurfelNode *node = tree->FindNodeByName(node_name);
      if (!node) {
        // fprintf(stderr, "Unable to find node %s for %s\n", node_name, filename);
        // return 0;
        continue;
      }

      // Find scan
      R3SurfelScan *scan = node->Scan();
#else
      // Find scan
      R3SurfelScan *scan = tree->FindScanByName(scan_name);
      if (!scan) {
        // fprintf(stderr, "Unable to find scan %s for %s\n", scan_name, filename);
        // return 0;
        continue;
      }

      // Find node
      R3SurfelNode *node = scan->Node();
#endif
      
      // Transform scan
      if (scan) {
        R3CoordSystem pose = scan->Pose();
        pose.Transform(transformation);
        scan->SetPose(pose);
      }
      
      // Transform node and all its decendents
      if (node) {
        RNArray<R3SurfelNode *> stack;
        stack.Insert(node);
        while (!stack.IsEmpty()) {
          R3SurfelNode *node = stack.Tail();
          stack.RemoveTail();
          node->Transform(transformation);
          for (int i = 0; i < node->NParts(); i++) {
            stack.InsertTail(node->Part(i));
          }
        }
      }

      // Update statistics
      count++;
    }
  }

  // Close configuration file
  fclose(fp);

  // Print statistics
  if (print_verbose) {
    printf("Tranformed nodes ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Nodes = %d\n", count);
    fflush(stdout);
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// TRANSFER FUNCTIONS
////////////////////////////////////////////////////////////////////////

#if 0

static int
TransferLabels(R3SurfelScene *scene1, R3SurfelScene *scene2)
{
  // Copy labels from scene2 to scene1
  for (int i = 0; i < scene2->NLabels(); i++) {
    R3SurfelLabel *label2 = scene2->Label(i);
    R3SurfelLabel *label1 = scene1->FindLabelByName(label2->Name());
    if (!label1) scene1->InsertLabel(new R3SurfelLabel(*label2));
  }

  // Return success
  return 1;
}



static int
TransferObjects(R3SurfelScene *scene1, R3SurfelScene *scene2)
{
  // Initialize an object index

  // Copy objects from scene2 to scene1
  R3SurfelObject *root = scene1->RootObject();
  for (int i = 0; i < scene2->NObjects(); i++) {
    R3SurfelObject *object2 = scene2->Object(i);
    if (object2 == scene2->RootObject()) continue;
    R2SurfelObject *object1 = new R3SurfelObject(object2->Name());
    scene1->InsertObject(object1, root);
    // Extract surfels nearby ones in object2 and move them into object1
  }

  // Copy object hierarchy from scene2 to scene1
  // Must be done in separate pass in case parents are created after children
  for (int i = 0; i < scene2->NObjects(); i++) {
    R3SurfelObject *object2 = scene2->Object(i);
    R3SurfelObject *object1 = objects_index[i];
    R3SurfelObject *parent2 = object2->Parent();
    R3SurfelObject *parent1 = objects_index[parent2->SceneIndex()];
    object1->SetParent(parent1);
  }

  // Copy label assignments from scene2 to scene1
  for (int i = 0; i < scene2->NLabelAssignments(); i++) {
    R3SurfelLabelAssignment *assignment2 = scene2->LabelAssignment(i);
    R3SurfelObject *object2 = assignment2->Object();
    R3SurfelObject *object1 = objects_index[object2->SceneIndex()];
    R3SurfelLabel *label2 = assignment2->Label();
    R3SurfelLabel *label1 = scene1->FindLabelByName(label2->Name());
    R3SurfelLabelAssignment *assignment1 = new R3SurfelLabelAssignment(object1, label1, assignment2->Confidence(), assignment2->Originaor());
    scene1->InsertLabelAssignment(assignment1);
  }

  // Return success
  return 1;
}



static int
TransferLabels(R3SurfelScene *scene1,  
  const char *label_scene_filename, const char *label_database_filename)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Allocate scene2
  R3SurfelScene *scene2 = new R3SurfelScene();
  if (!scene2) {
    fprintf(stderr, "Unable to allocate scene\n");
    return 0;
  }

  // Open scene2
  if (!scene2->OpenFile(label_scene_filename, label_database_filename, "r", "r")) {
    delete scene2;
    return 0;
  }

  // Transfer labels
  if (!TransferLabels(scene1, scene2)) return 0;

  // Transfer objects
  if (!TransferObjects(scene1, scene2)) return 0;

  // Print statistics
  if (print_verbose) {
    printf("Transferred labels from %s and %s ...\n", scene_filename, database_filename);
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Objects = %d\n", scene2->NObjects());
    printf("  # Labels = %d\n", scene2->NLabels());
    printf("  # Assignments = %d\n", scene2->NLabelAssignments());
    printf("  # Nodes = %d\n", scene2->Tree()->NNodes());
    printf("  # Blocks = %d\n", scene2->Tree()->Database()->NBlocks());
    printf("  # Surfels = %d\n", scene2->Tree()->Database()->NSurfels());
    fflush(stdout);
  }

  // Close scene
  if (!scene2->CloseFile()) return 0;

  // Delete scene
  delete scene2;

  // Return success
  return 1;
}

#endif



////////////////////////////////////////////////////////////////////////
// MASKING FUNCTIONS
////////////////////////////////////////////////////////////////////////

static int
Mask(R3SurfelScene *scene, const char *node_name, R3SurfelConstraint *constraint)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Get useful variables
  R3SurfelTree *tree = scene->Tree();
  if (!tree) {
    fprintf(stderr, "Scene has no surfel tree\n");
    return 0;
  }    

  // Get surfel database
  R3SurfelDatabase *database = tree->Database();
  if (!database) {
    fprintf(stderr, "Tree has no surfel database\n");
    return 0;
  }    

  // Find node
  R3SurfelNode *node = tree->RootNode();
  if (strcmp(node_name, "All") && strcmp(node_name, "Root")) {
    node = tree->FindNodeByName(node_name);
    if (!node) {
      fprintf(stderr, "Unable to find node with name %s\n", node_name);
      return 0;
    }
  }

  // Remove surfels not satistfying constraint
  RNArray<R3SurfelNode *> remove_nodes;
  tree->SplitLeafNodes(node, *constraint, NULL, &remove_nodes);
  for (int i = 0; i < remove_nodes.NEntries(); i++) {
    R3SurfelNode *node = remove_nodes.Kth(i);

    // Make array of blocks
    RNArray<R3SurfelBlock *> blocks;
    while (node->NBlocks() > 0) {
      R3SurfelBlock *block = node->Block(0);
      node->RemoveBlock(block);
      blocks.Insert(block);
    }

    // Remove/delete node
    tree->RemoveNode(node);
    delete node;

    // Remove/delete blocks
    for (int j = 0; j < blocks.NEntries(); j++) {
      R3SurfelBlock *block = blocks.Kth(j);
      database->RemoveBlock(block);
      delete block;
    }
  }

  // Return success
  return 1;

}



////////////////////////////////////////////////////////////////////////
// MULTIRESOLUTION FUNCTIONS
////////////////////////////////////////////////////////////////////////

static int
RemoveInteriorNodes(R3SurfelScene *scene)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Get surfel tree
  R3SurfelTree *tree = scene->Tree();
  if (!tree) {
    fprintf(stderr, "Scene has no surfel tree\n");
    return 0;
  }    

  // Create copy of nodes (because will edit as traverse)
  RNArray<R3SurfelNode *> interior_nodes;
  for (int i = 0; i < tree->NNodes(); i++) {
    R3SurfelNode *node = tree->Node(i);
    if (node->Name() && !strcmp(node->Name(), "Root")) continue;
    if (!node->Parent()) continue;
    if (node->NParts() == 0) continue;
    if (node->Object()) continue;
    assert(node->Tree() == node->Parent()->Tree());
    interior_nodes.Insert(node);
  }

  // Remove interior nodes
  for (int i = 0; i < interior_nodes.NEntries(); i++) {
    R3SurfelNode *node = interior_nodes.Kth(i);
    R3SurfelNode *parent = node->Parent();

    // Move parts into parent
    while (node->NParts() > 0) {
      R3SurfelNode *part = node->Part(0);
      part->SetParent(parent);
    }

    // Remove node
    tree->RemoveNode(node);
    delete node;
  }

  // Remove blocks from root node
  R3SurfelNode *node = tree->RootNode();
  while (node->NBlocks() > 0) {
    R3SurfelBlock *block = node->Block(0);
    node->RemoveBlock(block);
    tree->Database()->RemoveBlock(block);
    delete block;
  }

  // Print statistics
  if (print_verbose) {
    printf("Removed interior nodes ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Nodes = %d\n", tree->NNodes());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
SplitSurfelTreeNodes(R3SurfelScene *scene, const char *node_name,
  int max_parts_per_node, int max_blocks_per_node, 
  RNScalar max_node_complexity, RNScalar max_block_complexity,
  RNLength max_leaf_extent, RNLength max_block_extent, 
  int max_levels)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Get surfel tree
  R3SurfelTree *tree = scene->Tree();
  if (!tree) {
    fprintf(stderr, "Scene has no surfel tree\n");
    return 0;
  }    

  // Find node
  R3SurfelNode *node = NULL;
  if (strcmp(node_name, "All")) {
    node = tree->FindNodeByName(node_name);
    if (!node) {
      fprintf(stderr, "Unable to find node with name %s\n", node_name);
      return 0;
    }
  }

  // Check node
  if (node) {
    // Split nodes
    tree->SplitNodes(node,
      max_parts_per_node, max_blocks_per_node, 
      max_node_complexity, max_block_complexity, 
      max_leaf_extent, max_block_extent, 
      max_levels);

    // Print statistics
    if (print_verbose) {
      printf("Split nodes starting at %s ...\n", node_name);
      printf("  Time = %.2f seconds\n", start_time.Elapsed());
      printf("  # Nodes = %d\n", tree->NNodes());
      printf("  # Blocks = %d\n", tree->Database()->NBlocks());
      fflush(stdout);
    }
  }
  else {
    // Split nodes
    tree->SplitNodes(
      max_parts_per_node, max_blocks_per_node, 
      max_node_complexity, max_block_complexity, 
      max_leaf_extent, max_block_extent, 
      max_levels);

    // Print statistics
    if (print_verbose) {
      printf("Split all nodes  ...\n");
      printf("  Time = %.2f seconds\n", start_time.Elapsed());
      printf("  # Nodes = %d\n", tree->NNodes());
      printf("  # Blocks = %d\n", tree->Database()->NBlocks());
      fflush(stdout);
    }
  }

  // Return success
  return 1;
}



static int
CreateMultiresolutionNodes(R3SurfelScene *scene, const char *node_name,
  RNScalar min_complexity, RNScalar min_resolution, RNScalar min_multiresolution_factor)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Get surfel tree
  R3SurfelTree *tree = scene->Tree();
  if (!tree) {
    fprintf(stderr, "Scene has no surfel tree\n");
    return 0;
  }    

  // Temporary
  if (strcmp(node_name, "All")) {
    fprintf(stderr, "-create_multiresolution_nodes only supported for All nodes\n");
    return 0;
  }

  // Create multiresolution nodes
  tree->CreateMultiresolutionNodes(min_complexity, min_resolution, min_multiresolution_factor);

  // Print statistics
  if (print_verbose) {
    printf("Created multiresolution nodes  ...\n");
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Nodes = %d\n", tree->NNodes());
    fflush(stdout);
  }

  // Return success
  return 1;
}



static int
CreateMultiresolutionBlocks(R3SurfelScene *scene, const char *node_name,
  RNScalar multiresolution_factor, RNScalar max_node_complexity)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();

  // Get surfel tree
  R3SurfelTree *tree = scene->Tree();
  if (!tree) {
    fprintf(stderr, "Scene has no surfel tree\n");
    return 0;
  }    

  // Find node
  R3SurfelNode *node = NULL;
  if (strcmp(node_name, "All")) {
    node = tree->FindNodeByName(node_name);
    if (!node) {
      fprintf(stderr, "Unable to find node with name %s\n", node_name);
      return 0;
    }
  }

  // Check node
  if (node) {
    // Create multiresolution starting at node
    tree->CreateMultiresolutionBlocks(node, multiresolution_factor, max_node_complexity);

    // Print statistics
    if (print_verbose) {
      printf("Created multiresolution blocks for nodes starting at %s ...\n", node_name);
      printf("  Time = %.2f seconds\n", start_time.Elapsed());
      printf("  # Nodes = %d\n", tree->NNodes());
      printf("  # Blocks = %d\n", tree->Database()->NBlocks());
      fflush(stdout);
    }
  }
  else {
    // Create multiresolution starting at all root nodes
    tree->CreateMultiresolutionBlocks(multiresolution_factor, max_node_complexity);

    // Print statistics
    if (print_verbose) {
      printf("Created multiresolution blocks for all nodes  ...\n");
      printf("  Time = %.2f seconds\n", start_time.Elapsed());
      printf("  # Nodes = %d\n", tree->NNodes());
      printf("  # Blocks = %d\n", tree->Database()->NBlocks());
      fflush(stdout);
    }
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// SEGMENTATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

static int
CreateClusterObjects(R3SurfelScene *scene, 
  const char *parent_object_name, const char *parent_node_name, const char *source_node_name, 
  int max_neighbors, RNLength max_neighbor_distance, 
  RNLength max_offplane_distance, RNAngle max_normal_angle,
  int min_points_per_object, RNLength chunk_size)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Creating cluster objects ...\n");
    fflush(stdout);
  }

  // Find parent object
  R3SurfelObject *parent_object = scene->FindObjectByName(parent_object_name);
  if (!parent_object) {
    fprintf(stderr, "Unable to find object with name %s\n", parent_object_name);
    return 0;
  }

  // Find parent node
  R3SurfelNode *parent_node = scene->Tree()->FindNodeByName(parent_node_name);
  if (!parent_node) {
    fprintf(stderr, "Unable to find node with name %s\n", parent_node_name);
    return 0;
  }

  // Find source node
  R3SurfelNode *source_node = scene->Tree()->FindNodeByName(source_node_name);
  if (!source_node) {
    fprintf(stderr, "Unable to find node with name %s\n", source_node_name);
    return 0;
  }

  // Create cluster objects 
  RNArray<R3SurfelObject *> *objects = CreateClusterObjects(scene, 
    source_node, NULL, parent_object, parent_node,                                                   
    max_neighbors, max_neighbor_distance, 
    max_offplane_distance, max_normal_angle, 
    min_points_per_object, chunk_size);
  if (!objects) {
    fprintf(stderr, "No cluster objects created\n");
    return 0;
  }

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Objects = %d\n", objects->NEntries());
    fflush(stdout);
  }

  // Delete array of objects
  delete objects;

  // Return success
  return 1;
}



static int
CreatePlanarObjects(R3SurfelScene *scene, 
  const char *parent_object_name, const char *parent_node_name, const char *source_node_name, 
  int max_neighbors, RNLength max_neighbor_distance, 
  RNLength max_offplane_distance, RNAngle max_normal_angle,
  RNArea min_area, RNScalar min_density, int min_points,
  RNLength grid_spacing, RNScalar accuracy_factor, RNLength chunk_size)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Creating planar objects ...\n");
    fflush(stdout);
  }

  // Find parent object
  R3SurfelObject *parent_object = scene->FindObjectByName(parent_object_name);
  if (!parent_object) {
    fprintf(stderr, "Unable to find object with name %s\n", parent_object_name);
    return 0;
  }

  // Find parent node
  R3SurfelNode *parent_node = scene->Tree()->FindNodeByName(parent_node_name);
  if (!parent_node) {
    fprintf(stderr, "Unable to find node with name %s\n", parent_node_name);
    return 0;
  }

  // Find source node
  R3SurfelNode *source_node = scene->Tree()->FindNodeByName(source_node_name);
  if (!source_node) {
    fprintf(stderr, "Unable to find node with name %s\n", source_node_name);
    return 0;
  }

  // Create planar objects 
  RNArray<R3SurfelObject *> *objects = CreatePlanarObjects(scene, 
    source_node, NULL, parent_object, parent_node,                                                   
    max_neighbors, max_neighbor_distance, 
    max_offplane_distance, max_normal_angle,
    min_area, min_density, min_points, 
    grid_spacing, accuracy_factor, chunk_size);
  if (!objects) {
    fprintf(stderr, "No planar objects created\n");
    return 0;
  }

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Objects = %d\n", objects->NEntries());
    fflush(stdout);
  }

  // Delete array of objects
  delete objects;

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// OUTPUT FUNCTIONS
////////////////////////////////////////////////////////////////////////

static int
OutputBlobs(R3SurfelScene *scene, const char *directory_name)
{
  // Start statistics
  RNTime start_time;
  start_time.Read();
  if (print_verbose) {
    printf("Outputing blobs to %s ...\n", directory_name);
    fflush(stdout);
  }

  // Create directory
  char cmd[1024];
  sprintf(cmd, "mkdir -p %s", directory_name);
  system(cmd);

  // Output blob file for each object
  for (int i = 0; i < scene->NObjects(); i++) {
    R3SurfelObject *object = scene->Object(i);

    // Get object label
    R3SurfelLabel *label = object->GroundTruthLabel();
    if (!label) label = object->HumanLabel();
    int label_identifier = (label) ? label->Identifier() : 0;

    // Create pointset
    R3SurfelPointSet *pointset = object->PointSet();
    if (!pointset) continue;
    if (pointset->NPoints() == 0) { 
      delete pointset; 
      continue; 
    }

    // Create filename
    char filename[1024];
    R3Point centroid = object->Centroid();
    sprintf(filename, "%s/%d_%.3f_%.3f_%.3f.xyz", directory_name,
      label_identifier, centroid.X(), centroid.Y(), centroid.Z());

    // Open file
    FILE *fp = fopen(filename, "w");
    if (!fp) {
      fprintf(stderr, "Unable to open xyz file %s\n", filename);
      delete pointset;
      return 0;
    }

    // Write points to file
    for (int j = 0; j < pointset->NPoints(); j++) {
      R3SurfelPoint *point = pointset->Point(j);
      R3Point position = point->Position();
      // RNRgb rgb = point->Rgb();
      fprintf(fp, "%.6f %.6f %.6f\n", position.X(), position.Y(), position.Z());
    }

    // Close file
    fclose(fp);

    // Print message
    if (print_debug) {
      printf("%3d %8.3f %8.3f %8.3f : %6d %g\n", label_identifier, 
        centroid.X(), centroid.Y(), centroid.Z(),
        pointset->NPoints(), object->Complexity());
    }

    // Delete pointset
    delete pointset;
  }

  // Print statistics
  if (print_verbose) {
    printf("  Time = %.2f seconds\n", start_time.Elapsed());
    printf("  # Blobs = %d\n", scene->NObjects());
    fflush(stdout);
  }
  
  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// PROGRAM ARGUMENT PARSING
////////////////////////////////////////////////////////////////////////

static int
CheckForNumber(const char *str)
{
  // Check if string is a number
  for (const char *strp = str; *strp; strp++) {
    if (isdigit(*strp)) continue;
    if (*strp == '-') continue;
    if (*strp == '-') continue;
    if (*strp == '+') continue;
    if (*strp == '.') continue;
    return 0;
  }

  // Passed all tests
  return 1;
}



static int 
CheckForArgument(int argc, char **argv, const char *argument)
{
  // Check for -v in program arguments
  for (int i = 1; i < argc; i++) 
    if (!strcmp(argv[i], argument)) return 1;
  return 0;
}



R3SurfelConstraint *
ParseConstraint(int& argc, char **& argv)
{
  // Check arguments
  if (argc == 0) return NULL;

  // Check constraint type
  argc--; argv++;
  const char *constraint_type = *argv;
  if (!strcmp(constraint_type, "BoundingBox")) {
    // Read bounding box coordinates
    argc--; argv++; double x1 = atof(*argv);
    argc--; argv++; double y1 = atof(*argv);
    argc--; argv++; double z1 = atof(*argv);
    argc--; argv++; double x2 = atof(*argv);
    argc--; argv++; double y2 = atof(*argv);
    argc--; argv++; double z2 = atof(*argv);
    R3Box box(x1, y1, z1, x2, y2, z2);
    R3SurfelBoxConstraint *constraint = new R3SurfelBoxConstraint(box);
    return constraint;
  }
  else if (!strcmp(constraint_type, "OverheadGrid")) {
    // Read overhead grid
    argc--; argv++; const char *overhead_grid_name = *argv;
    R2Grid *overhead_grid = ReadGrid(overhead_grid_name);
    if (!overhead_grid) return NULL;

    // Parse comparison type
    argc--; argv++; const char *comparison_type_string = *argv;
    int comparison_type = 0;
    if (!strcmp(comparison_type_string, "NotEqual")) comparison_type = R3_SURFEL_CONSTRAINT_NOT_EQUAL;
    else if (!strcmp(comparison_type_string, "Equal")) comparison_type = R3_SURFEL_CONSTRAINT_EQUAL;
    else if (!strcmp(comparison_type_string, "Greater")) comparison_type = R3_SURFEL_CONSTRAINT_GREATER;
    else if (!strcmp(comparison_type_string, "GreaterOrEqual")) comparison_type = R3_SURFEL_CONSTRAINT_GREATER_OR_EQUAL;
    else if (!strcmp(comparison_type_string, "Less")) comparison_type = R3_SURFEL_CONSTRAINT_LESS;
    else if (!strcmp(comparison_type_string, "LessOrEqual")) comparison_type = R3_SURFEL_CONSTRAINT_LESS_OR_EQUAL;
    else { fprintf(stderr, "Unrecognized constraint comparison type: %s\n", comparison_type_string); return NULL; }

    // Parse surfel operand
    argc--; argv++; const char *surfel_operand_string = *argv;
    RNScalar surfel_operand_value = 0;
    int surfel_operand_type = R3_SURFEL_CONSTRAINT_OPERAND;
    if (!strcmp(surfel_operand_string, "X")) surfel_operand_type = R3_SURFEL_CONSTRAINT_X;
    else if (!strcmp(surfel_operand_string, "Y")) surfel_operand_type = R3_SURFEL_CONSTRAINT_Y;
    else if (!strcmp(surfel_operand_string, "Z")) surfel_operand_type = R3_SURFEL_CONSTRAINT_Z;
    else if (CheckForNumber(surfel_operand_string)) surfel_operand_value = atof(surfel_operand_string); 
    else { fprintf(stderr, "Unrecognized surfel operand: %s\n", surfel_operand_string); return NULL; }
  
    // Parse grid operand
    argc--; argv++; const char *grid_operand_string = *argv;
    RNScalar grid_operand_value = 0;
    int grid_operand_type = R3_SURFEL_CONSTRAINT_OPERAND;
    if (!strcmp(grid_operand_string, "Value")) grid_operand_type = R3_SURFEL_CONSTRAINT_VALUE;
    else if (CheckForNumber(grid_operand_string)) grid_operand_value = atof(grid_operand_string); 
    else { fprintf(stderr, "Unrecognized grid operand: %s\n", grid_operand_string); return NULL; }

    // Parse epsilon
    argc--; argv++; double epsilon = atof(*argv);

    // Create constraint
    R3SurfelOverheadGridConstraint *constraint = new R3SurfelOverheadGridConstraint(
      overhead_grid, comparison_type, 
      surfel_operand_type, grid_operand_type, 
      surfel_operand_value, grid_operand_value, 
      epsilon);
    return constraint;
  }

  // Did not recognize constraint type
  fprintf(stderr, "Unrecognized constraint type: %s\n", constraint_type); 
  return NULL;
}



////////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
  // Check program arguments
  if (argc < 3) {
    // Print usage
    fprintf(stderr, "Usage: surfelprocess scenefile databasefile [operations]\n");
    exit(-1);
  }

  // Parse program arguments
  scene_name = argv[1];
  database_name = argv[2];
  aerial_only = CheckForArgument(argc, argv, "-aerial_only");
  terrestrial_only = CheckForArgument(argc, argv, "-terrestrial_only");
  print_verbose = CheckForArgument(argc, argv, "-v");
  print_debug = CheckForArgument(argc, argv, "-debug");

  // Open scene
  R3SurfelScene *scene = OpenScene(scene_name, database_name);
  if (!scene) exit(-1);

  // Execute operations
  argc -= 3; argv += 3;
  while (argc > 0) {
    if (!strcmp(*argv, "-v")) print_verbose = 1;
    else if (!strcmp(*argv, "-debug")) print_debug = 1;
    else if (!strcmp(*argv, "-aerial_only")) aerial_only = 1;
    else if (!strcmp(*argv, "-terrestrial_only")) terrestrial_only = 1;
    else if (!strcmp(*argv, "-create_node")) { 
      argc--; argv++; char *node_name = *argv; 
      argc--; argv++; char *parent_name = *argv; 
      if (!CreateNode(scene, node_name, parent_name)) exit(-1);
    }
    else if (!strcmp(*argv, "-create_object")) { 
      argc--; argv++; char *object_name = *argv; 
      argc--; argv++; char *parent_name = *argv; 
      argc--; argv++; char *node_name = *argv; 
      if (!CreateObject(scene, object_name, parent_name, node_name)) exit(-1);
    }
    else if (!strcmp(*argv, "-create_label")) { 
      argc--; argv++; char *label_name = *argv; 
      argc--; argv++; char *parent_name = *argv; 
      if (!CreateLabel(scene, label_name, parent_name)) exit(-1);
    }
    else if (!strcmp(*argv, "-load_surfels")) { 
      argc--; argv++; char *surfels_filename = *argv; 
      argc--; argv++; char *node_name = *argv; 
      argc--; argv++; char *parent_node_name = *argv; 
      if (!LoadSurfels(scene, surfels_filename, 
        NULL, NULL, 
        node_name, parent_node_name)) exit(-1);
    }
    else if (!strcmp(*argv, "-load_surfels_list")) { 
      argc--; argv++; char *list_filename = *argv; 
      argc--; argv++; char *parent_node_name = *argv; 
      if (!LoadSurfelsList(scene, list_filename, 
        NULL, parent_node_name)) exit(-1);
    }
    else if (!strcmp(*argv, "-load_object")) { 
      argc--; argv++; char *surfels_filename = *argv; 
      argc--; argv++; char *object_name = *argv; 
      argc--; argv++; char *parent_object_name = *argv; 
      argc--; argv++; char *node_name = *argv; 
      argc--; argv++; char *parent_node_name = *argv; 
      if (!LoadSurfels(scene, surfels_filename, 
        object_name, parent_object_name, 
        node_name, parent_node_name)) exit(-1);
    }
    else if (!strcmp(*argv, "-load_object_list")) { 
      argc--; argv++; char *list_filename = *argv; 
      argc--; argv++; char *parent_object_name = *argv; 
      argc--; argv++; char *parent_node_name = *argv; 
      if (!LoadSurfelsList(scene, list_filename, 
        parent_object_name, parent_node_name)) exit(-1);
    }
    else if (!strcmp(*argv, "-load_label_list")) { 
      argc--; argv++; char *list_filename = *argv; 
      argc--; argv++; char *parent_label_name = *argv; 
      if (!LoadLabelList(scene, list_filename, parent_label_name)) exit(-1);
    }
    else if (!strcmp(*argv, "-load_assignment_list")) { 
      argc--; argv++; char *list_filename = *argv; 
      if (!LoadAssignmentList(scene, list_filename)) exit(-1);
    }
    else if (!strcmp(*argv, "-load_feature_list")) { 
      argc--; argv++; char *list_filename = *argv; 
      if (!LoadFeatureList(scene, list_filename)) exit(-1);
    }
    else if (!strcmp(*argv, "-load_google")) { 
      argc--; argv++; char *list_filename = *argv; 
      argc--; argv++; char *parent_node_name = *argv; 
      if (!LoadSurfelsFromGoogleStreetView(scene, list_filename, parent_node_name)) exit(-1);
    }
    else if (!strcmp(*argv, "-load_scene")) { 
      argc--; argv++; char *scene_filename = *argv; 
      argc--; argv++; char *database_filename = *argv; 
      argc--; argv++; char *parent_object_name = *argv; 
      argc--; argv++; char *parent_label_name = *argv; 
      argc--; argv++; char *parent_node_name = *argv; 
      if (!LoadScene(scene, scene_filename, database_filename, 
        parent_object_name, parent_label_name, parent_node_name)) {
        exit(-1);
      }
    }
    // else if (!strcmp(*argv, "-transfer_labels")) { 
    //   argc--; argv++; char *label_scene_filename = *argv; 
    //   argc--; argv++; char *label_database_filename = *argv; 
    //   if (!TransferLabels(scene, label_scene_filename, label_database_filename)) {
    //     exit(-1);
    //   }
    // }
    else if (!strcmp(*argv, "-mask")) { 
      argc--; argv++; char *source_node_name = *argv; 
      R3SurfelConstraint *constraint = ParseConstraint(argc, argv);
      if (!constraint) exit(-1);
      if (!Mask(scene, source_node_name, constraint)) exit(-1);
      delete constraint;
    }
    else if (!strcmp(*argv, "-remove_objects")) { 
      if (!RemoveObjects(scene)) exit(-1);
    }
    else if (!strcmp(*argv, "-remove_labels")) { 
      if (!RemoveLabels(scene)) exit(-1);
    }
    else if (!strcmp(*argv, "-remove_interior_nodes")) { 
      if (!RemoveInteriorNodes(scene)) exit(-1);
    }
    else if (!strcmp(*argv, "-transform_with_configuration_file")) { 
      argc--; argv++; char *configuration_filename = *argv; 
      if (!TransformWithConfigurationFile(scene, configuration_filename)) exit(-1);
    }
    else if (!strcmp(*argv, "-create_cluster_objects")) { 
      argc--; argv++; char *parent_object_name = *argv; 
      argc--; argv++; char *parent_node_name = *argv; 
      argc--; argv++; char *source_node_name = *argv; 
      argc--; argv++; int max_neighbors = atoi(*argv); 
      argc--; argv++; double max_neighbor_distance = atof(*argv); 
      argc--; argv++; double max_offplane_distance = atof(*argv); 
      argc--; argv++; double max_normal_angle = atof(*argv); 
      argc--; argv++; int min_points_per_object = atoi(*argv); 
      argc--; argv++; double chunk_size= atof(*argv); 
      if (!CreateClusterObjects(scene, parent_object_name, parent_node_name, source_node_name,
        chunk_size, max_neighbors, max_neighbor_distance, 
        max_offplane_distance, max_normal_angle, min_points_per_object)) {
        exit(-1);
      }
    }
    else if (!strcmp(*argv, "-create_planar_objects")) { 
      argc--; argv++; char *parent_object_name = *argv; 
      argc--; argv++; char *parent_node_name = *argv; 
      argc--; argv++; char *source_node_name = *argv; 
      argc--; argv++; int max_neighbors = atoi(*argv); 
      argc--; argv++; double max_neighbor_distance = atof(*argv); 
      argc--; argv++; double max_offplane_distance = atof(*argv); 
      argc--; argv++; double max_normal_angle = atof(*argv); 
      argc--; argv++; double min_area = atof(*argv); 
      argc--; argv++; double min_density = atof(*argv); 
      argc--; argv++; double min_points = atof(*argv); 
      argc--; argv++; double grid_spacing = atof(*argv); 
      argc--; argv++; double accuracy_factor = atof(*argv); 
      argc--; argv++; double chunk_size= atof(*argv); 
      if (!CreatePlanarObjects(scene, parent_object_name, parent_node_name, source_node_name,
        chunk_size, max_neighbors, max_neighbor_distance, max_offplane_distance, max_normal_angle,
        min_area, min_density, min_points, grid_spacing, accuracy_factor)) {
        exit(-1);
      }
    }
    else if (!strcmp(*argv, "-create_default_hierarchy")) { 
      const char *node_name = "Root";
      int max_parts_per_node = 8;
      int max_blocks_per_node = 32;
      RNScalar max_node_complexity = 1024;
      RNScalar max_block_complexity = 1024;
      RNLength max_leaf_extent = 10.0;
      RNLength max_block_extent = 10.0;
      RNScalar multiresolution_factor = 0.25;
      int max_levels = 64;
      if (!SplitSurfelTreeNodes(scene, node_name, 
        max_parts_per_node, max_blocks_per_node, 
        max_node_complexity, max_block_complexity,
        max_leaf_extent, max_block_extent, max_levels)) {
        exit(-1);
      }
      if (!CreateMultiresolutionBlocks(scene, node_name, 
        multiresolution_factor, max_node_complexity)) {
        exit(-1);
      }
    }
    else if (!strcmp(*argv, "-create_tree_hierarchy")) { 
      argc--; argv++; char *node_name = *argv; 
      argc--; argv++; int max_parts_per_node = atoi(*argv);
      argc--; argv++; int max_blocks_per_node = atoi(*argv);
      argc--; argv++; double max_node_complexity = atof(*argv); 
      argc--; argv++; double max_block_complexity = atof(*argv); 
      argc--; argv++; double max_leaf_extent = atof(*argv); 
      argc--; argv++; double max_block_extent = atof(*argv); 
      argc--; argv++; double multiresolution_factor = atof(*argv); 
      argc--; argv++; int max_levels = atoi(*argv); 
      if (!SplitSurfelTreeNodes(scene, node_name, 
        max_parts_per_node, max_blocks_per_node, 
        max_node_complexity, max_block_complexity,
        max_leaf_extent, max_block_extent, max_levels)) {
        exit(-1);
      }
      if (!CreateMultiresolutionBlocks(scene, node_name, 
        multiresolution_factor, max_node_complexity)) {
        exit(-1);
      }
    }
    else if (!strcmp(*argv, "-split_nodes")) { 
      argc--; argv++; char *node_name = *argv; 
      argc--; argv++; int max_parts_per_node = atoi(*argv);
      argc--; argv++; int max_blocks_per_node = atoi(*argv);
      argc--; argv++; double max_node_complexity = atof(*argv); 
      argc--; argv++; double max_block_complexity = atof(*argv); 
      argc--; argv++; double max_leaf_extent = atof(*argv); 
      argc--; argv++; double max_block_extent = atof(*argv);
      argc--; argv++; int max_levels = atoi(*argv); 
      if (!SplitSurfelTreeNodes(scene, node_name, 
        max_parts_per_node, max_blocks_per_node, 
        max_node_complexity, max_block_complexity,
        max_leaf_extent, max_block_extent, max_levels)) {
        exit(-1);
      }
    }
    else if (!strcmp(*argv, "-create_multiresolution_nodes")) { 
      argc--; argv++; char *node_name = *argv; 
      argc--; argv++; double min_complexity = atof(*argv); 
      argc--; argv++; double min_resolution = atof(*argv); 
      argc--; argv++; double min_multiresolution_factor = atof(*argv); 
      if (!CreateMultiresolutionNodes(scene, node_name, 
        min_complexity, min_resolution, min_multiresolution_factor)) {
        exit(-1);
      }
    }
    else if (!strcmp(*argv, "-create_multiresolution_blocks")) { 
      argc--; argv++; char *node_name = *argv; 
      argc--; argv++; double multiresolution_factor = atof(*argv); 
      argc--; argv++; double max_node_complexity = atof(*argv); 
      if (!CreateMultiresolutionBlocks(scene, node_name, 
        multiresolution_factor, max_node_complexity)) {
        exit(-1);
      }
    }
    else if (!strcmp(*argv, "-output_blobs")) { 
      argc--; argv++; char *blob_directory_name = *argv; 
      if (!OutputBlobs(scene, blob_directory_name)) exit(-1);
    }
    else { 
      fprintf(stderr, "Invalid operation: %s", *argv); 
      exit(1); 
    }
    argv++; argc--;
  }

  // Close scene
  if (!CloseScene(scene)) exit(-1);

  // Return success 
  return 0;
}



