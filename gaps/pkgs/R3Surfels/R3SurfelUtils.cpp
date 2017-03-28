// Source file for the surfel scene processing utilities



////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// Point set creation
////////////////////////////////////////////////////////////////////////

R3SurfelPointSet *
CreatePointSet(R3SurfelPointSet *pointset, 
  const R3SurfelConstraint *constraint)
{
  // Create point set
  R3SurfelPointSet *result = new R3SurfelPointSet();
  if (constraint) result->InsertPoints(pointset, *constraint);
  else result->InsertPoints(pointset);
  if (result->NPoints() == 0) { delete result; return NULL; }
  return result;
}



static int
InsertPoints(R3SurfelPointSet *pointset, 
  R3SurfelTree *tree, R3SurfelNode *node,
  const R3SurfelConstraint *constraint)
{
  // Check node
  if (constraint && !constraint->Check(node)) return 0;

  // Check if leaf node
  if (node->NParts() == 0) {
    // Insert blocks
    for (int i = 0; i < node->NBlocks(); i++) {
      R3SurfelBlock *block = node->Block(i);
      if (constraint) pointset->InsertPoints(block, *constraint);
      else pointset->InsertPoints(block);
    }
  }
  else {
    // Consider parts
    for (int i = 0; i < node->NParts(); i++) {
      R3SurfelNode *part = node->Part(i);
      InsertPoints(pointset, tree, part, constraint);
    }
  }

  // Return success
  return 1;
}



R3SurfelPointSet *
CreatePointSet(R3SurfelScene *scene, 
  R3SurfelNode *source_node,
  const R3SurfelConstraint *constraint)
{
  // Check tree
  R3SurfelTree *tree = scene->Tree();
  if (!tree) return NULL;

  // Check source node
  if (!source_node) source_node = tree->RootNode();

  // Create point set
  R3SurfelPointSet *pointset = new R3SurfelPointSet();
  if (!pointset) {
    fprintf(stderr, "Unable to allocate point set\n");
    return NULL;
  }

  // Insert nodes 
  InsertPoints(pointset, tree, source_node, constraint);

  // Check result
  if (pointset->NPoints() == 0) { 
    delete pointset; 
    return NULL; 
  }

  // Return point set
  return pointset;
}



R3SurfelPointSet *
CreatePointSet(R3SurfelScene *scene, R3Point& origin, 
  RNLength max_radius, RNLength min_height, RNLength max_height, RNLength max_spacing,
  RNVolume min_volume, RNVolume max_volume, int min_points)
{
  // Check origin Z coordinate
  R3SurfelPointSet *surfels2 = NULL;
  if (origin.Z() == RN_UNKNOWN) {
    // Extract points within XY radius
    R3SurfelCylinderConstraint cylinder_constraint(origin, max_radius);
    R3SurfelPointSet *surfels1 = CreatePointSet(scene, NULL, &cylinder_constraint);
    if (!surfels1) return NULL;
    
    // Check minimum number of points
    if ((min_points > 0) && (surfels1->NPoints() < min_points)) {
      delete surfels1;
      return NULL;
    }
    
    // Compute support plane
    RNScalar support_count = 0;
    R3Plane support_plane = EstimateSupportPlane(surfels1, 0.1, &support_count);

    // Move origin to support plane
    if ((support_count > 16) && (fabs(support_plane[2]) > 0.5)) {
      RNScalar support_z = -(origin[0]*support_plane[0] + origin[1]*support_plane[1] + support_plane[3]) / support_plane[2];
      origin[2] = support_z;
    }
    else {
      origin[2] = surfels1->BBox().ZMin();
    }

    // Remove points that are too low or too high
    R3SurfelCylinderConstraint z_constraint(origin, max_radius, origin[2] + min_height, origin[2] + max_height);
    surfels2 = CreatePointSet(surfels1, &z_constraint);
    delete surfels1;
    if (!surfels2) return NULL;
  }
  else {
    // Extract points with cylinder
    R3SurfelCylinderConstraint cylinder_constraint(origin, max_radius, origin[2] + min_height, origin[2] + max_height);
    surfels2 = CreatePointSet(scene, NULL, &cylinder_constraint);
    if (!surfels2) return NULL;
  }

  // Check minimum number of points
  if ((min_points > 0) && (surfels2->NPoints() < min_points)) {
    delete surfels2;
    return NULL;
  }

  // Remove points that are not connected
  R3SurfelPointSet *surfels3 = surfels2;
  if (max_spacing > 0) {
    surfels3 = CreateConnectedPointSet(surfels2, origin, max_radius,
      min_height, max_height, min_volume, max_volume, max_spacing, 256);
    delete surfels2;
    if (!surfels3) return NULL;
  }
  
  // Check minimum number of points
  if ((min_points > 0) && (surfels3->NPoints() < min_points)) {
    delete surfels3;
    return NULL;
  }

  // Return pointset
  return surfels3;
}



////////////////////////////////////////////////////////////////////////
// Point graph creation
////////////////////////////////////////////////////////////////////////

R3SurfelPointGraph *
CreatePointGraph(R3SurfelPointSet *pointset, 
  int max_neighbors, RNLength max_neighbor_distance)
{
  // Create point graph 
  return new R3SurfelPointGraph(*pointset, max_neighbors, max_neighbor_distance);
}



R3SurfelPointGraph *
CreatePointGraph(R3SurfelScene *scene, 
  R3SurfelNode *source_node, const R3SurfelConstraint *constraint, 
  int max_neighbors, RNLength max_neighbor_distance)
{
  // Create point graph with all surfels satisfying constraint
  R3SurfelPointSet *pointset = CreatePointSet(scene, source_node, constraint);
  if (!pointset) return NULL;
  R3SurfelPointGraph *graph = new R3SurfelPointGraph(*pointset, max_neighbors, max_neighbor_distance);
  delete pointset;
  return graph;
}



////////////////////////////////////////////////////////////////////////
// Block creation
////////////////////////////////////////////////////////////////////////

R3SurfelBlock *
CreateBlock(R3SurfelScene *scene, R3SurfelPointSet *pointset, 
  RNBoolean copy_surfels)
{
  // Get convenient variables
  if (pointset->NPoints() == 0) return NULL;
  R3SurfelTree *tree = scene->Tree();
  if (!tree) return NULL;
  R3SurfelDatabase *database = tree->Database();
  if (!database) return NULL;

  // Create block with copy of points
  R3SurfelBlock *block = new R3SurfelBlock(pointset);

  // Update properties
  block->UpdateProperties();

  // Insert block into database
  database->InsertBlock(block);

  // Remove and delete pointset from other blocks ???
  // tree->RemoveAndDeletePoints(scene, pointset);

  // Return block
  return block;
}



R3SurfelBlock *
CreateBlock(R3SurfelScene *scene, 
  R3SurfelNode *source_node, const R3SurfelConstraint *constraint, 
  RNBoolean copy_surfels)
{
  // Create point set
  R3SurfelPointSet *pointset = CreatePointSet(scene, source_node, constraint);
  if (!pointset) return NULL;

  // Create block
  R3SurfelBlock *block = CreateBlock(scene, pointset);

  // Delete point set
  delete pointset;

  // Return block
  return block;
}



////////////////////////////////////////////////////////////////////////
// Node creation
////////////////////////////////////////////////////////////////////////

R3SurfelNode *
CreateNode(R3SurfelScene *scene, R3SurfelPointSet *pointset, 
  R3SurfelNode *parent_node, const char *node_name, 
  RNBoolean copy_surfels)
{
  // Initialize result
  R3SurfelNode *node = NULL;

  // Get tree
  R3SurfelTree *tree = scene->Tree();
  if (!tree) return NULL;

  // Get parent node
  if (!parent_node) parent_node = tree->RootNode();

  // Check if should create copy of surfels for new node
  if (copy_surfels) {
    // Create block
    R3SurfelBlock *block = CreateBlock(scene, pointset);
    if (!block) return NULL;

    // Create node
    node = new R3SurfelNode(node_name);

    // Insert block
    node->InsertBlock(block);

    // Update properties
    node->UpdateProperties();

    // Insert node
    tree->InsertNode(node, parent_node);

    // Release block
    tree->Database()->ReleaseBlock(block);
  }
  else {
#if 0
    // Mark pointset
    pointset->SetMarks(TRUE);

    // Create constraint
    R3SurfelMarkConstraint constraint(TRUE, FALSE);
    
    // Split nodes based on constraint
    node = CreateNode(scene, tree->RootNode(), &constraint, parent_node, node_name, FALSE);

    // Unmark pointset
    pointset->SetMarks(FALSE);
    for (int i = 0; i < node->NParts(); i++) {
      R3SurfelNode *part = node->Part(i);
      part->SetMarks(FALSE);
    }
#else
    // Get surfel tree
    R3SurfelTree *tree = scene->Tree();
    if (!tree) return NULL;

    // Split leaf nodes
    RNArray<R3SurfelNode *> nodes;
    if (!tree->SplitNodes(*pointset, &nodes)) return NULL;
    if (nodes.IsEmpty()) return NULL;

    // Create node
    node = new R3SurfelNode(node_name);
    if (!node) return NULL;
    
    // Insert node into tree
    tree->InsertNode(node, parent_node);

    // Move nodes satisfying constraint into node
    for (int i = 0; i < nodes.NEntries(); i++) {
      R3SurfelNode *child = nodes.Kth(i);
      child->SetParent(node);
    }
    
    // Update properties
    node->UpdateProperties();
#endif
  }

  // Return node
  return node;
}



R3SurfelNode *
CreateNode(R3SurfelScene *scene, 
  R3SurfelNode *source_node, const R3SurfelConstraint *constraint,
  R3SurfelNode *parent_node, const char *node_name, 
  RNBoolean copy_surfels)
{
  // Initialize result
  R3SurfelNode *node = NULL;

  // Get tree
  R3SurfelTree *tree = scene->Tree();
  if (!tree) return NULL;

  // Get source node
  if (!source_node) source_node = tree->RootNode();

  // Get parent node
  if (!parent_node) parent_node = tree->RootNode();

  // Check if should create copy of surfels for new node
  if (copy_surfels) {
    // Create point set
    R3SurfelPointSet *pointset = CreatePointSet(scene, source_node, constraint);
    if (!pointset) return NULL;
    
    // Create node
    node = CreateNode(scene, pointset, parent_node, node_name);
    
    // Delete point set
    delete pointset;
  }
  else {
    // Get surfel tree
    R3SurfelTree *tree = scene->Tree();
    if (!tree) return NULL;

    // Split leaf nodes
    RNArray<R3SurfelNode *> nodes;
    if (!tree->SplitLeafNodes(source_node, *constraint, &nodes)) return NULL;
    if (nodes.IsEmpty()) return NULL;

    // Create node
    node = new R3SurfelNode(node_name);
    if (!node) return NULL;
    
    // Insert node into tree
    tree->InsertNode(node, parent_node);

    // Move nodes satisfying constraint into node
    for (int i = 0; i < nodes.NEntries(); i++) {
      R3SurfelNode *child = nodes.Kth(i);
      child->SetParent(node); 
    }
    
    // Update properties
    node->UpdateProperties();
  }

  // Return node
  return node;
}



////////////////////////////////////////////////////////////////////////
// Object creation
////////////////////////////////////////////////////////////////////////

int
SplitObject(R3SurfelObject *object, R3SurfelPointSet *pointset,
  R3SurfelObject **resultA, R3SurfelObject **resultB)
{
  // Read blocks
  object->ReadBlocks();

  // Set marks
  object->SetMarks(FALSE);
  pointset->SetMarks(TRUE);

  // Create constraint
  R3SurfelMultiConstraint constraint;
  R3SurfelMarkConstraint mark_constraint(TRUE, FALSE);
  R3SurfelObjectConstraint object_constraint(object);
  constraint.InsertConstraint(&mark_constraint);
  constraint.InsertConstraint(&object_constraint);

  // Split object
  int status = SplitObject(object, &constraint, resultA, resultB);

  // Release blocks
  object->ReleaseBlocks();

  // Return status
  return status;
}



int
SplitObject(R3SurfelObject *object, const R3SurfelConstraint *constraint,
  R3SurfelObject **resultA, R3SurfelObject **resultB)
{
  // Get useful variables
  if (!object) return 0;
  if (!constraint) return 0;
  if (object->NNodes() == 0) return 0;
  assert(strcmp(object->Name(), "Root"));
  R3SurfelScene *scene = object->Scene();
  if (!scene) return 0;
  R3SurfelTree *tree = scene->Tree();
  if (!tree) return 0;

  // Create constraint
  R3SurfelMultiConstraint multi_constraint;
  R3SurfelObjectConstraint object_constraint(object);
  multi_constraint.InsertConstraint(constraint);
  multi_constraint.InsertConstraint(&object_constraint);

  // Create array of nodes
  RNArray<R3SurfelNode *> nodes;
  for (int i = 0; i < object->NNodes(); i++) {
    R3SurfelNode *node = object->Node(i);
    nodes.Insert(node);
  }

  // Split all nodes
  RNArray<R3SurfelNode *> nodesA, nodesB;
  for (int i = 0; i < nodes.NEntries(); i++) {
    R3SurfelNode *node = nodes.Kth(i);
    tree->SplitLeafNodes(node, *constraint, &nodesA, &nodesB);
  }

  // Check nodesA
  if (nodesA.NEntries() == 0) {
    if (resultA) *resultA = NULL;
    if (resultB) *resultB = object; 
    return 0;
  }
    
  // Check nodesB
  if (nodesB.NEntries() == 0) {
    if (resultA) *resultA = object;
    if (resultB) *resultB = NULL; 
    return 0;
  }

  // Create new objects
  R3SurfelObject *objectA = new R3SurfelObject();
  R3SurfelObject *objectB = new R3SurfelObject();
  if (!objectA || !objectB) return 0;
      
  // Remove nodes from object
  while (object->NNodes() > 0) {
    R3SurfelNode *node = object->Node(0);
    object->RemoveNode(node);
  }
  
  // Insert nodes into objectA
  for (int j = 0; j < nodesA.NEntries(); j++) {
    R3SurfelNode *nodeA = nodesA.Kth(j);
    if (nodeA->Object()) continue;
    objectA->InsertNode(nodeA);
  }
  
  // Insert nodes into objectB
  for (int j = 0; j < nodesB.NEntries(); j++) {
    R3SurfelNode *nodeB = nodesB.Kth(j);
    if (nodeB->Object()) continue;
    objectB->InsertNode(nodeB);
  }
      
  // Copy result
  if (resultA) *resultA = objectA;
  if (resultB) *resultB = objectB;
  
  // Return success
  return 1;
}



R3SurfelObject *
CreateObject(R3SurfelScene *scene, 
  R3SurfelPointSet *pointset, 
  R3SurfelObject *parent_object, const char *object_name, 
  R3SurfelNode *parent_node, const char *node_name, 
  RNBoolean copy_surfels)
{
  // Get parent object
  if (!parent_object) parent_object = scene->RootObject();

  // Create object
  R3SurfelObject *object = new R3SurfelObject(object_name);
    
  // Insert object into scene
  scene->InsertObject(object, parent_object);

  // Check if should create copy of surfels for new node
  if (copy_surfels) {
    // Create node
    R3SurfelNode *node = CreateNode(scene, pointset, parent_node, node_name, copy_surfels);
    if (!node) { delete object; return NULL; }
    
    // Insert node into object
    object->InsertNode(node);
  }
  else {
    // Get surfel tree
    R3SurfelTree *tree = scene->Tree();
    if (!tree) return NULL;
    
    // Split nodes
    RNArray<R3SurfelNode *> nodes;
    tree->SplitNodes(*pointset, &nodes);
    if (nodes.IsEmpty()) { delete object; return NULL; }
    
    // Move nodes into object
    for (int i = 0; i < nodes.NEntries(); i++) {
      R3SurfelNode *node = nodes.Kth(i);
      R3SurfelObject *old_object = node->Object();
      if (old_object) old_object->RemoveNode(node);
      object->InsertNode(node);
    }
  }

  // Update properties
  object->UpdateProperties();

  // Return object
  return object;
}



R3SurfelObject *
CreateObject(R3SurfelScene *scene, 
  R3SurfelNode *source_node, const R3SurfelConstraint *constraint,
  R3SurfelObject *parent_object, const char *object_name, 
  R3SurfelNode *parent_node, const char *node_name, 
  RNBoolean copy_surfels)
{
  // Get tree
  R3SurfelTree *tree = scene->Tree();
  if (!tree) return NULL;

  // Get source node
  if (!source_node) source_node = tree->RootNode();

  // Get parent object
  if (!parent_object) parent_object = scene->RootObject();

  // Create object
  R3SurfelObject *object = new R3SurfelObject(object_name);
  if (!object) return NULL;

  // Insert object into scene
  scene->InsertObject(object, parent_object);

  // Check if should copy surfels
  if (copy_surfels) {
    // Create node
    R3SurfelNode *node = CreateNode(scene, source_node, constraint, parent_node, node_name, copy_surfels);
    if (!node) return NULL;
    
    // Insert node into object
    object->InsertNode(node);
    
  }
  else {
    // Get surfel tree
    R3SurfelTree *tree = scene->Tree();
    if (!tree) return NULL;

    // Split leaf nodes
    RNArray<R3SurfelNode *> nodes;
    tree->SplitLeafNodes(source_node, *constraint, &nodes);
    if (nodes.IsEmpty()) return NULL;

    // Insert nodes satisfying constraint into object
    for (int i = 0; i < nodes.NEntries(); i++) {
      R3SurfelNode *node = nodes.Kth(i);
      R3SurfelObject *old_object = node->Object();
      if (old_object) old_object->RemoveNode(node);
      object->InsertNode(node);
    }
  }

  // Update properties
  object->UpdateProperties();
    
  // Return object
  return object;
}



////////////////////////////////////////////////////////////////////////
// Connected point set creation (using graph)
////////////////////////////////////////////////////////////////////////

R3SurfelPointSet *
CreateConnectedPointSet(R3SurfelPointGraph *graph, int seed_index)
{
  // Initialize result
  R3SurfelPointSet *connected_pointset = new R3SurfelPointSet();
  if (graph->NPoints() == 0) return connected_pointset;

  // Find seed point 
  R3SurfelPoint *seed_point = graph->Point(seed_index);

  // Initialize marks
  for (int i = 0; i < graph->NPoints(); i++) {
    R3SurfelPoint *point = graph->Point(i);
    point->SetMark(FALSE);
  }

  // Flood fill from seed point
  RNArray<R3SurfelPoint *> stack;
  stack.Insert(seed_point);
  seed_point->SetMark(TRUE);
  while (!stack.IsEmpty()) {
    R3SurfelPoint *point = stack.Tail();
    stack.RemoveTail();
    connected_pointset->InsertPoint(*point);
    int point_index = graph->PointIndex(point);
    for (int i = 0; i < graph->NNeighbors(point_index); i++) {
      R3SurfelPoint *neighbor = graph->Neighbor(point_index, i);
      if (neighbor->IsMarked()) continue;
      stack.InsertTail(neighbor);
      neighbor->SetMark(TRUE);
    }
  }

  // Return connected pointset
  return connected_pointset;
}



R3SurfelPointSet *
CreateConnectedPointSet(R3SurfelPointGraph *graph, R3SurfelPoint *seed_point)
{
  // Find seed index
  int seed_index = graph->PointIndex(seed_point);

  // Check seed index
  if (seed_index < 0) return NULL;

  // Return connected component
  return CreateConnectedPointSet(graph, seed_index);
}



R3SurfelPointSet *
CreateConnectedPointSet(R3SurfelPointGraph *graph, const R3Point& center)
{
  // Find seed point (closest to center)
  int seed_index = -1;
  RNLength seed_distance = FLT_MAX;
  for (int i = 0; i < graph->NPoints(); i++) {
    R3SurfelPoint *point = graph->Point(i);
    RNLength distance = R3SquaredDistance(point->Position(), center);
    if (distance < seed_distance) {
      seed_distance = distance;
      seed_index = i;
    }
  }     

  // Check seed index
  if (seed_index < 0) return NULL;

  // Return connected component
  return CreateConnectedPointSet(graph, seed_index);
}



////////////////////////////////////////////////////////////////////////
// Connected point set creation (using grid)
////////////////////////////////////////////////////////////////////////

static int
MaskToLargestConnectedComponent(R3Grid *grid, RNScalar isolevel,
  RNScalar min_component_volume, RNScalar max_component_volume)
{
  // Check grid size
  if (grid->NEntries() == 0) return 0;

  // Get convenient variables
  RNScalar length_scale = grid->WorldToGridScaleFactor();
  RNScalar volume_scale = length_scale * length_scale * length_scale;
  int min_component_size = (int) (min_component_volume * volume_scale);
  int max_component_size = (int) (max_component_volume * volume_scale);
  int status = 0;

  // Allocate temporary memory for connectec components
  int *component_sizes = new int [ grid->NEntries() ];
  int *component_seeds = new int [ grid->NEntries() ];
  int *component_membership = new int [ grid->NEntries() ];

  // Compute connected component of grid
  int ncomponents = grid->ConnectedComponents(isolevel, grid->NEntries(), 
    component_seeds, component_sizes, component_membership);
  if (ncomponents > 0) {
    // Find largest connected component
    int largest_component = -1;
    int largest_component_size = 0;
    for (int i = 0; i < ncomponents; i++) {
      if ((min_component_size > 0) && (component_sizes[i] < min_component_size)) continue;
      if ((max_component_size > 0) && (component_sizes[i] > max_component_size)) continue;
      if (component_sizes[i] > largest_component_size) {
        largest_component_size = component_sizes[i];
        largest_component = i;
      }
    }

    // Zero all grid entries outside connected component
    if (largest_component >= 0) {
      status = 1;
      for (int i = 0; i < grid->NEntries(); i++) {
        if (component_membership[i] == largest_component) grid->SetGridValue(i, 1);
        else grid->SetGridValue(i, 0);
      }
    }
  }

  // Delete temporary memory
  delete [] component_sizes;
  delete [] component_seeds;
  delete [] component_membership;

  // Return status
  return status;
}



static int
MaskToSelectedConnectedComponent(R3Grid *grid, RNScalar isolevel, 
  const R3Point& center, 
  RNScalar min_component_volume, RNScalar max_component_volume)
{
  // Check grid size
  if (grid->NEntries() == 0) return 0;

  // Get convenient variables
  RNScalar length_scale = grid->WorldToGridScaleFactor();
  RNScalar volume_scale = length_scale * length_scale * length_scale;
  int min_component_size = (int) (min_component_volume * volume_scale);
  int max_component_size = (int) (max_component_volume * volume_scale);
  int status = 0;

  // Allocate temporary memory for connectec components
  int *component_sizes = new int [ grid->NEntries() ];
  int *component_seeds = new int [ grid->NEntries() ];
  int *component_membership = new int [ grid->NEntries() ];

  // Compute connected component of grid
  int ncomponents = grid->ConnectedComponents(isolevel, grid->NEntries(), 
    component_seeds, component_sizes, component_membership);
  if (ncomponents > 0) {
    // Check if grid cell containing center point is in a connected component
    int center_index = 0;
    R3Point grid_position = grid->GridPosition(center);
    int closest_component = -1;
    int center_i = (int) (grid_position[0] + 0.5);
    if ((center_i >= 0) && (center_i < grid->XResolution())) {
      int center_j = (int) (grid_position[1] + 0.5);
      if ((center_j >= 0) && (center_j < grid->YResolution())) {
        int center_k = (int) (grid_position[2] + 0.5);
        if ((center_k >= 0) && (center_k < grid->ZResolution())) {
          grid->IndicesToIndex(center_i, center_j, center_k, center_index);
          closest_component = component_membership[center_index];
          if (min_component_size > 0) {
            if (component_sizes[closest_component] < min_component_size) {
              closest_component = -1;
            }
          }
          if (max_component_size > 0) {
            if (component_sizes[closest_component] > max_component_size) {
              closest_component = -1;
            }
          }
        }
      }
    }

    // If not, find component whose size is big enough and whose seed is closest to center point
    if (closest_component < 0) {
      RNLength closest_dd = FLT_MAX;
      for (int i = 0; i < ncomponents; i++) {
        int seed_i, seed_j, seed_k;
        if ((min_component_size > 0) && (component_sizes[i] < min_component_size)) continue;
        if ((max_component_size > 0) && (component_sizes[i] > max_component_size)) continue;
        grid->IndexToIndices(component_seeds[i], seed_i, seed_j, seed_k);
        RNLength dd = R3SquaredDistance(R3Point(seed_i, seed_j, seed_k), grid_position);
        if (dd < closest_dd) { closest_dd = dd; closest_component = i; }       
      }
    }

    // Zero all grid entries outside selected connected component
    if (closest_component >= 0) {
      status = 1;
      for (int i = 0; i < grid->NEntries(); i++) {
        if (component_membership[i] == closest_component) grid->SetGridValue(i, 1);
        else grid->SetGridValue(i, 0);
      }
    }
  }

  // Delete temporary memory
  delete [] component_sizes;
  delete [] component_seeds;
  delete [] component_membership;

  // Return status
  return status;
}



static int 
MaskToSeededConnectedComponent(R3Grid& grid, const R3Grid& seeds, RNScalar isolevel)
{
  // Allocate grid of marks
  R3Grid marks(grid);
  marks.Clear(0);

  // Initialize stack 
  const RNScalar *grid_values = grid.GridValues();
  RNArray<const RNScalar *> stack;
  for (int i = 0; i < grid.NEntries(); i++) {
    RNScalar seed_value = seeds.GridValue(i);
    if (seed_value == 0) continue;
    RNScalar grid_value = grid.GridValue(i);
    if (grid_value > isolevel) {
      stack.Insert(&grid_values[i]);
      marks.SetGridValue(i, 1.0);
    }
    else {
      marks.SetGridValue(i, -1.0);
    }
  }

  // Find connected grid cells
  while (!stack.IsEmpty()) {
    // Pop tail off stack
    const RNScalar *grid_valuesp = stack.Tail();
    stack.RemoveTail();

    // Get grid index
    int grid_index = grid_valuesp - grid_values;
    assert(grid_index >= 0);
    assert(grid_index < grid.NEntries());
    assert(marks.GridValue(grid_index) == 1.0);
    assert(grid.GridValue(grid_index) > isolevel);

    // Mark connected neighbors
    int cx, cy, cz, neighbor_index;
    grid.IndexToIndices(grid_index, cx, cy, cz);
    for (int i = 0; i < 6; i++) {
      int ix = (i == 0) ? cx-1 : ((i == 1) ? cx+1 : cx);
      if ((ix >= 0) && (ix < grid.XResolution())) {
        int iy = (i == 2) ? cy-1 : ((i == 3) ? cy+1 : cy);
        if ((iy >= 0) && (iy < grid.YResolution())) {
          int iz = (i == 4) ? cz-1 : ((i == 5) ? cz+1 : cz);
          if ((iz >= 0) && (iz < grid.ZResolution())) {
            grid.IndicesToIndex(ix, iy, iz, neighbor_index);
            if (marks.GridValue(neighbor_index) == 0.0) {
              RNScalar neighbor_value = grid.GridValue(neighbor_index);
              if (neighbor_value > isolevel) {
                stack.Insert(&grid_values[neighbor_index]);
                marks.SetGridValue(neighbor_index, 1.0);
              }
              else {
                marks.SetGridValue(neighbor_index, -1.0);
              }
            }
          }
        }
      }
    }
  }

  // Mask grid
  marks.Threshold(0.5, 0.0, 1.0);
  grid.Mask(marks);

  // Return success
  return 1;
}



R3SurfelPointSet *
CreateConnectedPointSet(R3SurfelPointSet *pointset, 
  RNScalar min_volume, RNScalar max_volume, RNLength max_spacing)
{
  // Create grid
  R3Grid *grid = CreateGrid(pointset, max_spacing);
  if (!grid) return NULL;

  // Find largest connected component
  if (!MaskToLargestConnectedComponent(grid, 0.5, min_volume, max_volume)) { 
    delete grid; 
    return NULL; 
  }

  // Create connnected point set
  R3SurfelGridConstraint constraint(grid);
  R3SurfelPointSet *connected_pointset = CreatePointSet(pointset, &constraint);

  // Delete grid
  delete grid;

  // Return connected point set
  return connected_pointset;
}



R3SurfelPointSet *
CreateConnectedPointSet(R3SurfelPointSet *pointset, const R3Point &seed_position,
  RNScalar min_volume, RNScalar max_volume, RNLength grid_spacing, int max_grid_resolution)
{
  // Create grid
  R3Grid *grid = CreateGrid(pointset, grid_spacing, max_grid_resolution);
  if (!grid) return NULL;

  // Find largest connected component
  if (!MaskToSelectedConnectedComponent(grid, 0.5, seed_position, min_volume, max_volume)) { 
    delete grid; 
    return NULL; 
  }

  // Create connnected point set
  R3SurfelGridConstraint constraint(grid);
  R3SurfelPointSet *connected_pointset = CreatePointSet(pointset, &constraint);

  // Delete grid
  delete grid;

  // Return connected point set
  return connected_pointset;
}



R3SurfelPointSet *
CreateConnectedPointSet(R3SurfelPointSet *pointset, R3SurfelPoint *seed_point,
  RNScalar min_volume, RNScalar max_volume, RNLength grid_spacing, int max_grid_resolution)
{
  // Return connected component
  return CreateConnectedPointSet(pointset, seed_point->Position(), 
    min_volume, max_volume, grid_spacing, max_grid_resolution);
}



R3SurfelPointSet *
CreateConnectedPointSet(R3SurfelPointSet *pointset, R3SurfelPointSet *seedset,
  RNLength grid_spacing, int max_grid_resolution)
{
  // Just checking
  if (pointset->NPoints() == 0) return NULL;
  if (seedset->NPoints() == 0) return NULL;

  // Create pointset grid
  R3Grid *pointset_grid = CreateGrid(pointset, grid_spacing, max_grid_resolution);
  if (!pointset_grid) return NULL;
  if (pointset_grid->NEntries() == 0) { delete pointset_grid; return NULL; }

  // Create seedset grid
  R3Grid seedset_grid(*pointset_grid); seedset_grid.Clear(0);
  int step = seedset->NPoints() / seedset_grid.NEntries() + 1;
  for (int j = 0; j < seedset->NPoints(); j += step) {
    const R3SurfelPoint *point = seedset->Point(j);
    R3Point position = point->Position();
    seedset_grid.RasterizeWorldPoint(position, 1);
  }

  // Mask to part of grid connected to seeds
  MaskToSeededConnectedComponent(*pointset_grid, seedset_grid, 0.5);

  // Create connnected point set
  R3SurfelGridConstraint constraint(pointset_grid);
  R3SurfelPointSet *connected_pointset = CreatePointSet(pointset, &constraint);

  // Delete grids
  delete pointset_grid;

  // Return connected point set
  return connected_pointset;
}



R3SurfelPointSet *
CreateConnectedPointSet(R3SurfelPointSet *pointset, 
  const R3Point& seed_origin, RNScalar seed_radius, 
  RNScalar seed_min_height, RNScalar seed_max_height,
  RNScalar min_volume, RNScalar max_volume,
  RNLength grid_spacing, int max_grid_resolution)
{
  // Just checking
  if (!pointset) return NULL;
  if (pointset->NPoints() == 0) return NULL;

  // Create pointset grid
  R3Grid *pointset_grid = CreateGrid(pointset, grid_spacing, max_grid_resolution);
  if (!pointset_grid) return NULL;
  if (pointset_grid->NEntries() == 0) { delete pointset_grid; return NULL; }

  // Create seedset grid
  R3Grid seed_grid(*pointset_grid); seed_grid.Clear(0);
  RNScalar scale = seed_grid.WorldToGridScaleFactor();
  R3Point seed_grid_origin = seed_grid.GridPosition(seed_origin);
  RNScalar seed_grid_radius = scale * seed_radius;
  RNScalar seed_grid_min_height = scale * seed_min_height;
  RNScalar seed_grid_max_height = scale * seed_max_height;
  seed_grid.RasterizeGridBox(
    R3Box(seed_grid_origin.X()-seed_grid_radius, 
      seed_grid_origin.Y()-seed_grid_radius, 
      seed_grid_origin.Z()+seed_grid_min_height, 
      seed_grid_origin.X()+seed_grid_radius,  
      seed_grid_origin.Y()+seed_grid_radius, 
      seed_grid_origin.Z()+seed_grid_max_height), 
      1.0);

  // Mask seed grid to pointset grid
  seed_grid.Mask(*pointset_grid);

  // Mask to part of grid connected to seeds
  MaskToSeededConnectedComponent(*pointset_grid, seed_grid, 0.5);

  // Check volume
  if ((min_volume > 0) || (max_volume > 0)) {
    RNScalar volume = pointset_grid->Volume();
    if ((volume < min_volume) || (volume > max_volume)) {
      delete pointset_grid;
      return NULL;
    }
  }

  // Create connnected point set
  R3SurfelGridConstraint constraint(pointset_grid);
  R3SurfelPointSet *connected_pointset = CreatePointSet(pointset, &constraint);

  // Delete grid
  delete pointset_grid;

  // Return connected point set
  return connected_pointset;
}



////////////////////////////////////////////////////////////////////////
// Grid creation functions
////////////////////////////////////////////////////////////////////////

R3Grid *
CreateGrid(R3SurfelPointSet *pointset,
  RNLength grid_spacing, int max_resolution)
{
  // Check pointset
  if (!pointset) return NULL;
  if (pointset->NPoints() == 0) return NULL;

  // Compute bbox
  R3Box bbox = pointset->BBox();
  if (bbox.Volume() == 0) return NULL;
  bbox.Inflate(1.1);

  // Compute grid resolution
  int xres = (int) (bbox.XLength() / grid_spacing + 0.5);
  int yres = (int) (bbox.YLength() / grid_spacing + 0.5);
  int zres = (int) (bbox.ZLength() / grid_spacing + 0.5);
  if (xres > max_resolution) xres = max_resolution;
  if (yres > max_resolution) yres = max_resolution;
  if (zres > max_resolution) zres = max_resolution;
  if (xres < 2) xres = 2;
  if (yres < 2) yres = 2;
  if (zres < 2) zres = 2;
  
  // Create grid
  R3Grid *grid = new R3Grid(xres, yres, zres, bbox);
  if (!grid) {
    fprintf(stderr, "Unable to allocate grid\n");
    return NULL;
  }

  // Rasterize points into grid
  if (pointset->NPoints() > 0) {
    int step = pointset->NPoints() / grid->NEntries() + 1;
    for (int j = 0; j < pointset->NPoints(); j += step) {
      const R3SurfelPoint *point = pointset->Point(j);
      R3Point position = point->Position();
      grid->RasterizeWorldPoint(position, 1);
    }
  }

  // Return grid
  return grid;
}


R3Grid *
CreateGrid(R3SurfelScene *scene, 
  R3SurfelNode *source_node, const R3SurfelConstraint *constraint,
  RNLength grid_spacing, int max_resolution)
{
  // Create point set
  R3SurfelPointSet *pointset = CreatePointSet(scene, source_node, constraint);
  if (!pointset) return NULL;

  // Create grid
  R3Grid *grid = CreateGrid(pointset, grid_spacing, max_resolution);
  if (!grid) { delete pointset; return NULL; }

  // Delete pointset
  delete pointset;

  // Return grid
  return grid;
}



R3Grid *
CreateGrid(R3SurfelScene *scene, const R3Box& bbox,
  RNLength grid_spacing, int max_resolution)
{
  // Compute grid resolution
  int xres = (int) (bbox.XLength() / grid_spacing + 0.5);
  int yres = (int) (bbox.YLength() / grid_spacing + 0.5);
  int zres = (int) (bbox.ZLength() / grid_spacing + 0.5);
  if (xres > max_resolution) xres = max_resolution;
  if (yres > max_resolution) yres = max_resolution;
  if (zres > max_resolution) zres = max_resolution;
  if (xres < 2) xres = 2;
  if (yres < 2) yres = 2;
  if (zres < 2) zres = 2;
  
  // Create grid
  R3Grid *grid = new R3Grid(xres, yres, zres, bbox);
  if (!grid) {
    fprintf(stderr, "Unable to allocate grid\n");
    return NULL;
  }

  // Rasterize points into grid
  R3SurfelTree *tree = scene->Tree();
  R3SurfelDatabase *database = tree->Database();
  for (int i = 0; i < tree->NNodes(); i++) {
    R3SurfelNode *node = tree->Node(i);
    if (node->NParts() > 0) continue;
    if (!R3Intersects(bbox, node->BBox())) continue;
    for (int j = 0; j < node->NBlocks(); j++) {
      R3SurfelBlock *block = node->Block(j);
      if (!R3Intersects(bbox, block->BBox())) continue;
      const R3Point& origin = block->Origin();
      database->ReadBlock(block);
      for (int k = 0; k < block->NSurfels(); k++) {
        const R3Surfel *surfel = block->Surfel(k);
        double px = origin.X() + surfel->X();
        double py = origin.Y() + surfel->Y();
        double pz = origin.Z() + surfel->Z();
        R3Point position(px, py, pz);
        if (!R3Intersects(bbox, position)) continue;
        grid->RasterizeWorldPoint(position, 1.0);
      }
      database->ReleaseBlock(block);
    }
  }

  // Return grid
  return grid;
}



R3Grid *
CreateGrid(R3SurfelScene *scene, 
  RNLength grid_spacing, int max_resolution)
{
  // Compute bbox
  R3Box bbox = scene->BBox();
  if (bbox.Volume() == 0) return NULL;
  bbox.Inflate(1.1);

  // Compute grid resolution
  int xres = (int) (bbox.XLength() / grid_spacing + 0.5);
  int yres = (int) (bbox.YLength() / grid_spacing + 0.5);
  int zres = (int) (bbox.ZLength() / grid_spacing + 0.5);
  if (xres > max_resolution) xres = max_resolution;
  if (yres > max_resolution) yres = max_resolution;
  if (zres > max_resolution) zres = max_resolution;
  if (xres < 2) xres = 2;
  if (yres < 2) yres = 2;
  if (zres < 2) zres = 2;
  
  // Create grid
  R3Grid *grid = new R3Grid(xres, yres, zres, bbox);
  if (!grid) {
    fprintf(stderr, "Unable to allocate grid\n");
    return NULL;
  }

  // Rasterize points into grid
  R3SurfelTree *tree = scene->Tree();
  R3SurfelDatabase *database = tree->Database();
  for (int i = 0; i < tree->NNodes(); i++) {
    R3SurfelNode *node = tree->Node(i);
    if (node->NParts() > 0) continue;
    for (int j = 0; j < node->NBlocks(); j++) {
      R3SurfelBlock *block = node->Block(j);
      const R3Point& origin = block->Origin();
      database->ReadBlock(block);
      for (int k = 0; k < block->NSurfels(); k++) {
        const R3Surfel *surfel = block->Surfel(k);
        double px = origin.X() + surfel->X();
        double py = origin.Y() + surfel->Y();
        double pz = origin.Z() + surfel->Z();
        R3Point position(px, py, pz);
        grid->RasterizeWorldPoint(position, 1.0);
      }
      database->ReleaseBlock(block);
    }
  }

  // Return grid
  return grid;
}



////////////////////////////////////////////////////////////////////////
// Normal extraction
////////////////////////////////////////////////////////////////////////

R3Vector *
CreateNormals(R3SurfelPointGraph *graph, RNBoolean fast_and_approximate)
{
#if 0
  // Sorry
  RNAbort("CreateNormals has been deprecated.  Use UpdateNormals instead.");
  return NULL;
#else
  // Check graph
  if (graph->NPoints() == 0) return NULL;
  if (graph->MaxNeighbors() < 2) return NULL;

  // Allocate normals
  R3Vector *normals = new R3Vector [ graph->NPoints() ];
  if (!normals) {
    fprintf(stderr, "Unable to allocate normals\n");
    return NULL;
  }

  // Allocate temporary memory
  R3Point *positions = new R3Point [graph->MaxNeighbors() + 1];

  // Compute normals
  for (int i = 0; i < graph->NPoints(); i++) {
    const R3SurfelPoint *point = graph->Point(i);
    if (graph->NNeighbors(i) < 2) { 
      // Punt
      normals[i] = R3zero_vector; 
    }
    else {
      if (fast_and_approximate) {
        // Compute normal with vector cross product
        positions[0] = point->Position();
        int index1 = (int) (RNRandomScalar() * graph->NNeighbors(i));
        int index2 = (index1 + graph->NNeighbors(i)/2) % graph->NNeighbors(i);
        const R3SurfelPoint *neighbor1 = graph->Neighbor(i, index1);
        const R3SurfelPoint *neighbor2 = graph->Neighbor(i, index2);
        positions[1] = neighbor1->Position();
        positions[2] = neighbor2->Position();
        R3Vector v1 = positions[1] - positions[0];
        R3Vector v2 = positions[2] - positions[0];
        R3Vector n = v1 % v2;
        n.Normalize();
        normals[i] = n;
      }
      else {
        // Compute normal with PCA of neighborhood
        positions[0] = point->Position();
        for (int j = 0; j < graph->NNeighbors(i); j++) {
          const R3SurfelPoint *neighbor = graph->Neighbor(i, j);
          positions[j+1] = neighbor->Position();
        }
        int npositions = graph->NNeighbors(i) + 1;
        R3Point centroid = R3Centroid(npositions, positions);
        R3Triad triad = R3PrincipleAxes(centroid, npositions, positions);
        normals[i] = triad[2];
      }

      // Flip normal so that positive in max dimension
      int dim = normals[i].MaxDimension();
      if (normals[i][dim] < 0) normals[i].Flip();
    }
  }

  // Delete positions
  delete [] positions;

  // Return normals
  return normals;
#endif
}



R3Vector *
CreateNormals(R3SurfelPointSet *pointset, 
  int max_neighbors, RNLength max_neighbor_distance)
{
#if 0
  // Sorry
  RNAbort("CreateNormals has been deprecated.  Use UpdateNormals instead.");
  return NULL;
#else
  // Create graph
  R3SurfelPointGraph *graph = new R3SurfelPointGraph(*pointset, max_neighbors, max_neighbor_distance);
  if (!graph) {
    fprintf(stderr, "Unable to create graph\n");
    return NULL;
  }

  // Compute normals
  R3Vector *normals = CreateNormals(graph);

  // Delete graph
  delete graph;

  // Return normals
  return normals;
#endif
}



////////////////////////////////////////////////////////////////////////
// Node set creation
////////////////////////////////////////////////////////////////////////

static int
InsertNodes(R3SurfelNodeSet *nodeset, 
  R3SurfelTree *tree, R3SurfelNode *node,
  const R3SurfelConstraint *constraint,
  const R3Point& xycenter, RNLength xyradius,
  RNScalar center_resolution, RNScalar perimeter_resolution,
  RNScalar focus_exponent)
{
  // Check if node satisfies constraint
  if (constraint && !constraint->Check(node)) {
    return 0;
  }

  // Check if node is outside perimeter
  if ((perimeter_resolution == 0) && (xyradius > 0)) {
    RNScalar xydistance = XYDistance(xycenter, node->BBox());
    if (xydistance > xyradius) return 0;
  }

  // Check if this node is a leaf
  if (node->NParts() == 0) {
    nodeset->InsertNode(node);
    return 1;
  }

  // Check if this node has target resolution 
  if ((node->NBlocks() > 0) && (center_resolution > 0)) {
    // Compute the target resolution at node
    RNScalar target_resolution = center_resolution;
    if ((xyradius > 0) && (perimeter_resolution < center_resolution)) {
      RNLength xydistance =  XYDistance(xycenter, node->BBox());
      RNScalar t = 1 - xydistance / xyradius;
      target_resolution = perimeter_resolution + 
        (center_resolution - perimeter_resolution) * pow(t, focus_exponent);
    }
      
    // Check if this node is within the target resolution
    for (int i = 0; i < node->NParts(); i++) {
      R3SurfelNode *part = node->Part(i);
      if (part->Resolution() > target_resolution) {
        nodeset->InsertNode(node);
        return 1;
      }
    }
  }

  // Consider parts
  int status = 0;
  for (int i = 0; i < node->NParts(); i++) {
    R3SurfelNode *part = node->Part(i);
    status |= InsertNodes(nodeset, tree, part, constraint, xycenter, xyradius, 
      center_resolution, perimeter_resolution, focus_exponent);
  }

  // Return whether anything was added
  return status;
}



R3SurfelNodeSet *
CreateNodeSet(R3SurfelScene *scene, 
  R3SurfelNode *source_node, const R3SurfelConstraint *constraint,
  const R3Point& xycenter, RNLength xyradius,
  RNScalar center_resolution, RNScalar perimeter_resolution,
  RNScalar focus_exponent)
{
  // Get tree variables
  R3SurfelTree *tree = scene->Tree();
  if (!tree) return NULL;

  // Check source node
  if (!source_node) source_node = tree->RootNode();

  // Create node set
  R3SurfelNodeSet *nodeset = new R3SurfelNodeSet();
  if (!nodeset) {
    fprintf(stderr, "Unable to allocate node set\n");
    return NULL;
  }

  // Insert nodes
  InsertNodes(nodeset, tree, source_node, constraint, xycenter, xyradius,
    center_resolution, perimeter_resolution, focus_exponent);

  // Return node set
  return nodeset;
}



////////////////////////////////////////////////////////////////////////
// Object set creation
////////////////////////////////////////////////////////////////////////

R3SurfelObjectSet *
CreateObjectSet(R3SurfelScene *scene, 
  const R3SurfelConstraint *constraint)
{
  // Create object set
  R3SurfelObjectSet *objectset = new R3SurfelObjectSet();
  if (!objectset) {
    fprintf(stderr, "Unable to allocate object set\n");
    return NULL;
  }

  // Construct object set
  for (int i = 0; i < scene->NObjects(); i++) {
    R3SurfelObject *object = scene->Object(i);
    if (constraint && !constraint->Check(object)) continue;
    objectset->InsertObject(object);
  }


  // Return object set
  return objectset;
}



////////////////////////////////////////////////////////////////////////
// Plane estimation
////////////////////////////////////////////////////////////////////////

R3Plane
FitPlane(R3SurfelPointSet *pointset)
{
  // Allocate points
  int max_points = 1024;
  int npoints = pointset->NPoints();
  if (npoints > max_points) npoints = max_points;
  if (npoints == 0) return R3null_plane;
  R3Point *points = new R3Point [npoints];
  int step = pointset->NPoints() / npoints;
  if (step == 0) step = 1;
  for (int i = 0; i < npoints; i++) {
    const R3SurfelPoint *point = pointset->Point(i*step);
    points[i] = point->Position();
  }

  // Solve for plane
  R3Point centroid = R3Centroid(npoints, points);
  R3Triad triad = R3PrincipleAxes(centroid, npoints, points);
  R3Vector normal = triad[2];
  if (normal[2] < 0) normal.Flip();
  R3Plane plane(centroid, normal);

  // Delete points
  delete [] points;

  // Return plane
  return plane;
}



R3Plane
EstimateSupportPlane(R3SurfelPointSet *pointset,
  RNLength accuracy, RNScalar *npoints)
{
  // Check point set
  if (pointset->NPoints() == 0) {
    if (npoints) *npoints = 0;
    return R3null_plane;
  }

  // Determine zres
  const R3Box& bbox = pointset->BBox();
  RNScalar zmin = bbox.ZMin();
  RNScalar zlength = bbox.ZLength();
  if (zlength == 0) return R3Plane(0, 0, 1, -(bbox.ZMin()));
  int zres = (int) (2 * zlength / accuracy) + 4;

  // Initialize votes
  RNScalar *votes = new RNScalar [zres];
  for (int i = 0; i < zres; i++) votes[i] = 0;
  
  // Cast votes
  int step = 10 * pointset->NPoints() / zres + 1;
  for (int j = 0; j < pointset->NPoints(); j += step) {
    const R3SurfelPoint *point = pointset->Point(j);
    R3Point position = point->Position();
    int iz = (int) (zres * (position.Z() - zmin) / zlength);
    if (iz >= zres) iz = zres - 1;
    for (int k = iz; k >= 0; k--) votes[k] += 0.01;
    votes[iz]++;
  }

  // Blur votes
  RNScalar *copy = new RNScalar [ zres ];
  for (int i = 0; i < zres; i++) copy[i] = votes[i];
  votes[0] = 0.75*copy[0] + 0.25*copy[1];
  votes[zres-1] = 0.75*copy[zres-1] + 0.25*copy[zres-2];
  for (int i = 1; i < zres-1; i++) {
    votes[i] = 0.5 * copy[i];
    votes[i] += 0.25 * copy[i-1];
    votes[i] += 0.25 * copy[i+1];
  }
  delete [] copy;

  // Find z with most votes
  RNScalar best_z = 0;
  RNScalar best_vote = 0;
  for (int i = 0; i < zres; i++) {
    if (votes[i] > best_vote) {
      best_vote = votes[i];
      best_z = zlength * i / zres + zmin;
    }
  }

  // Delete votes
  delete [] votes;

  // Fill in return value
  if (npoints) *npoints = step * best_vote;

  // Return ground plane
  return R3Plane(0, 0, 1, -best_z);
}



R3Plane 
EstimateSupportPlane(R3SurfelScene *scene, 
  R3SurfelNode *source_node, const R3SurfelConstraint *constraint,
  RNLength accuracy, RNScalar *npoints)
{
  // Get points 
  R3SurfelPointSet *pointset = CreatePointSet(scene, source_node, constraint);
  if (!pointset) { 
    if (npoints) *npoints = 0;
    return R3null_plane; 
  }

  // Estimate support plane 
  R3Plane plane = EstimateSupportPlane(pointset, accuracy, npoints);

  // Delete points
  delete pointset;

  // Return plane
  return plane;
}



R3Plane 
EstimateSupportPlane(R3SurfelScene *scene, 
  const R3Point& center, RNLength radius,
  RNLength accuracy, RNScalar *npoints)
{
  // Estimate support plane for points within cylinder
  R3SurfelCylinderConstraint constraint(center, radius);
  return EstimateSupportPlane(scene, NULL, &constraint, accuracy, npoints);
}


RNCoord
EstimateSupportZ(R3SurfelScene *scene, 
  const R3Point& center, RNLength radius,
  RNLength accuracy, RNScalar *npoints)
{
 // Get points 
  R3SurfelCylinderConstraint cylinder_constraint(center, radius);
  R3SurfelPointSet *pointset = CreatePointSet(scene, NULL, &cylinder_constraint);
  if (!pointset) { 
    if (npoints) *npoints = 0;
    return -1; 
  }

  // Estimate support Z 
  R3Plane plane = EstimateSupportPlane(pointset, accuracy, npoints);
  RNScalar z = -(center[0]*plane[0] + center[1]*plane[1] + plane[3]) / plane[2];

  // Delete points
  delete pointset;

  // Return support z
  return z;
}



R3Plane
FitSupportPlane(R3SurfelPointSet *pointset,
  RNLength accuracy, RNScalar *npoints)
{
  // Estimate support plane
  R3Plane plane = EstimateSupportPlane(pointset, accuracy, npoints);

  // Create a set of points near plane
  R3SurfelPlaneConstraint constraint(plane, FALSE, TRUE, FALSE, 5 * accuracy);
  R3SurfelPointSet *plane_pointset = CreatePointSet(pointset, &constraint);

  // Fit plane to pointset
  if (plane_pointset->NPoints() > 3) {
    plane = FitPlane(plane_pointset);
    if (npoints) *npoints = plane_pointset->NPoints();
  }

  // Delete point set
  delete plane_pointset;

  // Return plane
  return plane;
}



R3Plane 
FitSupportPlane(R3SurfelScene *scene, 
  R3SurfelNode *source_node, const R3SurfelConstraint *constraint,
  RNLength accuracy, RNScalar *npoints)
{
  // Fit support plane for points within cylinder
  R3SurfelPointSet *pointset = CreatePointSet(scene, source_node, constraint);
  R3Plane plane = FitSupportPlane(pointset, accuracy, npoints);
  delete pointset;
  return plane;
}



R3Plane 
FitSupportPlane(R3SurfelScene *scene, 
  const R3Point& center, RNLength radius,
  RNLength accuracy, RNScalar *npoints)
{
  // Estimate support plane for points within cylinder
  R3SurfelCylinderConstraint constraint(center, radius);
  return FitSupportPlane(scene, NULL, &constraint, accuracy, npoints);
}



////////////////////////////////////////////////////////////////////////
// Planar grid creation
////////////////////////////////////////////////////////////////////////

struct SurfelPlanarGridData {
  RNArray<R3SurfelPoint *> points;
  R3Plane plane;
  R3Box bbox;
  RNScalar weight;
};


static int 
CompareSurfelPlanarGridDatas(const void *data1, const void *data2)
{
  SurfelPlanarGridData *grid_data1 = (SurfelPlanarGridData *) data1;
  SurfelPlanarGridData *grid_data2 = (SurfelPlanarGridData *) data2;
  if (grid_data1->weight > grid_data2->weight) return -1;
  else if (grid_data1->weight < grid_data2->weight) return 1;
  else return 0;
}



RNArray<R3PlanarGrid *> *
CreatePlanarGrids(R3SurfelPointGraph *graph, 
  RNLength max_offplane_distance, RNAngle max_normal_angle,
  RNArea min_area, RNScalar min_density, int min_points,
  RNLength grid_spacing, RNScalar accuracy_factor)
{
  // Get convenient variables
  RNLength offplane_distance_sigma = 0.5 * max_offplane_distance;
  RNLength offplane_distance_sigma_squared = offplane_distance_sigma * offplane_distance_sigma;
  RNScalar offplane_distance_factor = (offplane_distance_sigma_squared != 0) ? 1.0 / (-2.0 * offplane_distance_sigma_squared) : 1;
  RNLength normal_angle_sigma = 0.5 * max_normal_angle;
  RNAngle normal_angle_sigma_squared = normal_angle_sigma * normal_angle_sigma;
  RNScalar normal_angle_factor = (normal_angle_sigma_squared != 0) ? 1.0 / (-2.0 * normal_angle_sigma_squared) : 1;

  // Determine number of samples
  if (min_points <= 1) min_points = 1;
  RNScalar max_grids = graph->NPoints();
  RNScalar bbox_volume = graph->BBox().Volume();
  if ((min_points > 0) && (graph->NPoints() / min_points < max_grids)) max_grids = graph->NPoints() / min_points;
  if ((min_area > 0) && (bbox_volume / min_area < max_grids)) max_grids = bbox_volume / min_area;
  int nsamples = (int) (accuracy_factor * 10.0 * max_grids) + 1;
  if (nsamples > graph->NPoints()) nsamples = graph->NPoints();

  // Create temporary data
  int *point_marks = new int [ graph->NPoints() ];
  R3Vector *point_normals = CreateNormals(graph, FALSE);
  R3Point *point_positions = new R3Point [ graph->NPoints() ];
  RNScalar *point_weights = new RNScalar [ graph->NPoints() ];
  R3PlanarGrid **point_grids = new R3PlanarGrid * [ graph->NPoints() ];
  SurfelPlanarGridData *grid_datas = new SurfelPlanarGridData [ nsamples ];
  for (int i = 0; i < graph->NPoints(); i++) point_marks[i] = 0;
  for (int i = 0; i < graph->NPoints(); i++) point_grids[i] = NULL;

  // Generate grid data
  int grid_data_count = 0;
  for (int iter = 0; iter < nsamples; iter++) {
    // Compute plane 
    int seed_index = (int) (RNRandomScalar() * graph->NPoints());
    if (point_marks[seed_index]) continue;
    if (graph->NNeighbors(seed_index) < 2) continue;
    R3SurfelPoint *seed = graph->Point(seed_index);
    R3Point seed_position = seed->Position();
    const R3Vector& seed_normal = point_normals[seed_index];
    if (R3Contains(seed_normal, R3zero_vector)) continue;
    R3Plane plane(seed_position, seed_normal);

    // Find positions of points near plane
    int point_count = 0;
    RNScalar total_weight = 0;
    for (int i = 0; i < graph->NPoints(); i++) {
      R3SurfelPoint *point = graph->Point(i);
      if (graph->NNeighbors(seed_index) < 2) continue;
      R3Point position = point->Position();
      RNScalar offplane_distance = R3Distance(plane, position);
      if (offplane_distance > max_offplane_distance) continue;
      const R3Vector& normal = point_normals[i];
      if (R3Contains(normal, R3zero_vector)) continue;
      RNScalar dot = fabs(normal.Dot(seed_normal));
      RNAngle normal_angle = (dot < 1) ? acos(dot) : 0;
      if (normal_angle > max_normal_angle) continue;
      RNScalar weight = 1;
      weight *= exp(offplane_distance_factor * offplane_distance * offplane_distance);
      weight *= exp(normal_angle_factor * normal_angle * normal_angle);
      total_weight += weight;
      point_positions[point_count] = position;
      point_weights[point_count] = weight;
      point_count++;
    }

    // Check number of points
    if (total_weight < min_points) continue;

    // Re-compute plane to fit points
    R3Point centroid = R3Centroid(point_count, point_positions, point_weights);
    R3Triad triad = R3PrincipleAxes(centroid, point_count, point_positions, point_weights);
    if (R3Contains(triad[2], R3zero_vector)) continue;
    plane.Reset(centroid, triad[2]);

    // Fill grid data
    SurfelPlanarGridData *grid_data = &grid_datas[grid_data_count];
    grid_data->points.Empty();
    grid_data->plane = plane;
    grid_data->bbox = R3null_box;
    grid_data->weight = 0;
    for (int i = 0; i < graph->NPoints(); i++) {
      R3SurfelPoint *point = graph->Point(i);
      int point_index = graph->PointIndex(point);
      if (graph->NNeighbors(point_index) < 2) continue;
      R3Point position = point->Position();
      RNScalar offplane_distance = R3Distance(plane, position);
      if (offplane_distance > max_offplane_distance) continue;
      const R3Vector& normal = point_normals[i];
      if (R3Contains(normal, R3zero_vector)) continue;
      RNScalar dot = fabs(normal.Dot(plane.Normal()));
      RNAngle normal_angle = (dot < 1) ? acos(dot) : 0;
      if (normal_angle > max_normal_angle) continue;
      RNScalar weight = 1;
      weight *= exp(offplane_distance_factor * offplane_distance * offplane_distance);
      weight *= exp(normal_angle_factor * normal_angle * normal_angle);
      grid_data->points.Insert(point);
      grid_data->bbox.Union(position);
      grid_data->weight += weight;
      point_marks[point_index]++;
    }

    // Check grid
    if (grid_data->weight < min_points) continue;

    // Add grid data
    grid_data_count++;
  }

  ////////////////////

  // Sort grid data
  if (grid_data_count > 0) {
    qsort(grid_datas, grid_data_count, sizeof(SurfelPlanarGridData), CompareSurfelPlanarGridDatas);
  }

  ////////////////////

  // Create array of grids
  RNArray<R3PlanarGrid *> *grids = new RNArray<R3PlanarGrid *>();
  if (!grids) {
    fprintf(stderr, "Unable to allocate array of grids\n");
    return NULL;
  }

  // Create grids
  for (int i = 0; i < grid_data_count; i++) {
    int point_count = 0;
    SurfelPlanarGridData *grid_data = &grid_datas[i];
    R3PlanarGrid *grid = new R3PlanarGrid(grid_data->plane, grid_data->bbox, grid_spacing);
    for (int j = 0; j < grid_data->points.NEntries(); j++) {
      R3SurfelPoint *point = grid_data->points.Kth(j);
      int point_index = graph->PointIndex(point);
      if (graph->NNeighbors(point_index) < 2) continue;
      if (point_grids[point_index]) continue;
      point_grids[point_index] = grid;
      R3Point position = point->Position();
      RNScalar offplane_distance = R3Distance(grid_data->plane, position);
      if (offplane_distance > max_offplane_distance) continue;
      const R3Vector& normal = point_normals[point_index];
      if (R3Contains(normal, R3zero_vector)) continue;
      RNScalar dot = fabs(normal.Dot(grid_data->plane.Normal()));
      RNAngle normal_angle = (dot < 1) ? acos(dot) : 0;
      if (normal_angle > max_normal_angle) continue;
      RNScalar weight = 1;
      weight *= exp(offplane_distance_factor * offplane_distance * offplane_distance);
      weight *= exp(normal_angle_factor * normal_angle * normal_angle);
      grid->RasterizeWorldPoint(position, weight);
      point_count++;
    }

    // Check grid support
    if ((min_points > 0) && (point_count < min_points)) {
      delete grid;
      continue;
    }

    // Mask low-density areas and small-connected components
    if ((min_density > 0) || (min_area > 0)) {
      RNLength gridcell_per_meter = grid->WorldToGridScaleFactor();
      RNScalar min_grid_density = min_density / (gridcell_per_meter * gridcell_per_meter);
      RNScalar min_grid_area = min_area * gridcell_per_meter * gridcell_per_meter;
      grid->ConnectedComponentFilter(min_grid_density, min_grid_area, 0, 0, 0, R2_GRID_KEEP_VALUE);
      if (grid->Sum() == 0) { delete grid; continue; }
    }

    // Insert grid
    grids->Insert(grid);
  }

  ////////////////////

  // Delete temporary data
  delete [] point_marks;
  delete [] point_normals;
  delete [] point_positions;
  delete [] point_weights;
  delete [] point_grids;
  delete [] grid_datas;

  // Return grids
  return grids;
}



RNArray<R3PlanarGrid *> *
CreatePlanarGrids(R3SurfelScene *scene, 
  R3SurfelNode *source_node, const R3SurfelConstraint *constraint,
  int max_neighbors, RNLength max_neighbor_distance, 
  RNLength max_offplane_distance, RNAngle max_normal_angle,
  RNArea min_area, RNScalar min_density, int min_points,
  RNLength grid_spacing, RNScalar accuracy_factor,
  RNLength chunk_size)
{
  // Get convenient variables
  R3SurfelTree *tree = scene->Tree();
  if (!tree) return NULL;
  if (!source_node) source_node = tree->RootNode();
  const R3Box& source_bbox = source_node->BBox();
  int nxchunks = 1;
  int nychunks = 1;
  RNLength xchunk = source_bbox.XLength();
  RNLength ychunk = source_bbox.YLength();
  if (chunk_size > 0) {
    nxchunks = (int) (source_bbox.XLength() / chunk_size) + 1;
    nychunks = (int) (source_bbox.YLength() / chunk_size) + 1;
    xchunk = source_bbox.XLength() / nxchunks;
    ychunk = source_bbox.YLength() / nychunks;
  }

  // Create array of planar grids
  RNArray<R3PlanarGrid *> *grids = new RNArray<R3PlanarGrid *>();
  if (!grids) {
    fprintf(stderr, "Unable to allocate array of grids\n");
    return NULL;
  }

  // Fill array of planar grids chunk-by-chunk
  for (int j = 0; j < nychunks; j++) {
    for (int i = 0; i < nxchunks; i++) {
      // Compute chunk bounding box
      R3Box chunk_bbox = source_bbox;
      chunk_bbox[0][0] = source_bbox.XMin() + i     * xchunk;
      chunk_bbox[1][0] = source_bbox.XMin() + (i+1) * xchunk;
      chunk_bbox[0][1] = source_bbox.YMin() + j     * ychunk;
      chunk_bbox[1][1] = source_bbox.YMin() + (j+1) * ychunk;

      // Create graph
      R3SurfelBoxConstraint box_constraint(chunk_bbox);
      R3SurfelMultiConstraint multi_constraint;
      multi_constraint.InsertConstraint(&box_constraint);
      if (constraint) multi_constraint.InsertConstraint(constraint);
      R3SurfelPointGraph *graph = CreatePointGraph(scene, 
        source_node, &multi_constraint, 
        max_neighbors, max_neighbor_distance);
      if (!graph) continue;

      // Create planar grids
      RNArray<R3PlanarGrid *> *chunk_grids = CreatePlanarGrids(graph, 
        max_offplane_distance, max_normal_angle, 
        min_area, min_density, min_points, 
        grid_spacing, accuracy_factor);
      if (!chunk_grids) { delete graph; continue; }

      // Insert grids for chunk
      for (int k = 0; k < chunk_grids->NEntries(); k++) {
        R3PlanarGrid *grid = chunk_grids->Kth(k);
        printf("  %6d/%6d %6d/%6d %6d/%6d : %9.3f %9.3f\n", j, nychunks, i, nxchunks, k, chunk_grids->NEntries(), grid->L1Norm(), grid->Area());
        grids->Insert(grid);
      }

      // Delete array of grids for chunk
      delete chunk_grids;

      // Delete graph
      delete graph;
    }
  }

  // Return planar grids
  return grids;
}



RNArray<R3SurfelObject *> *
CreatePlanarObjects(R3SurfelScene *scene, 
  R3SurfelNode *source_node, const R3SurfelConstraint *constraint,
  R3SurfelObject *parent_object, R3SurfelNode *parent_node, RNBoolean copy_surfels,
  int max_neighbors, RNLength max_neighbor_distance, 
  RNLength max_offplane_distance, RNAngle max_normal_angle,
  RNArea min_area, RNScalar min_density, int min_points,
  RNLength grid_spacing, RNScalar accuracy_factor, 
  RNLength chunk_size)
{
  // Get convenient variables
  R3SurfelTree *tree = scene->Tree();
  if (!tree) return NULL;
  if (!source_node) source_node = tree->RootNode();
  const R3Box& source_bbox = source_node->BBox();
  int nxchunks = 1;
  int nychunks = 1;
  RNLength xchunk = source_bbox.XLength();
  RNLength ychunk = source_bbox.YLength();
  if (chunk_size > 0) {
    nxchunks = (int) (source_bbox.XLength() / chunk_size) + 1;
    nychunks = (int) (source_bbox.YLength() / chunk_size) + 1;
    xchunk = source_bbox.XLength() / nxchunks;
    ychunk = source_bbox.YLength() / nychunks;
  }

  // Create array of objects
  RNArray<R3SurfelObject *> *objects = new RNArray<R3SurfelObject *>();
  if (!objects) {
    fprintf(stderr, "Unable to allocate array of objects\n");
    return NULL;
  }

  // Fill array of objects chunk-by-chunk
  for (int j = 0; j < nychunks; j++) {
    for (int i = 0; i < nxchunks; i++) {
      // Compute chunk bounding box
      R3Box chunk_bbox = source_bbox;
      chunk_bbox[0][0] = source_bbox.XMin() + i     * xchunk;
      chunk_bbox[1][0] = source_bbox.XMin() + (i+1) * xchunk;
      chunk_bbox[0][1] = source_bbox.YMin() + j     * ychunk;
      chunk_bbox[1][1] = source_bbox.YMin() + (j+1) * ychunk;

      // Create graph
      R3SurfelBoxConstraint box_constraint(chunk_bbox);
      R3SurfelMultiConstraint multi_constraint;
      multi_constraint.InsertConstraint(&box_constraint);
      if (constraint) multi_constraint.InsertConstraint(constraint);
      R3SurfelPointGraph *graph = CreatePointGraph(scene, 
        source_node, &multi_constraint, 
        max_neighbors, max_neighbor_distance);
      if (!graph) continue;

      // Create planar grids
      RNArray<R3PlanarGrid *> *chunk_grids = CreatePlanarGrids(graph,
        max_offplane_distance, max_normal_angle, 
        min_area, min_density, min_points, 
        grid_spacing, accuracy_factor);
      if (!chunk_grids) { delete graph; continue; }

      // Delete graph
      delete graph;

      // Create objects 
      for (int k = 0; k < chunk_grids->NEntries(); k++) {
        R3PlanarGrid *grid = chunk_grids->Kth(k);
        char object_name[256];
        sprintf(object_name, "PlanarGrid%d\n", objects->NEntries());
        R3SurfelPlanarGridConstraint planar_grid_constraint(grid, max_offplane_distance);
        multi_constraint.InsertConstraint(&planar_grid_constraint);
        R3SurfelObject *object = CreateObject(scene, source_node, &multi_constraint, 
          parent_object, object_name, parent_node, object_name, copy_surfels);
        multi_constraint.RemoveConstraint(&planar_grid_constraint);
        printf("  %6d/%6d %6d/%6d %6d/%6d : %9.3g %9.3g\n", j, nychunks, i, nxchunks, k, chunk_grids->NEntries(), grid->L1Norm(), grid->Area());
        if (object) objects->Insert(object);
        delete grid;
      }

      // Delete array of grids
      delete chunk_grids;
    }
  }

  // Return objects
  return objects;
}



////////////////////////////////////////////////////////////////////////
// Hierarchical clustering
////////////////////////////////////////////////////////////////////////

struct R3SurfelCluster {
  R3SurfelCluster *parent;
  RNArray<R3SurfelPoint *> points; 
  R3Box bbox; 
  int id;
};

struct R3SurfelClusterPair {
  R3SurfelCluster *clusters[2];
  RNScalar similarity;
};



RNArray<R3SurfelObject *> *
CreateClusterObjects(R3SurfelScene *scene, R3SurfelPointGraph *graph, 
  R3SurfelObject *parent_object, R3SurfelNode *parent_node, 
  RNLength max_offplane_distance, RNAngle max_normal_angle,
  int min_points_per_object)
{
  // Get convenient variables
  R3SurfelTree *tree = scene->Tree();
  if (!tree) return 0;
  R3SurfelDatabase *database = tree->Database();
  if (!database) return 0;

  // Check number of points
  if (graph->NPoints() < min_points_per_object) return NULL;

  // Check parent object
  if (!parent_object) parent_object = scene->RootObject();

  // Check parent node
  if (!parent_node) parent_node = tree->RootNode();

  // Create normals
  R3Vector *normals = NULL;
  if ((max_offplane_distance > 0) || (max_normal_angle > 0)) {
    normals = CreateNormals(graph, FALSE);
  }

  // Create cluster for every point
  RNArray<R3SurfelCluster *> clusters;
  for (int i = 0; i < graph->NPoints(); i++) {
    R3SurfelPoint *point = graph->Point(i);
    R3Point position = point->Position();
    R3SurfelCluster *cluster = new R3SurfelCluster();
    cluster->parent = NULL;
    cluster->points.Insert(point);
    cluster->bbox.Reset(position, position);
    cluster->id = i;
    clusters.Insert(cluster);
  }

  // Check clusters
  if (clusters.IsEmpty()) {
    if (normals) delete [] normals;
    return NULL;  
  }

  // Create cluster pairs and add to heap
  R3SurfelClusterPair tmp;
  RNHeap<R3SurfelClusterPair *> heap(&tmp, &tmp.similarity, NULL, FALSE);;
  for (int index0 = 0; index0 < graph->NPoints(); index0++) {
    R3SurfelPoint *point0 = graph->Point(index0);
    R3SurfelCluster *cluster0 = clusters.Kth(index0);
    R3Point position0 = point0->Position();
    const R3Vector& normal0 = (normals) ? normals[index0] : R3zero_vector;
    R3Plane plane0(position0, normal0);
    for (int j = 0; j < graph->NNeighbors(index0); j++) {
      R3SurfelPoint *point1 = graph->Neighbor(index0, j);
      int index1 = graph->PointIndex(point1);
      if (index1 < index0) continue;
      R3Point position1 = point1->Position();

      // Compute plane similarity
      RNScalar plane_similarity = 1;
      if (max_offplane_distance > 0) {
        RNScalar offplane_distance = R3Distance(plane0, position1);
        if (offplane_distance > max_offplane_distance) continue;
        plane_similarity *= 1.0 - 0.9 * offplane_distance / max_offplane_distance;
      }

      // Compute normal similarity
      RNScalar normal_similarity = 1;
      if (max_normal_angle > 0) {
        if (normals) {
          const R3Vector& normal1 = normals[index1];
          RNScalar dot = fabs(normal0.Dot(normal1));
          RNAngle normal_angle = (dot < 1) ? acos(dot) : 0;
          if (normal_angle > max_normal_angle) continue;
          normal_similarity *= 1.0 - 0.9 * normal_angle / max_normal_angle;
        }
      }

      // Compute distance similarity
      RNScalar distance_similarity = 1.0;
      if (graph->MaxDistance() > 0) {
        RNScalar distance = R3Distance(position0, position1);
        distance_similarity = 1.0 - 0.99 * distance / graph->MaxDistance();
      }

      // Create pair
      R3SurfelCluster *cluster1 = clusters.Kth(index1);
      R3SurfelClusterPair *pair = new R3SurfelClusterPair();
      pair->clusters[0] = cluster0;
      pair->clusters[1] = cluster1;
      pair->similarity = distance_similarity * normal_similarity * plane_similarity;
      heap.Push(pair);
    }
  }

  // Check cluster pairs
  if (heap.IsEmpty()) {
    for (int i = 0; i < clusters.NEntries(); i++) delete clusters[i];
    if (normals) delete [] normals;
    return NULL;
  }

  // Merge clusters hierarchically
  int merge_count = 0;
  while (!heap.IsEmpty()) {
    // Get pair
    R3SurfelClusterPair *pair = heap.Pop();

    // Get clusters
    R3SurfelCluster *cluster0 = pair->clusters[0];
    R3SurfelCluster *cluster1 = pair->clusters[1];
    while (cluster0->parent) cluster0 = cluster0->parent;
    while (cluster1->parent) cluster1 = cluster1->parent;
    if (cluster0 == cluster1) continue;

    // Merge clusters
    cluster0->points.Append(cluster1->points);
    cluster0->bbox.Union(cluster1->bbox);
    cluster1->parent = cluster0;
    cluster1->points.Empty(TRUE);
    cluster1->bbox = R3null_box;

    // Delete pair
    delete pair;

    // Update count
    merge_count++;
  }

  // Create array of objects
  RNArray<R3SurfelObject *> *objects = new RNArray<R3SurfelObject *>();
  if (!objects) {
    fprintf(stderr, "Unable to create array of objects\n");
    for (int i = 0; i < clusters.NEntries(); i++) delete clusters[i];
    if (normals) delete [] normals;
    return NULL;
  }

  // Create objects with copies of surfels
  int object_count = 0;
  for (int i = 0; i < clusters.NEntries(); i++) {
    R3SurfelCluster *cluster = clusters.Kth(i);

    // Check cluster
    if (cluster->points.NEntries() < min_points_per_object) { delete cluster; continue; }
    if (cluster->parent) { delete cluster; continue; }

    // Create surfels
    R3Point origin = cluster->bbox.Centroid();
    R3Surfel *surfels = new R3Surfel [ cluster->points.NEntries() ];
    for (int j = 0; j < cluster->points.NEntries(); j++) {
      R3SurfelPoint *point = cluster->points.Kth(j);
      R3SurfelBlock *block = point->Block();
      const R3Surfel *surfel = point->Surfel();
      RNCoord x = surfel->X() + block->Origin().X() - origin.X();
      RNCoord y = surfel->Y() + block->Origin().Y() - origin.Y();
      RNCoord z = surfel->Z() + block->Origin().Z() - origin.Z();
      surfels[j].SetCoords(x, y, z);
      surfels[j].SetColor(surfel->Color());
      surfels[j].SetAerial(surfel->IsAerial());
    }

    // Create block
    R3SurfelBlock *block = new R3SurfelBlock(surfels, cluster->points.NEntries(), origin);
    if (!block) {
      fprintf(stderr, "Unable to create node\n");
      for (int i = 0; i < clusters.NEntries(); i++) delete clusters[i];
      if (normals) delete [] normals;
      delete [] surfels;
      return NULL;
    }

    // Delete surfels
    delete [] surfels;

    // Create node
    char node_name[256];
    sprintf(node_name, "O%d\n", objects->NEntries());
    R3SurfelNode *node = new R3SurfelNode(node_name);
    if (!node) {
      fprintf(stderr, "Unable to create node\n");
      for (int i = 0; i < clusters.NEntries(); i++) delete clusters[i];
      if (normals) delete [] normals;
      delete block;
      return NULL;
    }

    // Create object
    char object_name[256];
    sprintf(object_name, "O%d\n", objects->NEntries());
    R3SurfelObject *object = new R3SurfelObject(object_name);
    if (!object) {
      fprintf(stderr, "Unable to create object\n");
      for (int i = 0; i < clusters.NEntries(); i++) delete clusters[i];
      if (normals) delete [] normals;
      delete block;
      delete node;
      return NULL;
    }
     
    // Insert everything
    block->UpdateProperties();
    database->InsertBlock(block);
    node->InsertBlock(block);
    tree->InsertNode(node, parent_node);
    object->InsertNode(node);
    scene->InsertObject(object, parent_object);
    objects->Insert(object);

    // Delete cluster
    delete cluster;

    // Print debug statement
    printf("    %d %d\n", i, block->NSurfels());

    // Update count
    object_count++;
  }
  
  // Delete normals
  delete [] normals;

  // Return objects
  return objects;
}



RNArray<R3SurfelObject *> *
CreateClusterObjects(R3SurfelScene *scene, 
  R3SurfelNode *source_node, const R3SurfelConstraint *constraint,
  R3SurfelObject *parent_object, R3SurfelNode *parent_node, 
  int max_neighbors, RNLength max_neighbor_distance, 
  RNLength max_offplane_distance, RNAngle max_normal_angle,
  int min_points_per_object, 
  RNLength chunk_size)
{
  // Get convenient variables
  R3SurfelTree *tree = scene->Tree();
  if (!tree) return NULL;
  if (!source_node) source_node = tree->RootNode();
  const R3Box& source_bbox = source_node->BBox();
  int nxchunks = 1;
  int nychunks = 1;
  RNLength xchunk = source_bbox.XLength();
  RNLength ychunk = source_bbox.YLength();
  if (chunk_size > 0) {
    nxchunks = (int) (source_bbox.XLength() / chunk_size) + 1;
    nychunks = (int) (source_bbox.YLength() / chunk_size) + 1;
    xchunk = source_bbox.XLength() / nxchunks;
    ychunk = source_bbox.YLength() / nychunks;
  }

  // Create array of objects
  RNArray<R3SurfelObject *> *objects = new RNArray<R3SurfelObject *>();
  if (!objects) {
    fprintf(stderr, "Unable to allocate array of objects\n");
    return NULL;
  }

  // Fill array of objects chunk-by-chunk
  for (int j = 0; j < nychunks; j++) {
    for (int i = 0; i < nxchunks; i++) {
      // Compute chunk bounding box
      R3Box chunk_bbox = source_bbox;
      chunk_bbox[0][0] = source_bbox.XMin() + i     * xchunk;
      chunk_bbox[1][0] = source_bbox.XMin() + (i+1) * xchunk;
      chunk_bbox[0][1] = source_bbox.YMin() + j     * ychunk;
      chunk_bbox[1][1] = source_bbox.YMin() + (j+1) * ychunk;

      // Print debug message
      printf("  %6d/%6d %6d/%6d\n",j, nychunks, i, nxchunks);

      // Create graph
      R3SurfelBoxConstraint box_constraint(chunk_bbox);
      R3SurfelMultiConstraint multi_constraint;
      multi_constraint.InsertConstraint(&box_constraint);
      if (constraint) multi_constraint.InsertConstraint(constraint);
      R3SurfelPointGraph *graph = CreatePointGraph(scene, 
        source_node, &multi_constraint, 
        max_neighbors, max_neighbor_distance);
      if (!graph) continue;

      printf("    %9d : %9.3f %9.3f %9.3f  %9.3f %9.3f %9.3f\n",  graph->NPoints(),
            graph->BBox().XMin(), graph->BBox().YMin(), graph->BBox().ZMin(), 
            graph->BBox().XMax(), graph->BBox().YMax(), graph->BBox().ZMax());

      // Create objects
      RNArray<R3SurfelObject *> *chunk_objects = CreateClusterObjects(scene, graph, parent_object, parent_node, min_points_per_object);
      if (!chunk_objects) { delete graph; continue; }

      // Insert into result
      for (int k = 0; k < chunk_objects->NEntries(); k++) {
        R3SurfelObject *object = chunk_objects->Kth(k);
        char object_name[256];
        sprintf(object_name, "ClusterObject%d\n", objects->NEntries());
        object->SetName(object_name);
        objects->Insert(object);
      }

      // Delete array of chunk objects
      delete chunk_objects;

      // Delete graph
      delete graph;
    }
  }

  // Return objects
  return objects;
}



////////////////////////////////////////////////////////////////////////
// Basic geometric funcitons
////////////////////////////////////////////////////////////////////////

RNLength 
XYDistance(const R3Point& point1, const R3Point& point2)
{
  // Find distance in XY plane
  RNLength dx = point1.X() - point2.X();
  RNLength dy = point1.Y() - point2.Y();
  return sqrt(dx*dx + dy*dy);
}



RNLength 
XYDistanceSquared(const R3Point& point1, const R3Point& point2)
{
  // Find squared distance in XY plane
  RNLength dx = point1.X() - point2.X();
  RNLength dy = point1.Y() - point2.Y();
  return dx*dx + dy*dy;
}



RNLength 
XYDistance(const R3Point& point, const R3Box& box)
{
  // Find axial distances in XY plane
  RNLength dx, dy;
  if (RNIsGreater(point.X(), box.XMax())) dx = point.X() - box.XMax();
  else if (RNIsLess(point.X(), box.XMin())) dx = box.XMin()- point.X();
  else dx = 0.0;
  if (RNIsGreater(point.Y(), box.YMax())) dy = point.Y() - box.YMax();
  else if (RNIsLess(point.Y(), box.YMin())) dy = box.YMin()- point.Y();
  else dy = 0.0;
    
  // Return distance in XY between point and closest point in box 
  if (dy == 0.0) return dx;
  else if (dx == 0.0) return dy;
  else return sqrt(dx*dx + dy*dy);
}



RNLength 
XYDistanceSquared(const R3Point& point, const R3Box& box)
{
  // Find axial distances in XY plane
  RNLength dx, dy;
  if (RNIsGreater(point.X(), box.XMax())) dx = point.X() - box.XMax();
  else if (RNIsLess(point.X(), box.XMin())) dx = box.XMin()- point.X();
  else dx = 0.0;
  if (RNIsGreater(point.Y(), box.YMax())) dy = point.Y() - box.YMax();
  else if (RNIsLess(point.Y(), box.YMin())) dy = box.YMin()- point.Y();
  else dy = 0.0;
    
  // Return square of distance in XY between point and closest point in box 
  if (dy == 0.0) return dx*dx;
  else if (dx == 0.0) return dy*dy;
  else return dx*dx + dy*dy;
}



