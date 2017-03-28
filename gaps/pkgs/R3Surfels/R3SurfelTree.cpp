/* Source file for the R3 surfel tree class */



////////////////////////////////////////////////////////////////////////
// INCLUDE FILES
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// CONSTRUCTORS/DESTRUCTORS
////////////////////////////////////////////////////////////////////////

R3SurfelTree::
R3SurfelTree(void)
  : scene(NULL),
    database(NULL),
    nodes()
{
  // Create database
  database = new R3SurfelDatabase();
  database->tree = this;

  // Create root node
  R3SurfelNode *node = new R3SurfelNode("Root");
  InsertNode(node, NULL);
}



R3SurfelTree::
R3SurfelTree(const R3SurfelTree& tree)
{
  // Copy everything
  RNAbort("Not implemented");
}



R3SurfelTree::
~R3SurfelTree(void)
{
  // Delete scene
  if (scene) delete scene;

  // Delete everything
  // ???
}



////////////////////////////////////////////////////////////////////////
// ACCESS FUNCTIONS
////////////////////////////////////////////////////////////////////////

R3SurfelNode *R3SurfelTree::
FindNodeByName(const char *node_name) const
{
  // Search for node
  for (int i = 0; i < NNodes(); i++) {
    R3SurfelNode *node = Node(i);
    if (!node->Name()) continue;
    if (strcmp(node->Name(), node_name)) continue;
    return node;
  }

  // Not found
  return NULL;
}



////////////////////////////////////////////////////////////////////////
//  STRUCTURE MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelTree::
InsertNode(R3SurfelNode *node, R3SurfelNode *parent)
{
  // Just checking
  assert(node);
  assert(node->tree == NULL);
  assert(node->tree_index == -1);

  // Insert node into tree
  node->tree = this;
  node->tree_index = nodes.NEntries();
  nodes.Insert(node);

  // Update parent
  node->parent = parent;
  if (parent) parent->parts.Insert(node);

  // Update nodes
  node->UpdateAfterInsert(this);

  // Mark scene as dirty
  if (scene) scene->SetDirty();
}



void R3SurfelTree::
RemoveNode(R3SurfelNode *node)
{
  // Just checking
  assert(node);
  assert(node->tree == this);
  assert(node->tree_index >= 0);
  assert(nodes.Kth(node->tree_index) == node);

  // Update node
  node->UpdateBeforeRemove(this);

  // Update parent
  if (node->parent) node->parent->parts.Remove(node);

  // Update node
  node->parent = NULL;

  // Remove node from tree
  RNArrayEntry *entry = nodes.KthEntry(node->tree_index);
  R3SurfelNode *tail = nodes.Tail();
  tail->tree_index = node->tree_index;
  nodes.EntryContents(entry) = tail;
  nodes.RemoveTail();
  node->tree_index = -1;
  node->tree = NULL;

  // Mark scene as dirty
  if (scene) scene->SetDirty();
}



////////////////////////////////////////////////////////////////////////
//  HIGH-LEVEL MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

int R3SurfelTree::
CreateMultiresolutionBlocks(R3SurfelNode *node, RNScalar multiresolution_factor, RNScalar max_complexity, RNScalar max_resolution)
{
  // Check node
  if (node->NBlocks() > 0) return 1;
  if (node->NParts() == 0) return 1;

  // Create multiresolution blocks for parts
  for (int i = 0; i < node->NParts(); i++) {
    R3SurfelNode *part = node->Part(i);
    if (!CreateMultiresolutionBlocks(part, multiresolution_factor, max_complexity, max_resolution)) return 0;
  }

  // Compute some statistics
  RNScalar total_complexity = 0;
  RNScalar mean_resolution = 0;
  for (int i = 0; i < node->NParts(); i++) {
    R3SurfelNode *part = node->Part(i);
    mean_resolution += part->Resolution() / node->NParts();
    total_complexity += part->Complexity();
  }

  // Check statistics
  if (total_complexity == 0) return 1;
  if (mean_resolution == 0) return 1;
  
  // Compute target resolution
  RNScalar target_resolution = multiresolution_factor * mean_resolution;
  if ((max_resolution > 0) && (target_resolution > max_resolution)) {
    target_resolution = max_resolution;
  }
  if ((max_complexity > 0) && (total_complexity > 0)) {
    RNScalar max_res = max_complexity * mean_resolution / total_complexity;
    if (target_resolution > max_res) target_resolution = max_res;
  }

  // Construct set with surfels sampled from blocks of parts
  R3SurfelPointSet set;
  for (int i = 0; i < node->NParts(); i++) {
    R3SurfelNode *part = node->Part(i);
    for (int j = 0; j < part->NBlocks(); j++) {
      R3SurfelBlock *block = part->Block(j);
          
      // Read block
      database->ReadBlock(block);
          
      // Compute subsampling probability based on block resolution
      RNScalar block_resolution = block->Resolution();
      if (block_resolution == 0) continue;
      RNScalar probability = target_resolution / block_resolution;
          
      // Insert surfels from block
      if (probability >= 1) {
        // Insert all surfels from block
        set.InsertPoints(block);
      }
      else {
        // Insert subset of surfels from block
        for (int k = 0; k < block->NSurfels(); k++) {
          if (RNRandomScalar() > probability) continue;
          const R3Surfel *surfel = block->Surfel(k);
          R3SurfelPoint point(block, surfel);
          set.InsertPoint(point);
        }
      }

      // Release block
      database->ReleaseBlock(block);
    }
  }
      
  // Create block from set
  R3SurfelBlock *block = new R3SurfelBlock(&set);
  if (!block) return 0;

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

  // Return success
  return 1;
}



int R3SurfelTree::
CreateMultiresolutionBlocks(RNScalar multiresolution_factor, RNScalar max_complexity)
{
#if 0
  // Create multiresolution blocks for all nodes starting at Root
  for (int i = 0; i < NNodes(); i++) {
    R3SurfelNode *node = Node(i);
    CreateMultiresolutionBlocks(node, multiresolution_factor, max_complexity);
  }
#else
  CreateMultiresolutionBlocks(RootNode(), multiresolution_factor, max_complexity);
#endif
  
  // Return success
  return 1;
}



int R3SurfelTree::
CreateMultiresolutionNodes(RNScalar min_complexity, RNScalar min_resolution, RNScalar min_multiresolution_factor)
{
  // Just checking
  if (min_resolution <= 0) return 0;
  if (min_multiresolution_factor >= 1) return 0;

  // Make copy of list of nodes (since will change as edit)
  RNArray<R3SurfelNode *> copy;
  for (int i = 0; i < NNodes(); i++) {
    R3SurfelNode *node = Node(i);
    copy.Insert(node);
  }

  // Insert extra levels of tree based on resolution differences
  for (int i = 0; i < copy.NEntries(); i++) {
    R3SurfelNode *node = copy.Kth(i);
    if (node == RootNode()) continue;
    RNScalar node_resolution = node->Resolution();
    if (node_resolution <= min_resolution) continue;
    RNScalar node_complexity = node->Complexity();
    if (node_complexity < min_complexity) continue;
    R3SurfelNode *parent = node->Parent();
    if (!parent) continue;
    RNScalar parent_resolution = parent->Resolution();
    if (parent_resolution <= 0) parent_resolution = min_multiresolution_factor * min_resolution;
    RNScalar multiresolution_factor = parent_resolution / node_resolution;
    while (multiresolution_factor < min_multiresolution_factor) {
      // Create a new node and insert into tree between parent and node
      R3SurfelNode *new_node = new R3SurfelNode();
      InsertNode(new_node, parent);
      node->SetParent(new_node);

      // Create multiresoluiton blocks for new node
      if (!CreateMultiresolutionBlocks(new_node, 1, 0, min_multiresolution_factor * node_resolution)) {
        // Unroll changes
        node->SetParent(parent);
        RemoveNode(new_node);
        delete new_node;
        break;
      }

      // Update everything
      node = new_node;
      node_complexity = node->Complexity();
      if (node_complexity < min_complexity) break;
      node_resolution = node->Resolution();
      if (node_resolution < min_resolution) break;
      multiresolution_factor = parent_resolution / node_resolution;  
      if (multiresolution_factor < min_multiresolution_factor) break;
    }
  }

  // Return success
  return 1;
}



static int
SplitLeafNodeWithKMeansClusteringOfBlocks(R3SurfelTree *tree, R3SurfelNode *node, int nparts)
{
  // Just checking
  assert(node->NParts() == 0);
  assert(node->Tree() == tree);
  int status = 1;

  // Adjust number of parts
  if (nparts > node->NBlocks()) nparts = node->NBlocks();
  if (nparts == 1) return 0;

  ////// INITIALIZE KMEANS ////// 

  // Allocate temporary data for kmeans clustering
  R3Point *centroids = new R3Point [ nparts ];
  int *membership = new int [ node->NBlocks() ];

  // Initialize kmeans centroids
  for (int i = 0; i < nparts; i++) {
    int block_index = i * node->NBlocks() / nparts;
    R3SurfelBlock *block = node->Block(block_index);
    centroids[i] = block->Centroid();
  }

  // Intialize kmeans membership
  for (int i = 0; i < node->NBlocks(); i++) {
    membership[i] = -1;
  }


  ////// EXECUTE KMEANS ////// 

  // Interate K-means
  RNBoolean done = FALSE;
  const int max_iterations = 16;
  for (int i = 0; i < max_iterations; i++) {
    // Done when there are no changes in membership
    done = TRUE;

    // Update kmeans membership
    for (int j = 0; j < node->NBlocks(); j++) {
      R3SurfelBlock *block = node->Block(j);
      R3Point block_centroid = block->Centroid();
      RNLength closest_distance = FLT_MAX;
      for (int k = 0; k < nparts; k++) {
        RNLength distance = R3SquaredDistance(centroids[k], block_centroid);
        if (distance < closest_distance) {
          closest_distance = distance;
          if (membership[j] != k) done = FALSE;
          membership[j] = k;
        }
      }
    }

    // Check if there were no membership changes 
    if (done) break;
    
    // Update kmeans centroids
    for (int j = 0; j < nparts; j++) {
      // Find centroid
      int weight = 0;
      centroids[j] = R3zero_point;
      for (int k = 0; k < node->NBlocks(); k++) {
        if (membership[k] != j) continue;
        R3SurfelBlock *block = node->Block(k);
        centroids[j] += block->NSurfels() * block->Centroid();
        weight += block->NSurfels();
      }
      if (weight > 0) centroids[j] /= weight;
    }
  }

  ////// CREATE PARTS AND REDISTRIBUTE BLOCKS ////// 

  // Create temporary array of blocks (so that can remove blocks from node while iterating)
  RNArray<R3SurfelBlock *> blocks; 
  for (int i = 0; i < node->NBlocks(); i++) {
    R3SurfelBlock *block = node->Block(i);
    blocks.Insert(block);
  }

  // Create parts
  for (int i = 0; i < nparts; i++) {
    // Create part
    R3SurfelNode *part = new R3SurfelNode();

    // Insert part into tree
    tree->InsertNode(part, node);

    // Move blocks into part
    for (int j = 0; j < blocks.NEntries(); j++) {
      if (membership[j] != i) continue;
      R3SurfelBlock *block = blocks.Kth(j);
      node->RemoveBlock(block);
      part->InsertBlock(block);
    }

    // Check number of blocks in part
    if (part->NBlocks() == 0) {
      tree->RemoveNode(part);
      delete part;
      continue;
    }
  }

  // Just checking
  assert(node->NBlocks() == 0);

  // Check if result is only one part
  if (node->NParts() == 1) {
    R3SurfelNode *part = node->Part(0);
    for (int i = 0; i < blocks.NEntries(); i++) {
      R3SurfelBlock *block = blocks.Kth(i);
      part->RemoveBlock(block);
      node->InsertBlock(block);
    }
    tree->RemoveNode(part);
    delete part;
    status = 0;
  }


  ////// CLEAN UP ////// 

  delete [] membership;
  delete [] centroids;

  // Return whether anything was split
  return status;
}



static int
SplitInteriorNodeWithKMeansClusteringOfParts(R3SurfelTree *tree, R3SurfelNode *node, int nparts)
{
  // Just checking
  assert(node->NParts() > 0);
  assert(node->Tree() == tree);
  if (nparts >= node->NParts()) return 0;
  int status = 1;

  ////// INITIALIZE KMEANS ////// 

  // Allocate temporary data for kmeans clustering
  R3Point *centroids = new R3Point [ nparts ];
  int *membership = new int [ node->NParts() ];

  // Initialize kmeans centroids
  for (int i = 0; i < nparts; i++) {
    int part_index = i * node->NParts() / nparts;
    R3SurfelNode *part = node->Part(part_index);
    centroids[i] = part->Centroid();
  }

  // Intialize kmeans membership
  for (int i = 0; i < node->NParts(); i++) {
    membership[i] = -1;
  }

  ////// EXECUTE KMEANS ////// 

  // Interate K-means
  RNBoolean done = FALSE;
  const int max_iterations = 16;
  for (int i = 0; i < max_iterations; i++) {
    // Done when there are no changes in membership
    done = TRUE;

    // Update kmeans membership
    for (int j = 0; j < node->NParts(); j++) {
      R3SurfelNode *part = node->Part(j);
      R3Point part_centroid = part->Centroid();
      RNLength closest_distance = FLT_MAX;
      for (int k = 0; k < nparts; k++) {
        RNLength distance = R3SquaredDistance(centroids[k], part_centroid);
        if (distance < closest_distance) {
          closest_distance = distance;
          if (membership[j] != k) done = FALSE;
          membership[j] = k;
        }
      }
    }

    // Check if there were no membership changes 
    if (done) break;
    
    // Update kmeans centroids
    for (int j = 0; j < nparts; j++) {
      // Find centroid
      RNScalar weight = 0;
      centroids[j] = R3zero_point;
      for (int k = 0; k < node->NParts(); k++) {
        if (membership[k] != j) continue;
        R3SurfelNode *part = node->Part(k);
        centroids[j] += part->Complexity() * part->Centroid();
        weight += part->Complexity();
      }
      if (weight > 0) centroids[j] /= weight;
    }
  }


  ////// CREATE CHILDREN OBJECTS AND REDISTRIBUTE PARTS ////// 

  // Create temporary array of parts (so that can remove parts from node while iterating)
  RNArray<R3SurfelNode *> parts; 
  for (int i = 0; i < node->NParts(); i++) {
    R3SurfelNode *part = node->Part(i);
    parts.Insert(part);
  }

  // Create parts
  for (int i = 0; i < nparts; i++) {
    // Create child
    R3SurfelNode *child = new R3SurfelNode();

    // Insert child into tree
    tree->InsertNode(child, node);

    // Move parts into child
    for (int j = 0; j < parts.NEntries(); j++) {
      if (membership[j] != i) continue;
      R3SurfelNode *part = parts.Kth(j);
      tree->RemoveNode(part);
      tree->InsertNode(part, child);
    }

    // Check number of parts in child
    if (child->NParts() == 0) {
      tree->RemoveNode(child);
      delete child;
      continue;
    }
  }

  // Check if result is only one part
  if (node->NParts() == 1) {
    R3SurfelNode *part = node->Part(0);
    for (int i = 0; i < parts.NEntries(); i++) {
      R3SurfelNode *p = parts.Kth(i);
      tree->RemoveNode(p);
      tree->InsertNode(p, node);
    }
    tree->RemoveNode(part);
    delete part;
    status = 0;
  }


  ////// CLEAN UP ////// 

  delete [] membership;
  delete [] centroids;

  // Return whether anything was split
  return status;
}



int R3SurfelTree::
SplitNode(R3SurfelNode *node,  
    int max_parts_per_node, int max_blocks_per_node, 
    RNScalar max_leaf_complexity, RNScalar max_block_complexity,
    RNLength max_leaf_extent, RNLength max_block_extent,
    int max_levels)
{
  // Initialize return value
  int status = 0;

  // Just checking
  if (max_block_complexity > max_leaf_complexity) {
    max_block_complexity = max_leaf_complexity;
  }

  // Check if node is leaf
  if (node->NParts() == 0) {
    // Split blocks
    if (((max_block_complexity > 0) && (node->Complexity() > max_block_complexity)) ||
        ((max_block_extent > 0) && (node->BBox().LongestAxisLength() > max_block_extent))) {
      status |= SplitBlocks(node, max_block_complexity, max_block_extent);
    }

    // Check if node should be split 
    if (node->TreeLevel() < max_levels) {
      // Determine number of parts
      int nparts = 1;
      if (max_blocks_per_node > 0) {
        int n = node->NBlocks() / max_blocks_per_node + 1;
        if (n > nparts) nparts = n;
      }
      if (max_leaf_complexity > 0) {
        int n = (int) (node->Complexity() / max_leaf_complexity) + 1;
        if (n > nparts) nparts = n;
      }
      if (max_leaf_extent > 0) {
        int n = (int) (node->BBox().LongestAxisLength() / max_leaf_extent) + 1;
        if (n > nparts) nparts = n;
      }
      if (max_parts_per_node > 0) {
        if (nparts > max_parts_per_node) nparts = max_parts_per_node;
      }
      if (nparts > node->NBlocks()) {
        nparts = node->NBlocks();
      }

      // Split node into parts with K means clustering of blocks
      if (nparts > 1) {
        status |= SplitLeafNodeWithKMeansClusteringOfBlocks(this, node, nparts);
      }
    }
  }
  else {
    // Check if node should be split
    if (node->TreeLevel() < max_levels) {
      // Determine number of parts
      int nparts = node->NParts() / max_parts_per_node + 1;
      if (nparts > max_parts_per_node) nparts = max_parts_per_node;

      // Split node into parts with K means clustering of parts
      if (nparts > 1) {
        status |= SplitInteriorNodeWithKMeansClusteringOfParts(this, node, nparts);
      }
    }
  }

  // Return whether anything was split
  return status;
}



int R3SurfelTree::
SplitNodes(R3SurfelNode *start_node,
    int max_parts_per_node, int max_blocks_per_node, 
    RNScalar max_leaf_complexity, RNScalar max_block_complexity,
    RNLength max_leaf_extent, RNLength max_block_extent,
    int max_levels)
{
  // Initialize stack for depth first traversal from root node
  RNArray<R3SurfelNode *> stack;
  stack.Insert(start_node);
  int status = 0;

  // Split nodes into manageable sized chunks
  while (!stack.IsEmpty()) {
    // Pop node from stack
    R3SurfelNode *node = stack.Tail();
    stack.RemoveTail();

    // Split node
    status |= SplitNode(node, 
      max_parts_per_node, max_blocks_per_node, 
      max_leaf_complexity, max_block_complexity, 
      max_leaf_extent, max_block_extent, 
      max_levels);

    // Push parts onto stack
    for (int i = 0; i < node->NParts(); i++) {
      stack.Insert(node->Part(i));
    }
  }

  // Return whether any nodes were split
  return status;
}



int R3SurfelTree::
SplitNodes(int max_parts_per_node, int max_blocks_per_node, 
    RNScalar max_leaf_complexity, RNScalar max_block_complexity,
    RNLength max_leaf_extent, RNLength max_block_extent,
    int max_levels)
{
  // Split all nodes starting at root node
  return SplitNodes(RootNode(),
      max_parts_per_node, max_blocks_per_node, 
      max_leaf_complexity, max_block_complexity, 
      max_leaf_extent, max_block_extent, 
      max_levels);
}



int R3SurfelTree::
SplitNodes(R3SurfelPointSet& pointset,
  RNArray<R3SurfelNode *> *nodesA, RNArray<R3SurfelNode *> *nodesB)
{
  // Initialize return value (whether any node was split)
  int status = 0;

  // Create array of nodes containing points in point set
  RNArray<R3SurfelNode *> *pointset_nodes = pointset.Nodes();
  if (!pointset_nodes) return 0;

  // Split each node
  for (int i = 0; i < pointset_nodes->NEntries(); i++) {
    R3SurfelNode *node = pointset_nodes->Kth(i);

    // Split blocks in leaf node
    RNArray<R3SurfelBlock *> blocksA, blocksB;
    if (SplitBlocks(node, pointset, &blocksA, &blocksB)) {
      // Create new nodes
      R3SurfelNode *nodeA = new R3SurfelNode();
      R3SurfelNode *nodeB = new R3SurfelNode();
      
      // Move blocks from node to nodeA
      for (int i = 0; i < blocksA.NEntries(); i++) {
        R3SurfelBlock *block = blocksA.Kth(i);
        node->RemoveBlock(block);
        nodeA->InsertBlock(block);
      }
      
      // Move blocks from node to nodeB
      for (int i = 0; i < blocksB.NEntries(); i++) {
        R3SurfelBlock *block = blocksB.Kth(i);
        node->RemoveBlock(block);
        nodeB->InsertBlock(block);
      }
      
      // Insert new nodes into tree
      InsertNode(nodeA, node);
      InsertNode(nodeB, node);
      
      // Update return values
      if (nodesA) nodesA->Insert(nodeA);
      if (nodesB) nodesB->Insert(nodeB);
      status = 1;
    }
    else {
      if (blocksA.NEntries() > 0) {
        // Update return value
        if (nodesA) nodesA->Insert(node);
      }
      else {
        // Update return value
        if (nodesB) nodesB->Insert(node);
      }
    }
  }

  // Delete array of nodes containing points in point set
  delete pointset_nodes;

  // Return whether any nodes were split
  return status;
}



int R3SurfelTree::
SplitLeafNodes(R3SurfelNode *node, const R3SurfelConstraint& constraint, 
  RNArray<R3SurfelNode *> *nodesA, RNArray<R3SurfelNode *> *nodesB)
{
  // Initialize return value (whether any node was split)
  int status = 0;

  // Check if node is a leaf
  if (node->NParts() == 0) {
    // Check constraint
    int check = constraint.Check(node->BBox());
    if (check == R3_SURFEL_CONSTRAINT_PASS) {
      if (nodesA) nodesA->Insert(node);
    }
    else if (check == R3_SURFEL_CONSTRAINT_FAIL) {
      if (nodesB) nodesB->Insert(node);
    }
    else {
      // Split blocks in leaf node
      RNArray<R3SurfelBlock *> blocksA, blocksB;
      if (SplitBlocks(node, constraint, &blocksA, &blocksB)) {
        // Create new nodes
        R3SurfelNode *nodeA = new R3SurfelNode();
        R3SurfelNode *nodeB = new R3SurfelNode();

        // Move blocks from node to nodeA
        for (int i = 0; i < blocksA.NEntries(); i++) {
          R3SurfelBlock *block = blocksA.Kth(i);
          node->RemoveBlock(block);
          nodeA->InsertBlock(block);
        }
    
        // Move blocks from node to nodeB
        for (int i = 0; i < blocksB.NEntries(); i++) {
          R3SurfelBlock *block = blocksB.Kth(i);
          node->RemoveBlock(block);
          nodeB->InsertBlock(block);
        }
    
        // Insert new nodes into tree
        InsertNode(nodeA, node);
        InsertNode(nodeB, node);

        // Update return values
        if (nodesA) nodesA->Insert(nodeA);
        if (nodesB) nodesB->Insert(nodeB);
        status = 1;
      }
      else {
        // Blocks were all inside or outside constraint, update return values
        if (nodesA && !blocksA.IsEmpty()) nodesA->Insert(node);
        if (nodesB && !blocksB.IsEmpty()) nodesB->Insert(node);
      }
    }
  }
  else {
    // Check constraint
    if (!nodesA || !nodesB) {
      int check = constraint.Check(node->BBox());
      if ((check == R3_SURFEL_CONSTRAINT_PASS) && !nodesA) return 0;
      if ((check == R3_SURFEL_CONSTRAINT_FAIL) && !nodesB) return 0;
    }

    // Consider each child node
    for (int i = 0; i < node->NParts(); i++) {
      R3SurfelNode *part = node->Part(i);
      status |= SplitLeafNodes(part, constraint, nodesA, nodesB);
    }
  }

  // Return whether any nodes were split
  return status;
}



int R3SurfelTree::
SplitLeafNodes(const R3SurfelConstraint& constraint, 
  RNArray<R3SurfelNode *> *nodesA, RNArray<R3SurfelNode *> *nodesB)
{
  // Split leaf nodes starting at root
  return SplitLeafNodes(RootNode(), constraint, nodesA, nodesB);
}



////////////////////////////////////////////////////////////////////////
// BLOCK MANIPULATION FUNCTIONS
////////////////////////////////////////////////////////////////////////

int R3SurfelTree::
SplitBlocks(R3SurfelNode *node, R3SurfelPointSet& pointset,
  RNArray<R3SurfelBlock *> *blocksA, RNArray<R3SurfelBlock *> *blocksB)
{
  // Just checking
  assert(node);
  assert(node->Tree() == this);
  assert(node->tree_index >= 0);
  assert(nodes.Kth(node->tree_index) == node);
  int status = 0;
  
  // Create array of blocks containing points in point set
  RNArray<R3SurfelBlock *> *pointset_blocks = pointset.Blocks();;
  if (!pointset_blocks) return 0;

  // Create constraint to include only marked points
  const R3SurfelMarkConstraint constraint(TRUE, FALSE);

  // Split blocks 
  for (int i = 0; i < pointset_blocks->NEntries(); i++) {
    R3SurfelBlock *block = pointset_blocks->Kth(i);
    if (block->Node() != node) continue;
    block->SetMarks(FALSE);
    pointset.SetMarks(TRUE);
    R3SurfelBlock *blockA, *blockB;
    if (SplitBlock(node, block, constraint, &blockA, &blockB)) status = 1;
    if (blocksA && blockA) blocksA->Insert(blockA);
    if (blocksB && blockB) blocksB->Insert(blockB);
  }

  // Delete array of blocks containing points in point set
  delete pointset_blocks;

  // Return whether any blocks were split
  return status;
}



int R3SurfelTree::
SplitBlock(R3SurfelNode *node, R3SurfelBlock *block, const R3SurfelConstraint& constraint, 
  R3SurfelBlock **blockA, R3SurfelBlock **blockB)
{
  // Just checking
  assert(node && block);
  assert(block->Node() == node);

  // Check if block is fully outside constraint
  int check = constraint.Check(block);
  if (check == R3_SURFEL_CONSTRAINT_FAIL) { 
    if (blockA) *blockA = NULL;
    if (blockB) *blockB = block;
    return 0;
  }


  // Check if block is fully inside constraint
  if (check == R3_SURFEL_CONSTRAINT_PASS) { 
    if (blockA) *blockA = block;
    if (blockB) *blockB = NULL;
    return 0;
  }

  // Read block
  database->ReadBlock(block);

  // Partition surfels according to constraint
  RNArray<const R3Surfel *> subset1, subset2;
  for (int i = 0; i < block->NSurfels(); i++) {
    const R3Surfel *surfel = block->Surfel(i);
    if (constraint.Check(block, surfel)) subset1.Insert(surfel);
    else subset2.Insert(surfel);
  }

  // Create subset blocks 
  R3SurfelBlock *block1 = NULL;
  R3SurfelBlock *block2 = NULL;
  if (!database->InsertSubsetBlocks(block, subset1, subset2, &block1, &block2) || !block1 || !block2) {
    database->ReleaseBlock(block);
    if (blockA) *blockA = block1;
    if (blockB) *blockB = block2;
    return 0;
  }

  // Update node
  node->InsertBlock(block1); 
  node->InsertBlock(block2); 
  node->RemoveBlock(block);

  // Remove old block
  block->SetDirty(FALSE);
  database->ReleaseBlock(block);
  database->RemoveAndDeleteBlock(block);

  // Return new blocks
  if (blockA) *blockA = block1;
  if (blockB) *blockB = block2;

  // Return success
  return 1;
}



int R3SurfelTree::
SplitBlocks(R3SurfelNode *node, const R3SurfelConstraint& constraint, 
  RNArray<R3SurfelBlock *> *blocksA, RNArray<R3SurfelBlock *> *blocksB)
{
  // Just checking
  assert(node);
  assert(node->Tree() == this);
  assert(node->tree_index >= 0);
  assert(nodes.Kth(node->tree_index) == node);
  int status = 0;

  // Make a temporary array of blocks
  RNArray<R3SurfelBlock *> splittable_blocks;
  for (int i = 0; i < node->NBlocks(); i++) {
    R3SurfelBlock *block = node->Block(i);
    splittable_blocks.Insert(block);
  }

  // Split blocks 
  for (int i = 0; i < splittable_blocks.NEntries(); i++) {
    R3SurfelBlock *block = splittable_blocks.Kth(i);
    R3SurfelBlock *blockA, *blockB;
    if (SplitBlock(node, block, constraint, &blockA, &blockB)) status = 1;
    if (blocksA && blockA) blocksA->Insert(blockA);
    if (blocksB && blockB) blocksB->Insert(blockB);
  }

  // Return whether any blocks were split
  return status;
}



int R3SurfelTree::
SplitBlocks(R3SurfelNode *node, RNScalar max_complexity, RNScalar max_extent)
{
  // Just checking
  assert(node);
  assert(node->Tree() == this);
  assert(node->tree_index >= 0);
  assert(nodes.Kth(node->tree_index) == node);
  if ((max_extent <= 0) && (max_complexity <= 0)) return 0;
  int status = 0;
  
  // Make a temporary array of blocks
  RNArray<R3SurfelBlock *> splittable_blocks;
  for (int i = 0; i < node->NBlocks(); i++) {
    R3SurfelBlock *block = node->Block(i);
    splittable_blocks.Insert(block);
  }

  // Split blocks that are too big or have too much complexity
  while (!splittable_blocks.IsEmpty()) {
    R3SurfelBlock *block = splittable_blocks.Tail();
    splittable_blocks.RemoveTail();
    int dim = block->BBox().LongestAxis();
    if (((max_complexity > 0) && (block->NSurfels() > max_complexity)) || 
        ((max_extent > 0) && (block->BBox().AxisLength(dim) > max_extent))) {
      R3SurfelBlock *blockA, *blockB;
      R3Plane split(block->Centroid(), R3xyz_triad[dim]);
      R3SurfelHalfspaceConstraint constraint(R3Halfspace(split, 0));
      if (SplitBlock(node, block, constraint, &blockA, &blockB) && blockA && blockB) {
        splittable_blocks.Insert(blockA);
        splittable_blocks.Insert(blockB);
        status = 1;
      }
    }
  }

  // Return whether any blocks were split
  return status;
}



///////////////////////////////////////////////////////////////////////
// DISPLAY FUNCTIONS
////////////////////////////////////////////////////////////////////////

void R3SurfelTree::
Draw(RNFlags flags) const
{
  // Draw all nodes
  for (int i = 0; i < NNodes(); i++) {
    R3SurfelNode *node = Node(i);
    node->Draw(flags);
  } 
}



void R3SurfelTree::
Print(FILE *fp, const char *prefix, const char *suffix) const
{
  // Check fp
  if (!fp) fp = stdout;

  // Print tree header
  if (prefix) fprintf(fp, "%s", prefix);
  fprintf(fp, "%s", "Tree");
  if (suffix) fprintf(fp, "%s", suffix);
  fprintf(fp, "\n");

  // Add indentation to prefix
  char indented_prefix[1024];
  sprintf(indented_prefix, "%s  ", prefix);

  // Print all nodes
  for (int i = 0; i < NNodes(); i++) {
    R3SurfelNode *node = Node(i);
    node->Print(fp, indented_prefix, suffix);
  }
}



