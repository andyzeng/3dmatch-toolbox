////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "FET.h"



////////////////////////////////////////////////////////////////////////
// Constructors
////////////////////////////////////////////////////////////////////////

FETReconstruction::
FETReconstruction(void)
  : shapes(),
    matches(),
    features(),
    correspondences(),
    avg_feature_radius(0),
    max_correspondences(-1),
    max_euclidean_distance(RN_UNKNOWN), 
    max_normal_angle(RN_UNKNOWN),
    min_distinction(RN_UNKNOWN),
    max_distinction(RN_UNKNOWN),
    min_curvature(RN_UNKNOWN),
    max_curvature(RN_UNKNOWN),
    min_salience(RN_UNKNOWN),
    discard_boundaries(TRUE),
    discard_not_mutually_closest(FALSE),
    discard_outliers(TRUE),
    total_match_weight(0),
    total_trajectory_weight(0),
    total_inertia_weight(0),
    solver(RN_CSPARSE_SOLVER),
    bbox(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX)
{
  // Initialize parameters
  InitializeFeatureParameters();
  InitializeCorrespondenceParameters();
  InitializeOptimizationParameters();

#ifdef RN_USE_CERES
  solver = RN_CERES_SOLVER;
#endif
}



FETReconstruction::
FETReconstruction(const FETReconstruction& reconstruction)
  : shapes(),
    matches(),
    features(),
    correspondences(),
    avg_feature_radius(reconstruction.avg_feature_radius),
    max_correspondences(reconstruction.max_correspondences),
    max_euclidean_distance(reconstruction.max_euclidean_distance), 
    max_normal_angle(reconstruction.max_normal_angle),
    min_distinction(reconstruction.min_distinction),
    max_distinction(reconstruction.max_distinction),
    min_curvature(reconstruction.min_curvature),
    max_curvature(reconstruction.max_curvature),
    min_salience(reconstruction.min_salience),
    discard_boundaries(reconstruction.discard_boundaries),
    discard_not_mutually_closest(reconstruction.discard_not_mutually_closest),
    discard_outliers(reconstruction.discard_outliers),
    total_match_weight(reconstruction.total_match_weight),
    total_trajectory_weight(reconstruction.total_trajectory_weight),
    total_inertia_weight(reconstruction.total_inertia_weight),
    solver(reconstruction.solver),
    bbox(reconstruction.bbox)
{
  // Copy other stuff
  for (int i = 0; i < NUM_FEATURE_TYPES; i++) {
    max_descriptor_distances[i] = reconstruction.max_descriptor_distances[i];
    total_correspondence_weights[i] = reconstruction.total_correspondence_weights[i];
  }
}



FETReconstruction::
~FETReconstruction(void) 
{
  // Delete correspondences 
  while (NCorrespondences() > 0) {
    FETCorrespondence *correspondence = Correspondence(NCorrespondences()-1);
    delete correspondence;
  }

  // Delete features 
  while (NFeatures() > 0) {
    FETFeature *feature = Feature(NFeatures()-1);
    delete feature;
  }

  // Delete matches 
  while (NMatches() > 0) {
    FETMatch *match = Match(NMatches()-1);
    delete match;
  }

  // Delete shapes
  while (NShapes() > 0) {
    FETShape *shape = Shape(NShapes()-1);
    delete shape;
  }
}



////////////////////////////////////////////////////////////////////////
// Properties
////////////////////////////////////////////////////////////////////////

const R3Box& FETReconstruction::
BBox(void) const
{
  // Return bounding box
  if (bbox.IsEmpty()) ((FETReconstruction *) this)->UpdateBBox();
  return bbox;
}



RNLength FETReconstruction::
AverageFeatureRadius(void) const
{
  // Add average radii
  int count = 0;
  RNLength sum = 0;
  for (int i = 0; i < NShapes(); i++) {
    FETShape *shape = Shape(i);
    if (shape->NChildren() > 0) continue;
    sum += shape->AverageFeatureRadius();
    count++;
  }

  // Return average
  return (count > 0) ? sum / count : 0;
}



////////////////////////////////////////////////////////////////////////
// Shape access
////////////////////////////////////////////////////////////////////////

FETShape *FETReconstruction::
Shape(const char *name) const
{
  // Check name
  if (!name) return NULL;

  // Return first shape found with same name
  for (int i = 0; i < NShapes(); i++) {
    FETShape *shape = Shape(i);
    if (!shape->Name()) continue;
    if (!strcmp(name, shape->Name())) return shape;
  }

  // None found
  return NULL;
}



////////////////////////////////////////////////////////////////////////
// Shape manipulation
////////////////////////////////////////////////////////////////////////

void FETReconstruction::
InsertShape(FETShape *shape)
{
  // Just checking
  assert(shape->reconstruction_index == -1);
  assert(shape->reconstruction == NULL);

  // Insert shape
  shape->reconstruction = this;
  shape->reconstruction_index = shapes.NEntries();
  shapes.Insert(shape);

  // Update bounding box
  if (!bbox.IsEmpty()) bbox.Union(shape->BBox());

  // Insert matches
  for (int i = 0; i < shape->NMatches(); i++) {
    FETMatch *match = shape->Match(i);
    if (shape != match->Shape(0)) continue;
    InsertMatch(match);
  }

  // Insert features
  for (int i = 0; i < shape->NFeatures(); i++) {
    FETFeature *feature = shape->Feature(i);
    InsertFeature(feature);
  }
}



void FETReconstruction::
RemoveShape(FETShape *shape)
{
  // Just checking
  assert(shape->reconstruction_index >= 0);
  assert(shape->reconstruction_index < shapes.NEntries());
  assert(shape->reconstruction == this);

  // Remove features
  for (int i = 0; i < shape->NFeatures(); i++) {
    FETFeature *feature = shape->Feature(i);
    RemoveFeature(feature);
  }

  // Remove matches
  for (int i = 0; i < shape->NMatches(); i++) {
    FETMatch *match = shape->Match(i);
    if (shape != match->Shape(0)) continue;
    RemoveMatch(match);
  }

  // Remove shape
  RNArrayEntry *entry = shapes.KthEntry(shape->reconstruction_index);
  FETShape *tail = shapes.Tail();
  tail->reconstruction_index = shape->reconstruction_index;
  shapes.EntryContents(entry) = tail;
  shapes.RemoveTail();
  shape->reconstruction_index = -1;
  shape->reconstruction = NULL;

  // Invalidate bounding box
  InvalidateBBox();
}



////////////////////////////////////////////////////////////////////////
// Match manipulation
////////////////////////////////////////////////////////////////////////

void FETReconstruction::
InsertMatch(FETMatch *match)
{
  // Just checking
  assert(match->reconstruction_index == -1);
  assert(match->reconstruction == NULL);

  // Insert match
  match->reconstruction = this;
  match->reconstruction_index = matches.NEntries();
  matches.Insert(match);
}



void FETReconstruction::
RemoveMatch(FETMatch *match)
{
  // Just checking
  assert(match->reconstruction_index >= 0);
  assert(match->reconstruction_index < matches.NEntries());
  assert(match->reconstruction == this);

  // Remove match
  RNArrayEntry *entry = matches.KthEntry(match->reconstruction_index);
  FETMatch *tail = matches.Tail();
  tail->reconstruction_index = match->reconstruction_index;
  matches.EntryContents(entry) = tail;
  matches.RemoveTail();
  match->reconstruction_index = -1;
  match->reconstruction = NULL;
}



////////////////////////////////////////////////////////////////////////
// Feature manipulation
////////////////////////////////////////////////////////////////////////

void FETReconstruction::
InsertFeature(FETFeature *feature)
{
  // Just checking
  assert(feature->reconstruction_index == -1);
  assert(feature->reconstruction == NULL);

  // Insert feature
  feature->reconstruction = this;
  feature->reconstruction_index = features.NEntries();
  features.Insert(feature);

  // Insert correspondences
  for (int i = 0; i < feature->NCorrespondences(); i++) {
    FETCorrespondence *correspondence = feature->Correspondence(i);
    if (feature != correspondence->Feature(0)) continue;
    InsertCorrespondence(correspondence);
  }

  // Update bounding box
  if (!bbox.IsEmpty()) bbox.Union(feature->Position(TRUE));
}



void FETReconstruction::
RemoveFeature(FETFeature *feature)
{
  // Just checking
  assert(feature->reconstruction_index >= 0);
  assert(feature->reconstruction_index < features.NEntries());
  assert(feature->reconstruction == this);

  // Remove correspondences
  for (int i = 0; i < feature->NCorrespondences(); i++) {
    FETCorrespondence *correspondence = feature->Correspondence(i);
    if (feature != correspondence->Feature(0)) continue;
    RemoveCorrespondence(correspondence);
  }
  
  // Remove feature
  RNArrayEntry *entry = features.KthEntry(feature->reconstruction_index);
  FETFeature *tail = features.Tail();
  tail->reconstruction_index = feature->reconstruction_index;
  features.EntryContents(entry) = tail;
  features.RemoveTail();
  feature->reconstruction_index = -1;
  feature->reconstruction = NULL;

  // Update bounding box
  // ???
}



////////////////////////////////////////////////////////////////////////
// Correspondence manipulation
////////////////////////////////////////////////////////////////////////

void FETReconstruction::
InsertCorrespondence(FETCorrespondence *correspondence)
{
  // Just checking
  assert(correspondence->reconstruction_index == -1);
  assert(correspondence->reconstruction == NULL);

  // Insert correspondence
  correspondence->reconstruction = this;
  correspondence->reconstruction_index = correspondences.NEntries();
  correspondences.Insert(correspondence);
}



void FETReconstruction::
RemoveCorrespondence(FETCorrespondence *correspondence)
{
  // Just checking
  assert(correspondence->reconstruction_index >= 0);
  assert(correspondence->reconstruction_index < correspondences.NEntries());
  assert(correspondence->reconstruction == this);

  // Remove correspondence
  RNArrayEntry *entry = correspondences.KthEntry(correspondence->reconstruction_index);
  FETCorrespondence *tail = correspondences.Tail();
  tail->reconstruction_index = correspondence->reconstruction_index;
  correspondences.EntryContents(entry) = tail;
  correspondences.RemoveTail();
  correspondence->reconstruction_index = -1;
  correspondence->reconstruction = NULL;
}



////////////////////////////////////////////////////////////////////////
// FETReconstruction insertion
////////////////////////////////////////////////////////////////////////

void FETReconstruction::
CopyContents(const FETReconstruction& reconstruction)
{
  // Copy all shapes
  RNArray<FETShape *> copied_shapes;
  for (int i = 0; i < reconstruction.NShapes(); i++) {
    FETShape *shape1 = reconstruction.Shape(i);
    FETShape *shape2 = Shape(shape1->Name());
    if (!shape2) {
      shape2 = new FETShape(*shape1);
      InsertShape(shape2);
    }
    copied_shapes.Insert(shape2);
  }

  // Copy all matches
  RNArray<FETMatch *> copied_matches;
  for (int i = 0; i < reconstruction.NMatches(); i++) {
    FETMatch *match1 = reconstruction.Match(i);
    FETShape *shape10 = match1->Shape(0);
    FETShape *shape11 = match1->Shape(1);
    FETShape *shape20 = (shape10) ? copied_shapes.Kth(shape10->reconstruction_index) : NULL;
    FETShape *shape21 = (shape11) ? copied_shapes.Kth(shape11->reconstruction_index) : NULL;
    FETMatch *match2 = new FETMatch(*match1);
    InsertMatch(match2);
    if (shape20) shape20->InsertMatch(match2, 0);
    if (shape21) shape21->InsertMatch(match2, 1);
    copied_matches.Insert(match2);
  }

  // Copy all features
  RNArray<FETFeature *> copied_features;
  for (int i = 0; i < reconstruction.NFeatures(); i++) {
    FETFeature *feature1 = reconstruction.Feature(i);
    FETShape *shape1 = feature1->shape;
    FETShape *shape2 = (shape1) ? copied_shapes.Kth(shape1->reconstruction_index) : NULL;
    FETFeature *feature2 = new FETFeature(*feature1);
    InsertFeature(feature2);
    if (shape2) shape2->InsertFeature(feature2);
    copied_features.Insert(feature2);
  }

  // Copy all correspondences
  RNArray<FETCorrespondence *> copied_correspondences;
  for (int i = 0; i < reconstruction.NCorrespondences(); i++) {
    FETCorrespondence *correspondence1 = reconstruction.Correspondence(i);
    FETMatch *match1 = correspondence1->match;
    FETMatch *match2 = (match1) ? copied_matches.Kth(match1->reconstruction_index) : NULL;
    FETFeature *feature10 = correspondence1->Feature(0);
    FETFeature *feature11 = correspondence1->Feature(1);
    FETFeature *feature20 = (feature10) ? copied_features.Kth(feature10->reconstruction_index) : NULL;
    FETFeature *feature21 = (feature11) ? copied_features.Kth(feature11->reconstruction_index) : NULL;
    FETCorrespondence *correspondence2 = new FETCorrespondence(*correspondence1);
    InsertCorrespondence(correspondence2);
    if (feature20) feature20->InsertCorrespondence(correspondence2, 0);
    if (feature21) feature20->InsertCorrespondence(correspondence2, 1);
    if (match2) match2->InsertCorrespondence(correspondence2);
    copied_correspondences.Insert(correspondence2);
  }
}



////////////////////////////////////////////////////////////////////////
// Transformation manipulation
////////////////////////////////////////////////////////////////////////

void FETReconstruction::
ResetTransformations(void)
{
  // Reset transformations
  for (int i = 0; i < NShapes(); i++) {
    FETShape *shape = Shape(i);
    shape->ResetTransformation();
  }
}



void FETReconstruction::
PerturbTransformations(RNLength translation_magnitude, RNAngle rotation_magnitude)
{
  // Reset transformations
  for (int i = 0; i < NShapes(); i++) {
    FETShape *shape = Shape(i);
    shape->PerturbTransformation(translation_magnitude, rotation_magnitude);
  }
}



////////////////////////////////////////////////////////////////////////
// Alignment Properties
////////////////////////////////////////////////////////////////////////

RNScalar FETReconstruction::
RMSD(void) const
{
  // Compute ssd
  RNScalar ssd = 0;
  RNScalar weight = 0;
  for (int i = 0; i < NCorrespondences(); i++) {
    FETCorrespondence *correspondence = Correspondence(i);
    RNLength dd = correspondence->SquaredEuclideanDistance();
    if (dd == RN_UNKNOWN) continue;
    weight += correspondence->affinity;
    ssd += dd;
  }

  // Compute rmsd
  RNScalar rmsd = (weight > 0) ? sqrt(ssd/weight) : 0.0;
  return rmsd;
}



RNScalar FETReconstruction::
Score(RNScalar sigma) const
{
  // Check number of correspondences
  if (correspondences.NEntries() == 0) return 0;
  if (sigma == RN_UNKNOWN) sigma = 2.0 * avg_feature_radius;
  if (sigma == 0) return 0;

  // Check sigma
  if (RNIsNegativeOrZero(sigma)) return 0;

  // Compute ssd
  RNScalar sum = 0;
  RNScalar factor = -1.0 / (2.0 * sigma * sigma);
  for (int i = 0; i < NCorrespondences(); i++) {
    FETCorrespondence *correspondence = Correspondence(i);
    RNLength dd = correspondence->SquaredEuclideanDistance();
    if (dd == RN_UNKNOWN) continue;
    sum += exp(factor * dd);
  }

  // Return sum
  return sum;
}



RNScalar FETReconstruction::
InlierFraction(RNLength error_threshold) const
{
  // Check number of correspondences
  if (NCorrespondences() == 0) return 0.0;

  // Check error threshold
  if (error_threshold == RN_UNKNOWN) error_threshold = 2.0 * avg_feature_radius;

  // Count inliers
  int count = 0;
  for (int i = 0; i < NCorrespondences(); i++) {
    FETCorrespondence *correspondence = Correspondence(i);
    RNScalar error = correspondence->Error();
    if ((error < 0) || (error > error_threshold)) continue;
    count++;
  }

  // Return fraction of inliers
  RNScalar fraction = (double) count / (double) correspondences.NEntries();
  return fraction;
}




RNScalar FETReconstruction::
Error(void) const
{
  // XXX NOTE: THIS NEEDS TO HAVE GROUND TRUTH CORRESPONDENCES

  // Compute error in current correspondences
  RNScalar sum = 0;
  RNScalar weight = 0;
  for (int i = 0; i < NCorrespondences(); i++) {
    FETCorrespondence *correspondence = Correspondence(i);
    RNScalar error = correspondence->Error();
    if (error == RN_UNKNOWN) continue;
    weight += correspondence->affinity;
    sum += error;
  }

  // Return weighted average
  return (weight > 0) ? sum / weight : 0;
}



RNScalar FETReconstruction::
Affinity(FETFeature *feature1, FETFeature *feature2) const
{
  // Get/check shapes
  FETShape *shape1 = feature1->shape;
  FETShape *shape2 = feature2->shape;
  if (!shape1 || !shape2) return 0;
  
  // Check generator types
  if (((feature1->GeneratorType() == SILHOUETTE_FEATURE_TYPE) && (feature2->GeneratorType() == RIDGE_FEATURE_TYPE)) ||
      ((feature1->GeneratorType() == RIDGE_FEATURE_TYPE) && (feature2->GeneratorType() == SILHOUETTE_FEATURE_TYPE))) {
    // For ridge to match silhouette, viewing vectors should be "opposite"
    R3Vector vector1 = feature1->Position(TRUE) - feature1->Shape()->Viewpoint();
    R3Vector vector2 = feature2->Position(TRUE) - feature2->Shape()->Viewpoint();
    if (vector1.Dot(vector2) > 0) return 0;
  }
  else {
    // For other features, must be same generator type
    if (feature1->GeneratorType() != feature2->GeneratorType()) return 0;
  }

  // Initialize affinity
  RNScalar affinity = 1.0;
  const RNScalar min_affinity = 1E-3;

  // Compute salience affinity
  if (feature1->Salience() < 1.0) affinity *= feature1->Salience();
  if (feature2->Salience() < 1.0) affinity *= feature2->Salience();
  if (affinity < min_affinity) return 0;
  
#if 0
  // Compute position affinity
  // Could be ratio of closest distance to this distance if include non-closest correspondences
  RNScalar position_affinity = 1.0;
  if (max_euclidean_distance != RN_UNKNOWN) {
    RNLength euclidean_distance = sqrt(squared_euclidean_distance);
    position_affinity = 1.0 - euclidean_distance / max_euclidean_distance;
    affinity *= position_affinity;
    if (affinity < min_affinity) return 0;
  }
#endif

#if 0
  // Compute distinction affinity
  RNScalar distinction_affinity = 1.0;
  if ((min_distinction != RN_UNKNOWN) && (max_distinction > min_distinction)) {
    R3Point position1 = feature1->position;
    R3Point position2 = feature2->position;
    shape1->Transform(position1);
    shape2->Transform(position2);
    RNLength squared_euclidean_distance = R3SquaredDistance(position1, position2);
    if (squared_euclidean_distance > max_euclidean_distance * max_euclidean_distance) return 0; 
    RNScalar distinction = Distinction(feature1, feature2, squared_euclidean_distance);
    if (distinction == RN_UNKNOWN) return 0;
    if (distinction <= min_distinction) return 0;
    if (distinction > max_distinction) distinction = max_distinction;
    distinction_affinity = (distinction - min_distinction) / (max_distinction - min_distinction);
    affinity *= distinction_affinity;
    if (affinity < min_affinity) return 0;
  }
#endif

#if 0
  // Compute descriptor affinity
  RNScalar descriptor_affinity = 1.0;
  if (feature1->GeneratorType() == feature2->GeneratorType()) {
    if ((max_descriptor_distances[feature1->GeneratorType()] != RN_UNKNOWN) && (max_euclidean_distance != RN_UNKNOWN)) {
      RNScalar descriptor_distance = DescriptorDistance(feature1, feature2, max_euclidean_distance * max_euclidean_distance);
      if (descriptor_distance == RN_UNKNOWN) return 0;
      if (descriptor_distance > max_descriptor_distance) return 0;
      descriptor_affinity = 1.0 - descriptor_distance / max_descriptor_distances[feature1->GeneratorType()];
      affinity *= descriptor_affinity;
      if (affinity < min_affinity) return 0;
    }
  }
#endif
 
#if 0
  // Compute curvature affinity (THIS ASSUMES DESCRIPTORS STORE CURVATURE)
  RNScalar curvature_affinity = 1.0;
  if ((max_curvature != RN_UNKNOWN) && (max_euclidean_distance != RN_UNKNOWN)) {
    R3MeshPropertySet *descriptors1 = shape1->descriptors;
    R3MeshPropertySet *descriptors2 = shape2->descriptors;
    if (descriptors1 && descriptors2) {
      int nproperties = descriptors1->NProperties();
      if (descriptors2->NProperties() < nproperties) nproperties = descriptors2->NProperties();
      int property_index = NeighborhoodIndex(max_euclidean_distance * max_euclidean_distance, avg_feature_radius, nproperties);
      if (property_index >= 0) {
        R3MeshProperty *property1 = descriptors1->Property(property_index);
        R3MeshProperty *property2 = descriptors2->Property(property_index);
        RNScalar curvature1 = property1->VertexValue(vertex1);
        RNScalar curvature2 = property2->VertexValue(vertex2);
        if (curvature1 == RN_UNKNOWN) curvature1 = min_curvature;
        if (curvature2 == RN_UNKNOWN) curvature2 = min_curvature;
        RNScalar curvature = sqrt(fabs(curvature1) * fabs(curvature2));
        if ((min_curvature > 0) && (curvature < min_curvature)) curvature = min_curvature;
        if ((max_curvature > 0) && (curvature > max_curvature)) curvature = max_curvature;
        curvature_affinity = curvature / max_curvature;
        affinity *= curvature_affinity;
        if (affinity < min_affinity) return 0;
      }
    }
  }
#endif

  // Compute normal affinity
  RNScalar normal_affinity = 1.0;
  RNScalar direction_affinity = 1.0;
  if ((max_normal_angle != RN_UNKNOWN) && (max_normal_angle > 0)) {
    if ((feature1->ShapeType() != LINE_FEATURE_SHAPE) && (feature2->ShapeType() != LINE_FEATURE_SHAPE)) {
      // Check normal affinity
      R3Vector normal1 = feature1->normal;
      R3Vector normal2 = feature2->normal;
      shape1->Transform(normal1);
      shape2->Transform(normal2);
      RNAngle normal_angle = R3InteriorAngle(normal1, normal2);
      if (normal_angle > max_normal_angle) return 0;
      normal_affinity = 1.0 - normal_angle / max_normal_angle;
      affinity *= normal_affinity;
      if (affinity < min_affinity) return 0;
    }
    else if ((feature1->ShapeType() == LINE_FEATURE_SHAPE) && (feature2->ShapeType() == LINE_FEATURE_SHAPE)) {
      // Check direction affinity
      R3Vector direction1 = feature1->direction;
      R3Vector direction2 = feature2->direction;
      shape1->Transform(direction1);
      shape2->Transform(direction2);
      RNScalar dot = fabs(direction1.Dot(direction2));
      RNAngle direction_angle = (dot < 1.0) ? acos(dot) : 0.0;
      if (direction_angle > max_normal_angle) return 0;
      direction_affinity = 1.0 - direction_angle / max_normal_angle;
      affinity *= direction_affinity;
      if (affinity < min_affinity) return 0;
    }
  }
  
  // Return affinity
  return affinity;
}




RNScalar FETReconstruction::
Speckle(void) const
{
#if 0
  // Initialize speckle
  RNScalar speckle = 0.0;
  RNScalar max_distance = 10 * AverageFeatureRadius();

  // For every pair of shapes
  for (int i1 = 0; i1 < NShapes(); i1++) {
    Shape *shape1 = Shape(i1);
    const R3Affine& transformation1 = shape1->Transformation();
    R3Mesh *mesh1 = shape1->mesh;
    if (!shape1->mesh_kdtree) shape1->mesh_kdtree = new R3MeshSearchTree(mesh1);
    if (!shape1->mesh_kdtree) continue;

    for (int i2 = i1+1; i2 < NShapes(); i2++) {
      Shape *shape2 = Shape(i2);
      const R3Affine& transformation2 = shape2->Transformation();
      R3Mesh *mesh2 = shape2->mesh;

      for (int j2 = 0; j2 < shape2->points.NEntries(); j2++) {
        Point *point2 = shape2->points.Kth(j2);
        R3MeshVertex *vertex2 = point2->vertex;

        // Compute signed distance to closest point on mesh1
        R3Vector normal1;
        R3MeshIntersection closest1;
        R3Point position2 = point2->position;
        position2.Transform(transformation2);
        position2.InverseTransform(transformation1);
        shape1->mesh_kdtree->FindClosest(position2, closest1, 0, max_distance);
        if (closest1.type == R3_MESH_NULL_TYPE) continue;
        else if (closest1.type == R3_MESH_VERTEX_TYPE) normal1 = mesh1->VertexNormal(closest1.vertex);
        else if (closest1.type == R3_MESH_EDGE_TYPE) normal1 = mesh1->EdgeNormal(closest1.edge);
        else if (closest1.type == R3_MESH_FACE_TYPE) normal1 = mesh1->FaceNormal(closest1.face);
        R3Plane plane1(closest1.point, normal1);
        RNScalar d2 = R3SignedDistance(plane1, position2);

        // Count edge crossings
        for (int k2 = 0; k2 < mesh2->VertexValence(vertex2); k2++) {
          R3MeshEdge *edge2 = mesh2->EdgeOnVertex(vertex2, k2);
          R3MeshVertex *neighbor_vertex2 = mesh2->VertexAcrossEdge(edge2, vertex2);

          // Compute signed distance to closest point on mesh1
          R3Point neighbor_position2 = mesh2->VertexPosition(neighbor_vertex2);
          neighbor_position2.Transform(transformation2);
          neighbor_position2.InverseTransform(transformation1);
          shape1->mesh_kdtree->FindClosest(neighbor_position2, closest1, 0, max_distance);
          if (closest1.type == R3_MESH_NULL_TYPE) continue;
          else if (closest1.type == R3_MESH_VERTEX_TYPE) normal1 = mesh1->VertexNormal(closest1.vertex);
          else if (closest1.type == R3_MESH_EDGE_TYPE) normal1 = mesh1->EdgeNormal(closest1.edge);
          else if (closest1.type == R3_MESH_FACE_TYPE) normal1 = mesh1->FaceNormal(closest1.face);
          R3Plane plane1(closest1.point, normal1);
          RNScalar neighbor_d2 = R3SignedDistance(plane1, neighbor_position2);

          // Check for edge crossing
          if (neighbor_d2 * d2 < 0) {
            speckle += 1.0;
            break;
          }
        }
      }
    }
  }

  // Return speckle
  return speckle;
#else
  return 0;
#endif
}



////////////////////////////////////////////////////////////////////////
// Parameter setting functions
////////////////////////////////////////////////////////////////////////

void FETReconstruction::
InitializeFeatureParameters(void)
{
  // Compute average feature radius 
  avg_feature_radius = AverageFeatureRadius();
}



void FETReconstruction::
InitializeCorrespondenceParameters(void)
{
  // Set maximum number of correspondences 
  max_correspondences = 128 * 1024;

  // Set max correspondence distance{
  // max_euclidean_distance = 8 * avg_feature_radius;
  // max_euclidean_distance = 2 * avg_feature_radius;
  // max_euclidean_distance = 40 * avg_feature_radius;
  max_euclidean_distance = 2;

  // Turn off descriptor comparisons
  for (int i = 0; i < NUM_FEATURE_TYPES; i++) {
    max_descriptor_distances[i] = RN_UNKNOWN;
  }

  // Initialize max descriptor distances for some feature types
  max_descriptor_distances[SIFT_FEATURE_TYPE] = 0.25;

  // Initialize max normal angle
  max_normal_angle = RN_PI / 2.0;
}



void FETReconstruction::
InitializeOptimizationParameters(void)
{
  // Set initial weights
  total_match_weight = 0;
  total_inertia_weight = 0.001;
  total_trajectory_weight = 1000;
  for (int i = 0; i < NUM_FEATURE_TYPES; i++) {
    total_correspondence_weights[i] = 0;
  }

  // Set weights for some types of correspondences
  total_correspondence_weights[SIFT_FEATURE_TYPE] = 0;
  total_correspondence_weights[FAST_FEATURE_TYPE] = 0;
  total_correspondence_weights[CORNER_FEATURE_TYPE] = 1000;
  total_correspondence_weights[RIDGE_FEATURE_TYPE] = 500;
  total_correspondence_weights[VALLEY_FEATURE_TYPE] = 1000;
  total_correspondence_weights[SILHOUETTE_FEATURE_TYPE] = 1000;
  total_correspondence_weights[UNIFORM_FEATURE_TYPE] = 100;
  total_correspondence_weights[PLANE_FEATURE_TYPE] = 1000;
  total_correspondence_weights[STRUCTURE_FEATURE_TYPE] = 4000;
}



////////////////////////////////////////////////////////////////////////
// Parsing utility functions
////////////////////////////////////////////////////////////////////////

static int
CreateMeshFeatures(FETReconstruction *reconstruction, FETShape *shape,
  R3Mesh *mesh, R3MeshPropertySet *properties, RNScalar salience, RNLength min_spacing,
  RNBoolean create_vertex_features = TRUE, RNBoolean create_edge_features = TRUE, RNBoolean create_face_features = TRUE)
{
  // Just checking
  if (salience <= 0) salience = 1.0;
  
  // Create kdtree
  FETFeature tmp; int position_offset = (unsigned char *) &(tmp.position) - (unsigned char *) &tmp;
  R3Kdtree<FETFeature *> kdtree(mesh->BBox(), position_offset);

  // Create vertex features
  if (create_vertex_features) {
    for (int i = 0; i < mesh->NVertices(); i++) {
      R3MeshVertex *vertex = mesh->Vertex(i);
      if (mesh->VertexValence(vertex) < 2) continue;
      const R3Point& position = mesh->VertexPosition(vertex);
      int generator_type = CORNER_FEATURE_TYPE;
      int shape_type = POINT_FEATURE_SHAPE;
      RNFlags flags(FEATURE_IS_POINT);
    
      // Check min spacing
      if (min_spacing > 0) {
        if (kdtree.FindAny(position, 0, min_spacing)) continue;
      }

      // Compute radius
      RNLength radius = mesh->VertexAverageEdgeLength(vertex);
      if (min_spacing > radius) radius = min_spacing;
      if (radius == 0) continue;

      // Check curvature
      RNScalar curvature = mesh->VertexMeanCurvature(vertex);
      if (1 || (fabs(curvature) < 1.0 / radius)) {
        generator_type = PLANE_FEATURE_TYPE;
        shape_type = PLANE_FEATURE_SHAPE;
        flags.Add(FEATURE_IS_PLANAR);
      }

      // Check boundary
      if (mesh->IsVertexOnBoundary(vertex)) {
        generator_type = BORDER_FEATURE_TYPE;
        flags.Add(FEATURE_IS_ON_BORDER_BOUNDARY);
      }

      // Create feature
      FETFeature *feature = new FETFeature(reconstruction);
      feature->SetGeneratorType(generator_type);
      feature->SetShapeType(shape_type);
      feature->SetPosition(position);
      feature->SetNormal(mesh->VertexNormal(vertex));
      feature->SetRadius(radius);
      feature->SetFlags(flags);
      feature->SetSalience(10 * salience);

      // Create descriptor
      if (properties && (properties->NProperties() > 0)) {
        FETDescriptor descriptor(properties->NProperties());
        for (int j = 0; j < properties->NProperties(); j++) {
          R3MeshProperty *property = properties->Property(j);
          RNScalar value = property->VertexValue(i);
          descriptor.SetValue(j, value);
        }
        feature->SetDescriptor(descriptor);
      }

      // Insert feature into shape
      if (shape) shape->InsertFeature(feature);

      // Insert feature into kdtree
      kdtree.InsertPoint(feature);
    }
  }

  // Make sure min_spacing > 0
  if (min_spacing <= 0) min_spacing = 0.1;
  // if (min_spacing <= 0) min_spacing = mesh->AverageEdgeLength();

  // Create edge features
  if (create_edge_features) {
    for (int i = 0; i < mesh->NEdges(); i++) {
      R3MeshEdge *edge = mesh->Edge(i);
      RNAngle angle = mesh->EdgeInteriorAngle(edge);
      if (RNIsEqual(angle, RN_PI, 0.1)) continue;
      R3Span span = mesh->EdgeSpan(edge);
      RNLength length = span.Length();
      if (length < 2 * min_spacing) continue;
      int generator_type = (angle < RN_PI) ? RIDGE_FEATURE_TYPE : VALLEY_FEATURE_TYPE;
      R3Vector direction = span.Vector();
      RNFlags flags(FEATURE_IS_LINEAR);
      R3MeshFace *face0 = mesh->FaceOnEdge(edge, 0);
      R3MeshFace *face1 = mesh->FaceOnEdge(edge, 1);
      if (!face0 || !face1) flags.Add(FEATURE_IS_ON_BORDER_BOUNDARY);
      R3Vector normal = R3zero_vector;
      if (face0) normal += mesh->FaceNormal(face0);
      if (face1) normal += mesh->FaceNormal(face1);
      normal.Normalize();

      // Create features along edge
      for (RNScalar t = min_spacing; t < length; t+= min_spacing) {
        R3Point position = span.Point(t);

        // Check min spacing
        if (kdtree.FindAny(position, 0, min_spacing)) continue;

        // Create feature
        FETFeature *feature = new FETFeature(reconstruction);
        feature->SetGeneratorType(generator_type);
        feature->SetShapeType(LINE_FEATURE_SHAPE);
        feature->SetPosition(position);
        feature->SetDirection(direction);
        feature->SetNormal(normal);
        feature->SetRadius(min_spacing);
        feature->SetFlags(flags);
        feature->SetSalience(salience);
        feature->primitive_marker = i;

        // Insert feature into shape
        if (shape) shape->InsertFeature(feature);

        // Insert feature into kdtree
        kdtree.InsertPoint(feature);
      }
    }
  }
  
  // Create face features
  if (create_face_features) {
    RNScalar min_area = RN_PI * min_spacing * min_spacing;
    for (int i = 0; i < mesh->NFaces(); i++) {
      R3MeshFace *face = mesh->Face(i);
      RNArea area = mesh->FaceArea(face);
      if (area < min_area) continue;
      R3Vector normal = mesh->FaceNormal(face);
      int nfeatures = area / min_area;

      // Create features along face
      for (int j = 0; j < 2 * nfeatures; j++) {
        // Sample a position
        R3Point position = mesh->RandomPointOnFace(face);

        // Check min spacing
        if (kdtree.FindAny(position, 0, min_spacing)) continue;

        // Create feature
        FETFeature *feature = new FETFeature(reconstruction);
        feature->SetGeneratorType(PLANE_FEATURE_TYPE);
        feature->SetShapeType(PLANE_FEATURE_SHAPE);
        feature->SetPosition(position);
        feature->SetNormal(normal);
        feature->SetRadius(min_spacing);
        feature->SetSalience(0.1 * salience);
        feature->SetFlags(FEATURE_IS_PLANAR);
        feature->primitive_marker = i;

        // Insert feature into shape
        if (shape) shape->InsertFeature(feature);

        // Insert feature into kdtree
        kdtree.InsertPoint(feature);
      }
    }
  }
  
  // Return success
  return 1;
}



static int
ParseExternalCommand(FETReconstruction *reconstruction, FILE *fp, const char *cmd)
{
  // Parse external commands
  if (!strcmp(cmd, "bmesh")) {
    char mesh_name[4096];
    double mesh_tx, mesh_ty, mesh_tz, mesh_qa, mesh_qb, mesh_qc, mesh_qd;
    if (fscanf(fp, "%s%lf%lf%lf%lf%lf%lf%lf", mesh_name,
       &mesh_tx, &mesh_ty, &mesh_tz, &mesh_qa, &mesh_qb, &mesh_qc, &mesh_qd) != (unsigned int) 8) {
      fprintf(stderr, "Unable to read bmesh command\n");
      return 0;
    }
        
    // Determine transformation
    R3Vector mesh_translation(mesh_tx, mesh_ty, mesh_tz);
    R3Quaternion mesh_rotation(mesh_qd, -mesh_qa, -mesh_qb, -mesh_qc);
    R3Affine transformation = R3identity_affine;
    transformation.Translate(mesh_translation);
    transformation.Rotate(mesh_rotation);

    // Get base name
    char base_name[4096];
    char *strp = strchr(mesh_name, '/');
    if (strp) strncpy(base_name, strp+1, 4096);
    else strncpy(base_name, mesh_name, 4096);
    strp = strrchr(base_name, '.');
    if (strp) *strp = '\0';

    // Read mesh
    R3Mesh mesh;
    if (!mesh.ReadFile(mesh_name)) return 0;

    // Read mesh properties 
    R3MeshPropertySet properties(&mesh);
    char properties_name[4096];
    sprintf(properties_name, "descriptors/%s.arff", base_name);
    if (RNFileExists(properties_name)) {
      properties.Read(properties_name);
    }

    // Create shape
    FETShape *shape = new FETShape(reconstruction);    
    shape->initial_transformation = transformation;
    shape->current_transformation = transformation;
    shape->ground_truth_transformation = transformation;
    if (strp) shape->name = strdup(base_name);

    // Create features
    CreateMeshFeatures(reconstruction, shape, &mesh, &properties, 1.0, 0.0, TRUE, FALSE, FALSE);
  }
  else if (!strcmp(cmd, "mesh")) {
    int dummy;
    double m[16], inertia[9], salience, min_spacing;
    char mesh_name[4096], properties_name[4096];
    if (fscanf(fp, "%s%s%lf%lf%lf%lf %lf%lf%lf%lf%lf%lf%lf%lf %lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%d%d%d", mesh_name, properties_name, 
      &m[0], &m[1], &m[2], &m[3],  &m[4], &m[5], &m[6], &m[7],
      &m[8], &m[9], &m[10], &m[11],  &m[12], &m[13], &m[14], &m[15],
      &inertia[0], &inertia[1], &inertia[2], &inertia[3], &inertia[4], &inertia[5], &inertia[6], &inertia[7], &inertia[8],
      &salience, &min_spacing, &dummy, &dummy, &dummy) != (unsigned int) 32) {
      fprintf(stderr, "Unable to read mesh command\n");
      return 0;
    }

    // Get base name
    char base_name[4096];
    char *strp = strchr(mesh_name, '/');
    if (strp) strncpy(base_name, strp+1, 4096);
    else strncpy(base_name, mesh_name, 4096);
    strp = strrchr(base_name, '.');
    if (strp) *strp = '\0';

    // Read mesh
    R3Mesh mesh;
    if (!mesh.ReadFile(mesh_name)) return 0;

    // Read mesh properties 
    R3MeshPropertySet properties(&mesh);
    if (strcmp(properties_name, "none") && strcmp(properties_name, "None") && strcmp(properties_name, "NONE")) {
      if (!properties.Read(properties_name)) return 0;
    }

    // Create shape
    FETShape *shape = new FETShape(reconstruction);

    // Set inertias
    for (int i = 0; i < shape->max_variables; i++) shape->variable_inertias[i] = inertia[i];

    // Set transformation
    R3Affine transformation(R4Matrix(m), 0);
    shape->initial_transformation = transformation;
    shape->current_transformation = transformation;
    shape->ground_truth_transformation = transformation;
    if (strp) shape->name = strdup(base_name);

    // Create features
    CreateMeshFeatures(reconstruction, shape, &mesh, &properties, salience, min_spacing);
  }
  else if (!strcmp(cmd, "point")) {
    // Read point correspondence
    int shape_index0, shape_index1;
    double x0, y0, z0, x1, y1, z1;
    if (fscanf(fp, "%d%d%lf%lf%lf%lf%lf%lf", &shape_index0, &shape_index1, &x0, &y0, &z0, &x1, &y1, &z1) != (unsigned int) 8) {
      fprintf(stderr, "Unable to read point correspondence\n");
      return 0;
    }

    // Maciej's ground truth is for every fifth shape in original
    shape_index0 /= 2;
    shape_index1 /= 2;

    // Create point correspondence
    FETShape *shape0 = reconstruction->Shape(shape_index0);
    FETShape *shape1 = reconstruction->Shape(shape_index1);
    FETFeature *feature0 = new FETFeature(reconstruction, POINT_FEATURE_SHAPE, R3Point(x0, y0, z0));
    FETFeature *feature1 = new FETFeature(reconstruction, POINT_FEATURE_SHAPE, R3Point(x1, y1, z1));
    feature0->SetGeneratorType(CORNER_FEATURE_TYPE);
    feature1->SetGeneratorType(CORNER_FEATURE_TYPE);
    shape0->InsertFeature(feature0);
    shape1->InsertFeature(feature1);
    FETCorrespondence *correspondence = new FETCorrespondence(reconstruction, feature0, feature1);
    if (!correspondence) {
      fprintf(stderr, "Unable to create correspondence\n");
      return 0;
    }
  }
  else if (!strcmp(cmd, "inline")) {
    // Parse filename
    char inline_filename[4096];
    if (fscanf(fp, "%s", inline_filename) != (unsigned int) 1) {
      fprintf(stderr, "Unable to read inline command\n");
      return 0;
    }

    // Open file
    FILE *inline_fp = fopen(inline_filename, "r");
    if (!inline_fp) {
      fprintf(stderr, "Unable to open inline file %s\n", inline_filename);
      return 0;
    }

    // Read file
    reconstruction->ReadAscii(inline_fp);    

    // Close file
    fclose(inline_fp);
  }
  else if (!strcmp(cmd, "include")) {
    // Parse filename
    char include_filename[4096];
    if (fscanf(fp, "%s", include_filename) != (unsigned int) 1) {
      fprintf(stderr, "Unable to read include command\n");
      return 0;
    }

    // Read file  
    FETReconstruction tmp;
    if (!tmp.ReadFile(include_filename)) return 0;
    reconstruction->CopyContents(tmp);
  }
  else {
    // Command not recognized
    return 0;
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Input/output
////////////////////////////////////////////////////////////////////////

int FETReconstruction::
ReadFile(const char *filename)
{
  // Check filename
  if (!filename) return 0;

  // Parse input filename extension
  const char *extension;
  if (!(extension = strrchr(filename, '.'))) {
    printf("Filename %s has no extension (e.g., .fet)\n", filename);
    return 0;
  }

  // Read file of appropriate type
  if (!strncmp(extension, ".fcb", 4)) {
    if (!ReadBinaryFile(filename)) return 0;
  }
  else {
    if (!ReadAsciiFile(filename)) return 0;
  }

  // Return success
  return 1;
}



int FETReconstruction::
WriteFile(const char *filename) const
{
  // Parse input filename extension
  const char *extension;
  if (!(extension = strrchr(filename, '.'))) {
    printf("Filename %s has no extension (e.g., .fet)\n", filename);
    return 0;
  }

  // Write file of appropriate type
  if (!strncmp(extension, ".fcb", 4)) {
    if (!WriteBinaryFile(filename)) return 0;
  }
  else {
    if (!WriteAsciiFile(filename)) return 0;
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Ascii Input and Output
////////////////////////////////////////////////////////////////////////

int FETReconstruction::
ReadAsciiFile(const char *filename)
{
  // Check filename
  if (!filename) return 0;

  // Open file
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Unable to open reconstruction file %s\n", filename);
    return 0;
  }

  // Read file contents
  if (!ReadAscii(fp)) return 0;

  // Close file
  fclose(fp);

  // Return success
  return 1;
}

  

int FETReconstruction::
WriteAsciiFile(const char *filename) const
{
  // Check filename
  if (!filename) return 0;

  // Open file
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "Unable to open %s\n", filename);
    return 0;
  }

  // Read file contents
  if (!WriteAscii(fp)) {
    fprintf(stderr, "Unable to read %s\n", filename);
    return 0;
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



int FETReconstruction::
ReadAscii(FILE *fp)
{
  // Read ascii file
  int dummy;
  RNScalar ddummy;
  char cmd[1024];
  while (fscanf(fp, "%s", cmd) == (unsigned int) 1) {
    if (!strcmp(cmd, "H")) {
      int major_version, minor_version;
      int nshapes, nmatches, nfeatures, ncorrespondences;
      fscanf(fp, "%d", &major_version);
      fscanf(fp, "%d", &minor_version);
      fscanf(fp, "%d", &nshapes);
      fscanf(fp, "%d", &nmatches);
      fscanf(fp, "%d", &nfeatures);
      fscanf(fp, "%d", &ncorrespondences);
      for (int i = 0; i < 4; i++) fscanf(fp, "%d", &dummy);
    }
    else if (!strcmp(cmd, "X")) {
      // Read parameters
      fscanf(fp, "%lf ", &avg_feature_radius);
      for (int i = 0; i < 29; i++) fscanf(fp, "%lf ", &ddummy);
    }
    else if (!strcmp(cmd, "S")) {
      FETShape *shape = new FETShape(this);
      if (!shape->ReadAscii(fp)) return 0;
    }
    else if (!strcmp(cmd, "M")) {
      FETMatch *match = new FETMatch(this);
      if (!match->ReadAscii(fp)) return 0;
    }
    else if (!strcmp(cmd, "F")) {
      FETFeature *feature = new FETFeature(this);
      if (!feature->ReadAscii(fp)) return 0;
    }
    else if (!strcmp(cmd, "C")) {
      FETCorrespondence *correspondence = new FETCorrespondence(this);
      if (!correspondence->ReadAscii(fp)) return 0;
    }
    else if (!ParseExternalCommand(this, fp, cmd)) {
      fprintf(stderr, "Unrecognized command %s in reconstruction file\n", cmd);
      return 0;
    }
  }

  // Update parameters
  if (avg_feature_radius <= 0) InitializeFeatureParameters();
  InitializeCorrespondenceParameters();
  InitializeOptimizationParameters();

  // Return success
  return 1;
}



int FETReconstruction::
WriteAscii(FILE *fp) const
{
  // Write header
  int dummy = 0;
  int major_version = 0;
  int minor_version = 1;
  fprintf(fp, "H ");
  fprintf(fp, "%d ", major_version);
  fprintf(fp, "%d   ", minor_version);
  fprintf(fp, "%d ", NShapes());
  fprintf(fp, "%d ", NMatches());
  fprintf(fp, "%d ", NFeatures());
  fprintf(fp, "%d   ", NCorrespondences());
  for (int i = 0; i < 4; i++) fprintf(fp, "%d ", dummy);
  fprintf(fp, "\n");

  // Write parameters
  fprintf(fp, "X ");
  FETReconstruction *tmp = (FETReconstruction *) this;
  if (avg_feature_radius <= 0) tmp->InitializeFeatureParameters();
  fprintf(fp, "%g ", avg_feature_radius);
  for (int i = 0; i < 29; i++) fprintf(fp, "%d ", dummy);
  fprintf(fp, "\n");

  // Write shapes
  for (int i = 0; i < NShapes(); i++) {
    FETShape *shape = Shape(i);
    fprintf(fp, "S ");
    if (!shape->WriteAscii(fp)) return 0;
    fprintf(fp, "\n");
  }

  // Write matches
  for (int i = 0; i < NMatches(); i++) {
    FETMatch *match = Match(i);
    fprintf(fp, "M ");
    if (!match->WriteAscii(fp)) return 0;
    fprintf(fp, "\n");
  }

  // Write features
  for (int i = 0; i < NFeatures(); i++) {
    FETFeature *feature = Feature(i);
    fprintf(fp, "F ");
    if (!feature->WriteAscii(fp)) return 0;
    fprintf(fp, "\n");
  }

   // Write correspondences
  for (int i = 0; i < NCorrespondences(); i++) {
    FETCorrespondence *correspondence = Correspondence(i);
    fprintf(fp, "C ");
    if (!correspondence->WriteAscii(fp)) return 0;
    fprintf(fp, "\n");
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Binary Input and Output
////////////////////////////////////////////////////////////////////////

int FETReconstruction::
ReadBinaryFile(const char *filename)
{
  // Check filename
  if (!filename) return 0;

  // Open file
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open reconstruction file %s\n", filename);
    return 0;
  }

  // Read file contents
  if (!ReadBinary(fp)) return 0;

  // Close file
  fclose(fp);

  // Return success
  return 1;
}

  

int FETReconstruction::
WriteBinaryFile(const char *filename) const
{
  // Check filename
  if (!filename) return 0;

  // Open file
  FILE *fp = fopen(filename, "wb");
  if (!fp) {
    fprintf(stderr, "Unable to open %s\n", filename);
    return 0;
  }

  // Read file contents
  if (!WriteBinary(fp)) return 0;

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



int FETReconstruction::
ReadBinary(FILE *fp)
{
  // Read header
  int major_version, minor_version;
  fread(&major_version, sizeof(int), 1, fp);
  fread(&minor_version, sizeof(int), 1, fp);
  if ((major_version != 0) || (minor_version != 1)) {
    fprintf(stderr, "Unrecognized version %d %d\n", major_version, minor_version);
    return 0;
  }

  // Read header
  int nshapes, nmatches, nfeatures, ncorrespondences, dummy;
  fread(&nshapes, sizeof(int), 1, fp);
  fread(&nmatches, sizeof(int), 1, fp);
  fread(&nfeatures, sizeof(int), 1, fp);
  fread(&ncorrespondences, sizeof(int), 1, fp);
  for (int i = 0; i < 16; i++) fread(&dummy, sizeof(int), 1, fp);

  // Read parameters
  fread(&avg_feature_radius, sizeof(RNLength), 1, fp);
  for (int i = 0; i < 24; i++) fread(&dummy, sizeof(int), 1, fp);
  for (int i = 0; i < 127; i++) fread(&dummy, sizeof(int), 1, fp);

  // Read shapes
  for (int i = 0; i < nshapes; i++) {
    FETShape *shape = new FETShape(this);
    if (!shape->ReadBinary(fp)) return 0;
  }

  // Read matches
  for (int i = 0; i < nmatches; i++) {
    FETMatch *match = new FETMatch(this);
    if (!match->ReadBinary(fp)) return 0;
  }

  // Read features
  for (int i = 0; i < nfeatures; i++) {
    FETFeature *feature = new FETFeature(this);
    if (!feature->ReadBinary(fp)) return 0;
  }

  // Read correspondences
  for (int i = 0; i < ncorrespondences; i++) {
    FETCorrespondence *correspondence = new FETCorrespondence(this);
    if (!correspondence->ReadBinary(fp)) return 0;
  }

  // Update parameters
  if (avg_feature_radius <= 0) InitializeFeatureParameters();
  InitializeCorrespondenceParameters();
  InitializeOptimizationParameters();

  // Return success
  return 1;
}

  

int FETReconstruction::
WriteBinary(FILE *fp) const
{
  // Write header
  int dummy = 0;
  int major_version = 0;
  int minor_version = 1;
  int nshapes = NShapes();
  int nmatches = NMatches();
  int nfeatures = NFeatures();
  int ncorrespondences = NCorrespondences();
  fwrite(&major_version, sizeof(int), 1, fp);
  fwrite(&minor_version, sizeof(int), 1, fp);
  fwrite(&nshapes, sizeof(int), 1, fp);
  fwrite(&nmatches, sizeof(int), 1, fp);
  fwrite(&nfeatures, sizeof(int), 1, fp);
  fwrite(&ncorrespondences, sizeof(int), 1, fp);
  for (int i = 0; i < 16; i++) fwrite(&dummy, sizeof(int), 1, fp);

  // Write parameters
  FETReconstruction *tmp = (FETReconstruction *) this;
  if (avg_feature_radius <= 0) tmp->InitializeFeatureParameters();
  fwrite(&avg_feature_radius, sizeof(RNLength), 1, fp);
  for (int i = 0; i < 24; i++) fwrite(&dummy, sizeof(int), 1, fp);
  for (int i = 0; i < 127; i++) fwrite(&dummy, sizeof(int), 1, fp);

  // Write shapes
  for (int i = 0; i < nshapes; i++) {
    FETShape *shape = Shape(i);
    if (!shape->WriteBinary(fp)) return 0;
  }

  // Write matches
  for (int i = 0; i < nmatches; i++) {
    FETMatch *match = Match(i);
    if (!match->WriteBinary(fp)) return 0;
  }

  // Write features
  for (int i = 0; i < nfeatures; i++) {
    FETFeature *feature = Feature(i);
    if (!feature->WriteBinary(fp)) return 0;
  }

   // Write correspondences
  for (int i = 0; i < ncorrespondences; i++) {
    FETCorrespondence *correspondence = Correspondence(i);
    if (!correspondence->WriteBinary(fp)) return 0;
  }

  // Return success
  return 1;
}

  

////////////////////////////////////////////////////////////////////////
// Update functions
////////////////////////////////////////////////////////////////////////

void FETReconstruction::
InvalidateBBox(void)
{
  // Invalidate bbox
  bbox.Reset(R3Point(FLT_MAX, FLT_MAX, FLT_MAX), R3Point(-FLT_MAX, -FLT_MAX, -FLT_MAX));
}



void FETReconstruction::
UpdateBBox(void)
{
  // Initialize bbox
  bbox = R3null_box;

  // Update bounding box from shapes
  for (int i = 0; i < NShapes(); i++) {
    FETShape *shape = Shape(i);
    bbox.Union(shape->BBox());
  }

  // Update bounding box from features
  for (int i = 0; i < NFeatures(); i++) {
    FETFeature *feature = Feature(i);
    bbox.Union(feature->Position(TRUE));
  }
}



////////////////////////////////////////////////////////////////////////
// Correspondence creation
////////////////////////////////////////////////////////////////////////

void FETReconstruction::
CreateCorrespondences(FETShape *shape1, FETShape *shape2, RNScalar num_correspondences)
{
  // Check number of correspondences
  if (num_correspondences == 0) return;

  // Check bounding box distance
  if (max_euclidean_distance > 0) {
    R3Box bbox1 = shape1->BBox();
    R3Box bbox2 = shape2->BBox();
    if (R3Distance(bbox1, bbox2) > max_euclidean_distance) return;
  }

  // Find query features
  RNScalar total_salience = 0;
  RNArray<FETFeature *> query_features2;
  for (int i = 0; i < shape2->NFeatures(); i++) {
    FETFeature *feature2 = shape2->Feature(i);
    if ((min_salience > 0) && (feature2->Salience() < min_salience)) continue;
    if ((min_distinction > 0) && (feature2->Distinction() < min_distinction)) continue;
    if ((feature2->generator_type >= 0) && (total_correspondence_weights[feature2->generator_type] <= 0)) continue;
    if (feature2->IsOnBoundary() && (!feature2->IsOnSilhouetteBoundary())) continue;
    query_features2.Insert(feature2);
    total_salience += feature2->Salience();
  }
  
  // Compute correspondences for shape2 -> shape1
  for (int i = 0; i < query_features2.NEntries(); i++) {
    FETFeature *feature2 = query_features2.Kth(i);

    // Subsample features randomly
    if ((num_correspondences != RN_UNKNOWN) && (total_salience > 0)) {
      const RNScalar expected_inlier_probability = 0.1;
      RNScalar p = num_correspondences * feature2->Salience() / total_salience;
      p /= expected_inlier_probability;
      if ((p < 1.0) && (RNRandomScalar() > p)) continue;
    }

    // Find closest feature on shape1
    FETFeature *feature1 = shape1->FindClosestFeature(feature2, shape2->Transformation(), 
      0.0, max_euclidean_distance, max_descriptor_distances, max_normal_angle,
      min_distinction, min_salience, discard_boundaries);
    if (!feature1) continue;

    // Check if should discard because on boundary
    if (discard_boundaries && feature1->IsOnBoundary()) {  
      if (!feature1->IsOnSilhouetteBoundary() || !feature2->IsOnSilhouetteBoundary()) {
        continue;
      }
    }

    // Compute/check affinity
    RNScalar affinity = Affinity(feature1, feature2);
    if (affinity <= 0) continue;

    // Check if mutually closest
    if (discard_not_mutually_closest) {
      FETFeature *feature2a = shape2->FindClosestFeature(feature1, shape1->Transformation(), 
        0.0, max_euclidean_distance, max_descriptor_distances, max_normal_angle,
        min_distinction, min_salience, discard_boundaries);
      if (feature2a != feature2) continue;
    }

    // Create correspondence with sorted shapes
    if (shape1->reconstruction_index < shape2->reconstruction_index) {
      FETCorrespondence *correspondence = new FETCorrespondence(this, feature1, feature2, affinity);
      if (!correspondence) RNAbort("Unable to create correspondence");
    }
    else {
      FETCorrespondence *correspondence = new FETCorrespondence(this, feature2, feature1, affinity);
      if (!correspondence) RNAbort("Unable to create correspondence");
    }
  }
}



void FETReconstruction::
CreateCorrespondences(void)
{
  // Determine total saliences
  RNScalar *shape_saliences = new RNScalar [ NShapes() ];
  for (int i = 0; i < NShapes(); i++) {
    FETShape *shape = Shape(i);
    shape_saliences[i] = 0.0;
    for (int j = 0; j < shape->NFeatures(); j++) {
      FETFeature *feature = shape->Feature(j);
      shape_saliences[i] += feature->Salience();
    }
  }
  
  // Determine allocations of correspondences to shape/shape pairs
  R3Box intersection;
  RNScalar total_allocation = 0;
  if (max_correspondences > 0) {
    for (int i = 0; i < NShapes(); i++) {
      FETShape *shape1 = Shape(i);
      R3Box bbox1 = shape1->BBox();
      bbox1.Inflate(max_euclidean_distance);
      for (int j = 0; j < NShapes(); j++) {
        FETShape *shape2 = Shape(j);
        if (shape1 == shape2) continue;
        const R3Box& bbox2 = shape2->BBox();
        if (R3Intersects(bbox1, bbox2, &intersection)) {
          RNScalar allocation = shape_saliences[i] * shape_saliences[j] * intersection.Volume();
          total_allocation += allocation;
        }
      }
    }
  }

  // Create correspondences 
  for (int i = 0; i < NShapes(); i++) {
    FETShape *shape1 = Shape(i);
    R3Box bbox1 = shape1->BBox();
    bbox1.Inflate(max_euclidean_distance);
    for (int j = 0; j < NShapes(); j++) {
      FETShape *shape2 = Shape(j);
      if (shape1 == shape2) continue;
      const R3Box& bbox2 = shape2->BBox();
      if (R3Intersects(bbox1, bbox2, &intersection)) {
        RNScalar allocation = shape_saliences[i] * shape_saliences[j] * intersection.Volume();
        RNScalar num_correspondences = (total_allocation > 0) ? max_correspondences * allocation / total_allocation : RN_UNKNOWN;
        CreateCorrespondences(shape1, shape2, num_correspondences);
      }
    }
  }

  // Discard outliers
  if (discard_outliers) DiscardOutlierCorrespondences();

  // Select correspondences
  SelectCorrespondences(max_correspondences);
}

   

static void 
InternalDiscardOutliers(RNArray<FETCorrespondence *>& correspondences, RNScalar max_zscore)
{
  // Check correspondences
  if (max_zscore >= 10) return;
  if (correspondences.NEntries() < 5) return;

  // Allocate distances
  RNScalar *distances = new RNScalar [ correspondences.NEntries() ];

  // Sum distances
  RNScalar sum = 0;
  for (int i = 0; i < correspondences.NEntries(); i++) {
    FETCorrespondence *correspondence = correspondences.Kth(i);
    distances[i] = correspondence->EuclideanDistance();
    sum += distances[i];
  }

  // Compute mean
  RNScalar mean = sum / correspondences.NEntries();

  // Sum squared residuals
  RNScalar rss = 0;
  for (int i = 0; i < correspondences.NEntries(); i++) {
    RNScalar delta = distances[i] - mean;
    rss += delta * delta;
  }

  // Compute minimum affinity
  RNScalar stddev = sqrt(rss/correspondences.NEntries());
  RNScalar max_distance = mean + max_zscore * stddev;

  // Build array of outliers
  RNArray<FETCorrespondence *> outliers;
  for (int i = 0; i < correspondences.NEntries(); i++) {
    FETCorrespondence *correspondence = correspondences.Kth(i);
    if (distances[i] > max_distance) outliers.Insert(correspondence);
  }

  // Delete outliers
  for (int i = 0; i < outliers.NEntries(); i++) {
    FETCorrespondence *correspondence = outliers.Kth(i);
    delete correspondence;
  }

  // Delete distances
  delete [] distances;
}



void FETReconstruction::
DiscardOutlierCorrespondences(RNScalar max_zscore, int max_iterations, int min_correspondences)
{
  // Iteratively discard correspondences until convergence
  for (int i = 0; i < max_iterations; i++) {
    if (correspondences.NEntries() <= min_correspondences) break;
    int n = correspondences.NEntries();
    InternalDiscardOutliers(correspondences, max_zscore);
    if (correspondences.NEntries() >= n) break;
  }
}



void FETReconstruction::
SelectCorrespondences(int num_correspondences)
{
  // Check trivial conditions
  if (num_correspondences < 0) return;
  if (num_correspondences == 0) { TruncateCorrespondences(0); return; }
  if (NCorrespondences() == 0) return;

  // Delete correspondences randomly for now
  while (NCorrespondences() > num_correspondences) {
    int index = (int) (RNRandomScalar() * NCorrespondences());
    FETCorrespondence *correspondence = Correspondence(index);
    delete correspondence;
  }
}



void FETReconstruction::
TruncateCorrespondences(int num_correspondences)
{
  // Iteratively discard correspondences
  while (NCorrespondences() > num_correspondences) {
    FETCorrespondence *correspondence = Correspondence(NCorrespondences()-1);
    delete correspondence;
  }
}





////////////////////////////////////////////////////////////////////////
// Transformation optimization 
////////////////////////////////////////////////////////////////////////

static R3Affine
AligningTransformation(const RNArray<FETCorrespondence *>& correspondences,
  int translation = TRUE, int rotation = TRUE, int scale = FALSE)
{
  // Check the correspondences
  if (correspondences.NEntries() == 0) return R3identity_affine;

  // Create temporary array of positions1
  RNArray<R3Point *> positions1;
  for (int i = 0; i < correspondences.NEntries(); i++) {
    positions1.Insert(&(correspondences.Kth(i)->features[0]->position));
  }

  // Create temporary array of positions2
  RNArray<R3Point *> positions2;
  for (int i = 0; i < correspondences.NEntries(); i++) {
    positions2.Insert(&(correspondences.Kth(i)->features[1]->position));
  }

  // Create temporary array of weights
  RNScalar total_weight = 0;
  RNScalar *weights = new RNScalar [ correspondences.NEntries() ];
  for (int i = 0; i < correspondences.NEntries(); i++) {
    FETCorrespondence *correspondence = correspondences.Kth(i);
    weights[i] = correspondence->affinity;
    if (weights[i] == 0) weights[i] = RN_EPSILON;
    total_weight += weights[i];
  }

  // Compute aligning transformation
  R3Affine affine21;
  if (total_weight == 0) affine21 = R3identity_affine;
  else affine21 = R3AlignPoints(positions1, positions2, NULL /* weights */, translation, rotation, scale);

  // Delete weights
  delete [] weights;

  // Return aligning transformation
  return affine21;
}



void FETReconstruction::
AddPointPointCorrespondenceEquations(RNSystemOfEquations *system, 
  FETShape *shape1, FETShape *shape2, 
  const R3Point& position1, const R3Point& position2, 
  RNScalar w)
{
  // Get/check weight
  if (w <= 0) return;

  // Get point coordinates
  RNAlgebraic *px1, *py1, *pz1, *px2, *py2, *pz2;
  shape1->ComputeTransformedPointCoordinates(position1, px1, py1, pz1);
  shape2->ComputeTransformedPointCoordinates(position2, px2, py2, pz2);

  // Add equation representing differences between point coordinates
  RNAlgebraic *ex = new RNAlgebraic(RN_SUBTRACT_OPERATION, px2, px1);
  RNAlgebraic *ey = new RNAlgebraic(RN_SUBTRACT_OPERATION, py2, py1);
  RNAlgebraic *ez = new RNAlgebraic(RN_SUBTRACT_OPERATION, pz2, pz1);

  // Multiply by weight
  ex->Multiply(w);
  ey->Multiply(w);
  ez->Multiply(w);

  // Insert into system of equation
  system->InsertEquation(ex, w * max_euclidean_distance);
  system->InsertEquation(ey, w * max_euclidean_distance);
  system->InsertEquation(ez, w * max_euclidean_distance);
}



void FETReconstruction::
AddPointLineCorrespondenceEquations(RNSystemOfEquations *system, 
  FETShape *shape1, FETShape *shape2, 
  const R3Point& position1, const R3Point& position2, 
  const R3Vector& direction2,
  RNScalar w)
{
  // Get/check weight
  if (w <= 0) return;

  // Add point-plane equations
  int dim = direction2.MinDimension();
  R3Vector n2A = direction2 % R3xyz_triad[dim]; n2A.Normalize();
  R3Vector n2B = direction2 % n2A; n2B.Normalize();
  AddPointPlaneCorrespondenceEquations(system, shape1, shape2, position1, position2, n2A, 0.5 * w);
  AddPointPlaneCorrespondenceEquations(system, shape1, shape2, position1, position2, n2B, 0.5 * w);
}



void FETReconstruction::
AddPointPlaneCorrespondenceEquations(RNSystemOfEquations *system, 
  FETShape *shape1, FETShape *shape2, 
  const R3Point& position1, const R3Point& position2, 
  const R3Vector& normal2,
  RNScalar w)
{
  // Get/check weight
  if (w <= 0) return;

  // Get point coordinates
  RNAlgebraic *px1, *py1, *pz1, *px2, *py2, *pz2;
  shape1->ComputeTransformedPointCoordinates(position1, px1, py1, pz1);
  shape2->ComputeTransformedPointCoordinates(position2, px2, py2, pz2);

  // Get normal coordinates
  RNAlgebraic *nx2, *ny2, *nz2;
  shape2->ComputeTransformedVectorCoordinates(normal2, nx2, ny2, nz2);

  // Add equation representing point1-plane2 distance
  RNAlgebraic *dx, *dy, *dz, *d;
  dx = new RNAlgebraic(RN_SUBTRACT_OPERATION, px1, px2);
  dx = new RNAlgebraic(RN_MULTIPLY_OPERATION, dx, nx2);
  dy = new RNAlgebraic(RN_SUBTRACT_OPERATION, py1, py2);
  dy = new RNAlgebraic(RN_MULTIPLY_OPERATION, dy, ny2);
  dz = new RNAlgebraic(RN_SUBTRACT_OPERATION, pz1, pz2);
  dz = new RNAlgebraic(RN_MULTIPLY_OPERATION, dz, nz2);
  d = new RNAlgebraic(RN_ADD_OPERATION, dx, dy);
  d = new RNAlgebraic(RN_ADD_OPERATION, d, dz);
  d = new RNAlgebraic(RN_MULTIPLY_OPERATION, d, w);
  system->InsertEquation(d, w * max_euclidean_distance);
}



void FETReconstruction::
AddParallelVectorEquations(RNSystemOfEquations *system, 
  FETShape *shape1, FETShape *shape2, 
  const R3Vector& vector1, const R3Vector& vector2, 
  RNScalar w)
{
  // Get/check weight
  if (w <= 0) return;

  // Get point coordinates
  RNAlgebraic *vx1, *vy1, *vz1, *vx2, *vy2, *vz2;
  shape1->ComputeTransformedVectorCoordinates(vector1, vx1, vy1, vz1);
  shape2->ComputeTransformedVectorCoordinates(vector2, vx2, vy2, vz2);

  // Add equation representing differences between point coordinates
  RNAlgebraic *ex = new RNAlgebraic(RN_SUBTRACT_OPERATION, vx2, vx1);
  RNAlgebraic *ey = new RNAlgebraic(RN_SUBTRACT_OPERATION, vy2, vy1);
  RNAlgebraic *ez = new RNAlgebraic(RN_SUBTRACT_OPERATION, vz2, vz1);

  // Multiply by weight
  ex->Multiply(w);
  ey->Multiply(w);
  ez->Multiply(w);

  // Insert into system of equation
  system->InsertEquation(ex, w);
  system->InsertEquation(ey, w);
  system->InsertEquation(ez, w);
}



void FETReconstruction::
AddPerpendicularVectorEquations(RNSystemOfEquations *system, 
  FETShape *shape1, FETShape *shape2, 
  const R3Vector& vector1, const R3Vector& vector2, 
  RNScalar w)
{
  // Get/check weight
  if (w <= 0) return;

  // Get point coordinates
  RNAlgebraic *vx1, *vy1, *vz1, *vx2, *vy2, *vz2;
  shape1->ComputeTransformedVectorCoordinates(vector1, vx1, vy1, vz1);
  shape2->ComputeTransformedVectorCoordinates(vector2, vx2, vy2, vz2);

  // Add equation representing dot product between vectors
  RNAlgebraic *dx = new RNAlgebraic(RN_MULTIPLY_OPERATION, vx2, vx1);
  RNAlgebraic *dy = new RNAlgebraic(RN_MULTIPLY_OPERATION, vy2, vy1);
  RNAlgebraic *dz = new RNAlgebraic(RN_MULTIPLY_OPERATION, vz2, vz1);
  RNAlgebraic *e = new RNAlgebraic(RN_ADD_OPERATION, dx, dy);
  e = new RNAlgebraic(RN_ADD_OPERATION, e, dz);
  e->Multiply(w);

  // Insert into system of equation
  system->InsertEquation(e, w);
}



void FETReconstruction::
AddPairwiseTransformationEquations(RNSystemOfEquations *system,
  FETShape *shape1, FETShape *shape2, 
  const R3Affine& transformation21, RNScalar w)
{
  // Get/check weight
  if (w <= 0) return;

  // Get radius of bounding box
  RNBoolean tmp = shape2->BBox().IsEmpty();
  RNLength r = (tmp) ? 1.0 : shape2->BBox().DiagonalRadius();

  // Add equations to align shapes transformed by transformation21
  for (int i1 = -1; i1 <= 1; i1 += 2) {
    for (int i2 = -1; i2 <= 1; i2 += 2) {
      for (int i3 = -1; i3 <= 1; i3 += 2) {
        R3Point position2 = shape2->Centroid();
        position2[0] += i1*r; position2[1] += i2*r; position2[2] += i3*r;
        R3Point position1 = position2; position1.Transform(transformation21);  
        AddPointPointCorrespondenceEquations(system, shape1, shape2, position1, position2, w / 8.0);
      }
    }
  }
}



void FETReconstruction::
AddInertiaEquations(RNSystemOfEquations *system, RNScalar total_weight)
{
  // Check total weight
  if (total_weight <= 0) return;
  if (NShapes() == 0) return;

  // Compute total inertia
  RNScalar total_inertia = 0;
  for (int i = 0; i < NShapes(); i++) {
    FETShape *shape = Shape(i);
    for (int j = 0; j < shape->NVariables(); j++) {
      if (shape->variable_index[j] < 0) continue;
      if (shape->variable_inertias[j] >= RN_INFINITY) continue;
      total_inertia += shape->variable_inertias[j];
    }
  }

  // Determine factor to compute weight per shape
  if (total_inertia <= 0) return;
  RNScalar w = total_weight / total_inertia; 

  // Add inertia equations for each shape
  for (int i = 0; i < NShapes(); i++) {
    FETShape *shape = Shape(i);
    for (int j = 0; j < shape->NVariables(); j++) {
      if (shape->variable_index[j] < 0) continue;
      if (shape->variable_inertias[j] >= RN_INFINITY) continue;
      RNPolynomial *v = new RNPolynomial(w * shape->variable_inertias[j], shape->variable_index[j], 1.0);
      system->InsertEquation(v);
    }
  }
}



void FETReconstruction::
AddCorrespondenceEquations(RNSystemOfEquations *system, FETCorrespondence *correspondence, RNScalar weight)
{
  // Check total weight
  if (weight <= 0) return;

  // Get useful variables
  FETFeature *feature1 = correspondence->Feature(0);
  FETFeature *feature2 = correspondence->Feature(1);
  if (!feature1 || !feature2) return;
  FETShape *shape1 = feature1->shape;
  FETShape *shape2 = feature2->shape;
  if (!shape1 || !shape2) return;

  // Add equations for correspondence
  if (correspondence->relationship_type == COINCIDENT_RELATIONSHIP) {
    if ((feature1->shape_type == POINT_FEATURE_SHAPE) && (feature2->shape_type == POINT_FEATURE_SHAPE)) {
       AddPointPointCorrespondenceEquations(system, shape1, shape2, feature1->Position(), feature2->Position(), weight);
    }
    else {
      if (feature1->shape_type == LINE_FEATURE_SHAPE) {        
        AddPointLineCorrespondenceEquations(system, shape2, shape1, feature2->Position(), feature1->Position(), feature1->Direction(), weight);
      }
      else if (feature1->shape_type == PLANE_FEATURE_SHAPE) {        
        AddPointPlaneCorrespondenceEquations(system, shape2, shape1, feature2->Position(), feature1->Position(), feature1->Normal(), weight);
      }

      if (feature2->shape_type == LINE_FEATURE_SHAPE) {        
        AddPointLineCorrespondenceEquations(system, shape1, shape2, feature1->Position(), feature2->Position(), feature2->Direction(), weight);
      }
      else if (feature2->shape_type == PLANE_FEATURE_SHAPE) {        
        AddPointPlaneCorrespondenceEquations(system, shape1, shape2, feature1->Position(), feature2->Position(), feature2->Normal(), weight);
      }
    }
  }

  if (correspondence->relationship_type == PARALLEL_RELATIONSHIP) {
    if (feature1->shape_type == LINE_FEATURE_SHAPE) {
      if (feature2->shape_type == LINE_FEATURE_SHAPE) {
        AddParallelVectorEquations(system, shape1, shape2, feature1->Direction(), feature2->Direction(), weight);
      }
      else if (feature2->shape_type == PLANE_FEATURE_SHAPE) {
        AddPerpendicularVectorEquations(system, shape1, shape2, feature1->Direction(), feature2->Normal(), weight);
      }
    }
    else if (feature1->shape_type == PLANE_FEATURE_SHAPE) {
      if (feature2->shape_type == LINE_FEATURE_SHAPE) {
        AddPerpendicularVectorEquations(system, shape1, shape2, feature1->Normal(), feature2->Direction(), weight);
      }
      else if (feature2->shape_type == PLANE_FEATURE_SHAPE) {
        AddParallelVectorEquations(system, shape1, shape2, feature1->Normal(), feature2->Normal(), weight);
      }
    }
  }
  else if (correspondence->relationship_type == ANTIPARALLEL_RELATIONSHIP) {
    if (feature1->shape_type == LINE_FEATURE_SHAPE) {
      if (feature2->shape_type == LINE_FEATURE_SHAPE) {
        AddParallelVectorEquations(system, shape1, shape2, feature1->Direction(), -feature2->Direction(), weight);
      }
      else if (feature2->shape_type == PLANE_FEATURE_SHAPE) {
        AddPerpendicularVectorEquations(system, shape1, shape2, feature1->Direction(), feature2->Normal(), weight);
      }
    }
    else if (feature1->shape_type == PLANE_FEATURE_SHAPE) {
      if (feature2->shape_type == LINE_FEATURE_SHAPE) {
        AddPerpendicularVectorEquations(system, shape1, shape2, feature1->Normal(), feature2->Direction(), weight);
      }
      else if (feature2->shape_type == PLANE_FEATURE_SHAPE) {
        AddParallelVectorEquations(system, shape1, shape2, feature1->Normal(), -feature2->Normal(), weight);
      }
    }
  }
  else if (correspondence->relationship_type == PERPENDICULAR_RELATIONSHIP) {
    if (feature1->shape_type == LINE_FEATURE_SHAPE) {
      if (feature2->shape_type == LINE_FEATURE_SHAPE) {
        AddPerpendicularVectorEquations(system, shape1, shape2, feature1->Direction(), feature2->Direction(), weight);
      }
      else if (feature2->shape_type == PLANE_FEATURE_SHAPE) {
        AddParallelVectorEquations(system, shape1, shape2, feature1->Direction(), feature2->Normal(), weight);
      }
    }
    else if (feature1->shape_type == PLANE_FEATURE_SHAPE) {
      if (feature2->shape_type == LINE_FEATURE_SHAPE) {
        AddParallelVectorEquations(system, shape1, shape2, feature1->Normal(), feature2->Direction(), weight);
      }
      else if (feature2->shape_type == PLANE_FEATURE_SHAPE) {
        AddPerpendicularVectorEquations(system, shape1, shape2, feature1->Normal(), feature2->Normal(), weight);
      }
    }
  }
}



void FETReconstruction::
AddCorrespondenceEquations(RNSystemOfEquations *system, RNScalar *total_weights)
{
  // Check total weight
  if (correspondences.NEntries() == 0) return;

  // Compute total affinities
  RNScalar total_affinities[NUM_FEATURE_TYPES] = { 0.0 };
  for (int i = 0; i < correspondences.NEntries(); i++) {
    FETCorrespondence *correspondence = correspondences.Kth(i);
    FETFeature *feature1 = correspondence->Feature(0);
    FETFeature *feature2 = correspondence->Feature(1);
    if (!feature1 || !feature2) continue;
    RNScalar affinity = correspondence->affinity;
    if (affinity == 0) continue;
    int generator_type = feature1->GeneratorType();
    total_affinities[generator_type] += affinity;
  }

  // Determine factor to compute weight per correspondence
  RNScalar factors[NUM_FEATURE_TYPES] = { 0.0 };
  for (int i = 0; i < NUM_FEATURE_TYPES; i++) {
    if (total_weights[i] == 0) continue;
    if (total_affinities[i] <= 0) continue;
    factors[i] = total_weights[i] / total_affinities[i];
  }

  // Add equations for each correspondence
  for (int i = 0; i < correspondences.NEntries(); i++) {
    FETCorrespondence *correspondence = correspondences.Kth(i);
    FETFeature *feature1 = correspondence->Feature(0);
    FETFeature *feature2 = correspondence->Feature(1);
    if (!feature1 || !feature2) continue;
    RNScalar affinity = correspondence->affinity;
    if (affinity == 0) continue;
    int generator_type = feature1->GeneratorType();
    RNScalar weight = factors[generator_type] * affinity;
    AddCorrespondenceEquations(system, correspondence, weight);
  }
}



void FETReconstruction::
AddMatchEquations(RNSystemOfEquations *system, RNScalar total_weight)
{
  // Check total weight
  if (total_weight <= 0) return;
  if (NMatches() == 0) return;

  // Compute total score
  RNScalar total_score = 0;
  for (int i = 0; i <  NMatches(); i++) {
    // Match *match = Match(i);
    RNScalar score = 1.0; // match->Score(); TOO SLOW
    total_score += score;
  }

  // Determine factor to compute weight per child
  if (total_score <= 0) return;
  RNScalar w = total_weight / total_score; 

  // Add equations for each match
  for (int i = 0; i < NMatches(); i++) {
    FETMatch *match = Match(i);
    RNScalar score = 1.0; // match->Score(); TOO SLOW
    if (score == 0) continue;
    FETShape *shape1 = match->Shape(0);
    FETShape *shape2 = match->Shape(1);
    R3Affine transformation21 = match->Transformation();
    AddPairwiseTransformationEquations(system, shape1, shape2, transformation21, w * score);
  }
}



void FETReconstruction::
AddTrajectoryEquations(RNSystemOfEquations *system, RNScalar total_weight, RNScalar sigma)
{
  // Check total weight
  if (total_weight <= 0) return;
  if (NShapes() == 0) return;
  if (sigma <= 0) return;

  // Get convenient variables
  RNScalar min_affinity = 1E-3;
  RNScalar f = -1.0 / (2.0 * sigma * sigma);
  
  // Compute trajectory parameterization
  RNLength *parameterization = new RNLength [ NShapes() ];
  parameterization[0] = 0; 
  for (int i = 1; i < NShapes(); i++) {
    FETShape *shape0 = Shape(i-1);
    FETShape *shape1 = Shape(i);
    const R3Point& origin0 = shape0->Origin();
    const R3Point& origin1 = shape1->Origin();
    RNLength d = R3Distance(origin0, origin1);
    parameterization[i] = parameterization[i-1] + d;
  }

  // Determine total affinity
  RNScalar total_affinity = 0;
  for (int i = 0; i < NShapes(); i++) {
    for (int j = 1; j <= 16; j *= 2) {
      if (i-j >= 0) {
        RNScalar d = parameterization[i] - parameterization[i-j];
        RNScalar affinity = exp(f*d*d);
        if (affinity < min_affinity) continue;
        total_affinity += affinity;
      }
      if (i+j < NShapes()) {
        RNScalar d = parameterization[i+j] - parameterization[i];
        RNScalar affinity = exp(f*d*d);
        if (affinity < min_affinity) continue;
        total_affinity += affinity;
      }
    }
  }

  // Determine factor to compute weight per correspondence
  if (total_affinity > 0) {
    // Determine weighting scale factor
    RNScalar w = total_weight / total_affinity; 

    // Add equations
    for (int i = 0; i < NShapes(); i++) {
      FETShape *shape1 = Shape(i);
      for (int j = 1; j <= 16; j *= 2) {
        if (i-j >= 0) {
          RNScalar d = parameterization[i] - parameterization[i-j];
          RNScalar affinity = exp(f*d*d);
          if (affinity < min_affinity) continue;
          FETShape *shape2 = Shape(i-j);
          R3Affine transformation21 = R3identity_affine;
          transformation21.InverseTransform(shape1->initial_transformation);
          transformation21.Transform(shape2->initial_transformation);
          AddPairwiseTransformationEquations(system, shape1, shape2, transformation21, w * affinity);
        }
        if (i+j < NShapes()) {
          RNScalar d = parameterization[i+j] - parameterization[i];
          RNScalar affinity = exp(f*d*d);
          if (affinity < min_affinity) continue;
          FETShape *shape2 = Shape(i+j);
          R3Affine transformation21 = R3identity_affine;
          transformation21.InverseTransform(shape1->initial_transformation);
          transformation21.Transform(shape2->initial_transformation);
          AddPairwiseTransformationEquations(system, shape1, shape2, transformation21, w * affinity);
        }
      }
    }
  }
  
  // Delete parameterization
  delete [] parameterization;
}



void FETReconstruction::
OptimizeTransformationsWithClosedFormEquations(void)
{
  // Check the correspondences
  if (NShapes() != 2) return;
  if (correspondences.NEntries() == 0) return;

  // Compute the aligning transformation
  R3Affine transformation21 = AligningTransformation(correspondences);

  // Update transformation for second shape
  FETShape *shape1 = Shape(0);
  FETShape *shape2 = Shape(1);
  shape2->current_transformation = R3identity_affine;
  shape2->current_transformation.Transform(shape1->current_transformation);
  shape2->current_transformation.Transform(transformation21);
}



void FETReconstruction::
OptimizeTransformationsWithGlobalRelaxation(void)
{
#if 1
  // ELENA: COMMENT THIS OUT 
  if (NShapes() < 2) return;
  FETShape *shape0 = Shape(0);
  shape0->SetTransformation(R3identity_affine);
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < NMatches(); j++) {
      FETMatch *match = Match(j);
      FETShape *shape1 = match->Shape(0);
      FETShape *shape2 = match->Shape(1);
      R3Affine transformation21 = match->Transformation();
      if (shape1->reconstruction_index < shape2->reconstruction_index) {
        R3Affine transformation2 = R3identity_affine;
        transformation2.InverseTransform(shape1->Transformation());
        transformation2.Transform(transformation21);
        shape2->SetTransformation(transformation2);
      }
      else {
        R3Affine transformation1 = R3identity_affine;
        transformation1.Transform(shape2->Transformation());
        transformation1.InverseTransform(transformation21);
        shape1->SetTransformation(transformation1);
      }
    }
  }
#else
#endif
}



////////////////////////////////////////////////////////////////////////
// ICP alignment
////////////////////////////////////////////////////////////////////////

static R3Affine
ComputeTransformationWithICP(FETReconstruction *reconstruction1, FETShape *shape1A, FETShape *shape1B, int max_iterations)
{
  // Create a new reconstruction
  FETReconstruction *reconstruction2 = new FETReconstruction(*reconstruction1);
  reconstruction2->InitializeOptimizationParameters();

  // Copy shape1A
  FETShape *shape2A = new FETShape(*shape1A);
  reconstruction2->InsertShape(shape2A);
  for (int i = 0; i < shape1A->NFeatures(); i++) {
    FETFeature *feature1A = shape1A->Feature(i);
    FETFeature *feature2A = new FETFeature(*feature1A);
    reconstruction2->InsertFeature(feature2A);
    shape2A->InsertFeature(feature2A);
  }

  // Copy shape1B
  FETShape *shape2B = new FETShape(*shape1B);
  reconstruction2->InsertShape(shape2B);
  for (int i = 0; i < shape1B->NFeatures(); i++) {
    FETFeature *feature1B = shape1B->Feature(i);
    FETFeature *feature2B = new FETFeature(*feature1B);
    reconstruction2->InsertFeature(feature2B);
    shape2B->InsertFeature(feature2B);
  }

  // Optimize transformations with ICP
  reconstruction2->OptimizeTransformationsWithICP(max_iterations);

  // Compute transformation from shape2B to shape2A
  R3Affine transformation = R3null_affine;
  if (reconstruction2->NCorrespondences() > 0) {
    transformation = R3identity_affine;
    transformation.InverseTransform(shape2A->Transformation());
    transformation.Transform(shape2B->Transformation());
  }
 
  // Delete reconstruction
  delete reconstruction2;

  // Return transformation
  return transformation;
}


static FETMatch *
CreateMatchWithICP(FETReconstruction *reconstruction,
  FETShape *shape1, FETShape *shape2,
  int max_iterations = 1)
{
  // Compute transformation with ICP
  R3Affine transformation21 = ComputeTransformationWithICP(reconstruction, shape1, shape2, max_iterations);
  if (transformation21 == R3null_affine) return NULL;
  
  // Allocate match
  FETMatch *match = new FETMatch(reconstruction, shape1, shape2, transformation21);
  assert(match);

  // Return match
  return match;
}



void FETReconstruction::
EmptyMatches(void)
{
  // Delete all matchs
  while (NMatches() > 0) {
    FETMatch *match = Match(NMatches()-1);
    delete match;
  }
}
  


void FETReconstruction::
CreateMatchesWithICP(void)
{
  // Empty matches
  EmptyMatches();

  // Create matches between every pair of shapes
  for (int i = 0; i < NShapes(); i++) {
    FETShape *shape1 = Shape(i);
    for (int j = i+1; j < NShapes(); j++) {
      FETShape *shape2 = Shape(j);
      CreateMatchWithICP(this, shape1, shape2, 1);
    }
  }
}



void FETReconstruction::
OptimizeTransformationsWithICP(int max_iterations,
  int *result_niterations, RNBoolean *result_converged, RNScalar *result_compute_time,
  RNScalar *iteration_times, RNScalar *iteration_errors)
{
  // Start statistics
  RNTime step_time;
  RNScalar step1_time = 0;
  RNScalar step2_time = 0;
  RNScalar step3_time = 0;
  RNScalar total_step1_time = 0;
  RNScalar total_step2_time = 0;
  RNScalar total_step3_time = 0;

  // Get initial number of correspondences
  int saved_ncorrespondences = NCorrespondences();

  // Get initial compatibility thresholds
  RNScalar initial_max_euclidean_distance = max_euclidean_distance;
  RNAngle initial_max_normal_angle = max_normal_angle;

  // Set final compatibility thresholds
  RNScalar final_max_euclidean_distance = 0.1 * max_euclidean_distance;
  RNAngle final_max_normal_angle = 0.5 * max_normal_angle;

  // Initialize reduction factors
  RNScalar max_euclidean_distance_factor = 1.0;
  if ((initial_max_euclidean_distance != RN_UNKNOWN) && (initial_max_euclidean_distance > 0)) {
    max_euclidean_distance = initial_max_euclidean_distance;
    if (final_max_euclidean_distance != RN_UNKNOWN) {
      RNScalar ratio = final_max_euclidean_distance / initial_max_euclidean_distance;
      RNScalar exponent = 1.0 / max_iterations;
      max_euclidean_distance_factor = pow(ratio, exponent);
    }
  }
  RNScalar max_normal_angle_factor = 1.0;
  if ((initial_max_normal_angle != RN_UNKNOWN) && (initial_max_normal_angle > 0)) {
    max_normal_angle = initial_max_normal_angle;
    if (final_max_normal_angle != RN_UNKNOWN) {
      RNScalar ratio = final_max_normal_angle / initial_max_normal_angle;
      RNScalar exponent = 1.0 / max_iterations;
      max_normal_angle_factor = pow(ratio, exponent);
    }
  }

  // Iterate until max_iterations or converged
  int iteration = 0; 
  RNBoolean converged = FALSE;
  RNScalar compute_time = 0;
  while (iteration < max_iterations) {

    // Gather timing stats
    if (result_compute_time) {
      step_time.Read();
    }

    // Create correspondences
    TruncateCorrespondences(saved_ncorrespondences);
    CreateCorrespondences();

    // Create matches
    // EmptyMatches();
    // if (total_match_weight > 0) {
    //   CreateMatchesWithICP();
    // }

    // Check if there is anything to optimize
    if ((NCorrespondences() == 0) && (NMatches() == 0)) break;

    // Gather timing stats
    if (result_compute_time) {
      step1_time = step_time.Elapsed();
      total_step1_time += step1_time;
      compute_time += step1_time;
      step_time.Read();
    }

    // Update stats
    if (iteration_times) {
      if (iteration == 0) iteration_times[iteration] = 0;
      else iteration_times[iteration] = iteration_times[iteration-1] + step1_time + step2_time + step3_time;
    }
    if (iteration_errors) {
      iteration_errors[iteration] = Error();
    }
    if (result_compute_time) {
      step_time.Read();
    }

    // Optimize transformations to align correspondences
    OptimizeTransformations();

    // Gather timing stats
    if (result_compute_time) {
      step2_time = step_time.Elapsed();
      total_step2_time += step2_time;
      compute_time += step2_time;
      step_time.Read();
    }

    // Check for convergence
    converged = FALSE;
    if (1) {
    // if (((max_euclidean_distance == RN_UNKNOWN) || (final_max_euclidean_distance == RN_UNKNOWN) || (max_euclidean_distance <= final_max_euclidean_distance)) &&
    //    ((max_normal_angle == RN_UNKNOWN) || (final_max_normal_angle == RN_UNKNOWN) || (max_normal_angle <= final_max_normal_angle))) {
      static R4Matrix *previous_matrices = NULL;
      if (previous_matrices) {
        converged = TRUE;
        for (int j = 0; j < NShapes(); j++) {
          const R4Matrix& current_matrix = Shape(j)->current_transformation.Matrix();
          const R4Matrix& previous_matrix = previous_matrices[j];
          for (int k1 = 0; k1 < 4; k1++) {
            if (!converged) break;
            for (int k2 = 0; k2 < 4; k2++) {
              if (RNIsNotEqual(current_matrix[k1][k2], previous_matrix[k1][k2], 1E-4)) { 
                converged = FALSE;
                break; 
              }
            }
          }
        }
      }

      // Remember previous matrices
      if (!previous_matrices) previous_matrices = new R4Matrix [ NShapes() ];
      for (int j = 0; j < NShapes(); j++) previous_matrices[j] = Shape(j)->current_transformation.Matrix();
    }
    if (converged) break;

    // Gather timing stats
    if (result_compute_time) {
      step3_time = step_time.Elapsed();
      total_step3_time += step3_time;
      compute_time += step3_time;
    }

    // Print debugging message
    printf("  I %2d : %6.2f %6.2f : %9d %9d %6d : %9.6f %9.3f\n", iteration,
       max_euclidean_distance, max_normal_angle,
       NFeatures(), NCorrespondences(), NMatches(),
       RMSD(), Score());

    // Update outlier thresholds
    if (max_euclidean_distance != RN_UNKNOWN) {
      max_euclidean_distance *= max_euclidean_distance_factor;
      if ((final_max_euclidean_distance != RN_UNKNOWN) && (max_euclidean_distance < final_max_euclidean_distance)) {
        max_euclidean_distance = final_max_euclidean_distance;
      }
    }
    if (max_normal_angle != RN_UNKNOWN) {
      max_normal_angle *= max_normal_angle_factor;
      if ((final_max_normal_angle != RN_UNKNOWN) && (max_normal_angle < final_max_normal_angle)) {
        max_normal_angle = final_max_normal_angle;
      }
    }

    // Increment iteration
    iteration++;
  }

  // printf("S1=%g  S2=%g   S3=%g\n", total_step1_time, total_step2_time, total_step3_time);

  // Create final correspondences
  TruncateCorrespondences(saved_ncorrespondences);
  CreateCorrespondences();

  // Fill in final iteration stats
  assert(iteration <= max_iterations);
  if (iteration_times) {
    if (iteration == 0) iteration_times[iteration] = 0;
    else iteration_times[iteration] = iteration_times[iteration-1] + step1_time + step2_time + step3_time;
  }
  if (iteration_errors) {
    iteration_errors[iteration] = Error();
  }

  // Fill in results
  if (result_niterations) *result_niterations = iteration;
  if (result_converged) *result_converged = converged;
  if (result_compute_time) *result_compute_time = compute_time;
  if (iteration_times) {
    for (int i = iteration+1; i <= max_iterations; i++) {
      iteration_times[i] = iteration_times[iteration];
    }
  }
  if (iteration_errors) {
    for (int i = iteration+1; i <= max_iterations; i++) {
      iteration_errors[i] = iteration_errors[iteration];
    }
  }
}



////////////////////////////////////////////////////////////////////////
// RANSAC alignment
////////////////////////////////////////////////////////////////////////

static FETFeature *
SelectRandomFeature(const RNArray<FETFeature *>& features)
{
  // Compute total salience
  RNScalar total_salience = 0;
  for (int i = 0; i < features.NEntries(); i++) {
    FETFeature *feature = features.Kth(i);
    total_salience += feature->Salience();
  }
  
  // Check total salience
  if (total_salience > 0) {
    // Select feature according to salience distribution
    RNScalar current_salience = 0;
    RNScalar r = RNRandomScalar() * total_salience;
    for (int i = 0; i < features.NEntries(); i++) {
      FETFeature *feature = features.Kth(i);
      current_salience += feature->Salience();
      if (current_salience > r) return feature;
    }
  }

  // Select feature according to uniform distribution
  return features.Kth((int) RNRandomScalar() * features.NEntries());
}



static FETMatch *
CreateMatchWithRANSAC(FETReconstruction *reconstruction,
   FETShape *shape1, FETShape *shape2)
{
  // Just checking
  if (shape1->NFeatures() == 0) return NULL;
  if (shape2->NFeatures() == 0) return NULL;

  // Parameters
  // int max_icp_iterations = 4;
  int max_icp_iterations = 0;
  // int max_ransac_iterations_per_feature = 4;
  int max_ransac_iterations_per_feature = 1;
  int num_feature2_samples = 256;
  int num_inlier_samples = 128;
  RNScalar target_overlap = 0.1;
  RNLength target_distance = target_overlap * shape1->BBox().DiagonalRadius();
  RNLength generator_tolerance = reconstruction->max_euclidean_distance;
  RNLength inlier_tolerance = reconstruction->max_euclidean_distance;
  RNAngle max_normal_angle = reconstruction->max_normal_angle;
  if (generator_tolerance < 0) generator_tolerance = RN_INFINITY;
  if (inlier_tolerance < 0) generator_tolerance = RN_INFINITY;
  
  // Save the shape transformations
  R3Affine saved_transformation1 = shape1->current_transformation;
  R3Affine saved_transformation2 = shape2->current_transformation;

  // Make arrays of features within each generator type and shape
  RNArray<FETFeature *> features[NUM_FEATURE_TYPES][2];
  for (int i = 0; i < 2; i++) {
    FETShape *shape = (i == 0) ? shape1 : shape2;
    for (int j = 0; j < shape->NFeatures(); j++) {
      FETFeature *feature = shape->Feature(j);
      int generator_type = feature->GeneratorType();
      if (generator_type < 0) continue;
      if (generator_type >= NUM_FEATURE_TYPES) continue;
      features[generator_type][i].Insert(feature);
    }
  }

  // Compute descriptor distance limits within each generator type
  float max_descriptor_distance_squared[NUM_FEATURE_TYPES];
  for (int i = 0; i < NUM_FEATURE_TYPES; i++) {
    max_descriptor_distance_squared[i] = FLT_MAX;
    int mdds = features[i][0].NEntries() + features[i][1].NEntries();
    if (mdds > 2) {
      int ndds = 0;
      RNScalar *dds = new RNScalar [ mdds ];
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < features[i][j].NEntries(); k++) {
          FETFeature *featureA = features[i][j][k];
          FETFeature *featureB = features[i][j][(int) (RNRandomScalar() * features[i][j].NEntries())];
          const FETDescriptor& descriptorA = featureA->Descriptor();
          if (descriptorA.NValues() == 0) continue;
          const FETDescriptor& descriptorB = featureB->Descriptor();
          if (descriptorB.NValues() == 0) continue;
          RNScalar dd = descriptorA.SquaredDistance(descriptorB);
          assert(ndds < mdds);
          dds[ndds++] = dd;
        }
      }
      if (ndds > 0) {
        qsort(dds, ndds, sizeof(RNScalar), RNCompareScalars);
        max_descriptor_distance_squared[i] = dds[1 * shape1->NFeatures() / 10];
      }
      delete [] dds;
    }
  }
  
  // For N iterations
  RNScalar best_score = 0;
  R3Affine best_transformation = R3identity_affine;
  int niterations = max_ransac_iterations_per_feature * shape1->NFeatures();
  for (int i = 0; i < niterations; i++) {
    FETFeature *features1[3] = { NULL, NULL, NULL };
    FETFeature *features2[3] = { NULL, NULL, NULL };

    ////////

    // Get first feature point on shape1
    features1[0] = SelectRandomFeature(shape1->features);
    if (!features1[0]) continue;

    // Get second feature point on shape1
    RNArray<FETFeature *> listA, listB;
    RNLength d01 = (3*RNRandomScalar()/2 + 0.5) * target_distance;
    shape1->FindAllFeatures(features1[0], R3identity_affine, listA, d01 - generator_tolerance, d01 + generator_tolerance);
    if (listA.IsEmpty()) continue;
    features1[1] = SelectRandomFeature(listA);
    if (!features1[1]) continue;
    if (features1[1] == features1[0]) continue;
    R3Vector v01 = features1[1]->Position() - features1[0]->Position();
    RNAngle angle010 = R3InteriorAngle(v01, features1[0]->normal);
    RNAngle angle011 = R3InteriorAngle(v01, features1[1]->normal);
    d01 = v01.Length();

    // Get third feature point on shape1
    listA.Empty();
    RNLength d02 = (3*RNRandomScalar()/2 + 0.5) * target_distance;
    RNLength d12 = (3*RNRandomScalar()/2 + 0.5) * target_distance;
    shape1->FindAllFeatures(features1[0], R3identity_affine, listA, d02 - generator_tolerance, d02 + generator_tolerance);
    if (listA.IsEmpty()) continue;
    listB.Empty();
    for (int j = 0; j < listA.NEntries(); j++) {
      FETFeature *feature = listA.Kth(j);     
      if (feature == features1[0]) continue;
      if (feature == features1[1]) continue;

      // Check distance
      RNScalar d = R3Distance(features1[1]->Position(), feature->Position());
      if (d < d12 - generator_tolerance) continue;
      if (d > d12 + generator_tolerance) continue;

      // Passed tests
      listB.Insert(feature);
    }
    if (listB.IsEmpty()) continue;
    features1[2] = SelectRandomFeature(listB);
    if (!features1[2]) continue;
    R3Vector v02 = features1[2]->Position() - features1[0]->Position();
    R3Vector v12 = features1[2]->Position() - features1[1]->Position();
    RNAngle angle020 = R3InteriorAngle(v02, features1[0]->normal);
    RNAngle angle022 = R3InteriorAngle(v02, features1[2]->normal);
    RNAngle angle121 = R3InteriorAngle(v12, features1[1]->normal);
    RNAngle angle122 = R3InteriorAngle(v12, features1[2]->normal);
    d02 = v02.Length();
    d12 = v12.Length();

    ////////

    // Get first feature point on shape2
    // Do this with ANN
    listA.Empty();
    listB.Empty();
    // for (int j = 0; j < shape2->NFeatures(); j++) {
    //   FETFeature *feature = shape2->Feature(j);
    int generator_type = features1[0]->GeneratorType();
    RNScalar best_descriptor_distance_squared = max_descriptor_distance_squared[generator_type];
    for (int j = 0; j < num_feature2_samples; j++) {
      FETFeature *feature = SelectRandomFeature(features[generator_type][1]);
      RNScalar descriptor_distance_squared = feature->descriptor.SquaredDistance(features1[0]->descriptor);
      if (descriptor_distance_squared < best_descriptor_distance_squared) {
        best_descriptor_distance_squared = descriptor_distance_squared;
        features2[0] = feature;
      }
    }

    // Check if found first feature
    if (!features2[0]) continue;

    // Get second feature point on shape2
    listA.Empty();
    listB.Empty();
    FETFeature query1(*features2[0]);
    query1.SetDescriptor(features1[1]->descriptor);
    shape2->FindAllFeatures(&query1, R3identity_affine, listA, d01 - generator_tolerance, d01 + generator_tolerance);
    if (listA.IsEmpty()) continue;
    for (int j = 0; j < listA.NEntries(); j++) {
      FETFeature *feature = listA.Kth(j);     
      if (feature == features2[0]) continue;

      // Check angle relationship to v01
      if (max_normal_angle != RN_UNKNOWN) {
        R3Vector v = feature->Position() - features2[0]->Position();
        if (fabs(R3InteriorAngle(v, features2[0]->normal) - angle010) > max_normal_angle) continue;
        if (fabs(R3InteriorAngle(v, feature->normal) - angle011) > max_normal_angle) continue;
      }

      // Check descriptor relationship to features1[1]
      int generator_type = feature->GeneratorType();
      if (max_descriptor_distance_squared[generator_type] < FLT_MAX) {
        RNScalar descriptor_distance_squared = feature->descriptor.SquaredDistance(features1[1]->descriptor);
        if (descriptor_distance_squared > max_descriptor_distance_squared[generator_type]) continue;
      }

      // Passed all tests
      listB.Insert(feature);
    }

    // Check if found a second feature point on shape2
    if (listB.IsEmpty()) continue;
    features2[1] = SelectRandomFeature(listB);
    if (!features2[1]) continue;

    // Get third feature point on shape2
    listA.Empty();
    listB.Empty();
    FETFeature query2(*features2[0]);
    query2.SetDescriptor(features1[2]->descriptor);
    shape2->FindAllFeatures(&query2, R3identity_affine, listA, d02 - generator_tolerance, d02 + generator_tolerance);
    if (listA.IsEmpty()) continue;
    for (int j = 0; j < listA.NEntries(); j++) {
      FETFeature *feature = listA.Kth(j);     
      if (feature == features2[0]) continue;
      if (feature == features2[1]) continue;

      // Check distance relationship 
      RNLength d = R3Distance(feature->Position(), features2[0]->Position());
      if (d < d12 - generator_tolerance) continue;
      if (d > d12 + generator_tolerance) continue;

      // Check angle relationship to v02
      if (max_normal_angle != RN_UNKNOWN) {
        R3Vector v = feature->Position() - features2[0]->Position();
        if (fabs(R3InteriorAngle(v, features2[0]->normal) - angle020) > max_normal_angle) continue;
        if (fabs(R3InteriorAngle(v, feature->normal) - angle022) > max_normal_angle) continue;
      }

      // Check angle relationship to v12
      if (max_normal_angle != RN_UNKNOWN) {
        R3Vector v = feature->Position() - features2[1]->Position();
        if (fabs(R3InteriorAngle(v, features2[1]->normal) - angle121) > max_normal_angle) continue;
        if (fabs(R3InteriorAngle(v, feature->normal) - angle122) > max_normal_angle) continue;
      }

      // Check descriptor relationship to features1[2]
      int generator_type = feature->GeneratorType();
      if (max_descriptor_distance_squared[generator_type] < FLT_MAX) {
        RNScalar descriptor_distance_squared = feature->descriptor.SquaredDistance(features1[2]->descriptor);
        if (descriptor_distance_squared > max_descriptor_distance_squared[generator_type]) continue;
      }

      // Passed all tests
      listB.Insert(feature);
    }

    // Check if found a third feature point on shape2
    if (listB.IsEmpty()) continue;
    features2[2] = SelectRandomFeature(listB);
    if (!features2[2]) continue;

    // Find the transformation that minimizes the distance between the feature triplets
    R3Point points1[3], points2[3];
    points1[0] = features1[0]->Position();
    points1[1] = features1[1]->Position();
    points1[2] = features1[2]->Position();
    points2[0] = features2[0]->Position();
    points2[1] = features2[1]->Position();
    points2[2] = features2[2]->Position();

#if 1
    R4Matrix matrix21 = R3AlignPoints(3, points1, points2, NULL, TRUE, TRUE, 0);
    R3Affine transformation21(matrix21, 0);
#else
    R3CoordSystem cs1(points1[0], R3Triad(points1[1] - points1[0], points1[2] - points1[0]));
    R3CoordSystem cs2(points2[0], R3Triad(points2[1] - points2[0], points2[2] - points2[0]));
    R3Affine transformation21 = R3identity_affine;
    transformation21.Transform(R3Affine(cs1.Matrix(), 0));
    transformation21.InverseTransform(R3Affine(cs2.Matrix(), 0));
#endif

    // Check the normals for compatibility
    if (max_normal_angle > 0) {
      RNBoolean normals_are_compatible = TRUE;
      for (int j = 0; j < 3; j++) {
        R3Vector normal1 = features1[j]->Normal();
        R3Vector normal2 = features2[j]->Normal();
        normal2.Transform(transformation21);
        RNAngle angle = R3InteriorAngle(normal1, normal2);
        if (angle > max_normal_angle) {
          normals_are_compatible = FALSE;
          break;
        }
      }
      if (!normals_are_compatible) continue;
    }

    if (max_icp_iterations > 0) {
#if 0
      // Optimize transformation with ICP
      FETShape *s1 = new FETShape(*shape1);
      FETShape *s2 = new FETShape(*shape2);
      s1->current_transformation = R3identity_affine;
      s2->current_transformation = R3identity_affine;
      FETMatch match(s1, s2, transformation21);
      // XXX match.OptimizeTransformationsWithICP(8);
      // RNScalar score = match.Score();
      transformation21 = R3identity_affine;
      transformation21.InverseTransform(s1->Transformation());
      transformation21.Transform(s2->Transformation());
#endif
    }

    // Count the inliers
    RNScalar score = 0;
    int inlier_skip = shape1->NFeatures() / num_inlier_samples + 1;
    for (int j = 0; j < shape1->NFeatures(); j += inlier_skip) {
      FETFeature *feature1 = shape1->Feature(j);
      R3Point position1 = feature1->Position();
      position1.InverseTransform(transformation21);
      if (shape2->FindClosestFeature(position1, 0, inlier_tolerance)) {
        score += inlier_skip;
      }
    }

    // printf("%d %g\n", i, score);

    // Remember transformation, if best score
    if (score > best_score) {
      best_transformation = transformation21;
      best_score = score;
    }
  }

  if (best_score == 0) return NULL;

  R4Matrix m = best_transformation.Matrix();
  printf("  M %d %d : %g %d %d\n", shape1->reconstruction_index, shape2->reconstruction_index, 
    best_score, shape1->NFeatures(), shape2->NFeatures());
  printf("    %g %g %g %g\n", m[0][0], m[0][1], m[0][2], m[0][3]);
  printf("    %g %g %g %g\n", m[1][0], m[1][1], m[1][2], m[1][3]);
  printf("    %g %g %g %g\n", m[2][0], m[2][1], m[2][2], m[2][3]);
  printf("    %g %g %g %g\n", m[3][0], m[3][1], m[3][2], m[3][3]);

  // Create match
  FETMatch *match = new FETMatch(reconstruction, shape1, shape2, best_transformation, best_score);

  // Restore the saved transformations
  shape1->current_transformation = saved_transformation1;
  shape2->current_transformation = saved_transformation2;

  // Return match
  return match;
}



void FETReconstruction::
CreateMatchesWithRANSAC(void)
{
  // Empty matches
  EmptyMatches();

  // Create matches between every pair of shapes
  for (int i = 0; i < NShapes(); i++) {
    FETShape *shape1 = Shape(i);
    for (int j = i+1; j < NShapes(); j++) {
      FETShape *shape2 = Shape(j);
      CreateMatchWithRANSAC(this, shape1, shape2);
    }
  }
}



void FETReconstruction::
OptimizeTransformationsWithRANSAC(void)
{
  // Create pairwise matches
  CreateMatchesWithRANSAC();

  // Create global alignment
  OptimizeTransformationsWithGlobalRelaxation();
}



void FETReconstruction::
OptimizeTransformationsWithLinearSystemOfEquations(void)
{
  // Check the shapes
  if (NShapes() < 2) return;

#define PRINT_TIMING
#ifdef PRINT_TIMING
  RNTime step_time;
  step_time.Read();
#endif

  // Update variable indices
  int n = 0;
  for (int i = 0; i < NShapes(); i++) {
    FETShape *shape = Shape(i);
    shape->UpdateVariableIndex(n);
  }

  // Create system of equations
  RNSystemOfEquations equations(n);
  AddInertiaEquations(&equations, total_inertia_weight);
  AddMatchEquations(&equations, total_match_weight);
  AddTrajectoryEquations(&equations, total_trajectory_weight);
  AddCorrespondenceEquations(&equations, total_correspondence_weights);
  
// #define PRINT_EQUATIONS
#ifdef PRINT_EQUATIONS
  equations.Print();
  fflush(stdout);
#endif
  
  // Initialize variables
  double *x = new double [ n ];
  for (int i = 0; i < n; i++) x[i] = 0;

#ifdef PRINT_TIMING
  printf("T1 = %9.3f %9ld : %d %d %d : %d %d %d : %g\n", step_time.Elapsed(), RNMaxMemoryUsage(),
    NFeatures(), NCorrespondences(), NMatches(), 
    equations.NVariables(), equations.NEquations(), equations.NPartialDerivatives(),
    sqrt(equations.SumOfSquaredResiduals(x)));
  fflush(stdout);
  step_time.Read();
#endif

  // Solve system of equations
  if (equations.NEquations() >= n) {
    if (!equations.Minimize(x, solver, 1E-3)) {
      fprintf(stderr, "Unable to minimize system of equations\n");
      delete [] x;
      return;
    }
  }

#ifdef PRINT_TIMING
  printf("T2 = %9.3f %9ld : %g\n", step_time.Elapsed(), RNMaxMemoryUsage(),
   sqrt(equations.SumOfSquaredResiduals(x)));
  fflush(stdout);
  step_time.Read();
#endif

  // Extract solution
  for (int i = 0; i < NShapes(); i++) {
    FETShape *shape = Shape(i);
    shape->UpdateVariableValues(x);
  }

  // Delete variables
  delete [] x;

#ifdef PRINT_TIMING
  printf("T3 = %9.3f %9ld : %9d %15.6f %15.3f\n", step_time.Elapsed(), RNMaxMemoryUsage(),
    NCorrespondences(), RMSD(), Score());
  fflush(stdout);
#endif
}



void FETReconstruction::
OptimizeTransformations(void)
{
  // Check the correspondences
  if (NShapes() < 2) return;

  // Check the features (fill this in later)
  RNBoolean only_point_features = FALSE;

  // Compute total correspondence weight
  RNScalar total_correspondence_weight = 0;
  for (int i = 0; i < NUM_FEATURE_TYPES; i++) {
    total_correspondence_weight += total_correspondence_weights[i];
  }

  // Check the optimization method
  if ((NShapes() == 2) && 
      (only_point_features) && 
      ((NCorrespondences() > 0) && (total_correspondence_weight > 0)) && 
      ((NMatches() == 0) || (total_match_weight == 0)) && 
      (total_trajectory_weight == 0)) {
    OptimizeTransformationsWithClosedFormEquations();
  }
  else if ((NShapes() > 1) && 
    ((NCorrespondences() == 0) || ((total_correspondence_weight == 0))) &&
    ((NMatches() > 2) && (total_match_weight > 0)) && 
    (total_trajectory_weight == 0)) {
    OptimizeTransformationsWithGlobalRelaxation();
  }
  else {
    OptimizeTransformationsWithLinearSystemOfEquations();
  }
}



