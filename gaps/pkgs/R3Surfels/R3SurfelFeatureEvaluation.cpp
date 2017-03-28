/* Source file for the R3 surfel feature evaluation functions */



////////////////////////////////////////////////////////////////////////
// INCLUDE FILES
////////////////////////////////////////////////////////////////////////

#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// FEATURE UTILITY FUNCTIONS
////////////////////////////////////////////////////////////////////////

int
CreateFeatures(R3SurfelScene *scene)
{
  // Check if features are already created
  if (scene->NFeatures() > 0) return 1;

  // Create pointset features
  scene->InsertFeature(new R3SurfelPointSetFeature("PointSetZMax", 0, 16, 0.5));
  scene->InsertFeature(new R3SurfelPointSetFeature("PointSetZMean", 0, 8, 1));
  scene->InsertFeature(new R3SurfelPointSetFeature("PointSetZStddev", 0, 8, 1));      
  scene->InsertFeature(new R3SurfelPointSetFeature("PointSetXYMax", 0, 4, 0.5));      
  scene->InsertFeature(new R3SurfelPointSetFeature("PointSetXYStddev1", 0, 2, 1));      
  scene->InsertFeature(new R3SurfelPointSetFeature("PointSetXYStddev2", 0, 2, 1));

  // Return success
  return 1;
}



int
EvaluateFeatures(R3SurfelScene *scene)
{
  // Evaluate features for every object 
  for (int i = 0; i < scene->NObjects(); i++) {
    R3SurfelObject *object = scene->Object(i);
    object->UpdateFeatureVector();
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// POINTSET FEATURE FUNCTIONS
////////////////////////////////////////////////////////////////////////


R3SurfelPointSetFeature::
R3SurfelPointSetFeature(const char *name, RNScalar minimum, RNScalar maximum, RNScalar weight)
 : R3SurfelFeature(name, minimum, maximum, weight) 
{
}



R3SurfelPointSetFeature::
~R3SurfelPointSetFeature(void)
{
}



int R3SurfelPointSetFeature::
Type(void) const
{
  // Return feature type
  return R3_SURFEL_POINTSET_FEATURE_TYPE;
}



int R3SurfelPointSetFeature::
UpdateFeatureVector(R3SurfelObject *object, R3SurfelFeatureVector& vector) const
{
  // Get/check scene
  R3SurfelScene *scene = Scene();
  int scene_index = SceneIndex();
  if (!scene) return 0;
  assert(scene == object->Scene());
  assert(scene_index >= 0);

  // Check if feature value is already up to date
  if (vector.Value(scene_index) != RN_UNKNOWN) return 1;

  // Update all pointset features
  R3SurfelFeature *z_max_feature = scene->FindFeatureByName("PointSetZMax");
  R3SurfelFeature *z_mean_feature = scene->FindFeatureByName("PointSetZMean");
  R3SurfelFeature *z_stddev_feature = scene->FindFeatureByName("PointSetZStddev");
  R3SurfelFeature *xy_max_feature = scene->FindFeatureByName("PointSetXYMax");
  R3SurfelFeature *xy_stddev1_feature = scene->FindFeatureByName("PointSetXYStddev1");
  R3SurfelFeature *xy_stddev2_feature = scene->FindFeatureByName("PointSetXYStddev2");

  // Extract surfel point set
  R3SurfelPointSet pointset;
  for (int i = 0; i < object->NNodes(); i++) {
    R3SurfelNode *node = object->Node(i);
    for (int j = 0; j < node->NBlocks(); j++) {
      R3SurfelBlock *block = node->Block(j);
      pointset.InsertPoints(block);
    }
  }

  // Check point set
  if (pointset.NPoints() == 0) {
    if (z_max_feature) vector.SetValue(z_max_feature->SceneIndex(), 0);
    if (z_mean_feature) vector.SetValue(z_mean_feature->SceneIndex(), 0);
    if (z_stddev_feature) vector.SetValue(z_stddev_feature->SceneIndex(), 0);
    if (xy_max_feature) vector.SetValue(xy_max_feature->SceneIndex(), 0);
    if (xy_stddev1_feature) vector.SetValue(xy_stddev1_feature->SceneIndex(), 0);
    if (xy_stddev2_feature) vector.SetValue(xy_stddev2_feature->SceneIndex(), 0);
    return 1;
  }

  // Compute sums, mins, maxs
  RNScalar x_sum = 0;
  RNScalar y_sum = 0;
  RNScalar z_sum = 0;
  RNScalar z_min = FLT_MAX;
  RNScalar z_max = -FLT_MAX;
  for (int i = 0; i < pointset.NPoints(); i++) {
    const R3SurfelPoint *point = pointset.Point(i);
    R3Point position = point->Position();
    if (position.Z() < z_min) z_min = position.Z();
    if (position.Z() > z_max) z_max = position.Z();
    x_sum += position.X();
    y_sum += position.Y();
    z_sum += position.Z();
  }

  // Compute means
  RNScalar x_mean = x_sum / pointset.NPoints();
  RNScalar y_mean = y_sum / pointset.NPoints();
  RNScalar z_mean = z_sum / pointset.NPoints();

  // Compute Z residuals and XY covariances
  RNScalar z_ssd = 0;
  RNScalar sq_r_max = -FLT_MAX;
  RNScalar xy_covariance[4] = { 0 };
  for (int i = 0; i < pointset.NPoints(); i++) {
    const R3SurfelPoint *point = pointset.Point(i);
    R3Point position = point->Position();
    RNScalar x_delta = position.X() - x_mean;
    RNScalar y_delta = position.Y() - y_mean;
    RNScalar z_delta = position.Z() - z_mean;
    z_ssd += z_delta * z_delta;
    RNScalar sq_r = x_delta*x_delta + y_delta*y_delta;
    if (sq_r > sq_r_max) sq_r_max = sq_r;
    xy_covariance[0] += x_delta * x_delta;
    xy_covariance[1] += x_delta * y_delta;
    xy_covariance[2] += y_delta * x_delta;
    xy_covariance[3] += y_delta * y_delta;
  }

  // Compute Z standard deviation
  RNScalar z_variance = z_ssd / pointset.NPoints();
  RNScalar z_stddev = sqrt(z_variance);

  // Normalize XY covariances
  for (int i = 0; i < 4; i++) {
    xy_covariance[i] /= pointset.NPoints();
  }

  // Calculate XY values
  RNScalar U[4], W[2], Vt[4];
  RNSvdDecompose(2, 2, xy_covariance, U, W, Vt); 
  RNScalar xy_stddev1 = sqrt(W[0]);
  RNScalar xy_stddev2 = sqrt(W[1]);
  assert(xy_stddev1 >= xy_stddev2);
  RNScalar xy_max = sqrt(sq_r_max);

  // Set vector values 
  if (z_max_feature) vector.SetValue(z_max_feature->SceneIndex(), z_max - z_min);
  if (z_mean_feature) vector.SetValue(z_mean_feature->SceneIndex(), z_mean - z_min);
  if (z_stddev_feature) vector.SetValue(z_stddev_feature->SceneIndex(), z_stddev);
  if (xy_max_feature) vector.SetValue(xy_max_feature->SceneIndex(), xy_max);
  if (xy_stddev1_feature) vector.SetValue(xy_stddev1_feature->SceneIndex(), xy_stddev1);
  if (xy_stddev2_feature) vector.SetValue(xy_stddev2_feature->SceneIndex(), xy_stddev2);

  // Return success
  return 1;
}




////////////////////////////////////////////////////////////////////////
// OVERHEAD GRID FEATURE FUNCTIONS
////////////////////////////////////////////////////////////////////////

R3SurfelOverheadGridFeature::
R3SurfelOverheadGridFeature(const char *filename, const char *featurename, RNScalar minimum, RNScalar maximum, RNScalar weight)
  : R3SurfelFeature(featurename, minimum, maximum, weight),
    filename(NULL),
    world_to_grid_matrix(R3null_matrix),
    fp(NULL)
{
  // Initialize resolution
  grid_resolution[0] = 0;
  grid_resolution[1] = 0;

  // Open file
  fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open overhead grid file %s\n", filename);
    return;
  }

  // Read grid resolution from file
  if (fread(&grid_resolution, sizeof(int), 2, fp) != 2) {
    fprintf(stderr, "Unable to read grid file %s", filename);
    fclose(fp);
    grid_resolution[0] = 0;
    grid_resolution[1] = 0;
    fp = NULL;
    return;
  }

  // Read world_to_grid transformation from file
  RNScalar m[9];
  if (fread(m, sizeof(RNScalar), 9, fp) != 9) {
    fprintf(stderr, "Invalid format for grid file %s", filename);
    fclose(fp);
    grid_resolution[0] = 0;
    grid_resolution[1] = 0;
    fp = NULL;
    return;
  }

  // Create world_to_grid transformation matrix
  world_to_grid_matrix = R3Matrix(m);

  // Copy filename
  this->filename = strdup(filename);
}



R3SurfelOverheadGridFeature::
~R3SurfelOverheadGridFeature(void)
{
  // Delete filename
  if (filename) free(filename);

  // Close file
  if (fp) fclose(fp);
}



int R3SurfelOverheadGridFeature::
Type(void) const
{
  // Return feature type
  return R3_SURFEL_OVERHEAD_GRID_FEATURE_TYPE;
}



int R3SurfelOverheadGridFeature::
UpdateFeatureVector(R3SurfelObject *object, R3SurfelFeatureVector& vector) const
{
  // Get/check scene
  R3SurfelScene *scene = Scene();
  int scene_index = SceneIndex();
  if (!scene) return 0;
  assert(scene == object->Scene());
  assert(scene_index >= 0);

  // Check if feature value is already up to date
  if (vector.Value(scene_index) != RN_UNKNOWN) return 1;

  // Check if file is open
  if (!fp) {
    fprintf(stderr, "Can't evaluate overhead grid feature because grid file is not open.\n");
    vector.SetValue(SceneIndex(), 0);
    return 0;
  }

  // Get object centroid
  R3Point centroid = object->Centroid();

  // Compute grid coordinates
  R2Point grid_position = world_to_grid_matrix * R2Point(centroid.X(), centroid.Y());
  int ix = (int) (grid_position[0] + 0.5);
  if ((ix < 0) || (ix >= grid_resolution[0])) return 0;
  int iy = (int) (grid_position[1] + 0.5);
  if ((iy < 0) || (iy >= grid_resolution[1])) return 0;

  // Seek to correct position in file
  int grid_index = grid_resolution[0] * iy + ix;
  unsigned int file_offset = 2*sizeof(int) + 9*sizeof(RNScalar) + grid_index*sizeof(RNScalar);
  if (fseek(fp, file_offset, SEEK_SET)) {
    fprintf(stderr, "Unable to seek to offset %d in grid file %s\n", file_offset, filename);
    vector.SetValue(SceneIndex(), 0);
    return 0;
  }

  // Read value
  RNScalar value;
  if (fread(&value, sizeof(RNScalar), 1, fp) != (unsigned int) 1) {
    fprintf(stderr, "Unable to read value from grid file %s\n", filename);
    vector.SetValue(SceneIndex(), 0);
    return 0;
  }

  // Set vector values 
  vector.SetValue(SceneIndex(), value);

  // Return success
  return 1;
}



