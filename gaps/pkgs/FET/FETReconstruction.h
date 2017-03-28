////////////////////////////////////////////////////////////////////////
// FETReconstruction class definition
////////////////////////////////////////////////////////////////////////

struct FETReconstruction {
public:
  // Constructor
  FETReconstruction(void);
  FETReconstruction(const FETReconstruction& reconstruction);
  ~FETReconstruction(void);

  // Shape access
  int NShapes(void) const;
  FETShape *Shape(int k) const;
  FETShape *Shape(const char *name) const;
  
  // Geometric properties
  R3Point Centroid(void) const;
  const R3Box& BBox(void) const;

  // Alignment properties
  RNScalar Speckle(void) const;
  RNScalar RMSD(void) const;
  RNScalar Score(RNScalar sigma = RN_UNKNOWN) const;
  RNScalar Error(void) const;
  RNScalar InlierFraction(RNLength error_threshold = RN_UNKNOWN) const;
  
  // Transformation manipulation
  void ResetTransformations(void);
  void PerturbTransformations(RNLength translation_magnitude, RNAngle rotation_magnitude);
  void OptimizeTransformations(void);

  // Input/output
  int ReadFile(const char *filename);
  int ReadAsciiFile(const char *filename);
  int ReadBinaryFile(const char *filename);
  int ReadAscii(FILE *fp);
  int ReadBinary(FILE *fp);
  int WriteFile(const char *filename) const;
  int WriteAsciiFile(const char *filename) const;
  int WriteBinaryFile(const char *filename) const;
  int WriteAscii(FILE *fp) const;
  int WriteBinary(FILE *fp) const;

public:
  // Internal property functions
  RNLength AverageFeatureRadius(void) const;
  RNScalar Affinity(FETFeature *feature1, FETFeature *feature2) const;

  // Internal access
  int NMatches(void) const;
  FETMatch *Match(int k) const;
  int NFeatures(void) const;
  FETFeature *Feature(int k) const;
  int NCorrespondences(void) const;
  FETCorrespondence *Correspondence(int k) const;
  
  // Internal manipulation
  void InsertShape(FETShape *shape);
  void RemoveShape(FETShape *shape);
  void InsertMatch(FETMatch *match);
  void RemoveMatch(FETMatch *match);
  void InsertFeature(FETFeature *feature);
  void RemoveFeature(FETFeature *feature);
  void InsertCorrespondence(FETCorrespondence *correspondence);
  void RemoveCorrespondence(FETCorrespondence *correspondence);
  void CopyContents(const FETReconstruction& reconstruction);
  
  // Internal updates
  void InvalidateBBox();
  void UpdateBBox();

  // Internal optimization
  void OptimizeTransformationsWithGlobalRelaxation(void);
  void OptimizeTransformationsWithRANSAC(void);
  void OptimizeTransformationsWithICP(int max_iterations = 64,
    int *result_niterations = NULL, RNBoolean *result_converged = NULL, RNScalar *result_compute_time = NULL,
    RNScalar *iteration_times = NULL, RNScalar *iteration_errors = NULL);

  // Internal transformation functions
  void OptimizeTransformationsWithClosedFormEquations(void);
  void OptimizeTransformationsWithLinearSystemOfEquations(void);
 
  // Internal system of equation functions
  void AddPointPointCorrespondenceEquations(RNSystemOfEquations *system, FETShape *shape1, FETShape *shape2, 
    const R3Point& position1, const R3Point& position2, RNScalar w);
  void AddPointLineCorrespondenceEquations(RNSystemOfEquations *system, FETShape *shape1, FETShape *shape2, 
    const R3Point& position1, const R3Point& position2, const R3Vector& direction2, RNScalar w);
  void AddPointPlaneCorrespondenceEquations(RNSystemOfEquations *system, FETShape *shape1, FETShape *shape2, 
    const R3Point& position1, const R3Point& position2, const R3Vector& normal2, RNScalar w);
  void AddParallelVectorEquations(RNSystemOfEquations *system, FETShape *shape1, FETShape *shape2, 
    const R3Vector& vector1, const R3Vector& vector2, RNScalar w);
  void AddPerpendicularVectorEquations(RNSystemOfEquations *system, FETShape *shape1, FETShape *shape2, 
    const R3Vector& vector1, const R3Vector& vector2, RNScalar w);
  void AddPairwiseTransformationEquations(RNSystemOfEquations *system, FETShape *shape1, FETShape *shape2, 
    const R3Affine& transformation, RNScalar w);

  // Internal system of equation functions
  void AddInertiaEquations(RNSystemOfEquations *system, RNScalar w);
  void AddTrajectoryEquations(RNSystemOfEquations *system, RNScalar w, RNScalar sigma = 0.25);
  void AddMatchEquations(RNSystemOfEquations *system, RNScalar w);
  void AddCorrespondenceEquations(RNSystemOfEquations *system, RNScalar *w);
  void AddCorrespondenceEquations(RNSystemOfEquations *system, FETCorrespondence *correspondence, RNScalar w);

 // Correspondence manipulation
  void EmptyCorrespondences(void);
  void CreateCorrespondences(void);
  void CreateCorrespondences(FETShape *shape1, FETShape *shape2, RNScalar max_correspondences = RN_UNKNOWN);
  void DiscardOutlierCorrespondences(RNScalar max_zscore = 3, int max_iterations = 8, int min_correspondences = 5);
  void SelectCorrespondences(int max_correspondences = -1);
  void TruncateCorrespondences(int max_correspondences = 0);

  // Internal match functions
  void EmptyMatches(void);
  void CreateMatches(void);
  void CreateMatchesWithICP(void);
  void CreateMatchesWithRANSAC(void);

  // Internal parameter setting functions
  void InitializeFeatureParameters(void);
  void InitializeCorrespondenceParameters(void);
  void InitializeOptimizationParameters(void);

public:
  RNArray<FETShape *> shapes;
  RNArray<FETMatch *> matches;
  RNArray<FETFeature *> features;
  RNArray<FETCorrespondence *> correspondences;

  // Feature parameters
  RNLength avg_feature_radius;

  // Correspondence parameters
  int max_correspondences;
  RNLength max_euclidean_distance; 
  RNLength max_descriptor_distances[NUM_FEATURE_TYPES]; 
  RNAngle max_normal_angle;
  RNScalar min_distinction;
  RNScalar max_distinction;
  RNScalar min_curvature;
  RNScalar max_curvature;
  RNScalar min_salience;
  RNBoolean discard_boundaries;
  RNBoolean discard_not_mutually_closest;
  RNBoolean discard_outliers;

  // Transformation parameters
  RNScalar total_match_weight;
  RNScalar total_correspondence_weights[NUM_FEATURE_TYPES];
  RNScalar total_trajectory_weight;
  RNScalar total_inertia_weight;
  int solver;

  // Geometry parameters
  R3Box bbox;
};



////////////////////////////////////////////////////////////////////////
// Inline functions
////////////////////////////////////////////////////////////////////////


inline int FETReconstruction::
NShapes(void) const
{
  // Return number of shapes
  return shapes.NEntries();
}



inline FETShape *FETReconstruction::
Shape(int k) const
{
  // Return kth shape
  return shapes.Kth(k);
}



inline int FETReconstruction::
NFeatures(void) const
{
  // Return number of features
  return features.NEntries();
}



inline FETFeature *FETReconstruction::
Feature(int k) const
{
  // Return kth feature
  return features.Kth(k);
}




inline int FETReconstruction::
NCorrespondences(void) const
{
  // Return number of correspondences
  return correspondences.NEntries();
}



inline FETCorrespondence *FETReconstruction::
Correspondence(int k) const
{
  // Return kth correspondence
  return correspondences.Kth(k);
}




inline int FETReconstruction::
NMatches(void) const
{
  // Return number of matches
  return matches.NEntries();
}



inline FETMatch *FETReconstruction::
Match(int k) const
{
  // Return kth match
  return matches.Kth(k);
}



inline R3Point FETReconstruction::
Centroid(void) const
{
  // Return centroid
  return BBox().Centroid();
}



inline void FETReconstruction::
EmptyCorrespondences(void)
{
  // Delete all correspondences
  TruncateCorrespondences(0);
}
  


