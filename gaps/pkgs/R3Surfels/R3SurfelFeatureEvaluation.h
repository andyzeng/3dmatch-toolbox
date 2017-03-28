/* Include file for the R3 surfel feature evaluation functions */



////////////////////////////////////////////////////////////////////////
// POINTSET FEATURE DEFINITIONS
////////////////////////////////////////////////////////////////////////

class R3SurfelPointSetFeature : public R3SurfelFeature {
public:
  R3SurfelPointSetFeature(const char *name = NULL, RNScalar minimum = -FLT_MAX, RNScalar maximum = FLT_MAX, RNScalar weight = 1);
  virtual ~R3SurfelPointSetFeature(void);
  virtual int Type(void) const;
  virtual int UpdateFeatureVector(R3SurfelObject *object, R3SurfelFeatureVector& vector) const;
};

class R3SurfelOverheadGridFeature : public R3SurfelFeature {
public:
  R3SurfelOverheadGridFeature(const char *filename, const char *featurename = NULL, RNScalar minimum = -FLT_MAX, RNScalar maximum = FLT_MAX, RNScalar weight = 1);
  virtual ~R3SurfelOverheadGridFeature(void);
  virtual int Type(void) const;
  virtual int UpdateFeatureVector(R3SurfelObject *object, R3SurfelFeatureVector& vector) const;

public:
  char *filename;
  R3Matrix world_to_grid_matrix;
  int grid_resolution[2];
  FILE *fp;
};



////////////////////////////////////////////////////////////////////////
// TOP-LEVEL FUNCTIONS
////////////////////////////////////////////////////////////////////////

int CreateFeatures(R3SurfelScene *scene);
int EvaluateFeatures(R3SurfelScene *scene);




