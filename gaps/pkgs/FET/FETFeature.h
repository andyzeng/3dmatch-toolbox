////////////////////////////////////////////////////////////////////////
// Feature class definition
////////////////////////////////////////////////////////////////////////

struct FETFeature {
public:
  // Constructor
  FETFeature(FETReconstruction *reconstruction = NULL, int shape_type = 0, 
    const R3Point& position = R3Point(0,0,0), const R3Vector& direction = R3Vector(0,0,0), const R3Vector& normal = R3Vector(0,0,0),
    RNLength radius = 0, const FETDescriptor& descriptor = FETDescriptor(), const RNRgb& color = RNRgb(0.5,0.5,0.5), RNFlags flags = 0);
  FETFeature(const FETFeature& feature);
  ~FETFeature(void);

  // Reconstruction access
  FETReconstruction *Reconstruction(void) const;

  // Shape access
  FETShape *Shape(void) const;

  // Correspondence access
  int NCorrespondences(void) const;
  FETCorrespondence *Correspondence(int k) const;
  
  // Properties
  int ShapeType(void) const;
  R3Point Position(RNBoolean transformed = FALSE) const;
  R3Vector Direction(RNBoolean transformed = FALSE) const;
  R3Vector Normal(RNBoolean transformed = FALSE) const;
  RNLength Radius(RNBoolean transformed = FALSE) const;
  RNScalar Salience(void) const;
  RNScalar Distinction(void) const;
  const FETDescriptor& Descriptor(void) const;
  const RNRgb& Color(void) const;
  int GeneratorType(void) const;
  RNBoolean IsPoint(void) const;
  RNBoolean IsLinear(void) const;
  RNBoolean IsPlanar(void) const;
  RNBoolean IsOnSilhouetteBoundary(void) const;
  RNBoolean IsOnShadowBoundary(void) const;
  RNBoolean IsOnBorderBoundary(void) const;
  RNBoolean IsOnBoundary(void) const;

  // Manipulation
  void Transform(const R3Affine& transformation);
  void SetShapeType(int shape_type);
  void SetPosition(const R3Point& position, RNBoolean transformed = FALSE);
  void SetDirection(const R3Vector& direction, RNBoolean transformed = FALSE);
  void SetNormal(const R3Vector& normal, RNBoolean transformed = FALSE);
  void SetRadius(RNLength radius, RNBoolean transformed = FALSE);
  void SetSalience(RNScalar salience);
  void SetDistinction(RNScalar distinction);
  void SetDescriptor(const FETDescriptor& descriptor);
  void SetColor(const RNRgb& color);
  void SetGeneratorType(int generator_type);
  void SetFlags(const RNFlags& flags);
  
  // Relationship
  RNLength EuclideanDistance(const R3Point& position) const;
  RNLength SquaredEuclideanDistance(const R3Point& position) const;

  // Display
  void Draw(void) const;

  // Input/output
  int ReadAscii(FILE *fp);
  int ReadBinary(FILE *fp);
  int WriteAscii(FILE *fp) const;
  int WriteBinary(FILE *fp) const;

public:
  // Internal correspondence manipulation
  void InsertCorrespondence(FETCorrespondence *correspondence, int k);
  void RemoveCorrespondence(FETCorrespondence *correspondence, int k);
  void SortCorrespondences(void);

public:
  // Membership stuff
  FETReconstruction *reconstruction;
  int reconstruction_index;
  FETShape *shape;
  int shape_index;

  // Correspondence stuff
  RNArray<FETCorrespondence *> correspondences;
  
  // Geometry stuff
  int shape_type;
  R3Point position;
  R3Vector direction;
  R3Vector normal;
  RNLength radius;

  // Other stuff
  RNScalar salience;
  RNScalar distinction;
  FETDescriptor descriptor;
  RNRgb color;
  RNFlags flags;

  // Temporary
  int generator_type;
  int primitive_marker;
};



////////////////////////////////////////////////////////////////////////
// Shape types
////////////////////////////////////////////////////////////////////////

enum {
  NULL_FEATURE_SHAPE,
  POINT_FEATURE_SHAPE,
  LINE_FEATURE_SHAPE,
  PLANE_FEATURE_SHAPE,
  NUM_FEATURE_SHAPES
};



////////////////////////////////////////////////////////////////////////
// Generator types
////////////////////////////////////////////////////////////////////////

#define UNKNOWN_FEATURE_TYPE        0
#define SIFT_FEATURE_TYPE           1
#define FAST_FEATURE_TYPE           2
#define CORNER_FEATURE_TYPE         3
#define BORDER_FEATURE_TYPE        11
#define SILHOUETTE_FEATURE_TYPE    12
#define SHADOW_FEATURE_TYPE        13
#define POLE_FEATURE_TYPE          14
#define RIDGE_FEATURE_TYPE         15
#define VALLEY_FEATURE_TYPE        16
#define UNIFORM_FEATURE_TYPE       21
#define PLANE_FEATURE_TYPE         22
#define STRUCTURE_FEATURE_TYPE     31
#define NUM_FEATURE_TYPES          32

 

////////////////////////////////////////////////////////////////////////
// Feature flags
////////////////////////////////////////////////////////////////////////

#define FEATURE_IS_POINT                     0x01
#define FEATURE_IS_LINEAR                    0x02
#define FEATURE_IS_PLANAR                    0x04
#define FEATURE_IS_ON_SILHOUETTE_BOUNDARY    0x10
#define FEATURE_IS_ON_SHADOW_BOUNDARY        0x20
#define FEATURE_IS_ON_BORDER_BOUNDARY        0x40
#define FEATURE_IS_ON_BOUNDARY (FEATURE_IS_ON_SILHOUETTE_BOUNDARY | FEATURE_IS_ON_SHADOW_BOUNDARY | FEATURE_IS_ON_BORDER_BOUNDARY)



////////////////////////////////////////////////////////////////////////
// Feature compatibility parameters
////////////////////////////////////////////////////////////////////////

struct FETCompatibilityParameters {
public:
  FETCompatibilityParameters(
    RNLength max_euclidean_distance = RN_UNKNOWN, 
    RNLength *max_descriptor_distances = NULL,
    RNAngle max_normal_angle = RN_UNKNOWN,
    RNScalar min_distinction = RN_UNKNOWN,
    RNScalar min_salience = RN_UNKNOWN,
    RNBoolean discard_boundaries = FALSE);
public:
  RNLength max_euclidean_distance_squared;
  RNLength max_descriptor_distance_squared[NUM_FEATURE_TYPES];
  RNAngle min_normal_dot_product;
  RNScalar min_distinction;
  RNScalar min_salience;
  RNBoolean discard_boundaries;
};

int AreFeaturesCompatible(FETFeature *feature1, FETFeature *feature2, void *data);



////////////////////////////////////////////////////////////////////////
// Comparison function
////////////////////////////////////////////////////////////////////////

int FETCompareFeatures(const void *data1, const void *data2);



////////////////////////////////////////////////////////////////////////
// Inline feature functions
////////////////////////////////////////////////////////////////////////

inline FETReconstruction *FETFeature::
Reconstruction(void) const
{
  // Return reconstruction
  return reconstruction;
}



inline FETShape *FETFeature::
Shape(void) const
{
  // Return shape
  return shape;
}



inline int FETFeature::
NCorrespondences(void) const
{
  // Return number of correspondences
  return correspondences.NEntries();
}



inline FETCorrespondence *FETFeature::
Correspondence(int k) const
{
  // Return kth correspondence
  return correspondences.Kth(k);
}



inline int FETFeature::
ShapeType(void) const
{
  // Return shape type
  return shape_type;
}



inline RNScalar FETFeature::
Salience(void) const
{
  // Return salience
  return salience;
}



inline RNScalar FETFeature::
Distinction(void) const
{
  // Return distinction
  return distinction;
}



inline const FETDescriptor& FETFeature::
Descriptor(void) const
{
  // Return descriptor
  return descriptor;
}



inline const RNRgb& FETFeature::
Color(void) const
{
  // Return color
  return color;
}



inline int FETFeature::
GeneratorType(void) const
{
  // Return generator type
  return generator_type;
}



inline RNBoolean FETFeature::
IsPoint(void) const
{
  // Return whether feature is a point
  return flags[FEATURE_IS_POINT];
}



inline RNBoolean FETFeature::
IsLinear(void) const
{
  // Return whether feature is linear
  return flags[FEATURE_IS_LINEAR];
}



inline RNBoolean FETFeature::
IsPlanar(void) const
{
  // Return whether feature is planar
  return flags[FEATURE_IS_PLANAR];
}



inline RNBoolean FETFeature::
IsOnSilhouetteBoundary(void) const
{
  // Return whether feature is on silhouette boundary
  return flags[FEATURE_IS_ON_SILHOUETTE_BOUNDARY];
}



inline RNBoolean FETFeature::
IsOnShadowBoundary(void) const
{
  // Return whether feature is on shadow boundary
  return flags[FEATURE_IS_ON_SHADOW_BOUNDARY];
}



inline RNBoolean FETFeature::
IsOnBorderBoundary(void) const
{
  // Return whether feature is on shadow boundary
  return flags[FEATURE_IS_ON_BORDER_BOUNDARY];
}



inline RNBoolean FETFeature::
IsOnBoundary(void) const
{
  // Return whether feature is on boundary
  return flags[FEATURE_IS_ON_BOUNDARY];
}



inline void FETFeature::
SetShapeType(int shape_type)
{
  // Set shape type
  this->shape_type = shape_type;
}



inline void FETFeature::
SetSalience(RNScalar salience)
{
  // Set salience
  this->salience = salience;
}



inline void FETFeature::
SetDistinction(RNScalar distinction)
{
  // Set distinction
  this->distinction = distinction;
}



inline void FETFeature::
SetDescriptor(const FETDescriptor& descriptor)
{
  // Set descriptor
  this->descriptor = descriptor;
}



inline void FETFeature::
SetColor(const RNRgb& color)
{
  // Set color
  this->color = color;
}



inline void FETFeature::
SetGeneratorType(int generator_type)
{
  // Set generator type
  this->generator_type = generator_type;
}



inline void FETFeature::
SetFlags(const RNFlags& flags)
{
  // Set flags
  this->flags = flags;
}




