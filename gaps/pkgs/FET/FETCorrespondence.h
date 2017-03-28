////////////////////////////////////////////////////////////////////////
// Correspondence class definition
////////////////////////////////////////////////////////////////////////

struct FETCorrespondence {
public:
  // Constructors
  FETCorrespondence(FETReconstruction *reconstruction = NULL,
    FETFeature *feature1 = NULL, FETFeature *feature2 = NULL, 
    RNScalar affinity = 1.0, int relationship_type = 0);
  FETCorrespondence(const FETCorrespondence& correspondence);
  ~FETCorrespondence(void);

  // Reconstruction access
  FETReconstruction *Reconstruction(void) const;

  // Match access
  FETMatch *Match(void) const;

  // Feature access
  FETFeature *Feature(int k) const;
  FETFeature *OtherFeature(FETFeature *feature) const;
  int FeatureIndex(FETFeature *feature) const;

  // Properties
  RNScalar Affinity(void) const;
  int RelationshipType(void) const;

  // Relationships
  RNLength EuclideanDistance(void) const;
  RNLength SquaredEuclideanDistance(void) const;
  RNLength DescriptorDistance(void) const;
  RNLength SquaredDescriptorDistance(void) const;
  RNAngle NormalAngle(void) const;
  RNLength Error(void) const;

  // Input/output
  int ReadAscii(FILE *fp);
  int ReadBinary(FILE *fp);
  int WriteAscii(FILE *fp) const;
  int WriteBinary(FILE *fp) const;

public:
  FETReconstruction *reconstruction;
  int reconstruction_index;
  FETMatch *match;
  int match_index;
  FETFeature *features[2];
  int feature_indices[2];
  RNScalar affinity;
  int relationship_type;
};



////////////////////////////////////////////////////////////////////////
// Relationship types
////////////////////////////////////////////////////////////////////////

enum {
  COINCIDENT_RELATIONSHIP,
  PARALLEL_RELATIONSHIP,
  ANTIPARALLEL_RELATIONSHIP,
  PERPENDICULAR_RELATIONSHIP
};



////////////////////////////////////////////////////////////////////////
// Comparision function
////////////////////////////////////////////////////////////////////////

int FETCompareCorrespondences(const void *data1, const void *data2);



////////////////////////////////////////////////////////////////////////
// Inline correspondence functions
////////////////////////////////////////////////////////////////////////

inline FETReconstruction *FETCorrespondence::
Reconstruction(void) const
{
  // Return reconstruction
  return reconstruction;
}



inline FETMatch *FETCorrespondence::
Match(void) const
{
  // Return match
  return match;
}



inline FETFeature *FETCorrespondence::
Feature(int k) const
{
  // Return feature
  assert((k >= 0) && (k < 2));
  return features[k];
}



inline FETFeature *FETCorrespondence::
OtherFeature(FETFeature *feature) const
{
  // Return feature
  if (feature == features[0]) return features[1];
  else if (feature == features[1]) return features[0];
  else return NULL;
}



inline int FETCorrespondence::
FeatureIndex(FETFeature *feature) const
{
  // Return index of feature in correspondence
  if (!feature) return -1;
  if (feature == features[0]) return 0;
  else if (feature == features[1]) return 1;
  else return -1;
}



inline RNScalar FETCorrespondence::
Affinity(void) const
{
  // Return affinity
  return affinity;
}



inline int FETCorrespondence::
RelationshipType(void) const
{
  // Return relationship type
  return relationship_type;
}
