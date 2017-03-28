////////////////////////////////////////////////////////////////////////
// Match class definition
////////////////////////////////////////////////////////////////////////

struct FETMatch {
public:
  // Constructor
  FETMatch(FETReconstruction *reconstruction = NULL,
    FETShape *shape1 = NULL, FETShape *shape2 = NULL,
    const R3Affine& transformation21 = R3identity_affine,
    RNScalar affinity = 1.0);
  FETMatch(const FETMatch& match);
  ~FETMatch(void);

  // Reconstruction access
  FETReconstruction *Reconstruction(void) const;

  // Shape access
  int NShapes(void) const;
  FETShape *Shape(int k) const;
  FETShape *OtherShape(FETShape *shape) const;
  int ShapeIndex(FETShape *shape) const;
  
  // Correspondence access
  int NCorrespondences(void) const;
  FETCorrespondence *Correspondence(int k) const;

  // Properties
  const R3Affine& Transformation(void) const;
  RNScalar Affinity(void) const;
  
  // Input/output
  int ReadAscii(FILE *fp);
  int ReadBinary(FILE *fp);
  int WriteAscii(FILE *fp) const;
  int WriteBinary(FILE *fp) const;

public:
  // Internal manipulation
  void InsertCorrespondence(FETCorrespondence *correspondence);
  void RemoveCorrespondence(FETCorrespondence *correspondence);

public:
  FETReconstruction *reconstruction;
  int reconstruction_index;
  FETShape *shapes[2];
  int shape_indices[2];
  RNArray<FETCorrespondence *> correspondences;
  R3Affine current_transformation;
  R3Affine initial_transformation;
  R3Affine ground_truth_transformation;
  RNScalar affinity;
};



////////////////////////////////////////////////////////////////////////
// Inline functions
////////////////////////////////////////////////////////////////////////

inline FETReconstruction *FETMatch::
Reconstruction(void) const
{
  // Return reconstruction
  return reconstruction;
}



inline int FETMatch::
NShapes(void) const
{
  // Return number of shapes
  return 2;
}



inline FETShape *FETMatch::
Shape(int k) const
{
  // Return kth shape
  return shapes[k];
}



inline FETShape *FETMatch::
OtherShape(FETShape *shape) const
{
  // Return kth shape
  if (shapes[0] == shape) return shapes[1];
  if (shapes[1] == shape) return shapes[0];
  return NULL;
}



inline int FETMatch::
ShapeIndex(FETShape *shape) const
{
  // Return index of shape in correspondence
  if (!shape) return -1;
  if (shape == shapes[0]) return 0;
  else if (shape == shapes[1]) return 1;
  else return -1;
}



inline int FETMatch::
NCorrespondences(void) const
{
  // Return number of correspondences
  return correspondences.NEntries();
}



inline FETCorrespondence *FETMatch::
Correspondence(int k) const
{
  // Return kth correspondence
  return correspondences.Kth(k);
}



inline const R3Affine& FETMatch::
Transformation(void) const
{
  // Return current_transformation from shapes[1] to shapes[0]
  return current_transformation;
}



inline RNScalar FETMatch::
Affinity(void) const
{
  // Return affinity of the match
  return affinity;
}



