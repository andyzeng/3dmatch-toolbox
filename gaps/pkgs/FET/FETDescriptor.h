////////////////////////////////////////////////////////////////////////
// Class definition
////////////////////////////////////////////////////////////////////////

struct FETDescriptor {
public:
  // Constructors
  FETDescriptor(int nvalues = 0, float *values = NULL);
  FETDescriptor(const FETDescriptor& descriptor);
  ~FETDescriptor(void);

  // Properties
  int NValues(void) const;
  float Value(int k) const;
  float L1Norm(void) const;

  // Manipulation
  FETDescriptor& operator=(const FETDescriptor& descriptor);
  void Reset(int nvalues = 0, float *values = NULL);
  void SetValue(int k, float value);

  // Input/output
  int ReadAscii(FILE *fp);
  int ReadBinary(FILE *fp);
  int WriteAscii(FILE *fp) const;
  int WriteBinary(FILE *fp) const;

  // Relationships
  float SquaredDistance(const FETDescriptor& descriptor, float unknown_penalty = 10) const;
  float Distance(const FETDescriptor& descriptor, float unknown_penalty = 10) const;

public:
  // Internal data
  float *values;
  int nvalues;
};
