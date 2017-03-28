/* Include file for the R3 surfel vector vector class */



////////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
////////////////////////////////////////////////////////////////////////

class R3SurfelFeatureVector {
public:
  //////////////////////////////////////////
  //// CONSTRUCTOR/DESTRUCTOR FUNCTIONS ////
  //////////////////////////////////////////

  // Constructor functions
  R3SurfelFeatureVector(int nvalues = 0);
  R3SurfelFeatureVector(const R3SurfelFeatureVector& vector);

  // Destructor function
  virtual ~R3SurfelFeatureVector(void);


  //////////////////////////
  //// ACCESS FUNCTIONS ////
  //////////////////////////

  // Value access functions
  int NValues(void) const;
  RNScalar Value(int k) const;
  RNScalar operator[](int k) const;


  /////////////////////////////////////////
  //// MANIPULATION FUNCTIONS ////
  /////////////////////////////////////////

  // Arithmetic operators
  R3SurfelFeatureVector& operator=(const R3SurfelFeatureVector& vector);
  R3SurfelFeatureVector& operator+=(const R3SurfelFeatureVector& vector);
  R3SurfelFeatureVector& operator-=(const R3SurfelFeatureVector& vector);
  R3SurfelFeatureVector& operator*=(const R3SurfelFeatureVector& vector);
  R3SurfelFeatureVector& operator/=(const R3SurfelFeatureVector& vector);
  R3SurfelFeatureVector& operator+=(RNScalar a);
  R3SurfelFeatureVector& operator-=(RNScalar a);
  R3SurfelFeatureVector& operator*=(RNScalar a);
  R3SurfelFeatureVector& operator/=(RNScalar a);

  // More arithmetic operators
  friend R3SurfelFeatureVector operator+(const R3SurfelFeatureVector& vector1, const R3SurfelFeatureVector& vector2);
  friend R3SurfelFeatureVector operator-(const R3SurfelFeatureVector& vector1, const R3SurfelFeatureVector& vector2);
  friend R3SurfelFeatureVector operator*(const R3SurfelFeatureVector& vector1, const R3SurfelFeatureVector& vector2);
  friend R3SurfelFeatureVector operator/(const R3SurfelFeatureVector& vector1, const R3SurfelFeatureVector& vector2);
  friend R3SurfelFeatureVector operator+(const R3SurfelFeatureVector& vector, const RNScalar a);
  friend R3SurfelFeatureVector operator+(const RNScalar a, const R3SurfelFeatureVector& vector);
  friend R3SurfelFeatureVector operator-(const R3SurfelFeatureVector& vector, const RNScalar a);
  friend R3SurfelFeatureVector operator-(const RNScalar a, const R3SurfelFeatureVector& vector);
  friend R3SurfelFeatureVector operator*(const R3SurfelFeatureVector& vector, const RNScalar a);
  friend R3SurfelFeatureVector operator*(const RNScalar a, const R3SurfelFeatureVector& vector);
  friend R3SurfelFeatureVector operator/(const R3SurfelFeatureVector& vector, const RNScalar a);

  // Vector size manipulation functions
  virtual void Resize(int nvalues);

  // Vector value manipulation functions
  virtual void SetValue(int k, RNScalar value);
  virtual void Clear(RNScalar value = RN_UNKNOWN);


  //////////////////////////////
  //// COMPARISON FUNCTIONS ////
  //////////////////////////////

  RNLength EuclideanDistance(const R3SurfelFeatureVector& vector) const;
  RNLength EuclideanDistanceSquared(const R3SurfelFeatureVector& vector) const;
  RNScalar Correlation(const R3SurfelFeatureVector& vector) const;


  ///////////////////////////
  //// DISPLAY FUNCTIONS ////
  ///////////////////////////

  // Print function
  virtual void Print(FILE *fp = NULL, const char *prefix = NULL, const char *suffix = NULL) const;


  ////////////////////////////////////////////////////////////////////////
  // INTERNAL STUFF BELOW HERE
  ////////////////////////////////////////////////////////////////////////

private:
  RNScalar *values;
  int nvalues;
};



////////////////////////////////////////////////////////////////////////
// INLINE FUNCTION DEFINITIONS
////////////////////////////////////////////////////////////////////////

inline int R3SurfelFeatureVector::
NValues(void) const
{
  // Return number of values
  return nvalues;
}



inline RNScalar R3SurfelFeatureVector::
Value(int k) const
{
  // Return kth value
  assert((k >= 0) && (k < nvalues));
  return values[k];
}



inline RNScalar R3SurfelFeatureVector::
operator[](int k) const
{
  // Return kth value
  return Value(k);
}



inline RNLength R3SurfelFeatureVector::
EuclideanDistance(const R3SurfelFeatureVector& vector) const
{
  // Return Euclidean distance
  return sqrt(EuclideanDistanceSquared(vector));
}






