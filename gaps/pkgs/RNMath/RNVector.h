// Include file for vector class



// Class definition

class RNVector {
public:
  // Constructor/destructor
  RNVector(void);
  RNVector(int n, RNScalar *values = NULL);
  RNVector(const RNVector& vector);
  ~RNVector(void);

  // Entry access
  int NRows(void) const;
  int NValues(void) const;
  RNScalar Value(int i) const;
  void SetValue(int i, RNScalar value);
  RNScalar operator[](int i) const;

  // Property functions/operators
  RNBoolean IsZero(void) const;
  RNLength Length(void) const;
  RNVector Subvector(int min_row, int max_row) const;
  RNScalar Dot(const RNVector& vector) const;
  RNBoolean operator==(const RNVector& vector) const;
  RNBoolean operator!=(const RNVector& vector) const;

  // Vector mainpulation
  void Negate(void);
  void Normalize(void);
  void Add(const RNVector& vector);
  void Subtract(const RNVector& vector);
  void Multiply(RNScalar a);
  void Reset(int nvalues, RNScalar *values = NULL);
 
  // Assignment operators
  RNVector& operator=(const RNVector& vector);
  RNVector& operator+=(const RNVector& vector);
  RNVector& operator-=(const RNVector& vector);
  RNVector& operator*=(RNScalar a);
  RNVector& operator/=(RNScalar a);

  // Arithmetic operators
  friend RNVector operator-(const RNVector& vector);
  friend RNVector operator+(const RNVector& vector1, const RNVector& vector2);
  friend RNVector operator-(const RNVector& vector1, const RNVector& vector2);
  friend RNVector operator*(RNScalar a, const RNVector& vector);
  friend RNVector operator*(const RNVector& vector, RNScalar a);
  friend RNVector operator/(const RNVector& vector, RNScalar a);

  // Friends
  friend class RNDenseMatrix;

protected:
  RNScalar *values;
  int nvalues;
};



// Inline functions

inline int RNVector::
NRows(void) const
{
  // Return number of values
  return nvalues;
}



inline int RNVector::
NValues(void) const
{
  // Return number of values
  return NRows();
}




inline RNScalar RNVector::
Value(int i) const
{
  // Return ith value
  assert((i >= 0) && (i < nvalues));
  return values[i];
}




inline void RNVector::
SetValue(int i, RNScalar value)
{
  // Return number of values
  assert((i >= 0) && (i < nvalues));
  values[i] = value;
}




inline RNScalar RNVector::
operator[](int i) const
{
  // Return number of values
  return Value(i);
}




