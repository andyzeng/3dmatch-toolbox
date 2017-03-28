// Include file for dense matrix class



// Class definition

class RNDenseMatrix : public RNMatrix {
public:
  // Constructor/destructor
  RNDenseMatrix(void);
  RNDenseMatrix(int nrows, int ncols, RNScalar *values = NULL);
  RNDenseMatrix(const RNDenseMatrix& matrix);
  RNDenseMatrix(const RNMatrix& matrix);
  virtual ~RNDenseMatrix(void);

  // Entry access
  virtual int NRows(void) const;
  virtual int NColumns(void) const;
  virtual RNScalar Value(int i, int j) const;
  virtual void SetValue(int i, int j, RNScalar value);
  RNScalar *operator[](int i) const;
  RNScalar *operator[](int i);

  // Property functions/operators
  virtual RNBoolean IsDense(void) const;
  virtual RNBoolean IsSparse(void) const;
  virtual RNBoolean IsZero(void) const;
  virtual RNBoolean IsSymmetric(void) const;
  virtual RNScalar Determinant(void) const;
  virtual RNDenseMatrix Transpose(void) const;
  virtual RNDenseMatrix Inverse(void) const;
  virtual RNDenseMatrix Submatrix(int min_row, int max_row, int min_col, int max_col) const;
  virtual RNBoolean operator==(const RNDenseMatrix& matrix) const;
  virtual RNBoolean operator!=(const RNDenseMatrix& matrix) const;

  // Matrix mainpulation
  virtual void Negate(void);
  virtual void Flip(void);
  virtual void Invert(void);
  virtual void Add(RNScalar a);
  virtual void Subtract(RNScalar a);
  virtual void Multiply(RNScalar a);
  virtual void Divide(RNScalar a);
  virtual void Add(const RNDenseMatrix& matrix);
  virtual void Subtract(const RNDenseMatrix& matrix);
  virtual void Multiply(const RNDenseMatrix& matrix);
  virtual void Reset(int nrows, int ncolumns, RNScalar *values = NULL);
 
  // Factorization
  int DecomposeLU(RNDenseMatrix& L, RNDenseMatrix& U) const;
  int DecomposeSVD(RNDenseMatrix& U, RNVector& S, RNDenseMatrix& Vt) const;
  int DecomposeEigen(RNVector& eigenvalues, RNDenseMatrix& eigenvectors);

  // Assignment operators
  virtual RNDenseMatrix& operator=(const RNDenseMatrix& matrix);
  virtual RNDenseMatrix& operator+=(const RNDenseMatrix& matrix);
  virtual RNDenseMatrix& operator-=(const RNDenseMatrix& matrix);
  virtual RNDenseMatrix& operator*=(const RNDenseMatrix& matrix);
  virtual RNDenseMatrix& operator*=(RNScalar a);
  virtual RNDenseMatrix& operator/=(RNScalar a);

  // Arithmetic operators
  friend RNDenseMatrix operator-(const RNDenseMatrix& matrix);
  friend RNDenseMatrix operator+(const RNDenseMatrix& matrix1, const RNDenseMatrix& matrix2);
  friend RNDenseMatrix operator-(const RNDenseMatrix& matrix1, const RNDenseMatrix& matrix2);
  friend RNDenseMatrix operator*(RNScalar a, const RNDenseMatrix& matrix);
  friend RNDenseMatrix operator*(const RNDenseMatrix& matrix, RNScalar a);
  friend RNDenseMatrix operator*(const RNDenseMatrix& matrix1, const RNDenseMatrix& matrix2);
  friend RNDenseMatrix operator/(const RNDenseMatrix& matrix, RNScalar a);
  friend RNVector operator*(const RNDenseMatrix& matrix, const RNVector& vector);

  // I/O functions
  virtual int ReadFile(const char *filename);
  virtual int ReadASCIIFile(const char *filename);
  virtual int ReadSquareBinaryFile(const char *filename);
  virtual int WriteFile(const char *filename) const;
  virtual int WriteASCIIFile(const char *filename) const;
  virtual int WriteSquareBinaryFile(const char *filename) const;

protected:
  RNScalar *values;;
  int nrows;
  int ncols;
};



// Inline functions and operators

inline RNScalar *RNDenseMatrix::
operator[](int i) const
{
  // Return pointer to ith row of values
  return &values[i*nrows];
}



inline RNScalar *RNDenseMatrix::
operator[](int i) 
{
  // Return pointer to ith row of values
  return &values[i*nrows];
}



