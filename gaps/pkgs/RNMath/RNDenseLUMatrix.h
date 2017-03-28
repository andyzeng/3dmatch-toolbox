// Include file for abstract matrix class



// Class definition

class RNDenseLUMatrix : public RNDenseMatrix {
public:
  // Constructor/destructor
  RNDenseLUMatrix(void);
  RNDenseLUMatrix(int nrows, int ncols, RNScalar *values = NULL);
  RNDenseLUMatrix(const RNDenseLUMatrix& matrix);
  RNDenseLUMatrix(const RNDenseMatrix& matrix);
  RNDenseLUMatrix(const RNMatrix& matrix);
  virtual ~RNDenseLUMatrix(void);

  // Property functions/operators
  virtual RNScalar Determinant(void) const;
  virtual RNDenseMatrix Inverse(void) const;

  // Factoring functions/operations
  int DecomposeLU(RNDenseMatrix& L, RNDenseMatrix& U) const;

protected:
  // Utility functions
  void Decompose(void);
  void BackSubstitute(RNDenseMatrix &b, RNDenseMatrix &x) const;

private:
  int *pivots;
};




