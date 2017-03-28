// Include file for abstract matrix class



// Class definition

class RNMatrix {
public:
  // Constructor/destructor
  RNMatrix(void);
  RNMatrix(const RNMatrix& matrix);
  virtual ~RNMatrix(void);

  // Entry access
  virtual int NRows(void) const = 0;
  virtual int NColumns(void) const = 0;
  virtual RNScalar Value(int i, int j) const = 0;
  virtual void SetValue(int i, int j, RNScalar value) = 0;

  // Property functions/operators
  virtual RNBoolean IsDense(void) const = 0;
  virtual RNBoolean IsSparse(void) const = 0;
};




