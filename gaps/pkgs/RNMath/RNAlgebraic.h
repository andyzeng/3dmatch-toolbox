// Include file for algebraic expression class



////////////////////////////////////////////////////////////////////////
// Class definition
////////////////////////////////////////////////////////////////////////

class RNAlgebraic {
public:
  // Constructor/destructor
  RNAlgebraic(void);
  RNAlgebraic(RNPolynomial *polynomial);
  RNAlgebraic(RNAlgebraic *algebraic);
  RNAlgebraic(RNScalar c, int v, RNScalar e);
  RNAlgebraic(RNScalar c, int nv, const int *v = NULL, const RNScalar *e = NULL, RNBoolean already_sorted = FALSE);
  RNAlgebraic(const RNPolynomial& polynomial, int dummy);
  RNAlgebraic(const RNAlgebraic& algebraic);
  RNAlgebraic(int operation, RNScalar      operand1, RNScalar      operand2);
  RNAlgebraic(int operation, RNScalar      operand1, RNPolynomial *operand2);
  RNAlgebraic(int operation, RNScalar      operand1, RNAlgebraic  *operand2);
  RNAlgebraic(int operation, RNPolynomial *operand1, RNScalar      operand2);
  RNAlgebraic(int operation, RNPolynomial *operand1, RNPolynomial *operand2);
  RNAlgebraic(int operation, RNPolynomial *operand1, RNAlgebraic  *operand2);
  RNAlgebraic(int operation, RNAlgebraic  *operand1, RNScalar      operand2);
  RNAlgebraic(int operation, RNAlgebraic  *operand1, RNPolynomial *operand2);
  RNAlgebraic(int operation, RNAlgebraic  *operand1, RNAlgebraic  *operand2);
  virtual ~RNAlgebraic(void);

  // Property functions
  int NVariables(void) const;
  int NPartialDerivatives(void) const;
  RNBoolean IsZero(void) const;
  RNBoolean IsOne(void) const;
  RNBoolean IsConstant(void) const;
  RNBoolean IsLinear(void) const;
  RNBoolean IsQuadratic(void) const;
  RNBoolean IsPolynomial(void) const;
  RNBoolean IsAlgebraic(void) const;
  RNBoolean HasVariable(int v) const;

  // Access functions
  int Operation(void) const;
  int NOperands(void) const;
  RNAlgebraic *Operand(int k) const;
  RNPolynomial *Polynomial(void) const;

  // Manipulation functions
  void SetOperation(int operation);
  void Reset(int operation, RNAlgebraic *algebraic1, RNAlgebraic *algebraic2);
  void Reset(RNPolynomial *polynomial);

  // More manipulation functions
  void Empty(void);
  void Negate(void);
  void Add(RNScalar constant);
  void Subtract(RNScalar constant);
  void Multiply(RNScalar factor);
  void Divide(RNScalar factor);
  void Add(const RNPolynomial& polynomial);
  void Subtract(const RNPolynomial& polynomial);
  void Multiply(const RNPolynomial& polynomial);
  void Divide(const RNPolynomial& polynomial);
  void Add(const RNAlgebraic& algebraic);
  void Subtract(const RNAlgebraic& algebraic);
  void Multiply(const RNAlgebraic& algebraic);
  void Divide(const RNAlgebraic& algebraic);

  // Assignment operators
  RNAlgebraic& operator=(const RNAlgebraic& algebraic);
  RNAlgebraic& operator+=(const RNAlgebraic& algebraic);
  RNAlgebraic& operator-=(const RNAlgebraic& algebraic);
  RNAlgebraic& operator*=(const RNAlgebraic& algebraic);
  RNAlgebraic& operator/=(const RNAlgebraic& algebraic);
  RNAlgebraic& operator=(const RNPolynomial& polynomial);
  RNAlgebraic& operator+=(const RNPolynomial& polynomial);
  RNAlgebraic& operator-=(const RNPolynomial& polynomial);
  RNAlgebraic& operator*=(const RNPolynomial& polynomial);
  RNAlgebraic& operator/=(const RNPolynomial& polynomial);
  RNAlgebraic& operator=(RNScalar scalar);
  RNAlgebraic& operator+=(RNScalar scalar);
  RNAlgebraic& operator-=(RNScalar scalar);
  RNAlgebraic& operator*=(RNScalar scalar);
  RNAlgebraic& operator/=(RNScalar scalar);
  
  // Arithmetic operators
  friend RNAlgebraic operator-(const RNAlgebraic& algebraic);
  friend RNAlgebraic operator+(const RNAlgebraic& algebraic1, const RNAlgebraic& algebraic2);
  friend RNAlgebraic operator+(const RNAlgebraic& algebraic, const RNPolynomial& polynomial);
  friend RNAlgebraic operator+(const RNAlgebraic& algebraic, RNScalar scalar);
  friend RNAlgebraic operator+(const RNPolynomial& polynomial, const RNAlgebraic& algebraic);
  friend RNAlgebraic operator+(RNScalar scalar, const RNAlgebraic& algebraic);
  friend RNAlgebraic operator-(const RNAlgebraic& algebraic1, const RNAlgebraic& algebraic2);
  friend RNAlgebraic operator-(const RNAlgebraic& algebraic, const RNPolynomial& polynomial);
  friend RNAlgebraic operator-(const RNAlgebraic& algebraic, RNScalar scalar);
  friend RNAlgebraic operator-(const RNPolynomial& polynomial, const RNAlgebraic& algebraic);
  friend RNAlgebraic operator-(RNScalar scalar, const RNAlgebraic& algebraic);
  friend RNAlgebraic operator*(const RNAlgebraic& algebraic1, const RNAlgebraic& algebraic2);
  friend RNAlgebraic operator*(const RNAlgebraic& algebraic, const RNPolynomial& polynomial);
  friend RNAlgebraic operator*(const RNAlgebraic& algebraic, RNScalar scalar);
  friend RNAlgebraic operator*(const RNPolynomial& polynomial, const RNAlgebraic& algebraic);
  friend RNAlgebraic operator*(RNScalar scalar, const RNAlgebraic& algebraic);
  friend RNAlgebraic operator/(const RNAlgebraic& algebraic1, const RNAlgebraic& algebraic2);
  friend RNAlgebraic operator/(const RNAlgebraic& algebraic, const RNPolynomial& polynomial);
  friend RNAlgebraic operator/(const RNAlgebraic& algebraic, RNScalar scalar);
  friend RNAlgebraic operator/(const RNPolynomial& polynomial, const RNAlgebraic& algebraic);
  friend RNAlgebraic operator/(RNScalar scalar, const RNAlgebraic& algebraic);

  // Evaluation functions
  RNScalar Evaluate(const RNScalar *x) const;

  // Partial derivative functions
  RNScalar PartialDerivative(const RNScalar *x, int variable) const;

  // Print functions
  void Print(FILE *fp = stdout, int indent = 0) const;

public:
  // Internal functions (for constructors)
  void Construct(int op, RNScalar      operand1, RNScalar      operand2,  RNBoolean force = FALSE);
  void Construct(int op, RNScalar      operand1, RNPolynomial *operand2,  RNBoolean force = FALSE);
  void Construct(int op, RNScalar      operand1, RNAlgebraic  *operand2,  RNBoolean force = FALSE);
  void Construct(int op, RNPolynomial *operand1, RNScalar      operand2,  RNBoolean force = FALSE);
  void Construct(int op, RNPolynomial *operand1, RNPolynomial *operand2,  RNBoolean force = FALSE);
  void Construct(int op, RNPolynomial *operand1, RNAlgebraic  *operand2,  RNBoolean force = FALSE);
  void Construct(int op, RNAlgebraic  *operand1, RNScalar      operand2,  RNBoolean force = FALSE);
  void Construct(int op, RNAlgebraic  *operand1, RNPolynomial *operand2,  RNBoolean force = FALSE);
  void Construct(int op, RNAlgebraic  *operand1, RNAlgebraic  *operand2,  RNBoolean force = FALSE);

  // Internal functions (for CERES)
  RNScalar Evaluate(double const* const* x) const;
  RNScalar PartialDerivative(double const* const* x, int variable) const;

  // Internal functions (for counting unique variables)
  void UpdateVariableRange(int& min_v, int& max_v) const;
  void UpdateVariableIndex(int max_variables, int& variable_count, 
    int *variable_marks, int current_mark, 
    int *index_to_variable = NULL, int *variable_to_index = NULL,
    RNBoolean remap_variables = FALSE) const;

  // Internal functions (for sanity checking)
  RNBoolean IsValid(void) const;

private:
  int operation;
  RNAlgebraic *operands[2];
  RNPolynomial *polynomial;
};



////////////////////////////////////////////////////////////////////////
// Operation definitions 
////////////////////////////////////////////////////////////////////////

#define RN_ZERO_OPERATION           0
#define RN_POLYNOMIAL_OPERATION     1
#define RN_ADD_OPERATION            2
#define RN_SUBTRACT_OPERATION       3
#define RN_MULTIPLY_OPERATION       4
#define RN_DIVIDE_OPERATION         5
#define RN_POW_OPERATION            6
#define RN_NUM_ALGEBRAIC_OPERATIONS 7



////////////////////////////////////////////////////////////////////////
// Inline functions
////////////////////////////////////////////////////////////////////////

inline int RNAlgebraic::
NPartialDerivatives(void) const
{
  // Return number of partial derivatives
  return NVariables();
}



inline RNBoolean RNAlgebraic::
IsAlgebraic(void) const
{
  // Trivially true
  return TRUE;
}



inline int RNAlgebraic::
Operation(void) const
{
  // Return operation
  return operation;
}



inline int RNAlgebraic::
NOperands(void) const
{
  // Return number of operands in algebraic
  return 2;
}



inline RNAlgebraic *RNAlgebraic::
Operand(int k) const
{
  // Return kth operand
  assert((k >= 0) && (k < NOperands()));
  return operands[k];
}



inline RNPolynomial *RNAlgebraic::
Polynomial(void) const
{
  // Return polynomial
  return polynomial;
}



inline RNAlgebraic& RNAlgebraic::
operator+=(RNScalar a)
{
  // Add constant
  Add(a);
  return *this;
}



inline RNAlgebraic& RNAlgebraic::
operator-=(RNScalar a)
{
  // Subtract constant
  Subtract(a);
  return *this;
}



inline RNAlgebraic& RNAlgebraic::
operator*=(RNScalar a)
{
  // Multiply by constant
  Multiply(a);
  return *this;
}



inline RNAlgebraic& RNAlgebraic::
operator/=(RNScalar a)
{
  // Divide by constant
  Divide(a);
  return *this;
}



inline RNAlgebraic& RNAlgebraic::
operator+=(const RNPolynomial& p)
{
  // Add polynomial
  Add(p);
  return *this;
}



inline RNAlgebraic& RNAlgebraic::
operator-=(const RNPolynomial& p)
{
  // Subtract polynomial
  Subtract(p);
  return *this;
}



inline RNAlgebraic& RNAlgebraic::
operator*=(const RNPolynomial& p)
{
  // Multiply polynomial
  Multiply(p);
  return *this;
}



inline RNAlgebraic& RNAlgebraic::
operator/=(const RNPolynomial& p)
{
  // Multiply polynomial
  Divide(p);
  return *this;
}



inline RNAlgebraic& RNAlgebraic::
operator+=(const RNAlgebraic& algebraic)
{
  // Add algebraic
  Add(algebraic);
  return *this;
}



inline RNAlgebraic& RNAlgebraic::
operator-=(const RNAlgebraic& algebraic)
{
  // Subtract algebraic
  Subtract(algebraic);
  return *this;
}



inline RNAlgebraic& RNAlgebraic::
operator*=(const RNAlgebraic& algebraic)
{
  // Multiply algebraic
  Multiply(algebraic);
  return *this;
}



inline RNAlgebraic& RNAlgebraic::
operator/=(const RNAlgebraic& algebraic)
{
  // Multiply algebraic
  Divide(algebraic);
  return *this;
}



inline RNAlgebraic
operator-(const RNAlgebraic& algebraic)
{
  // Return negated algebraic
  RNAlgebraic result(algebraic);
  result.Negate();
  return result;
}



inline RNAlgebraic 
operator+(const RNAlgebraic& algebraic1, const RNAlgebraic& algebraic2)
{
  // Return sum
  RNAlgebraic result(algebraic1);
  result.Add(algebraic2);
  return result;
}



inline RNAlgebraic 
operator+(const RNAlgebraic& algebraic, const RNPolynomial& polynomial)
{
  // Return sum
  RNAlgebraic result(algebraic);
  result.Add(polynomial);
  return result;
}



inline RNAlgebraic 
operator+(const RNAlgebraic& algebraic, RNScalar constant)
{
  // Return sum
  RNAlgebraic result(algebraic);
  result.Add(constant);
  return result;
}



inline RNAlgebraic 
operator+(const RNPolynomial& polynomial, const RNAlgebraic& algebraic)
{
  // Return sum
  RNAlgebraic result(polynomial, 0);
  result.Add(algebraic);
  return result;
}



inline RNAlgebraic 
operator+(RNScalar constant, const RNAlgebraic& algebraic)
{
  // Return sum
  RNAlgebraic result(algebraic);
  result.Add(constant);
  return result;
}



inline RNAlgebraic 
operator-(const RNAlgebraic& algebraic1, const RNAlgebraic& algebraic2)
{
  // Return difference
  RNAlgebraic result(algebraic1);
  result.Subtract(algebraic2);
  return result;
}



inline RNAlgebraic 
operator-(const RNAlgebraic& algebraic, const RNPolynomial& polynomial)
{
  // Return difference
  RNAlgebraic result(algebraic);
  result.Subtract(polynomial);
  return result;
}



inline RNAlgebraic 
operator-(const RNAlgebraic& algebraic, RNScalar constant)
{
  // Return difference
  RNAlgebraic result(algebraic);
  result.Subtract(constant);
  return result;
}



inline RNAlgebraic 
operator-(const RNPolynomial& polynomial, const RNAlgebraic& algebraic)
{
  // Return difference
  RNAlgebraic result(polynomial, 0);
  result.Subtract(algebraic);
  return result;
}



inline RNAlgebraic 
operator-(RNScalar constant, const RNAlgebraic& algebraic)
{
  // Return difference
  RNAlgebraic result(algebraic);
  result.Negate();
  result.Add(constant);
  return result;
}



inline RNAlgebraic 
operator*(const RNAlgebraic& algebraic1, const RNAlgebraic& algebraic2)
{
  // Return product
  RNAlgebraic result(algebraic1);
  result.Multiply(algebraic2);
  return result;
}



inline RNAlgebraic 
operator*(const RNAlgebraic& algebraic, const RNPolynomial& polynomial)
{
  // Return product
  RNAlgebraic result(algebraic);
  result.Multiply(polynomial);
  return result;
}



inline RNAlgebraic 
operator*(const RNAlgebraic& algebraic, RNScalar a)
{
  // Return product
  RNAlgebraic result(algebraic);
  result.Multiply(a);
  return result;
}



inline RNAlgebraic 
operator*(const RNPolynomial& polynomial, const RNAlgebraic& algebraic)
{
  // Return product
  RNAlgebraic result(polynomial, 0);
  result.Multiply(algebraic);
  return result;
}


inline RNAlgebraic 
operator*(RNScalar a, const RNAlgebraic& algebraic)
{
  // Return product
  RNAlgebraic result(algebraic);
  result.Multiply(a);
  return result;
}



inline RNAlgebraic 
operator/(const RNAlgebraic& algebraic1, const RNAlgebraic& algebraic2)
{
  // Return quotient
  RNAlgebraic result(algebraic1);
  result.Divide(algebraic2);
  return result;
}



inline RNAlgebraic 
operator/(const RNAlgebraic& algebraic, const RNPolynomial& polynomial)
{
  // Return quotient
  RNAlgebraic result(algebraic);
  result.Divide(polynomial);
  return result;
}



inline RNAlgebraic 
operator/(const RNAlgebraic& algebraic, RNScalar a)
{
  // Return quotient
  RNAlgebraic result(algebraic);
  result.Divide(a);
  return result;
}



inline RNAlgebraic 
operator/(const RNPolynomial& polynomial, const RNAlgebraic& algebraic)
{
  // Return quotient
  RNAlgebraic result(polynomial, 0);
  result.Divide(algebraic);
  return result;
}


inline RNAlgebraic 
operator/(RNScalar a, const RNAlgebraic& algebraic)
{
  // Return quotent
  RNAlgebraic result(a, 0);
  result.Divide(algebraic);
  return result;
}


