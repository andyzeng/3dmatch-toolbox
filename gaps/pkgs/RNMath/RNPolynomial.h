// Include file for polynomial class



////////////////////////////////////////////////////////////////////////
// Declarations
////////////////////////////////////////////////////////////////////////

class RNPolynomialTerm;



////////////////////////////////////////////////////////////////////////
// Polynomial
////////////////////////////////////////////////////////////////////////

class RNPolynomial {
public:
  // Constructor/destructor
  RNPolynomial(void);
  RNPolynomial(const RNPolynomial& polynomial);
  RNPolynomial(RNScalar c, int v, RNScalar e);
  RNPolynomial(RNScalar c, int nv, const int *v = NULL, const RNScalar *e = NULL, RNBoolean already_sorted = FALSE);
  ~RNPolynomial(void);

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
  RNScalar Degree(void) const;

  // Access functions
  int NTerms(void) const;
  RNPolynomialTerm *Term(int k) const;

  // Manipulation functions
  void Empty(void);
  void Negate(void);
  void Add(RNScalar constant);
  void Subtract(RNScalar constant);
  void Multiply(RNScalar factor);
  void Divide(RNScalar factor);
  void Add(const RNPolynomial& polynomial);
  void Subtract(const RNPolynomial& polynomial);
  void Multiply(const RNPolynomial& polynomial);

  // Assignment operators
  RNPolynomial& operator=(const RNPolynomial& polynomial);
  RNPolynomial& operator+=(const RNPolynomial& polynomial);
  RNPolynomial& operator-=(const RNPolynomial& polynomial);
  RNPolynomial& operator*=(const RNPolynomial& polynomial);
  RNPolynomial& operator=(RNScalar a);
  RNPolynomial& operator+=(RNScalar a);
  RNPolynomial& operator-=(RNScalar a);
  RNPolynomial& operator*=(RNScalar a);
  RNPolynomial& operator/=(RNScalar a);
  
  // Arithmetic operators
  friend RNPolynomial operator-(const RNPolynomial& polynomial);
  friend RNPolynomial operator+(const RNPolynomial& polynomial1, const RNPolynomial& polynomial2);
  friend RNPolynomial operator+(const RNPolynomial& polynomial, RNScalar a);
  friend RNPolynomial operator+(RNScalar a, const RNPolynomial& polynomial);
  friend RNPolynomial operator-(const RNPolynomial& polynomial1, const RNPolynomial& polynomial2);
  friend RNPolynomial operator-(const RNPolynomial& polynomial, RNScalar a);
  friend RNPolynomial operator-(RNScalar a, const RNPolynomial& polynomial);
  friend RNPolynomial operator*(const RNPolynomial& polynomial1, const RNPolynomial& polynomial2);
  friend RNPolynomial operator*(const RNPolynomial& polynomial, RNScalar a);
  friend RNPolynomial operator*(RNScalar a, const RNPolynomial& polynomial);
  friend RNPolynomial operator/(const RNPolynomial& polynomial, RNScalar a);

  // Construction functions
  void AddTerm(RNScalar c, RNBoolean already_unique = FALSE);
  void AddTerm(RNScalar c, int v, RNScalar e, RNBoolean already_unique = FALSE);
  void AddTerm(RNScalar c, int n, const int *v, const RNScalar *e,
    RNBoolean already_sorted = FALSE, RNBoolean already_unique = FALSE);

  // Evaluation functions
  RNScalar Evaluate(const RNScalar *x) const;

  // Partial derivative functions
  RNScalar PartialDerivative(const RNScalar *x, int variable) const;

  // Print functions
  void Print(FILE *fp = stdout) const;

public:
  // Internal functions (for CERES)
  RNScalar Evaluate(double const* const* x) const;
  RNScalar PartialDerivative(double const* const* x, int variable) const;

  // Internal functions (for finding similar terms)
  RNPolynomialTerm *FindTermWithSameVariables(const RNPolynomialTerm *query) const;
  RNPolynomialTerm *FindTermWithVariables(int n, int *v, RNScalar *e) const;

  // Internal functions (for counting unique variables)
  void UpdateVariableRange(int& min_v, int& max_v) const;
  void UpdateVariableIndex(int max_variables, int& variable_count, 
    int *variable_marks, int current_mark, 
    int *index_to_variable = NULL, int *variable_to_index = NULL,
    RNBoolean remap_variables = FALSE) const;

private:
  RNArray<RNPolynomialTerm *> terms;
};



////////////////////////////////////////////////////////////////////////
// Polynomial term
////////////////////////////////////////////////////////////////////////

class RNPolynomialTerm {
public:
  // Constructor/destructor
  RNPolynomialTerm(RNScalar c = 0.0, int nv = 0, const int *v = NULL, const RNScalar *e = NULL, 
    RNBoolean already_sorted = FALSE, RNBoolean already_unique = FALSE);
  RNPolynomialTerm(const RNPolynomialTerm& term);
  ~RNPolynomialTerm(void);

  // Property functions
  RNBoolean IsZero(void) const;
  RNBoolean IsOne(void) const;
  RNBoolean IsConstant(void) const;
  RNBoolean IsLinear(void) const;
  RNBoolean IsQuadratic(void) const;
  RNBoolean HasVariable(int v) const;
  RNScalar Degree(void) const;

  // Access functions
  int NVariables(void) const;
  int Variable(int k) const;
  RNScalar Coefficient(void) const;
  RNScalar Exponent(int k) const;
  const int *Variables(void) const;
  const RNScalar *Exponents(void) const;
  RNPolynomial *Polynomial(void) const;

  // Manipulation functions
  void Empty(void);
  void Negate(void);
  void Multiply(RNScalar factor);
  void Divide(RNScalar factor);
  void SetCoefficient(RNScalar c);
  void SetVariable(int k, int v);
  void SetExponent(int k, RNScalar e);

  // Evaluation functions
  RNScalar Evaluate(const RNScalar *x) const;

  // Partial derivative functions
  int NPartialDerivatives(void) const;
  RNScalar PartialDerivative(const RNScalar *x, int variable) const;

  // Print functions
  void Print(FILE *fp = stdout) const;

public:
  // More internal functions (for CERES)
  RNScalar Evaluate(double const* const* x) const;
  RNScalar PartialDerivative(double const* const* x, int variable) const;

  // Internal functions (for finding similar terms)
  RNBoolean HasSameVariables(const RNPolynomialTerm *query) const;
  RNBoolean HasVariables(int n, const int *v, const RNScalar *e) const;

  // Internal functions (for counting unique variables)
  void UpdateVariableRange(int& min_v, int& max_v) const;
  void UpdateVariableIndex(int max_variables, int& variable_count, 
    int *variable_marks, int current_mark, 
    int *index_to_variable = NULL, int *variable_to_index = NULL,
    RNBoolean remap_variables = FALSE) const;

private:
  friend class RNPolynomial;
  RNPolynomial *polynomial;
  int n;
  RNScalar c;
// #define RN_POLYNOMIAL_TERM_STATIC_MEMORY
#ifdef RN_POLYNOMIAL_TERM_STATIC_MEMORY
  static const int max_variables = 4;
  int v[max_variables];
  RNScalar e[max_variables];
#else
  int *v;
  RNScalar *e;
#endif
};



////////////////////////////////////////////////////////////////////////
// Inline functions for polynomial term
////////////////////////////////////////////////////////////////////////

inline int RNPolynomialTerm::
NVariables(void) const
{
  // Return the number of variables
  return n;
}



inline int RNPolynomialTerm::
NPartialDerivatives(void) const
{
  // Return the number of partial derivatives
  return n;
}



inline int RNPolynomialTerm::
Variable(int k) const
{
  // Return the index of the kth variable
  return v[k];
}



inline RNScalar RNPolynomialTerm::
Exponent(int k) const
{
  // Return the exponent of the kth variable
  return e[k];
}



inline RNScalar RNPolynomialTerm::
Coefficient(void) const
{
  // Return the coefficient
  return c;
}



inline const int *RNPolynomialTerm::
Variables(void) const
{
  // Return the variables
  return v;
}



inline const RNScalar *RNPolynomialTerm::
Exponents(void) const
{
  // Return the exponents
  return e;
}



inline RNPolynomial *RNPolynomialTerm::
Polynomial(void) const
{
  // Return the polynomial this term is part of
  return polynomial;
}



inline void RNPolynomialTerm::
Negate(void)
{
  // Negate coefficient
  c = -c;
}



inline void RNPolynomialTerm::
Multiply(RNScalar factor) 
{
  // Multiply term by a constant factor
  c *= factor;
}



inline void RNPolynomialTerm::
Divide(RNScalar factor) 
{
  // Multiply term by a constant factor
  if (RNIsZero(factor)) return;
  c /= factor;
}



inline void RNPolynomialTerm::
SetCoefficient(RNScalar c)
{
  // Set coefficient
  this->c = c;
}



inline void RNPolynomialTerm::
SetVariable(int k, int v)
{
  // Set index of kth variable
  this->v[k] = v;
}



inline void RNPolynomialTerm::
SetExponent(int k, RNScalar e)
{
  // Set exponent of kth variable
  this->e[k] = e;
}



inline RNBoolean RNPolynomialTerm::
HasSameVariables(const RNPolynomialTerm *query) const
{
  // Check if query has same variables
  return HasVariables(query->NVariables(), query->Variables(), query->Exponents());
}



////////////////////////////////////////////////////////////////////////
// Inline functions for polynomial
////////////////////////////////////////////////////////////////////////

inline RNBoolean RNPolynomial::
IsPolynomial(void) const
{
  // Trivially true
  return TRUE;
}



inline RNBoolean RNPolynomial::
IsAlgebraic(void) const
{
  // Trivially true
  return TRUE;
}



inline int RNPolynomial::
NTerms(void) const
{
  // Return number of terms
  return terms.NEntries();
}



inline int RNPolynomial::
NPartialDerivatives(void) const
{
  // Return the number of partial derivatives
  return NVariables();
}



inline RNPolynomialTerm *RNPolynomial::
Term(int k) const
{
  // Return kth term
  return terms.Kth(k);
}



inline void RNPolynomial::
AddTerm(RNScalar c, RNBoolean already_unique)
{
  // Add term
  AddTerm(c, 0, NULL, NULL, TRUE, already_unique);
}



inline void RNPolynomial::
AddTerm(RNScalar c, int v1, RNScalar e1, RNBoolean already_unique)
{
  // Add term
  AddTerm(c, 1, &v1, &e1, TRUE, already_unique);
}



inline void RNPolynomial::
Add(RNScalar constant)
{
  // Add constant 
  AddTerm(constant);
}



inline void RNPolynomial::
Subtract(RNScalar constant)
{
  // Subtract constant 
  Add(-constant);
}



inline RNPolynomial& RNPolynomial::
operator+=(const RNPolynomial& polynomial)
{
  // Add polynomial
  Add(polynomial);
  return *this;
}



inline RNPolynomial& RNPolynomial::
operator-=(const RNPolynomial& polynomial)
{
  // Subtract polynomial
  Subtract(polynomial);
  return *this;
}



inline RNPolynomial& RNPolynomial::
operator*=(const RNPolynomial& polynomial)
{
  // Multiply polynomial
  Multiply(polynomial);
  return *this;
}



inline RNPolynomial& RNPolynomial::
operator=(RNScalar a)
{
  // Assign constant
  Empty();
  Add(a);
  return *this;
}



inline RNPolynomial& RNPolynomial::
operator+=(RNScalar a)
{
  // Add constant
  Add(a);
  return *this;
}



inline RNPolynomial& RNPolynomial::
operator-=(RNScalar a)
{
  // Subtract constant
  Subtract(a);
  return *this;
}



inline RNPolynomial& RNPolynomial::
operator*=(RNScalar a)
{
  // Multiply by constant
  Multiply(a);
  return *this;
}



inline RNPolynomial& RNPolynomial::
operator/=(RNScalar a)
{
  // Divide by constant
  Divide(a);
  return *this;
}



inline RNPolynomial
operator-(const RNPolynomial& polynomial)
{
  // Return negated polynomial
  RNPolynomial result(polynomial);
  result.Multiply(-1.0);
  return result;
}



inline RNPolynomial 
operator+(const RNPolynomial& polynomial1, const RNPolynomial& polynomial2)
{
  // Return sum
  RNPolynomial result(polynomial1);
  result.Add(polynomial2);
  return result;
}



inline RNPolynomial 
operator+(const RNPolynomial& polynomial, RNScalar constant)
{
  // Return sum
  RNPolynomial result(polynomial);
  result.Add(constant);
  return result;
}



inline RNPolynomial 
operator+(RNScalar constant, const RNPolynomial& polynomial)
{
  // Return sum
  RNPolynomial result(polynomial);
  result.Add(constant);
  return result;
}



inline RNPolynomial 
operator-(const RNPolynomial& polynomial1, const RNPolynomial& polynomial2)
{
  // Return difference
  RNPolynomial result(polynomial1);
  result.Subtract(polynomial2);
  return result;
}



inline RNPolynomial 
operator-(const RNPolynomial& polynomial, RNScalar constant)
{
  // Return difference
  RNPolynomial result(polynomial);
  result.Subtract(constant);
  return result;
}



inline RNPolynomial 
operator-(RNScalar constant, const RNPolynomial& polynomial)
{
  // Return difference
  RNPolynomial result(polynomial);
  result.Negate();
  result.Add(constant);
  return result;
}



inline RNPolynomial 
operator*(const RNPolynomial& polynomial1, const RNPolynomial& polynomial2)
{
  // Return product
  RNPolynomial result(polynomial1);
  result.Multiply(polynomial2);
  return result;
}



inline RNPolynomial 
operator*(const RNPolynomial& polynomial, RNScalar a)
{
  // Return product
  RNPolynomial result(polynomial);
  result.Multiply(a);
  return result;
}



inline RNPolynomial 
operator*(RNScalar a, const RNPolynomial& polynomial)
{
  // Return product
  RNPolynomial result(polynomial);
  result.Multiply(a);
  return result;
}



inline RNPolynomial 
operator/(const RNPolynomial& polynomial, RNScalar a)
{
  // Return quotient
  RNPolynomial result(polynomial);
  result.Divide(a);
  return result;
}






