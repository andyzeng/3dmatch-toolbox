// Include file for equation class



////////////////////////////////////////////////////////////////////////
// Class definition
////////////////////////////////////////////////////////////////////////

class RNEquation : public RNExpression {
public:
  // Constructor/destructor
  RNEquation(void);
  RNEquation(RNPolynomial *polynomial);
  RNEquation(RNAlgebraic *algebraic);
  RNEquation(RNScalar c, int nv, const int *v = NULL, const RNScalar *e = NULL, RNBoolean already_sorted = FALSE);
  RNEquation(int operation, RNAlgebraic *operand1, RNAlgebraic *operand2);
  RNEquation(const RNPolynomial& polynomial, int dummy);
  RNEquation(const RNExpression& expression, int dummy);
  RNEquation(const RNEquation& equation);
  virtual ~RNEquation(void);

  // Access functions
  RNSystemOfEquations *SystemOfEquations(void) const;
  int SystemIndex(void) const;

  // Evaluation functions
  RNScalar EvaluateResidual(const RNScalar *x) const;

private:
  friend class RNSystemOfEquations;
  RNSystemOfEquations *system;
  int system_index;

public:
  // Temporary
  RNScalar ResidualThreshold(void) const { return residual_threshold; };
  void SetResidualThreshold(RNScalar threshold) { this->residual_threshold = threshold; };
  RNScalar residual_threshold;
};



////////////////////////////////////////////////////////////////////////
// Inline functions
////////////////////////////////////////////////////////////////////////

inline RNSystemOfEquations *RNEquation::
SystemOfEquations(void) const
{
  // Return system of equations this equation belongs to
  return system;
}



inline int RNEquation::
SystemIndex(void) const
{
  // Return index of this equation in its system of equations
  return system_index;
}



inline RNScalar RNEquation::
EvaluateResidual(const RNScalar *x) const
{
  // Evaluate expression
  return Evaluate(x);
}



