// Source file for equation class



////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "RNMath/RNMath.h"



////////////////////////////////////////////////////////////////////////
// Functions
////////////////////////////////////////////////////////////////////////

RNEquation::
RNEquation(void)
  : RNExpression(),
    system(NULL),
    system_index(-1),
    residual_threshold(0)
{
}



RNEquation::
RNEquation(RNPolynomial *polynomial)
  : RNExpression(polynomial),
    system(NULL),
    system_index(-1),
    residual_threshold(0)
{
}



RNEquation::
RNEquation(RNAlgebraic *algebraic)
  : RNExpression(algebraic),
    system(NULL),
    system_index(-1),
    residual_threshold(0)
{
}



RNEquation::
RNEquation(RNScalar c, int nv, const int *v, const RNScalar *e, RNBoolean already_sorted)
  : RNExpression(c, nv, v, e, already_sorted),
    system(NULL),
    system_index(-1),
    residual_threshold(0)
{
}



RNEquation::
RNEquation(int operation, RNExpression *operand1, RNExpression *operand2)
  : RNExpression(operation, operand1, operand2),
    system(NULL),
    system_index(-1),
    residual_threshold(0)
{
}



RNEquation::
RNEquation(const RNPolynomial& polynomial, int dummy)
  : RNExpression(polynomial, dummy),
    system(NULL),
    system_index(-1),
    residual_threshold(0)
{
}



RNEquation::
RNEquation(const RNExpression& expression, int dummy)
  : RNExpression(expression),
    system(NULL),
    system_index(-1),
    residual_threshold(0)
{
}



RNEquation::
RNEquation(const RNEquation& equation)
  : RNExpression(equation),
    system(NULL),
    system_index(-1),
    residual_threshold(equation.residual_threshold)
{
}



RNEquation::
~RNEquation(void)
{
  // Remove from system of equations
  if (system) system->RemoveEquation(this);
}



