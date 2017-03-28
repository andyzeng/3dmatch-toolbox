// Source file for algebraic expression class



////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "RNMath/RNMath.h"



////////////////////////////////////////////////////////////////////////
// Functions
////////////////////////////////////////////////////////////////////////

RNAlgebraic::
RNAlgebraic(void)
  : operation(RN_ZERO_OPERATION),
    polynomial(NULL)
{
  // Initialize operands
  operands[0] = NULL;
  operands[1] = NULL;
}



RNAlgebraic::
RNAlgebraic(RNPolynomial *p)
  : operation(RN_ZERO_OPERATION),
    polynomial(NULL)
{
  // NOTE: THIS TAKES OVER MANAGEMENT OF MEMORY FOR POLYNOMIAL
  // SPECIFICALLY, POLYNOMIAL WILL BE DELETED WHEN ALGEBRAIC IS DECONSTRUCTED

  // Initialize operands
  operands[0] = NULL;
  operands[1] = NULL;

  // Check polynomial
  if (!p) {
    return;
  }
  else if (p->IsZero()) {
    delete p;
    return;
  }

  // Assign polynomial
  operation = RN_POLYNOMIAL_OPERATION;
  this->polynomial = p;

  // Just checking
  assert(IsValid());
}



RNAlgebraic::
RNAlgebraic(RNAlgebraic *algebraic)
  : operation(RN_ZERO_OPERATION),
    polynomial(NULL)
{
  // NOTE: THIS TAKES OVER MANAGEMENT OF MEMORY FOR ALGEBRAIC 
  // SPECIFICALLY, ALGEBRAIC WILL BE DELETED WHEN THIS ALGEBRAIC IS DECONSTRUCTED

  // Initialize operands
  operands[0] = NULL;
  operands[1] = NULL;

  // Check algebraic
  if (!algebraic) {
    return;
  }
  else if (algebraic->IsZero()) {
    delete algebraic;
    return;
  }

  // Copy stuff from algebraic
  operation = algebraic->operation;
  polynomial = algebraic->polynomial;
  operands[0] = algebraic->operands[0];
  operands[1] = algebraic->operands[1];

  // Delete algebraic
  algebraic->operation = RN_ZERO_OPERATION;
  algebraic->operands[0] = NULL;
  algebraic->operands[1] = NULL;
  algebraic->polynomial = NULL;
  delete algebraic;

  // Just checking
  assert(IsValid());
}



RNAlgebraic::
RNAlgebraic(RNScalar c, int v, RNScalar e)
  : operation(RN_ZERO_OPERATION),
    polynomial(NULL)
{
  // Initialize operands
  operands[0] = NULL;
  operands[1] = NULL;

  // Initialize polynomial
  if (c != 0) {
    polynomial = new RNPolynomial(c, v, e);
    operation = RN_POLYNOMIAL_OPERATION;
  }

  // Just checking
  assert(IsValid());
}



RNAlgebraic::
RNAlgebraic(RNScalar c, int nv, const int *v, const RNScalar *e, RNBoolean already_sorted)
  : operation(RN_ZERO_OPERATION),
    polynomial(NULL)
{
  // Initialize operands
  operands[0] = NULL;
  operands[1] = NULL;

  // Initialize polynomial
  if (c != 0) {
    polynomial = new RNPolynomial(c, nv, v, e, already_sorted);
    operation = RN_POLYNOMIAL_OPERATION;
  }

  // Just checking
  assert(IsValid());
}



RNAlgebraic::
RNAlgebraic(const RNPolynomial& p, int dummy)
  : operation(RN_ZERO_OPERATION),
    polynomial(NULL)
{
  // Initialize operands
  operands[0] = NULL;
  operands[1] = NULL;

  // Check polynomial
  if (p.IsZero()) return;

  // Initialize polynomial
  this->operation = RN_POLYNOMIAL_OPERATION;
  this->polynomial = new RNPolynomial(p);

  // Just checking
  assert(IsValid());
}



RNAlgebraic::
RNAlgebraic(const RNAlgebraic& algebraic)
  : operation(RN_ZERO_OPERATION),
    polynomial(NULL)
{
  // Initialize operands
  operands[0] = NULL;
  operands[1] = NULL;

  // Check algebraic
  if (algebraic.IsZero()) return;

  // Copy operands
  switch(algebraic.operation) {
  case RN_ZERO_OPERATION:
    return;

  case RN_POLYNOMIAL_OPERATION:
    assert(algebraic.polynomial);
    operation = algebraic.operation;
    polynomial = new RNPolynomial(*(algebraic.polynomial));
    return;

  default:
    assert(algebraic.operands[0]);
    assert(algebraic.operands[1]);
    operation = algebraic.operation;
    operands[0] = new RNAlgebraic(*(algebraic.operands[0]));
    operands[1] = new RNAlgebraic(*(algebraic.operands[1]));
    return;
  }

  // Just checking
  assert(IsValid());
}



RNAlgebraic::
RNAlgebraic(int op, RNScalar operand1, RNScalar operand2)
  : operation(RN_ZERO_OPERATION),
    polynomial(NULL)
{
  // Initialize operands
  operands[0] = NULL;
  operands[1] = NULL;

  // Construct 
  Construct(op, operand1, operand2);
}



RNAlgebraic::
RNAlgebraic(int op, RNScalar operand1, RNPolynomial *operand2)
  : operation(RN_ZERO_OPERATION),
    polynomial(NULL)
{
  // Initialize operands
  operands[0] = NULL;
  operands[1] = NULL;

  // Construct 
  Construct(op, operand1, operand2);
}



RNAlgebraic::
RNAlgebraic(int op, RNScalar operand1, RNAlgebraic *operand2)
  : operation(RN_ZERO_OPERATION),
    polynomial(NULL)
{
  // Initialize operands
  operands[0] = NULL;
  operands[1] = NULL;

  // Construct 
  Construct(op, operand1, operand2);
}



RNAlgebraic::
RNAlgebraic(int op, RNPolynomial *operand1, RNScalar operand2)
  : operation(RN_ZERO_OPERATION),
    polynomial(NULL)
{
  // Initialize operands
  operands[0] = NULL;
  operands[1] = NULL;

  // Construct 
  Construct(op, operand1, operand2);
}



RNAlgebraic::
RNAlgebraic(int op, RNPolynomial *operand1, RNPolynomial *operand2)
  : operation(RN_ZERO_OPERATION),
    polynomial(NULL)
{
  // Initialize operands
  operands[0] = NULL;
  operands[1] = NULL;

  // Construct 
  Construct(op, operand1, operand2);
}



RNAlgebraic::
RNAlgebraic(int op, RNPolynomial *operand1, RNAlgebraic *operand2)
  : operation(RN_ZERO_OPERATION),
    polynomial(NULL)
{
  // Initialize operands
  operands[0] = NULL;
  operands[1] = NULL;

  // Construct 
  Construct(op, operand1, operand2);
}



RNAlgebraic::
RNAlgebraic(int op, RNAlgebraic *operand1, RNScalar operand2)
  : operation(RN_ZERO_OPERATION),
    polynomial(NULL)
{
  // Initialize operands
  operands[0] = NULL;
  operands[1] = NULL;

  // Construct 
  Construct(op, operand1, operand2);
}



RNAlgebraic::
RNAlgebraic(int op, RNAlgebraic *operand1, RNPolynomial *operand2)
  : operation(RN_ZERO_OPERATION),
    polynomial(NULL)
{
  // Initialize operands
  operands[0] = NULL;
  operands[1] = NULL;

  // Construct 
  Construct(op, operand1, operand2);
}



RNAlgebraic::
RNAlgebraic(int op, RNAlgebraic *operand1, RNAlgebraic *operand2)
  : operation(RN_ZERO_OPERATION),
    polynomial(NULL)
{
  // Initialize operands
  operands[0] = NULL;
  operands[1] = NULL;

  // Construct 
  Construct(op, operand1, operand2);
}



RNAlgebraic::
~RNAlgebraic(void)
{
  // Just checking
  assert(IsValid());

  // Delete everything
  if (polynomial) delete polynomial;
  if (operands[0]) delete operands[0];
  if (operands[1]) delete operands[1];
}



void RNAlgebraic::
Construct(int op, RNScalar operand1, RNScalar operand2, RNBoolean force)
{
  // Condense scalars based on operation
  switch(op) {
  case RN_ZERO_OPERATION:
    break;

  case RN_ADD_OPERATION: {
    RNScalar sum = operand1 + operand2;
    if (sum == 0) break;
    operation = RN_POLYNOMIAL_OPERATION;
    polynomial = new RNPolynomial(sum, 0);
    break; }

  case RN_SUBTRACT_OPERATION: {
    RNScalar difference = operand1 - operand2;
    if (difference == 0) break;
    operation = RN_POLYNOMIAL_OPERATION;
    polynomial = new RNPolynomial(difference, 0);
    break; }

  case RN_MULTIPLY_OPERATION: {
    RNScalar product = operand1 * operand2;
    if (product == 0) break;
    operation = RN_POLYNOMIAL_OPERATION;
    polynomial = new RNPolynomial(product, 0);
    break; }

  case RN_DIVIDE_OPERATION:
    if (operand2 == 0) {
      operation = RN_POLYNOMIAL_OPERATION;
      polynomial = new RNPolynomial(RN_INFINITY, 0);
    }
    else {
      RNScalar quotient = operand1 / operand2;
      if (quotient == 0) break;
      operation = RN_POLYNOMIAL_OPERATION;
      polynomial = new RNPolynomial(quotient, 0);
    }
    break;

  case RN_POW_OPERATION:
    if (operand2 == 0) {
      operation = RN_POLYNOMIAL_OPERATION;
      polynomial = new RNPolynomial(1.0, 0);
    }
    else {
      RNScalar result = pow(operand1, operand2);
      if (result == 0) break;
      operation = RN_POLYNOMIAL_OPERATION;
      polynomial = new RNPolynomial(result, 0);
    }
    break;

  default:
    RNAbort("Invalid operation %d\n", op);
    break;
  }

  // Just checking
  assert(IsValid());
}



void RNAlgebraic::
Construct(int op, RNScalar operand1, RNPolynomial *operand2, RNBoolean force)
{
  // NOTE: THIS TAKES OVER MANAGEMENT OF MEMORY FOR OPERAND2
  // SPECIFICALLY, OPERANDS WILL BE DELETED WHEN ALGEBRAIC IS DECONSTRUCTED
  assert(operand2);

  // Check type of operand2
  if (!force && (operand2->NTerms() == 1) && (operand2->Term(0)->Degree() == 0)) {
    Construct(op, operand1, operand2->Term(0)->Coefficient());
    delete operand2;
  }
  else {
    switch(op) {
    case RN_ZERO_OPERATION:
      break;

    case RN_ADD_OPERATION: 
      if ((operand1 == 0) && (operand2->IsZero())) {
        delete operand2;
      }
      else {
        operation = RN_POLYNOMIAL_OPERATION;
        this->polynomial = operand2;
        this->polynomial->Add(operand1);
      }
      break; 

    case RN_SUBTRACT_OPERATION: 
      if ((operand1 == 0) && (operand2->IsZero())) {
        delete operand2;
      }
      else {
        operation = RN_POLYNOMIAL_OPERATION;
        this->polynomial = operand2;
        this->polynomial->Negate();
        this->polynomial->Add(operand1);
      }
      break; 

    case RN_MULTIPLY_OPERATION: 
      if ((operand1 == 0) || (operand2->IsZero())) {
        delete operand2;
      }
      else if (operand2->IsOne()) {
        operation = RN_POLYNOMIAL_OPERATION;
        this->polynomial = new RNPolynomial(operand1, 0);
        delete operand2;
      }
      else {
        operation = RN_POLYNOMIAL_OPERATION;
        this->polynomial = operand2;
        this->polynomial->Multiply(operand1);
      }
      break; 

    case RN_DIVIDE_OPERATION:
      if (operand1 == 0) {
        delete operand2;
      }
      else if (operand2->IsZero()) {
        operation = RN_POLYNOMIAL_OPERATION;
        polynomial = new RNPolynomial(RN_INFINITY, 0);
        delete operand2;
      }  
      else if (operand2->IsOne()) {
        operation = RN_POLYNOMIAL_OPERATION;
        this->polynomial = new RNPolynomial(operand1, 0);
        delete operand2;
      }
      else {      
        Construct(op, new RNAlgebraic(operand1, 0), new RNAlgebraic(operand2), TRUE);
      }
      break;

    case RN_POW_OPERATION:
      if (operand1 == 0) {
        delete operand2;
      }
      else if (operand2->IsZero()) {
        operation = RN_POLYNOMIAL_OPERATION;
        this->polynomial = new RNPolynomial(1.0, 0);
        delete operand2;
      }
      else if (operand2->IsOne()) {
        operation = RN_POLYNOMIAL_OPERATION;
        this->polynomial = new RNPolynomial(operand1, 0);
        delete operand2;
      }
      else {      
        Construct(op, new RNAlgebraic(operand1, 0), new RNAlgebraic(operand2), TRUE);
      }
      break;

    default:
      RNAbort("Invalid operation %d\n", op);
      break;
    }
  }

  // Just checking
  assert(IsValid());
}



void RNAlgebraic::
Construct(int op, RNScalar operand1, RNAlgebraic *operand2, RNBoolean force)
{
  // NOTE: THIS TAKES OVER MANAGEMENT OF MEMORY FOR OPERAND2
  // SPECIFICALLY, OPERANDS WILL BE DELETED WHEN ALGEBRAIC IS DECONSTRUCTED
  assert(operand2);
  assert(operand2->IsValid());

  // Check type of operand2
  if (!force && (operand2->operation == RN_POLYNOMIAL_OPERATION)) {
    // Construct with polynomial
    Construct(op, operand1, operand2->polynomial);
    operand2->operation = RN_ZERO_OPERATION;
    operand2->operands[0] = NULL;
    operand2->operands[1] = NULL;
    operand2->polynomial = NULL;
    delete operand2;
  }
  else {
    // Construct with algebraic
    Construct(op, new RNAlgebraic(operand1, 0), operand2, TRUE);
  }
}



void RNAlgebraic::
Construct(int op, RNPolynomial *operand1, RNScalar operand2, RNBoolean force)
{
  // NOTE: THIS TAKES OVER MANAGEMENT OF MEMORY FOR OPERAND1
  // SPECIFICALLY, OPERANDS WILL BE DELETED WHEN ALGEBRAIC IS DECONSTRUCTED
  assert(operand1);

  // Check type of operand1
  if (!force && (operand1->NTerms() == 1) && (operand1->Term(0)->Degree() == 0)) {
    Construct(op, operand1->Term(0)->Coefficient(), operand2);
    delete operand1;
  }
  else {
    switch(op) {
    case RN_ZERO_OPERATION:
      break;

    case RN_ADD_OPERATION:
      if ((operand1->IsZero()) && (operand2 == 0)) {
        delete operand1;
      }
      else {
        operation = RN_POLYNOMIAL_OPERATION;
        this->polynomial = operand1;
        this->polynomial->Add(operand2);
      }
      break; 

    case RN_SUBTRACT_OPERATION: 
      if ((operand1->IsZero()) && (operand2 == 0)) {
        delete operand1;
      }
      else {
        operation = RN_POLYNOMIAL_OPERATION;
        this->polynomial = operand1;
        this->polynomial->Subtract(operand2);
      }
      break; 

    case RN_MULTIPLY_OPERATION: 
      if ((operand1->IsZero()) || (operand2 == 0)) {
        delete operand1;
      }
      else {
        operation = RN_POLYNOMIAL_OPERATION;
        this->polynomial = operand1;
        this->polynomial->Multiply(operand2);
      }
      break; 

    case RN_DIVIDE_OPERATION:
      if (operand1->IsZero()) {
        delete operand1;
      }
      else if (operand2 == 0) {
        operation = RN_POLYNOMIAL_OPERATION;
        polynomial = new RNPolynomial(RN_INFINITY, 0);
        delete operand1;
      }  
      else if (operand1->IsOne()) {
        operation = RN_POLYNOMIAL_OPERATION;
        this->polynomial = new RNPolynomial(operand2, 0);
        delete operand1;
      }
      else {
        operation = RN_POLYNOMIAL_OPERATION;
        this->polynomial = operand1;
        this->polynomial->Divide(operand2);
      }
      break;

    case RN_POW_OPERATION:
      if (operand1->IsZero()) {
        delete operand1;
      }
      else if (operand2 == 0) {
        operation = RN_POLYNOMIAL_OPERATION;
        this->polynomial = new RNPolynomial(1.0, 0);
        delete operand1;
      }
      else if (operand2 == 1.0) {
        operation = RN_POLYNOMIAL_OPERATION;
        this->polynomial = operand1;
      }
      else {      
        Construct(op, new RNAlgebraic(operand1), new RNAlgebraic(operand2, 0), TRUE);
      }
      break;

    default:
      RNAbort("Invalid operation %d\n", op);
      break;
    }
  }

  // Just checking
  assert(IsValid());
}



void RNAlgebraic::
Construct(int op, RNPolynomial *operand1, RNPolynomial *operand2, RNBoolean force)
{
  // NOTE: THIS TAKES OVER MANAGEMENT OF MEMORY FOR OPERANDS
  // SPECIFICALLY, OPERANDS WILL BE DELETED WHEN ALGEBRAIC IS DECONSTRUCTED
  assert(operand1 && operand2);

  // Check types of operands
  if (!force && (operand1->NTerms() == 1) && (operand1->Term(0)->Degree() == 0)) {
    Construct(op, operand1->Term(0)->Coefficient(), operand2);
    delete operand1;
  }
  else if (!force && (operand2->NTerms() == 1) && (operand2->Term(0)->Degree() == 0)) {
    Construct(op, operand1, operand2->Term(0)->Coefficient());
    delete operand2;
  }
  else {
    switch(op) {
    case RN_ZERO_OPERATION:
      break;

    case RN_ADD_OPERATION: 
      if (operand1->IsZero() && operand2->IsZero()) {
        delete operand1;
        delete operand2;
      }
      else {
        operation = RN_POLYNOMIAL_OPERATION;
        this->polynomial = operand1;
        this->polynomial->Add(*operand2);
        delete operand2;
      }
      break; 

    case RN_SUBTRACT_OPERATION: 
      if (operand1->IsZero() && operand2->IsZero()) {
        delete operand1;
        delete operand2;
      }
      else {
        operation = RN_POLYNOMIAL_OPERATION;
        this->polynomial = operand1;
        this->polynomial->Subtract(*operand2);
        delete operand2;
      }
      break; 

    case RN_MULTIPLY_OPERATION: 
      if (operand1->IsZero() || operand2->IsZero()) {
        delete operand1;
        delete operand2;
      }
      else if (operand1->IsOne()) {
        operation = RN_POLYNOMIAL_OPERATION;
        this->polynomial = operand2;
        delete operand1;
      }
      else if (operand2->IsOne()) {
        operation = RN_POLYNOMIAL_OPERATION;
        this->polynomial = operand1;
        delete operand2;
      }
      else if ((operand1->NTerms() == 1) && (operand2->NTerms() == 1)) {
        operation = RN_POLYNOMIAL_OPERATION;
        this->polynomial = operand1;
        this->polynomial->Multiply(*operand2);
        delete operand2;
      }
      else {
        Construct(op, new RNAlgebraic(operand1), new RNAlgebraic(operand2), TRUE);
      }
      break; 

    case RN_DIVIDE_OPERATION:
      if (operand1->IsZero()) {
        delete operand1;
        delete operand2;
      }
      else if (operand2->IsZero()) {
        operation = RN_POLYNOMIAL_OPERATION;
        polynomial = new RNPolynomial(RN_INFINITY, 0);
        delete operand1;
        delete operand2;
      }
      else if (operand2->IsOne()) {
        operation = RN_POLYNOMIAL_OPERATION;
        polynomial = operand1;
        delete operand2;
      }
      else {
        Construct(op, new RNAlgebraic(operand1), new RNAlgebraic(operand2), TRUE);
      }
      break;

    case RN_POW_OPERATION:
      if (operand1->IsZero()) {
        delete operand1;
        delete operand2;
      }
      else if (operand2->IsZero()) {
        operation = RN_POLYNOMIAL_OPERATION;
        this->polynomial = new RNPolynomial(1.0, 0);
        delete operand1;
        delete operand2;
      }
      else if (operand2->IsOne()) {
        operation = RN_POLYNOMIAL_OPERATION;
        this->polynomial = operand1;
        delete operand2;
      }
      else {      
        Construct(op, new RNAlgebraic(operand1), new RNAlgebraic(operand2), TRUE);
      }
      break;

    default:
      RNAbort("Invalid operation %d\n", op);
      break;
    }
  }
  
  // Just checking
  assert(IsValid());
}



void RNAlgebraic::
Construct(int op, RNPolynomial *operand1, RNAlgebraic *operand2, RNBoolean force)
{
  // NOTE: THIS TAKES OVER MANAGEMENT OF MEMORY FOR OPERANDS
  // SPECIFICALLY, OPERANDS WILL BE DELETED WHEN ALGEBRAIC IS DECONSTRUCTED

  // Check types of operands
  if (!force && (operand2->operation == RN_POLYNOMIAL_OPERATION)) {
    Construct(op, operand1, operand2->polynomial);
    operand2->operation = RN_ZERO_OPERATION;
    operand2->operands[0] = NULL;
    operand2->operands[1] = NULL;
    operand2->polynomial = NULL;
    delete operand2;
  }
  else if (!force && (operand1->NTerms() == 1) && (operand1->Term(0)->Degree() == 0)) {
    Construct(op, operand1->Term(0)->Coefficient(), operand2);
    delete operand1;
  }
  else {
    Construct(op, new RNAlgebraic(operand1), operand2, TRUE);
  }
}



void RNAlgebraic::
Construct(int op, RNAlgebraic *operand1, RNScalar operand2, RNBoolean force)
{
  // Check type of operand1
  if (!force && (operand1->operation == RN_POLYNOMIAL_OPERATION)) {
    Construct(op, operand1->polynomial, operand2);
    operand1->operation = RN_ZERO_OPERATION;
    operand1->operands[0] = NULL;
    operand1->operands[1] = NULL;
    operand1->polynomial = NULL;
    delete operand1;
  }
  else {
    Construct(op, operand1, new RNAlgebraic(operand2, 0), TRUE);
  }
}



void RNAlgebraic::
Construct(int op, RNAlgebraic *operand1, RNPolynomial *operand2, RNBoolean force)
{
  // NOTE: THIS TAKES OVER MANAGEMENT OF MEMORY FOR OPERANDS
  // SPECIFICALLY, OPERANDS WILL BE DELETED WHEN ALGEBRAIC IS DECONSTRUCTED

  // Check types of operands
  if (!force && (operand1->operation == RN_POLYNOMIAL_OPERATION)) {
    Construct(op, operand1->polynomial, operand2);
    operand1->operation = RN_ZERO_OPERATION;
    operand1->operands[0] = NULL;
    operand1->operands[1] = NULL;
    operand1->polynomial = NULL;
    delete operand1;
  }
  else if (!force && (operand2->NTerms() == 1) && (operand2->Term(0)->Degree() == 0)) {
    Construct(op, operand1, operand2->Term(0)->Coefficient());
    delete operand2;
  }
  else {
    Construct(op, operand1, new RNAlgebraic(operand2), TRUE);
  }
}



void RNAlgebraic::
Construct(int op, RNAlgebraic *operand1, RNAlgebraic *operand2, RNBoolean force)
{
  // NOTE: THIS TAKES OVER MANAGEMENT OF MEMORY FOR OPERANDS
  // SPECIFICALLY, OPERANDS WILL BE DELETED WHEN ALGEBRAIC IS DECONSTRUCTED
  assert(operand1 && operand2);
  assert(operand1->IsValid());
  assert(operand2->IsValid());

  // Check types of operands
  if (!force && (operand1->operation == RN_POLYNOMIAL_OPERATION)) {
    Construct(op, operand1->polynomial, operand2);
    operand1->operation = RN_ZERO_OPERATION;
    operand1->operands[0] = NULL;
    operand1->operands[1] = NULL;
    operand1->polynomial = NULL;
    delete operand1;
  }
  else if (!force && (operand2->operation == RN_POLYNOMIAL_OPERATION)) {
    Construct(op, operand1, operand2->polynomial);
    operand2->operation = RN_ZERO_OPERATION;
    operand2->operands[0] = NULL;
    operand2->operands[1] = NULL;
    operand2->polynomial = NULL;
    delete operand2;
  }
  else {
    switch(op) {
    case RN_ZERO_OPERATION:
      delete operand1;
      delete operand2;
      break;

    case RN_ADD_OPERATION:
    case RN_SUBTRACT_OPERATION:
      if (operand1->IsZero()) {
        if (operand2->IsZero()) {
          // Both operands are zero -- delete both 
          delete operand1;
          delete operand2;
        }
        else {
          // Operand1 is zero - copy stuff from operand2
          operation = operand2->operation;
          polynomial = operand2->polynomial;
          operands[0] = operand2->operands[0];
          operands[1] = operand2->operands[1];
          if (op == RN_SUBTRACT_OPERATION) Negate();
        
          // Delete operands
          operand2->operation = RN_ZERO_OPERATION;
          operand2->operands[0] = NULL;
          operand2->operands[1] = NULL;
          operand2->polynomial = NULL;
          delete operand1;
          delete operand2;
        }
      }
      else if (operand2->IsZero()) {
        // Operand2 is zero - copy stuff from operand1
        operation = operand1->operation;
        polynomial = operand1->polynomial;
        operands[0] = operand1->operands[0];
        operands[1] = operand1->operands[1];
      
        // Delete operands
        operand1->operation = RN_ZERO_OPERATION;
        operand1->operands[0] = NULL;
        operand1->operands[1] = NULL;
        operand1->polynomial = NULL;
        delete operand1;
        delete operand2;
      }
      else {
        // Initialize binary operation
        operation = op;
        operands[0] = operand1;
        operands[1] = operand2;
      }
      break;

    case RN_MULTIPLY_OPERATION:
      if ((operand1->IsZero()) || (operand2->IsZero())) {
        delete operand1;
        delete operand2;
      }
      else if ((operand1->IsOne()) && (operand2->IsOne())) {
        operation = RN_POLYNOMIAL_OPERATION;
        polynomial = new RNPolynomial(1.0, 0);
        delete operand1;
        delete operand2;
      }
      else if (operand1->IsOne()) {
        // Operand1 is one - copy stuff from operand2
        operation = operand2->operation;
        polynomial = operand2->polynomial;
        operands[0] = operand2->operands[0];
        operands[1] = operand2->operands[1];
        
        // Delete operands
        operand2->operation = RN_ZERO_OPERATION;
        operand2->operands[0] = NULL;
        operand2->operands[1] = NULL;
        operand2->polynomial = NULL;
        delete operand1;
        delete operand2;
      }
      else if (operand2->IsOne()) {
        // Operand2 is one - copy stuff from operand1
        operation = operand1->operation;
        polynomial = operand1->polynomial;
        operands[0] = operand1->operands[0];
        operands[1] = operand1->operands[1];
        
        // Delete operands
        operand1->operation = RN_ZERO_OPERATION;
        operand1->operands[0] = NULL;
        operand1->operands[1] = NULL;
        operand1->polynomial = NULL;
        delete operand1;
        delete operand2;
      }
      else {
        // Initialize binary operation
        operation = op;
        operands[0] = operand1;
        operands[1] = operand2;
      }
      break;

    case RN_DIVIDE_OPERATION:
      if (operand1->IsZero()) {
        // Numerator is zero
        delete operand1;
        delete operand2;
      }
      else if (operand2->IsZero()) {
        // Denominator is zero
        operation = RN_POLYNOMIAL_OPERATION;
        polynomial = new RNPolynomial(RN_INFINITY, 0);
        delete operand1;
        delete operand2;
      }
      else if (operand2->IsOne()) {
        // Operand2 is one - copy stuff from operand1
        operation = operand1->operation;
        polynomial = operand1->polynomial;
        operands[0] = operand1->operands[0];
        operands[1] = operand1->operands[1];
        
        // Delete operands
        operand1->operation = RN_ZERO_OPERATION;
        operand1->operands[0] = NULL;
        operand1->operands[1] = NULL;
        operand1->polynomial = NULL;
        delete operand1;
        delete operand2;
      }
      else {
        // Initialize binary operation
        operation = op;
        operands[0] = operand1;
        operands[1] = operand2;
      }
      break;

    case RN_POW_OPERATION:
      if (operand1->IsZero()) {
        // Base is zero
        delete operand1;
        delete operand2;
      }
      else if ((operand1->IsOne()) || (operand2->IsZero())) {
        operation = RN_POLYNOMIAL_OPERATION;
        polynomial = new RNPolynomial(1.0, 0);
        delete operand1;
        delete operand2;
      }
      else if (operand2->IsOne()) {
        // Operand2 is one - copy stuff from operand1
        operation = operand1->operation;
        polynomial = operand1->polynomial;
        operands[0] = operand1->operands[0];
        operands[1] = operand1->operands[1];
        
        // Delete operands
        operand1->operation = RN_ZERO_OPERATION;
        operand1->operands[0] = NULL;
        operand1->operands[1] = NULL;
        operand1->polynomial = NULL;
        delete operand1;
        delete operand2;
      }
      else {
        // Initialize binary operation
        operation = op;
        operands[0] = operand1;
        operands[1] = operand2;
      }
      break;

    default:
      RNAbort("Invalid operation %d\n", op);
      break;
    }
  }
  
  // Just checking
  assert(IsValid());
}



RNBoolean RNAlgebraic::
IsZero(void) const
{
  // Return if algebraic is definitely zero for all variable settings
  switch(operation) {
  case RN_ZERO_OPERATION:
    return TRUE;

  case RN_POLYNOMIAL_OPERATION:
    return polynomial->IsZero();

  case RN_ADD_OPERATION:
  case RN_SUBTRACT_OPERATION:
    return (operands[0]->IsZero() && operands[1]->IsZero());

  case RN_MULTIPLY_OPERATION:
    return (operands[0]->IsZero() || operands[1]->IsZero());

  case RN_DIVIDE_OPERATION:
    return (operands[0]->IsZero() && !operands[1]->IsZero());

  case RN_POW_OPERATION:
    return (operands[0]->IsZero());
  }

  // Should never get here
  RNAbort("Unrecognized operation %d in algebraic");
  return FALSE;
}



RNBoolean RNAlgebraic::
IsOne(void) const
{
  // Return if algebraic algebraic is constant
  switch(operation) {
  case RN_ZERO_OPERATION:
    return FALSE;

  case RN_POLYNOMIAL_OPERATION:
    return polynomial->IsOne();

  case RN_ADD_OPERATION:
    return ((operands[0]->IsOne() && operands[1]->IsZero()) ||
            (operands[1]->IsOne() && operands[0]->IsZero()));

  case RN_SUBTRACT_OPERATION:
    return (operands[0]->IsOne() && operands[1]->IsZero());

  case RN_MULTIPLY_OPERATION:
  case RN_DIVIDE_OPERATION:
    return (operands[0]->IsOne() && operands[1]->IsOne());

  case RN_POW_OPERATION:
    if (operands[0]->IsOne()) return TRUE;
    if (operands[1]->IsZero()) return TRUE;
    return FALSE;
  }

  // Should never get here
  RNAbort("Unrecognized operation %d in algebraic");
  return FALSE;
}



RNBoolean RNAlgebraic::
IsConstant(void) const
{
  // Return if algebraic algebraic is constant
  switch(operation) {
  case RN_ZERO_OPERATION:
    return TRUE;

  case RN_POLYNOMIAL_OPERATION:
    return polynomial->IsConstant();

  case RN_ADD_OPERATION:
  case RN_SUBTRACT_OPERATION:
  case RN_MULTIPLY_OPERATION:
  case RN_DIVIDE_OPERATION:
    return (operands[0]->IsConstant() && operands[1]->IsConstant());

  case RN_POW_OPERATION:
    if (operands[0]->IsConstant() && operands[1]->IsConstant()) return TRUE;
    if (operands[1]->IsZero()) return TRUE;
    return FALSE;
  }

  // Should never get here
  RNAbort("Unrecognized operation %d in algebraic");
  return FALSE;
}



RNBoolean RNAlgebraic::
IsLinear(void) const
{
  // Return if algebraic is linear
  switch(operation) {
  case RN_ZERO_OPERATION:
    return TRUE;

  case RN_POLYNOMIAL_OPERATION:
    return polynomial->IsLinear();

  case RN_ADD_OPERATION:
  case RN_SUBTRACT_OPERATION:
    return (operands[0]->IsLinear() && operands[1]->IsLinear());

  case RN_MULTIPLY_OPERATION:
    if (operands[0]->IsLinear() && operands[1]->IsConstant()) return TRUE;
    if (operands[0]->IsConstant() && operands[1]->IsLinear()) return TRUE;
    return FALSE;

  case RN_DIVIDE_OPERATION:
    return (operands[0]->IsLinear() && operands[1]->IsConstant());

  case RN_POW_OPERATION:
    return (operands[1]->IsZero());
  }

  // Should never get here
  RNAbort("Unrecognized operation %d in algebraic");
  return FALSE;
}



RNBoolean RNAlgebraic::
IsQuadratic(void) const
{
  // Return if algebraic is quadratic
  switch(operation) {
  case RN_ZERO_OPERATION:
    return TRUE;

  case RN_POLYNOMIAL_OPERATION:
    return polynomial->IsQuadratic();

  case RN_ADD_OPERATION:
  case RN_SUBTRACT_OPERATION:
    return (operands[0]->IsQuadratic() && operands[1]->IsQuadratic());

  case RN_MULTIPLY_OPERATION:
    if (operands[0]->IsQuadratic() && operands[1]->IsConstant()) return TRUE;
    if (operands[0]->IsConstant() && operands[1]->IsQuadratic()) return TRUE;
    if (operands[0]->IsLinear() && operands[1]->IsLinear()) return TRUE;
    return FALSE;

  case RN_DIVIDE_OPERATION:
    return (operands[0]->IsQuadratic() && operands[1]->IsConstant());

  case RN_POW_OPERATION:
    return (operands[1]->IsZero());
  }

  // Should never get here
  RNAbort("Unrecognized operation %d in algebraic");
  return FALSE;
}



RNBoolean RNAlgebraic::
IsPolynomial(void) const
{
  // Return if algebraic is polynomial
  switch(operation) {
  case RN_ZERO_OPERATION:
    return TRUE;

  case RN_POLYNOMIAL_OPERATION:
    return TRUE;

  case RN_ADD_OPERATION:
  case RN_SUBTRACT_OPERATION:
  case RN_MULTIPLY_OPERATION:
    return (operands[0]->IsPolynomial() && operands[1]->IsPolynomial());

  case RN_DIVIDE_OPERATION:
    return (operands[0]->IsPolynomial() && operands[1]->IsConstant());

  case RN_POW_OPERATION:
    return (operands[1]->IsConstant());
  }

  // Should never get here
  RNAbort("Unrecognized operation %d in algebraic");
  return FALSE;
}



RNBoolean RNAlgebraic::
HasVariable(int variable) const
{
  // Return whether algebraic has a variable
  switch(operation) {
  case RN_ZERO_OPERATION:
    return FALSE;

  case RN_POLYNOMIAL_OPERATION:
    return polynomial->HasVariable(variable);
   
  default:
    if (operands[0]->HasVariable(variable)) return TRUE;
    if (operands[1]->HasVariable(variable)) return TRUE;
    return FALSE;
  }
}



int RNAlgebraic::
NVariables(void) const
{
  // Return whether algebraic has a variable
  switch(operation) {
  case RN_ZERO_OPERATION:
    return 0;

  case RN_POLYNOMIAL_OPERATION:
    return polynomial->NVariables();

  default: {
    // Find variable index range
    int min_v = INT_MAX, max_v = -1;
    UpdateVariableRange(min_v, max_v);

    // Check variable index range
    if (max_v < 0) return 0;
    if (min_v == INT_MAX) return 0;
    if (max_v < min_v) return 0;
    if (max_v == min_v) return 1;
    if (max_v == min_v + 1) return 2;
   
    // Allocate index stuff
    int nvariables = 0;
    int *marks = new int [ max_v + 1 ];
    for (int k = 0; k < max_v + 1; k++) marks[k] = 0;
    UpdateVariableIndex(max_v + 1, nvariables, marks, 1);
    delete [] marks;

    // Return count of unique variables
    return nvariables; }
  }
}



void RNAlgebraic::
Empty(void)
{
  // Just checking
  assert(IsValid());

  // Delete everything
  operation = RN_ZERO_OPERATION;
  if (operands[0]) { delete operands[0]; operands[0] = NULL; }
  if (operands[1]) { delete operands[1]; operands[1] = NULL; }
  if (polynomial) { delete polynomial; polynomial = NULL; }

  // Just checking
  assert(IsValid());
}



void RNAlgebraic::
SetOperation(int operation)
{
  // Assign operation
  this->operation = operation;
}



void RNAlgebraic::
Reset(int operation, RNAlgebraic *operand1, RNAlgebraic *operand2)
{
  // Just checking
  assert(IsValid());
  assert((operation != RN_ZERO_OPERATION) && (operation != RN_POLYNOMIAL_OPERATION));
  assert(operand1 && operand2);

  // Reset everything
  Empty();

  // Set operation
  this->operation = operation;
  this->operands[0] = operand1;
  this->operands[1] = operand2;

  // Just checking
  assert(IsValid());
}



void RNAlgebraic::
Reset(RNPolynomial *polynomial)
{
  // NOTE: THIS TAKES OVER MANAGEMENT OF MEMORY FOR POLYNOMIAL
  // SPECIFICALLY, POLYNOMIAL WILL BE DELETED WHEN ALGEBRAIC IS DECONSTRUCTED

  // Empty everything
  Empty();

  // Set polynomial
  if (polynomial) {
    operation = RN_POLYNOMIAL_OPERATION;
    this->polynomial = polynomial;
  }
}



void RNAlgebraic::
Negate(void)
{
  // Just checking
  assert(IsValid());

  // Flip sign
  switch(operation) {
  case RN_ZERO_OPERATION:
    return;

  case RN_POLYNOMIAL_OPERATION:
    polynomial->Negate();
    return;

  case RN_ADD_OPERATION:
    operands[0]->Negate();
    operands[1]->Negate();
    return;

  case RN_SUBTRACT_OPERATION: {
    RNAlgebraic *swap = operands[0];
    operands[0] = operands[1];
    operands[1] = swap;
    return; }

  case RN_MULTIPLY_OPERATION:
  case RN_DIVIDE_OPERATION:
    operands[0]->Negate();
    return;

  case RN_POW_OPERATION: 
    Multiply(-1.0);
    return;
  }

  // Should never get here
  RNAbort("Unrecognized operation %d in algebraic");
}



void RNAlgebraic::
Add(RNScalar a)
{
  // Just checking
  assert(IsValid());

  // Check value
  if (a == 0.0) return;

  // Add scalar
  switch(operation) {
  case RN_ZERO_OPERATION: 
    operation = RN_POLYNOMIAL_OPERATION;
    polynomial = new RNPolynomial(a, 0);
    return;

  case RN_POLYNOMIAL_OPERATION:
    polynomial->Add(a);
    return;

  default: {
    RNAlgebraic *e = new RNAlgebraic(operation, operands[0], operands[1]);
    operation = RN_ADD_OPERATION;
    operands[0] = new RNAlgebraic(a, 0);
    operands[1] = e;
    return; }
  }
}



void RNAlgebraic::
Subtract(RNScalar a)
{
  // Just checking
  assert(IsValid());

  // Check value
  if (a == 0.0) return;

  // Subtract scalar
  switch(operation) {
  case RN_ZERO_OPERATION: 
    operation = RN_POLYNOMIAL_OPERATION;
    polynomial = new RNPolynomial(-a, 0);
    return;
    
  case RN_POLYNOMIAL_OPERATION:
    polynomial->Subtract(a);
    return;
    
  default: {
    RNAlgebraic *e = new RNAlgebraic(operation, operands[0], operands[1]);
    operation = RN_SUBTRACT_OPERATION;
    operands[0] = e;
    operands[1] = new RNAlgebraic(a, 0);
    return; }
  }
}



void RNAlgebraic::
Multiply(RNScalar factor)
{
  // Just checking
  assert(IsValid());

  // Check factor 
  if (factor == 0.0) {
    Empty();
  }
  else if (factor != 1.0) {
    switch(operation) {
    case RN_ZERO_OPERATION: 
      return;
    
    case RN_POLYNOMIAL_OPERATION:
      polynomial->Multiply(factor);
      return;
    
    default: {
      RNAlgebraic *e = new RNAlgebraic(operation, operands[0], operands[1]);
      operation = RN_MULTIPLY_OPERATION;
      operands[0] = new RNAlgebraic(factor, 0);
      operands[1] = e;
      return; }
    }
  }
}



void RNAlgebraic::
Divide(RNScalar factor)
{
  // Just checking
  assert(IsValid());

  // Check factor
  if (factor == 0.0) {
    if (IsZero()) {
      Empty();
    }
    else {
      Empty();
      operation = RN_POLYNOMIAL_OPERATION;
      polynomial = new RNPolynomial(RN_INFINITY, 0);
    }
  }
  else if (factor != 1.0) {
    switch(operation) {
    case RN_ZERO_OPERATION: 
      return;
      
    case RN_POLYNOMIAL_OPERATION:
      polynomial->Divide(factor);
      return;
    
    default: {
      RNAlgebraic *e = new RNAlgebraic(operation, operands[0], operands[1]);
      operation = RN_DIVIDE_OPERATION;
      operands[0] = new RNAlgebraic(factor, 0);
      operands[1] = e;
      return; }
    }
  }
}



void RNAlgebraic::
Add(const RNPolynomial& a)
{
  // Just checking
  assert(IsValid());

  // Check value
  if (a.IsZero()) return;

  // Add scalar
  switch(operation) {
  case RN_ZERO_OPERATION: 
    operation = RN_POLYNOMIAL_OPERATION;
    polynomial = new RNPolynomial(a);
    return;

  case RN_POLYNOMIAL_OPERATION:
    polynomial->Add(a);
    return;

  default: {
    RNAlgebraic *e = new RNAlgebraic(operation, operands[0], operands[1]);
    operation = RN_ADD_OPERATION;
    operands[0] = new RNAlgebraic(a, 0);
    operands[1] = e;
    return; }
  }
}



void RNAlgebraic::
Subtract(const RNPolynomial& a)
{
  // Just checking
  assert(IsValid());

  // Check value
  if (a.IsZero()) return;

  // Subtract scalar
  switch(operation) {
  case RN_ZERO_OPERATION: 
    operation = RN_POLYNOMIAL_OPERATION;
    polynomial = new RNPolynomial(a);
    polynomial->Negate();
    return;
    
  case RN_POLYNOMIAL_OPERATION:
    polynomial->Subtract(a);
    return;
    
  default: {
    RNAlgebraic *e = new RNAlgebraic(operation, operands[0], operands[1]);
    operation = RN_SUBTRACT_OPERATION;
    operands[0] = e;
    operands[1] = new RNAlgebraic(a, 0);
    return; }
  }
}



void RNAlgebraic::
Multiply(const RNPolynomial& factor)
{
  // Just checking
  assert(IsValid());

  // Check factor 
  if (factor.IsZero()) {
    Empty();
  }
  else if (!factor.IsOne()) {
    switch(operation) {
    case RN_ZERO_OPERATION: 
      return;
    
    case RN_POLYNOMIAL_OPERATION:
      assert(polynomial);
      if (TRUE || (factor.NTerms() == 1) || (polynomial->NTerms() == 1)) {
        polynomial->Multiply(factor);
      }
      else {
        operation = RN_MULTIPLY_OPERATION;
        operands[0] = new RNAlgebraic(polynomial);
        operands[1] = new RNAlgebraic(factor, 0);
        polynomial = NULL;
      }
      return;
    
    default: {
      RNAlgebraic *e = new RNAlgebraic(operation, operands[0], operands[1]);
      operation = RN_MULTIPLY_OPERATION;
      operands[0] = new RNAlgebraic(factor, 0);
      operands[1] = e;
      return; }
    }
  }
}



void RNAlgebraic::
Divide(const RNPolynomial& factor)
{
  // Just checking
  assert(IsValid());

  // Check factor
  if (factor.IsZero()) {
    if (IsZero()) {
      Empty();
    }
    else {
      Empty();
      operation = RN_POLYNOMIAL_OPERATION;
      polynomial = new RNPolynomial(RN_INFINITY, 0);
    }
  }
  else if (!factor.IsOne()){
    switch(operation) {
    case RN_ZERO_OPERATION: 
      return;
      
    case RN_POLYNOMIAL_OPERATION:
      operation = RN_DIVIDE_OPERATION;
      operands[0] = new RNAlgebraic(polynomial);
      operands[1] = new RNAlgebraic(factor, 0);
      polynomial = NULL;
      return;
    
    default: {
      RNAlgebraic *e = new RNAlgebraic(operation, operands[0], operands[1]);
      operation = RN_DIVIDE_OPERATION;
      operands[0] = new RNAlgebraic(factor, 0);
      operands[1] = e;
      return; }
    }
  }
}



void RNAlgebraic::
Add(const RNAlgebraic& algebraic)
{
  // Just checking
  assert(IsValid());
  assert(algebraic.IsValid());

  // Check algebraic
  if (algebraic.IsZero()) return;

  // Add algebraic
  switch(operation) {
  case RN_ZERO_OPERATION: 
    *this = algebraic;
    return;

  case RN_POLYNOMIAL_OPERATION:
    if (algebraic.operation == RN_POLYNOMIAL_OPERATION) {
      polynomial->Add(*(algebraic.polynomial));
    }
    else {
      operands[0] = new RNAlgebraic(algebraic);
      operands[1] = new RNAlgebraic(polynomial);
      operation = RN_ADD_OPERATION;
      polynomial = NULL;
    }
    return;

  default: {
    RNAlgebraic *e = new RNAlgebraic(operation, operands[0], operands[1]);
    operands[0] = new RNAlgebraic(algebraic);
    operands[1] = e;
    operation = RN_ADD_OPERATION;
    return; }
  }
}



void RNAlgebraic::
Subtract(const RNAlgebraic& algebraic)
{
  // Just checking
  assert(IsValid());
  assert(algebraic.IsValid());

  // Check algebraic
  if (algebraic.IsZero()) return;

  // Subtract algebraic
  switch(operation) {
  case RN_ZERO_OPERATION: 
    *this = algebraic;
    Negate();
    return;

  case RN_POLYNOMIAL_OPERATION:
    if (algebraic.operation == RN_POLYNOMIAL_OPERATION) {
      polynomial->Subtract(*(algebraic.polynomial));
    }
    else {
      operands[1] = new RNAlgebraic(algebraic);
      operands[0] = new RNAlgebraic(polynomial);
      operation = RN_SUBTRACT_OPERATION;
      polynomial = NULL;
    }
    return;

  default: {
    RNAlgebraic *e = new RNAlgebraic(operation, operands[0], operands[1]);
    operands[1] = new RNAlgebraic(algebraic);
    operands[0] = e;
    operation = RN_SUBTRACT_OPERATION;
    return; }
  }
}



void RNAlgebraic::
Multiply(const RNAlgebraic& algebraic)
{
  // Just checking
  assert(IsValid());
  assert(algebraic.IsValid());

  // Check factor 
  if (algebraic.IsZero()) {
    Empty();
  }
  else if (!algebraic.IsOne()) {
    switch(operation) {
    case RN_ZERO_OPERATION: 
      return;
      
    case RN_POLYNOMIAL_OPERATION:
      if (algebraic.operation == RN_POLYNOMIAL_OPERATION) {
        if (TRUE || (polynomial->NTerms() == 1) || (algebraic.polynomial->NTerms() == 1)) {
          polynomial->Multiply(*(algebraic.polynomial));
        }
        else {
          operands[1] = new RNAlgebraic(algebraic);
          operands[0] = new RNAlgebraic(polynomial);
          operation = RN_MULTIPLY_OPERATION;
          polynomial = NULL;
        }
      }
      else {
        operands[1] = new RNAlgebraic(algebraic);
        operands[0] = new RNAlgebraic(polynomial);
        operation = RN_MULTIPLY_OPERATION;
        polynomial = NULL;
      }
      return;
    
    default: {
      RNAlgebraic *e = new RNAlgebraic(operation, operands[0], operands[1]);
      operands[0] = new RNAlgebraic(algebraic);
      operands[1] = e;
      operation = RN_MULTIPLY_OPERATION;
      return; }
    }
  }
}



void RNAlgebraic::
Divide(const RNAlgebraic& algebraic)
{
  // Just checking
  assert(IsValid());
  assert(algebraic.IsValid());

  // Check factor
  if (algebraic.IsZero()) {
    if (IsZero()) {
      Empty();
    }
    else {
      Empty();
      operation = RN_POLYNOMIAL_OPERATION;
      polynomial = new RNPolynomial(RN_INFINITY, 0);
    }
  }
  else if (!algebraic.IsOne()) {
    switch(operation) {
    case RN_ZERO_OPERATION: 
      return;
      
    case RN_POLYNOMIAL_OPERATION:
      operands[1] = new RNAlgebraic(algebraic);
      operands[0] = new RNAlgebraic(polynomial);
      operation = RN_DIVIDE_OPERATION;
      polynomial = NULL;
      return;
    
    default: {
      RNAlgebraic *e = new RNAlgebraic(operation, operands[0], operands[1]);
      operands[0] = new RNAlgebraic(algebraic);
      operands[1] = e;
      operation = RN_DIVIDE_OPERATION;
      return; }
    }
  }
}



RNAlgebraic& RNAlgebraic::
operator=(RNScalar a)
{
  // Empty this algebraic
  Empty();

  // Assign constant
  operation = RN_POLYNOMIAL_OPERATION;
  polynomial = new RNPolynomial(a, 0);

  // Return this
  return *this;
}



RNAlgebraic& RNAlgebraic::
operator=(const RNPolynomial& p)
{
  // Empty this algebraic
  Empty();

  // Assign polynomial
  operation = RN_POLYNOMIAL_OPERATION;
  polynomial = new RNPolynomial(p);

  // Return this
  return *this;
}



RNAlgebraic& RNAlgebraic::
operator=(const RNAlgebraic& algebraic)
{
  // Just checking
  assert(algebraic.IsValid());

  // Empty this algebraic
  Empty();

  // Copy operation
  this->operation = algebraic.operation;

  // Copy operands
  switch(operation) {
  case RN_ZERO_OPERATION:
    break;

  case RN_POLYNOMIAL_OPERATION:
    polynomial = new RNPolynomial(*(algebraic.polynomial));
    break;

  default:
    operands[0] = new RNAlgebraic(*(algebraic.operands[0]));
    operands[1] = new RNAlgebraic(*(algebraic.operands[1]));
    break;
  }

  // Return this
  return *this;
}



RNScalar RNAlgebraic::
Evaluate(const RNScalar *x) const
{
  // Return value of algebraic
  switch(operation) {
  case RN_ZERO_OPERATION:
    return 0;
  
  case RN_POLYNOMIAL_OPERATION: 
    return polynomial->Evaluate(x);

  case RN_ADD_OPERATION: 
    return operands[0]->Evaluate(x) + operands[1]->Evaluate(x);
  
  case RN_SUBTRACT_OPERATION: 
    return operands[0]->Evaluate(x) - operands[1]->Evaluate(x);
  
  case RN_MULTIPLY_OPERATION: 
    return operands[0]->Evaluate(x) * operands[1]->Evaluate(x);

  case RN_DIVIDE_OPERATION: {
    RNScalar v1 = operands[1]->Evaluate(x);
    if (RNIsZero(v1, RN_SMALL_EPSILON)) return RN_INFINITY;
    RNScalar v0 = operands[0]->Evaluate(x);
    return v0 / v1; }
  
  case RN_POW_OPERATION: {
    RNScalar v0 = operands[0]->Evaluate(x);
    if (v0 == 0) return 0;
    RNScalar v1 = operands[1]->Evaluate(x);
    if (v1 == 0) return 1;
    return pow(v0, v1); }
  }

  // Should never get here
  RNAbort("Invalid algebraic operation: %d", operation);
  return 0.0;
}



RNScalar RNAlgebraic::
Evaluate(double const* const* x) const
{
  // Return value of algebraic
  switch(operation) {
  case RN_ZERO_OPERATION: 
    return 0;
  
  case RN_POLYNOMIAL_OPERATION: 
    return polynomial->Evaluate(x);

  case RN_ADD_OPERATION: 
    return operands[0]->Evaluate(x) + operands[1]->Evaluate(x);
  
  case RN_SUBTRACT_OPERATION: 
    return operands[0]->Evaluate(x) - operands[1]->Evaluate(x);
  
  case RN_MULTIPLY_OPERATION: 
    return operands[0]->Evaluate(x) * operands[1]->Evaluate(x);

  case RN_DIVIDE_OPERATION: {
    RNScalar v1 = operands[1]->Evaluate(x);
    if (RNIsZero(v1, RN_SMALL_EPSILON)) return RN_INFINITY;
    RNScalar v0 = operands[0]->Evaluate(x);
    return v0 / v1; }
  
  case RN_POW_OPERATION: {
    RNScalar v0 = operands[0]->Evaluate(x);
    if (v0 == 0) return 0;
    RNScalar v1 = operands[1]->Evaluate(x);
    if (v1 == 0) return 1;
    return pow(v0, v1); }
  }

  // Should never get here
  RNAbort("Invalid algebraic operation: %d", operation);
  return 0.0;
}



RNScalar RNAlgebraic::
PartialDerivative(const RNScalar *x, int variable) const
{
  // Return partial derivative
  switch(operation) {
  case RN_ZERO_OPERATION: 
    return 0;
    
  case RN_POLYNOMIAL_OPERATION:
    return polynomial->PartialDerivative(x, variable);

  case RN_ADD_OPERATION: 
    return operands[0]->PartialDerivative(x, variable) + operands[1]->PartialDerivative(x, variable);

  case RN_SUBTRACT_OPERATION: 
    return operands[0]->PartialDerivative(x, variable) - operands[1]->PartialDerivative(x, variable);

  case RN_MULTIPLY_OPERATION: {
    // Product rule
    RNScalar v0 = operands[0]->Evaluate(x);
    RNScalar v1 = operands[1]->Evaluate(x);
    RNScalar d0 = operands[0]->PartialDerivative(x, variable);
    RNScalar d1 = operands[1]->PartialDerivative(x, variable);
    return d0*v1 + v0*d1; }

  case RN_DIVIDE_OPERATION: {
    // Quotion rule
    RNScalar v1 = operands[1]->Evaluate(x);
    RNScalar v1_squared = v1 * v1;
    if (RNIsZero(v1_squared, RN_SMALL_EPSILON)) return RN_INFINITY;
    RNScalar v0 = operands[0]->Evaluate(x);
    RNScalar d0 = operands[0]->PartialDerivative(x, variable);
    RNScalar d1 = operands[1]->PartialDerivative(x, variable);
    return (d0*v1 - v0*d1) / v1_squared; }
  
  case RN_POW_OPERATION: {
    // Power rule
    RNScalar v0 = operands[0]->Evaluate(x);
    if (RNIsZero(v0, RN_SMALL_EPSILON)) return RN_INFINITY;
    RNScalar v1 = operands[1]->Evaluate(x);
    RNScalar d0 = operands[0]->PartialDerivative(x, variable);
    RNScalar d1 = operands[1]->PartialDerivative(x, variable);
    return (d0*v1/v0 + d1*log(v0)); }
  }    

  // Should never get here
  RNAbort("Invalid algebraic operation: %d", operation);
  return 0.0;
}



RNScalar RNAlgebraic::
PartialDerivative(double const* const* x, int variable) const
{
  // Return partial derivative
  switch(operation) {
  case RN_ZERO_OPERATION: 
    return 0;
    
  case RN_POLYNOMIAL_OPERATION: 
    return polynomial->PartialDerivative(x, variable);

  case RN_ADD_OPERATION: 
    return operands[0]->PartialDerivative(x, variable) + operands[1]->PartialDerivative(x, variable);

  case RN_SUBTRACT_OPERATION: 
    return operands[0]->PartialDerivative(x, variable) - operands[1]->PartialDerivative(x, variable);

  case RN_MULTIPLY_OPERATION: {
    // Product rule
    RNScalar v0 = operands[0]->Evaluate(x);
    RNScalar v1 = operands[1]->Evaluate(x);
    RNScalar d0 = operands[0]->PartialDerivative(x, variable);
    RNScalar d1 = operands[1]->PartialDerivative(x, variable);
    return d0*v1 + v0*d1; }

  case RN_DIVIDE_OPERATION: {
    // Quotion rule
    RNScalar v1 = operands[1]->Evaluate(x);
    RNScalar v1_squared = v1 * v1;
    if (RNIsZero(v1_squared, RN_SMALL_EPSILON)) return RN_INFINITY;
    RNScalar v0 = operands[0]->Evaluate(x);
    RNScalar d0 = operands[0]->PartialDerivative(x, variable);
    RNScalar d1 = operands[1]->PartialDerivative(x, variable);
    return (d0*v1 - v0*d1) / v1_squared; }
  
  case RN_POW_OPERATION: {
    // Power rule
    RNScalar v0 = operands[0]->Evaluate(x);
    if (RNIsZero(v0, RN_SMALL_EPSILON)) return RN_INFINITY;
    RNScalar v1 = operands[1]->Evaluate(x);
    RNScalar d0 = operands[0]->PartialDerivative(x, variable);
    RNScalar d1 = operands[1]->PartialDerivative(x, variable);
    return (d0*v1/v0 + d1*log(v0)); }
  }    

  // Should never get here
  RNAbort("Invalid algebraic operation: %d", operation);
  return 0.0;
}



void RNAlgebraic::
Print(FILE *fp, int indent) const
{
  // Print algebraic
  switch(operation) {
  case RN_ZERO_OPERATION:
    for (int i = 0; i < indent; i++) fprintf(fp, " ");
    fprintf(fp, "0\n");
    return;
    
  case RN_POLYNOMIAL_OPERATION: 
    for (int i = 0; i < indent; i++) fprintf(fp, " ");
    polynomial->Print(fp);
    return;

  default:
    for (int i = 0; i < indent; i++) fprintf(fp, " ");
    fprintf(fp, "[ %d\n", operation);
    operands[0]->Print(fp, indent+2);
    operands[1]->Print(fp, indent+2);
    for (int i = 0; i < indent; i++) fprintf(fp, " ");
    fprintf(fp, "]\n");
  }
}



void RNAlgebraic::
UpdateVariableRange(int& min_v, int& max_v) const
{
  // Update variable indices
  switch(operation) {
  case RN_ZERO_OPERATION: 
    return;
    
  case RN_POLYNOMIAL_OPERATION: 
    polynomial->UpdateVariableRange(min_v, max_v);
    return;

  default:
    operands[0]->UpdateVariableRange(min_v, max_v);
    operands[1]->UpdateVariableRange(min_v, max_v);
    return;
  }
}



void RNAlgebraic::
UpdateVariableIndex(int max_variables, int& variable_count, 
  int *variable_marks, int current_mark, 
  int *index_to_variable, int *variable_to_index,
  RNBoolean remap_variables) const
{
  // Just checking
  assert(variable_to_index || !remap_variables);

  // Update variable indices
  switch(operation) {
  case RN_ZERO_OPERATION: 
    return;
    
  case RN_POLYNOMIAL_OPERATION: 
    polynomial->UpdateVariableIndex(max_variables, variable_count, 
      variable_marks, current_mark, index_to_variable, variable_to_index, remap_variables);
    return;

  default:
    operands[0]->UpdateVariableIndex(max_variables, variable_count, 
      variable_marks, current_mark, index_to_variable, variable_to_index, remap_variables);
    operands[1]->UpdateVariableIndex(max_variables, variable_count, 
      variable_marks, current_mark, index_to_variable, variable_to_index, remap_variables);
    return;
  }
}


void DebugBreakpoint(void) {};


RNBoolean RNAlgebraic::
IsValid(void) const
{
  // Check operation
  if ((operation < 0) || (operation >= RN_NUM_ALGEBRAIC_OPERATIONS)) return FALSE; 

  // Check invariants
  switch(operation) {
  case RN_ZERO_OPERATION: 
    if (polynomial) { printf("E1\n"); DebugBreakpoint(); return FALSE; }
    if (operands[0]) { printf("E2\n"); DebugBreakpoint(); return FALSE; }
    if (operands[1]) { printf("E3\n"); DebugBreakpoint(); return FALSE; }
    break;
    
  case RN_POLYNOMIAL_OPERATION: 
    if (!polynomial) { printf("E4\n"); DebugBreakpoint(); return FALSE; }
    if (operands[0]) { printf("E5\n"); DebugBreakpoint(); return FALSE; }
    if (operands[1]) { printf("E6\n"); DebugBreakpoint(); return FALSE; }
    break;

  default:
    if (polynomial) { printf("E7\n"); DebugBreakpoint(); return FALSE; }
    if (!operands[0]) { printf("E8\n"); DebugBreakpoint(); return FALSE; }
    if (!operands[1]) { printf("E9\n"); DebugBreakpoint(); return FALSE; }
    break;
  }

  // Passed all tests
  return TRUE;
}
