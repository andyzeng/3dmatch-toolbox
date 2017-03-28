// Include file for RNMath package

#ifndef __RN__MATH__H__
#define __RN__MATH__H__




// Dependency include files

#include "RNBasics/RNBasics.h"



// Declarations

class RNVector;
class RNMatrix;
class RNDenseMatrix;
class RNPolynomial;
class RNPolynomialTerm;
class RNAlgebraic;
typedef RNAlgebraic RNExpression;
class RNEquation;
class RNSystemOfEquations;



// Matrix classes

#include "RNMath/RNLapack.h"
#include "RNMath/RNVector.h"
#include "RNMath/RNMatrix.h"
#include "RNMath/RNDenseMatrix.h"
#include "RNMath/RNDenseLUMatrix.h"


// Expression and equation classes

#include "RNMath/RNPolynomial.h"
#include "RNMath/RNAlgebraic.h"
#include "RNMath/RNEquation.h"
#include "RNMath/RNSystemOfEquations.h"



#endif
