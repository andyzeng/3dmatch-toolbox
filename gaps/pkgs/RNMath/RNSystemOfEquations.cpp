// Source file for system of equations class



////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "RNMath/RNMath.h"



////////////////////////////////////////////////////////////////////////
// Functions
////////////////////////////////////////////////////////////////////////

RNSystemOfEquations::
RNSystemOfEquations(int nvariables)
  : nvariables(nvariables),
    equations()
{
  // Allocate memory for variable counting
  index_to_variable = new int [ nvariables ];
  variable_to_index = new int [ nvariables ];
  variable_marks = new int [ nvariables ];
  for (int i = 0; i < nvariables; i++) variable_marks[i] = 0;
  current_mark = 1;
}



RNSystemOfEquations::
RNSystemOfEquations(const RNSystemOfEquations& system)
  : nvariables(system.nvariables),
    equations()
{
  // Allocate memory for variable counting
  index_to_variable = new int [ nvariables ];
  variable_to_index = new int [ nvariables ];
  variable_marks = new int [ nvariables ];
  for (int i = 0; i < nvariables; i++) variable_marks[i] = 0;
  current_mark = 1;

  // Copy equations
  equations.Resize(system.NEquations());
  for (int i = 0; i < system.NEquations(); i++) {
    RNEquation *equation = system.Equation(i);
    InsertEquation(new RNEquation(*equation));
  }
}



RNSystemOfEquations::
~RNSystemOfEquations(void)
{
  // Delete memory for variable counting
  if (index_to_variable) delete [] index_to_variable;
  if (variable_to_index) delete [] variable_to_index;
  if (variable_marks) delete [] variable_marks;

  // Delete all equations
  while (NEquations() > 0) {
    RNEquation *equation = equations.Tail();
    RemoveEquation(equation);
    delete equation;
  }
}



int RNSystemOfEquations::
NPartialDerivatives(void) const
{
  // Return number of partial derivatives
  int count = 0;
  for (int i = 0; i < NEquations(); i++) {
    RNEquation *equation = Equation(i);
    equation->UpdateVariableIndex(nvariables, count, variable_marks, current_mark);
    ((RNSystemOfEquations *) this)->current_mark++;
  }
  return count;
}



RNBoolean RNSystemOfEquations::
IsLinear(void) const
{
  // Check whether all equations are linear
  for (int i = 0; i < NEquations(); i++) {
    RNEquation *equation = Equation(i);
    if (!equation->IsLinear()) return FALSE;
  }

  // Passed all tests
  return TRUE;
}



RNBoolean RNSystemOfEquations::
IsQuadratic(void) const
{
  // Check whether all equations are quadratic
  for (int i = 0; i < NEquations(); i++) {
    RNEquation *equation = Equation(i);
    if (!equation->IsQuadratic()) return FALSE;
  }

  // Passed all tests
  return TRUE;
}



RNBoolean RNSystemOfEquations::
IsPolynomial(void) const
{
  // Check whether all equations are polynomial
  for (int i = 0; i < NEquations(); i++) {
    RNEquation *equation = Equation(i);
    if (!equation->IsPolynomial()) return FALSE;
  }

  // Passed all tests
  return TRUE;
}



RNBoolean RNSystemOfEquations::
IsAlgebraic(void) const
{
  // Check whether all equations are algebraic
  for (int i = 0; i < NEquations(); i++) {
    RNEquation *equation = Equation(i);
    if (!equation->IsAlgebraic()) return FALSE;
  }

  // Passed all tests
  return TRUE;
}



RNBoolean RNSystemOfEquations::
HasVariable(int v) const
{
  // Check whether any equation has variable v
  for (int i = 0; i < NEquations(); i++) {
    RNEquation *equation = Equation(i);
    if (equation->HasVariable(v)) return TRUE;
  }

  // No equation has variable v
  return FALSE;
}



void RNSystemOfEquations::
InsertEquation(RNPolynomial *polynomial)
{
  // Check polynomial
  if (polynomial->IsConstant()) { delete polynomial; return; }

  // Insert equation
  RNEquation *equation = new RNEquation(polynomial);
  InsertEquation(equation);
}



void RNSystemOfEquations::
InsertEquation(RNAlgebraic *algebraic)
{
  // Check algebraic
  if (algebraic->IsConstant()) { delete algebraic; return; }

  // Insert equation
  RNEquation *equation = new RNEquation(algebraic);
  InsertEquation(equation);
}



void RNSystemOfEquations::
InsertEquation(RNPolynomial *polynomial, RNScalar residual_threshold)
{
  // Check polynomial
  if (polynomial->IsConstant()) { delete polynomial; return; }

  // Insert equation
  RNEquation *equation = new RNEquation(polynomial);
  equation->SetResidualThreshold(residual_threshold);
  InsertEquation(equation);
}



void RNSystemOfEquations::
InsertEquation(RNAlgebraic *algebraic, RNScalar residual_threshold)
{
  // Check algebraic
  if (algebraic->IsConstant()) { delete algebraic; return; }

  // Insert equation
  RNEquation *equation = new RNEquation(algebraic);
  equation->SetResidualThreshold(residual_threshold);
  InsertEquation(equation);
}



void RNSystemOfEquations::
InsertEquation(RNEquation *equation)
{
  // Check equation
  if (equation->IsConstant()) { delete equation; return; }

  // Just checking
  assert(!equation->system);
  assert(equation->system_index == -1);
  assert(!equations.FindEntry(equation));

  // Insert equation
  equation->system = this;
  equation->system_index = equations.NEntries();
  equations.Insert(equation);
}



void RNSystemOfEquations::
RemoveEquation(RNEquation *equation)
{
  // Just checking
  assert(equation->system == this);
  assert(equation->system_index >= 0);
  assert(equations.FindEntry(equation));

  // Remove equation
  RNArrayEntry *entry = equations.KthEntry(equation->system_index);
  assert(entry && (equations.EntryContents(entry) == equation));
  RNEquation *tail = equations.Tail();
  tail->system_index = equation->system_index;
  equations.EntryContents(entry) = tail;
  equations.RemoveTail();
  equation->system_index = -1;
  equation->system = NULL;
}



void RNSystemOfEquations::
EvaluateResiduals(const RNScalar *x, RNScalar *y) const
{
  // Evaluate equations
  for (int i = 0; i < NEquations(); i++) {
    RNEquation *equation = Equation(i);
    y[i] = equation->Evaluate(x);
  }
}



RNScalar RNSystemOfEquations::
SumOfSquaredResiduals(const RNScalar *x) const
{
  // Check stuff
  if (NVariables() == 0) return 0.0;
  if (NEquations() == 0) return 0.0;
  
  // Allocate residuals
  RNScalar *y = new RNScalar [ NEquations() ];
  
  // Compute residuals
  EvaluateResiduals(x, y);

  // Sum squared residuals
  RNScalar sum = 0;
  for (int i = 0; i < NEquations(); i++) {
    sum += y[i] * y[i];
  }

  // Free residuals
  delete [] y;

  // Return sum of squared residuals
  return sum;
}



void RNSystemOfEquations::
PrintEquations(FILE *fp) const
{
  // Check file
  if (!fp) fp = stdout;
  
  // Print equation
  fprintf(fp, "fp, %d %d %d\n", NEquations(), NVariables(), NPartialDerivatives());
  for (int i = 0; i < NEquations(); i++) {
    RNEquation *equation = Equation(i);
    equation->Print(fp);
  }
}



void RNSystemOfEquations::
PrintValues(const RNScalar *x, FILE *fp) const
{
  // Check file
  if (!fp) fp = stdout;
  
  // Print values
  fprintf(fp, "Values: %d\n", NVariables());
  for (int i = 0; i < NVariables(); i++) {
    fprintf(fp, "  %12d %12.6f ", i, x[i]);
    if ((i > 0) && ((i % 6) == 0)) fprintf(fp, "\n");
  }
  fprintf(fp, "\n");
}



void RNSystemOfEquations::
PrintResiduals(const RNScalar *x, FILE *fp) const
{
  // Check file
  if (!fp) fp = stdout;
  
  // Print residuals
  double sum = 0;
  double *y = new double [ NEquations() ];
  EvaluateResiduals(x, y);
  fprintf(fp, "Residuals: %d\n", NEquations());
  for (int i = 0; i < NEquations(); i++) {
    fprintf(fp, "  %12d %12.6f\n", i, y[i]);
    sum += y[i]*y[i];
  }
  fprintf(fp, "%g\n", sum);
  delete [] y;
}



void RNSystemOfEquations::
Print(FILE *fp) const
{
  // Print equations
  PrintEquations(fp);
}



