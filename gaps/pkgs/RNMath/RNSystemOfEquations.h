// Include file for system of equation class



////////////////////////////////////////////////////////////////////////
// Just making sure
////////////////////////////////////////////////////////////////////////

#ifndef __RN_SYSTEM_OF_EQUATIONS__
#define __RN_SYSTEM_OF_EQUATIONS__



////////////////////////////////////////////////////////////////////////
// Class definition
////////////////////////////////////////////////////////////////////////

class RNSystemOfEquations {
public:
  // Constructor/destructor
  RNSystemOfEquations(int nvariables = 0);
  RNSystemOfEquations(const RNSystemOfEquations& system);
  ~RNSystemOfEquations(void);

  // Property functions
  int NVariables(void) const;
  int NEquations(void) const;
  int NPartialDerivatives(void) const;
  RNBoolean IsLinear(void) const;
  RNBoolean IsQuadratic(void) const;
  RNBoolean IsPolynomial(void) const;
  RNBoolean IsAlgebraic(void) const;
  RNBoolean HasVariable(int v) const;

  // Access functions
  RNEquation *Equation(int k) const;

  // Manipulation functions
  void InsertEquation(RNPolynomial *polynomial);
  void InsertEquation(RNAlgebraic *algebraic);
  void InsertEquation(RNEquation *equation);
  void RemoveEquation(RNEquation *equation);

  // Evaluation functions
  void EvaluateResiduals(const RNScalar *x, RNScalar *y) const;
  RNScalar SumOfSquaredResiduals(const RNScalar *x) const;

  // Optimization functions
  int Minimize(RNScalar *x, int solver = 0, RNScalar tolerance = RN_EPSILON) const;

  // Print functions
  void PrintEquations(FILE *fp = stdout) const;
  void PrintValues(const RNScalar *x, FILE *fp = stdout) const;
  void PrintResiduals(const RNScalar *x, FILE *fp = stdout) const;
  void Print(FILE *fp = stdout) const;

public:
  // Do not use these
  void InsertEquation(RNPolynomial *polynomial, RNScalar residual_threshold);
  void InsertEquation(RNAlgebraic *algebraic, RNScalar residual_threshold);
  

public:
  int *index_to_variable;
  int *variable_to_index;
  int *variable_marks;
  int current_mark;

private:
  int nvariables;
  RNArray<RNEquation *> equations;
};



////////////////////////////////////////////////////////////////////////
// Inline functions 
////////////////////////////////////////////////////////////////////////

inline int RNSystemOfEquations::
NVariables(void) const
{
  // Return number of variables
  return nvariables;
}



inline int RNSystemOfEquations::
NEquations(void) const
{
  // Return number of equations
  return equations.NEntries();
}



inline RNEquation *RNSystemOfEquations::
Equation(int k) const
{
  // Return Kth equation
  return equations.Kth(k);
}



////////////////////////////////////////////////////////////////////////
// System of equation solvers
////////////////////////////////////////////////////////////////////////

enum {
  RN_CERES_SOLVER,
  RN_MINPACK_SOLVER,
  RN_SPLM_SOLVER,
  RN_CSPARSE_SOLVER,
  RN_NUM_SOLVERS
};



////////////////////////////////////////////////////////////////////////
// Select dependencies (can be set in compile flags by app)
////////////////////////////////////////////////////////////////////////

// #define RN_USE_SPLM
// #define RN_NO_SPLM
// #define RN_USE_MINPACK
// #define RN_NO_MINPACK
// #define RN_USE_CERES
// #define RN_NO_CERES
// #define RN_USE_CSPARSE
// #define RN_NO_CSPARSE



////////////////////////////////////////////////////////////////////////
// CERES Stuff
////////////////////////////////////////////////////////////////////////

#ifdef RN_NO_CERES
#undef RN_USE_CERES
#endif
#ifdef RN_USE_CERES

#include "ceres/ceres.h"

class CeresCostFunction : public ceres::CostFunction {
private:
  RNEquation *equation;
  int nvariables;
public:
  CeresCostFunction(RNEquation *equation = NULL, int nvariables = 0) 
    : equation(equation), nvariables(nvariables)
  {
    set_num_residuals(1);
    for (int i = 0; i < nvariables; i++) {
      mutable_parameter_block_sizes()->push_back(1);
    }
  };

  virtual ~CeresCostFunction(void) 
  {
  };

  virtual bool Evaluate(double const* const* x, double* residual, double** jacobian) const 
  {
    // Evaluate residual
    if (residual != NULL) {
      residual[0] = equation->Evaluate(x);
    }

    // Evaluate Jacobian, if asked for.
    if (jacobian != NULL) {
      for (int v = 0; v < nvariables; v++) {
        jacobian[v][0] = equation->PartialDerivative(x, v);
      }

#if 0
      // Check versus numerical partial derivative
      for (int v = 0; v < nvariables; v++) {
        double tmp0, tmp1, tmp2;
        double **xp = (double **) x;
        tmp0 = x[v][0];
        xp[v][0] = tmp0 + 0.001;
        Evaluate(x, &tmp1, NULL);
        xp[v][0] = tmp0 - 0.001;
        Evaluate(x, &tmp2, NULL);
        xp[v][0] = tmp0;
        double dydx = (tmp1 - tmp2) / 0.002;
        if (RNIsNotEqual(dydx, jacobian[v][0])) {
          printf("PD %d : %g %g : %g %g\n", v, tmp1, tmp2, dydx, jacobian[v][0]);
        }
      }
#endif
    }

    // Return success
    return true;
  }
};



static int 
MinimizeCERES(const RNSystemOfEquations *system, RNScalar *io, RNScalar tolerance)
{
  // Get convenient variables
  int n = system->NVariables();

  // Allocate ceres stuff
  ceres::Problem *problem = new ceres::Problem();
  ceres::Solver::Options *options = new ceres::Solver::Options();
  ceres::Solver::Summary *summary = new ceres::Solver::Summary();

  // Copy system of equations (so can remap variables) !!!
  RNSystemOfEquations system_copy(*system);

  // Allocate and initialize parameter data
  double *x = new double [ n ];
  for (int i = 0; i < n; i++) x[i] = io[i];

  // Create ceres residual blocks
  for (int i = 0; i < system_copy.NEquations(); i++) {
    RNEquation *equation = system_copy.Equation(i);

    // Remap variables
    int variable_count = 0;
    equation->UpdateVariableIndex(system_copy.NVariables(), variable_count, 
      system_copy.variable_marks, system_copy.current_mark++, 
      system_copy.index_to_variable, system_copy.variable_to_index, TRUE);
    if (variable_count == 0) continue;

    // Create cost function and residual block stuff
    std::vector<double *> variable_ptr;
    for (int j = 0; j < variable_count; j++) {
      int v = system_copy.index_to_variable[j];
      variable_ptr.push_back(&x[v]);
    }

    // Create cost function
    ceres::CostFunction *cost_function = new CeresCostFunction(equation, variable_count);

    // Create loss function
    ceres::LossFunction *loss_function = NULL;
    if (equation->ResidualThreshold() > 0) {
      loss_function = new ceres::HuberLoss(equation->ResidualThreshold());
    }

    // Add residual block
    problem->AddResidualBlock(cost_function, loss_function, variable_ptr);
  }

  // Run the solver
  // options->max_num_iterations = 128;
  options->num_threads = 12;
  options->num_linear_solver_threads = 12; 
  // options->check_gradients = true;
  // options->gradient_check_relative_precision = 1E-1;
  // options->numeric_derivative_relative_step_size = 1E-3;
  // options->trust_region_strategy_type = ceres::DOGLEG;
  options->linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;  
  options->max_solver_time_in_seconds = 12 * 60 * 60; 
  options->function_tolerance = tolerance;
  // options->gradient_tolerance = 1E-12;
  // options->parameter_tolerance = 1E-12;
  // options->min_relative_decrease = 1E-6;
  // options->use_nonmonotonic_steps = true;
  // options->minimizer_progress_to_stdout = true;
  Solve(*options, problem, summary);

  // Print report
  // std::cout << summary->BriefReport() << "\n";

  // Copy solved solution into result 
  for (int i = 0; i < n; i++) io[i] = x[i]; 

  // Delete data
  delete [] x;

  // Delete ceres stuff
  delete summary;
  delete options;
  delete problem;

  // Return success
  return 1;
}

#else

static int 
MinimizeCERES(const RNSystemOfEquations *system, RNScalar *io, RNScalar tolerance)
{
  // Print error message
  fprintf(stderr, "Cannot minimize equation: Ceres solver disabled during compile.\n");
  fprintf(stderr, "Enable it by adding -DRN_USE_CERES and -lceres xxx to compilation and link commands.\n");
  return 0;
}

#endif



////////////////////////////////////////////////////////////////////////
// SPLM Stuff
////////////////////////////////////////////////////////////////////////

#ifdef RN_NO_SPLM
#undef RN_USE_SPLM
#endif
#ifdef RN_USE_SPLM

#include "splm/splm.h"

static void
SPLMFunction(double *x, double *y, int n, int m, void *data)
{
  // x is vector of length n with current variable values
  // y is vector of length m with returned function values
  // n is number of variables
  // m is number of equations
  // data is a user-data variable

  // Get convenient variables
  RNSystemOfEquations *system = (RNSystemOfEquations *) data;
  assert(m == system->NEquations());
  assert(n == system->NVariables());

  // Evaluate residuals
  system->EvaluateResiduals(x, y);
}



static void
SPLMJacobian(double *x, struct splm_ccsm *jac, int n, int m, void *data)
{
  // p is a user-data variable
  // m is number of equations
  // n is number of variables
  // x is vector of length n with current variable values
  // jac needs to be filled in with non-zero elements of jacobian

  // Get convenient variables
  RNSystemOfEquations *system = (RNSystemOfEquations *) data;
  assert(m == system->NEquations());
  assert(n == system->NVariables());
  assert(jac->nnz == system->NPartialDerivatives());

  // Allocate triplets
  splm_stm sm;
  splm_stm_allocval(&sm, m, n, jac->nnz);

#if 0
  // Fill triplets
  int ntriplets = 0;
  for (int i = 0; i < system->NEquations(); i++) {
    RNEquation *equation = system->Equation(i);
    for (int v = 0; v < n; v++) {
      if (equation->HasVariable(v)) {
        RNScalar d = equation->PartialDerivative(x, v);
        splm_stm_nonzeroval(&sm, i, v, d);
        ntriplets++;
      }
    }
  }
#else
  int ntriplets = 0;
  for (int i = 0; i < system->NEquations(); i++) {
    RNEquation *equation = system->Equation(i);
    int count = 0;
    equation->UpdateVariableIndex(system->NVariables(), count, 
      system->variable_marks, system->current_mark++, 
      system->index_to_variable);
    for (int j = 0; j < count; j++) {
      int v = system->index_to_variable[j];
      RNScalar d = equation->PartialDerivative(x, v);
      splm_stm_nonzeroval(&sm, i, v, d);
      ntriplets++;
    }
  }      
#endif

  // Just checking
  if (ntriplets != jac->nnz) {
    fprintf(stderr, "Mismatching number of derivatives: %d %d\n", ntriplets, jac->nnz);
    abort();
  }

  // Convert from triplets to CSM
  splm_stm2ccsm(&sm, jac);

  // Free triplets
  splm_stm_free(&sm);
}



static int 
MinimizeSPLM(const RNSystemOfEquations *system, RNScalar *io, RNScalar tolerance)
{
  // Get convenient variables
  const int n = system->NVariables();
  const int m = system->NEquations();
  const int jnnz = system->NPartialDerivatives();

  // Allocate temporary data
  double *x = new double [ n ];
  double *y = new double [ m ];

  // Initialize values and residuals
  for (int i = 0; i < n; i++) x[i] = io[i];
  for (int i = 0; i < m; i++) y[i] = 0;

  // Set options to control tolerance
  // double opts[SPLM_OPTSSZ];
  // ???
  
  // Run the solver
  double info[SPLM_INFO_SZ];
  int status = sparselm_derccs(SPLMFunction, SPLMJacobian, x, y, n, 0, m, jnnz, -1, 100, NULL, info, (void *) system);
  if (status == SPLM_ERROR) fprintf(stderr, "Error in SPLM solver\n");
  else { for (int i = 0; i < n; i++) io[i] = x[i]; }

#if 0
  // Print debug info
  printf("SPLM Info: ");
  for (int i = 0; i < SPLM_INFO_SZ; i++) 
    printf("%g ", info[i]);
  printf("\n");
#endif

  // Delete temporary data
  delete [] x;
  delete [] y;

  // Return status 
  return (status == SPLM_ERROR) ? 0 : 1;
}

#else

static int 
MinimizeSPLM(const RNSystemOfEquations *system, RNScalar *io, RNScalar tolerance)
{
  // Print error message
  fprintf(stderr, "Cannot minimize equation: SPLM solver disabled during compile.\n");
  fprintf(stderr, "Enable it by adding -DRN_USE_SPLM and -lsplm to compilation and link commands.\n");
  return 0;
}

#endif



////////////////////////////////////////////////////////////////////////
// MINPACK STUFF
////////////////////////////////////////////////////////////////////////

#ifdef RN_NO_MINPACK
#undef RN_USE_MINPACK
#endif
#ifdef RN_USE_MINPACK

#include "minpack/minpack.h"


static int 
MinpackFunction(void *data, int m, int n, const double *x, double *y, int iflag)
{
  // data is a user-data variable
  // m is number of equations
  // n is number of variables
  // x is vector of length n with current variable values
  // y is vector of length m with returned function values
  // return a negative value to terminate

  // Get convenient variables
  RNSystemOfEquations *system = (RNSystemOfEquations *) data;
  assert(m == system->NEquations());
  assert(n == system->NVariables());

  // Evaluate function
  system->EvaluateResiduals(x, y);

  // Return success
  return 1;
}



#define USE_LMDER1
#ifdef USE_LMDER1

static int 
MinpackJacobian(void *data, int m, int n, const double *x, double *jacobian, int ldjacobian, int iflag)
{
  // data is a user-data variable
  // m is number of equations
  // n is number of variables
  // x is vector of length n with current variable values
  // y is vector of length m with returned function values
  // return a negative value to terminate

  // Get convenient variables
  RNSystemOfEquations *system = (RNSystemOfEquations *) data;
  assert(m == system->NEquations());
  assert(n == system->NVariables());
  assert(ldjacobian == m);

#if 0
  // Evaluate jacobian
  for (int i = 0; i < system->NEquations(); i++) {
    RNEquation *equation = system->Equation(i);
    for (int v = 0; v < n; v++) {
      RNScalar d = equation->PartialDerivative(x, v);
      jacobian[v*ldjacobian+i] += term->PartialDerivative(x, v);
    }
  }
#else
  // Initialize jacobian
  for (int i = 0; i < system->NEquations(); i++) {
    for (int v = 0; v < n; v++) {
      jacobian[v*ldjacobian+i] = 0;
    }
  }
  // Evaluate jacobian
  for (int i = 0; i < system->NEquations(); i++) {
    RNEquation *equation = system->Equation(i);
    int count = 0;
    equation->UpdateVariableIndex(system->NVariables(), count, 
      system->variable_marks, system->current_mark++, 
      system->index_to_variable);
    for (int j = 0; j < count; j++) {
      int v = system->index_to_variable[j];
      RNScalar d = equation->PartialDerivative(x, v);
      jacobian[v*ldjacobian+i] = d;
    }
  }  
#endif    

  // Return success
  return 1;
}



static int 
MinpackCallback(void *data, int m, int n, const double *x, double *y, double *jacobian, int ldjacobian, int iflag)
{
  // data is a user-data variable
  // m is number of equations
  // n is number of variables
  // x is vector of length n with current variable values
  // y is vector of length m with returned function values
  // jacobian is vector of length m*n with returned jacobian values
  // if iflag=1 fill in y, else if iflag=2 fill in jacobian
  // return a negative value to terminate

  // Call appropriate function
  if (iflag == 2) return MinpackJacobian(data, m, n, x, jacobian, ldjacobian, iflag);
  else return MinpackFunction(data, m, n, x, y, iflag);

  // Return success
  return 1;
}

#endif



static int 
MinimizeMINPACK(const RNSystemOfEquations *system, RNScalar *io, RNScalar tolerance)
{
  // Get convenient variables
  const int m = system->NEquations();
  const int n = system->NVariables();

  // Allocate temporary data
  double *x = new double [ n ];
  double *y = new double [ m ];

  // Initialize values and residuals
  for (int i = 0; i < n; i++) x[i] = io[i];
  for (int i = 0; i < m; i++) y[i] = 0;

#ifdef USE_LMDER1
  // Allocate temporary data
  int lwa = 5*n+m;
  double *wa = new double [ lwa ];
  double *jacobian = new double [ m * n ];
  int *ipvt = new int [ n ];
  double tol = tolerance;   // 1E-3; // sqrt(dpmpar(1));

  // Run the solver
  int status = lmder1(MinpackCallback, (void *) system, m, n, x, y, jacobian, m, tol, ipvt, wa, lwa);
  if (status == 0) fprintf(stderr, "Error in Minpack solver\n");
  else { for (int i = 0; i < n; i++) io[i] = x[i]; }

  // Delete temporary data
  delete [] ipvt;
  delete [] jacobian;
  delete [] wa;
#else
  // Allocate temporary data
  int lwa = m*n+5*n+m;
  int *iwa = new int [ n ];
  double *wa = new double [ lwa ];
  double tol = 1E-3; // sqrt(dpmpar(1));

  // Run the solver
  int status = lmdif1(MinpackFunction, (void *) system, m, n, x, y, tol, iwa, wa, lwa);
  if (status == 0) fprintf(stderr, "Error in Minpack solver\n");
  else { for (int i = 0; i < n; i++) io[i] = x[i]; }

  // Delete temporary data
  delete [] iwa;
  delete [] wa;
#endif

  // Delete values and residuals
  delete [] x;
  delete [] y;

  // Return status 
  return status;
}

#else

static int 
MinimizeMINPACK(const RNSystemOfEquations *system, RNScalar *io, RNScalar tolerance)
{
  // Print error message
  fprintf(stderr, "Cannot minimize equation: Minpack solver disabled during compile.\n");
  fprintf(stderr, "Enable it by adding -DRN_USE_MINPACK and -lminpack to compilation and link commands.\n");
  return 0;
}

#endif



////////////////////////////////////////////////////////////////////////
// CSPARSE Stuff
////////////////////////////////////////////////////////////////////////

#ifdef RN_NO_CSPARSE
#undef RN_USE_CSPARSE
#endif
#ifdef RN_USE_CSPARSE

#include "CSparse/CSparse.h"

static int 
MinimizeCSPARSE(const RNSystemOfEquations *system, RNScalar *io, RNScalar tolerance)
{
  // Get convenient variables
  const int n = system->NVariables();
  const int mm = system->NEquations();
  const int max_nz = system->NPartialDerivatives();

  // Allocate matrix
  cs *a = cs_spalloc (0, n, max_nz, 1, 1);
  if (!a) {
    fprintf(stderr, "Unable to allocate cs matrix: %d %d\n", n, max_nz);
    return 0;
  }
    
  // Allocate B vector
  double *b = new double [ mm ];
  for (int i = 0; i < mm; i++) b[i] = 0;

  // Allocate X vector
  double *x = new double [ n ];
  for (int i = 0; i < n; i++) x[i] = 0;

  // Allocate temporary data for rows
  double *lhs = new double [ n ];
  for (int i = 0; i < n; i++) lhs[i] = 0;

  // Fill matrix
  int m = 0;
  for (int i = 0; i < system->NEquations(); i++) {
    RNEquation *equation = system->Equation(i);

    // Initialize constant term
    double rhs = -equation->Evaluate(x);

    // Mark variables in equation
    int nz = 0;
    int variable_count = 0;
    RNSystemOfEquations *tmp = (RNSystemOfEquations *) system;
    equation->UpdateVariableIndex(n, variable_count, tmp->variable_marks, tmp->current_mark++, tmp->index_to_variable);
    for (int j = 0; j < variable_count; j++) {
      int v = tmp->index_to_variable[j];
      lhs[v] = equation->PartialDerivative(x, v);
      if (lhs[v] != 0) nz++;
    }

    // Add data to matrix if there are nonzero entries 
    if (nz > 0) {
      assert(m < mm);
      for (int j = 0; j < variable_count; j++) {
        int v = tmp->index_to_variable[j];
        if (lhs[v] == 0) continue;
        cs_entry(a, m, v, lhs[v]);
      }
      b[m] = rhs;
      m++;
    }
  }

  // Just checking
  assert(a->m == m);
  assert(a->n == n);
  assert(a->n == system->NVariables());
  assert(a->m <= system->NEquations());
  assert(a->nz <= system->NPartialDerivatives());

  // Setup aT * a * x = aT * b        
  cs *A = cs_compress(a);
  assert(A);
  cs *AT = cs_transpose (A, 1);
  assert(AT);
  cs *ATA = cs_multiply (AT, A);
  assert(ATA);
  cs_gaxpy(AT, b, x);

  // Solve linear system
  // int status = cs_lusol (1, ATA, x, RN_EPSILON);
  int status = cs_cholsol (1, ATA, x);
  if (status == 0) fprintf(stderr, "Error in CSPARSE solver\n");
  else { for (int i = 0; i < n; i++) io[i] = x[i]; }

  // Delete stuff
  cs_spfree(A);
  cs_spfree(AT);
  cs_spfree(ATA);
  cs_spfree(a);
  delete [] b;
  delete [] x;
  delete [] lhs;

  // Return status
  return status;
}

#else

static int 
MinimizeCSPARSE(const RNSystemOfEquations *system, RNScalar *io, RNScalar tolerance)
{
  // Print error message
  fprintf(stderr, "Cannot minimize equation: CSparse solver disabled during compile.\n");
  fprintf(stderr, "Enable it by adding -DRN_USE_CSPARSE and -lCSparse to compilation and link commands.\n");
  return 0;
}

#endif




inline int RNSystemOfEquations::
Minimize(RNScalar *x, int solver, RNScalar tolerance) const
{
  // Check solver
  if (solver == RN_SPLM_SOLVER) return MinimizeSPLM(this, x, tolerance);
  else if (solver == RN_MINPACK_SOLVER) return MinimizeMINPACK(this, x, tolerance);
  else if (solver == RN_CERES_SOLVER) return MinimizeCERES(this, x, tolerance);
  else if (solver == RN_CSPARSE_SOLVER) return MinimizeCSPARSE(this, x, tolerance);
  fprintf(stderr, "System of equation solver not recognized: %d\n", solver);
  return 0;
}



#endif
