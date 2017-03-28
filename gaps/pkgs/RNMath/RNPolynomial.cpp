// Source file for polynomial classes



////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "RNMath/RNMath.h"



////////////////////////////////////////////////////////////////////////
// Utility function
////////////////////////////////////////////////////////////////////////

#if 0
static int
RNComparePolynomialTerms(const RNPolynomialTerm *term1, const RNPolynomialTerm *term1)
{
  // Check number of variables
  if (term1->NVariables() < term2->NVariables()) return -1;
  else if (term1->NVariables() > term2->NVariables()) return 1;

  // Compare variables and exponents 
  for (int i = 0; i < term1->NVariables(); i++) {
    // This assumes variables are sorted
    if (term1->Variable(i) < term2->Variable(i)) return -1;
    else if (term1->Variable(i) > term2->Variable(i)) return 1;
    if (term1->Exponent(i) < term2->Exponent(i)) return -1;
    else if (term1->Exponent(i) > term2->Exponent(i)) return 1;
  }

  // Terms have the same variables and exponents
  return 0;
}
#endif



////////////////////////////////////////////////////////////////////////
// Polynomial
////////////////////////////////////////////////////////////////////////

RNPolynomial::
RNPolynomial(void)
  : terms()
{
}



RNPolynomial::
RNPolynomial(const RNPolynomial& polynomial)
  :terms()
{
  // Copy polynomial terms
  terms.Resize(polynomial.NTerms());
  for (int i = 0; i < polynomial.NTerms(); i++) {
    RNPolynomialTerm *polynomial_term = polynomial.Term(i);
    RNPolynomialTerm *term = new RNPolynomialTerm(*polynomial_term);
    terms.Insert(term); 
    term->polynomial = this; 
  }
}



RNPolynomial::
RNPolynomial(RNScalar c, int v, RNScalar e)
  : terms()
{
  // Add term
  AddTerm(c, v, e, TRUE);
}



RNPolynomial::
RNPolynomial(RNScalar c, int nv, const int *v, const RNScalar *e, 
    RNBoolean already_sorted)
  : terms()
{
  // Add term
  AddTerm(c, nv, v, e, already_sorted);
}



RNPolynomial::
~RNPolynomial(void)
{
  // Delete terms
  Empty();
}



RNScalar RNPolynomial::
Degree(void) const
{
  // Return highest degree of any term
  RNScalar max_degree = 0;
  for (int i = 0; i < NTerms(); i++) {
    RNPolynomialTerm *term = Term(i);
    RNScalar degree = term->Degree();
    if (degree > max_degree) max_degree = degree;
  }
  return max_degree;
}



int RNPolynomial::
NVariables(void) const
{
  // Check if there are no terms
  if (NTerms() == 0) return 0;

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
  return nvariables;
}



RNBoolean RNPolynomial::
IsZero(void) const
{
  // Return whether polynomial is definitely zero
  if (NTerms() == 0) return TRUE;
  else return FALSE;
}



RNBoolean RNPolynomial::
IsOne(void) const
{
  // Return whether polynomial is definitely zero
  if (NTerms() != 1) return FALSE;
  RNPolynomialTerm *term = Term(0);
  return term->IsOne();
}



RNBoolean RNPolynomial::
IsConstant(void) const
{
  // Check if no terms
  if (NTerms() == 0) return TRUE;

  // Check if multiple terms
  if (NTerms() > 1) return FALSE;

  // Check term
  RNPolynomialTerm *term = Term(0);
  if (term->IsConstant()) return TRUE;
  else return FALSE;
}



RNBoolean RNPolynomial::
IsLinear(void) const
{
  // Check each term
  for (int i = 0; i < NTerms(); i++) {
    RNPolynomialTerm *term = Term(i);
    if (!term->IsLinear()) return FALSE;
  }

  // Passed all tests
  return TRUE;
}



RNBoolean RNPolynomial::
IsQuadratic(void) const
{
  // Check each term
  for (int i = 0; i < NTerms(); i++) {
    RNPolynomialTerm *term = Term(i);
    if (!term->IsQuadratic()) return FALSE;
  }

  // Passed all tests
  return TRUE;
}



RNBoolean RNPolynomial::
HasVariable(int variable) const
{
  // Check terms
  for (int i = 0; i < NTerms(); i++) {
    RNPolynomialTerm *term = Term(i);
    if (term->HasVariable(variable)) return TRUE;
  }

  // Not found
  return FALSE;
}



void RNPolynomial::
Empty(void)
{
  // Delete all terms
  for (int i = 0; i < NTerms(); i++) {
    RNPolynomialTerm *term = Term(i);
    delete term;
  }

  // Empty array of terms
  terms.Empty();
}



void RNPolynomial::
Negate(void)
{
  // Negate all terms
  for (int i = 0; i < NTerms(); i++) {
    RNPolynomialTerm *term = Term(i);
    term->Negate();
  }
}



void RNPolynomial::
Multiply(RNScalar factor)
{
  // Check if factor is zero
  if (factor == 0.0) {
    // Delete all terms
    Empty();
  }
  else {
    // Multiply all terms by factor
    for (int i = 0; i < NTerms(); i++) {
      RNPolynomialTerm *term = Term(i);
      term->Multiply(factor);
    }
  }
}



void RNPolynomial::
Divide(RNScalar factor)
{
  // Multiply all terms by factor
  if (RNIsZero(factor)) return;
  Multiply(1.0/factor);
}



void RNPolynomial::
Add(const RNPolynomial& polynomial)
{
  // Add polynomial
  for (int i = 0; i < polynomial.NTerms(); i++) {
    RNPolynomialTerm *term = polynomial.Term(i);
    AddTerm(term->Coefficient(), term->NVariables(), term->Variables(), term->Exponents(), TRUE, TRUE);
  }
}



void RNPolynomial::
Subtract(const RNPolynomial& polynomial)
{
  // Subtract polynomial
  for (int i = 0; i < polynomial.NTerms(); i++) {
    RNPolynomialTerm *term = polynomial.Term(i);
    AddTerm(-(term->Coefficient()), term->NVariables(), term->Variables(), term->Exponents(), TRUE, TRUE);
  }
}



void RNPolynomial::
Multiply(const RNPolynomial& polynomial)
{
  // Check special cases for speed
  if (IsZero()) {
    // Do nothing
  }
  else if (polynomial.IsZero()) {
    Multiply(0.0);
  }
  else if (polynomial.IsConstant()) {
    // Polynomial is constant (multiply by factor)
    assert(polynomial.NTerms() == 1);
    RNPolynomialTerm *term = polynomial.Term(0);
    assert(term->IsConstant());
    Multiply(term->Coefficient());
  }
  else {
    // Convenient variables
    RNPolynomial polynomial1(*this);
    RNPolynomial polynomial2(polynomial);
    const int max_factors = 1024;
    int v [ max_factors ];
    RNScalar e [ max_factors ];
    int n = 0;
    
    // Start from scratch
    Empty();
    
    // For each term of this polynomial
    for (int i = 0; i < polynomial1.NTerms(); i++) {
      RNPolynomialTerm *term1 = polynomial1.Term(i);
      
      // Start with factors from this term
      n = 0;
      for (int k = 0; k < term1->NVariables(); k++) {
        if (n >= max_factors) break;
        v[n] = term1->Variable(k);
        e[n] = term1->Exponent(k);
        n++;
      }
      
      // For each term of other polynomial
      for (int j = 0; j < polynomial2.NTerms(); j++) {
        RNPolynomialTerm *term2 = polynomial2.Term(j);
        
        // Include factors from other term
        n = term1->NVariables();
        for (int k = 0; k < term2->NVariables(); k++) {
          if (n >= max_factors) break;
          v[n] = term2->Variable(k);
          e[n] = term2->Exponent(k);
          n++;
        }
        
        // Add term to this polynomial
        AddTerm(term1->Coefficient() * term2->Coefficient(), n, v, e);
      }
    }
  }
}



RNPolynomial& RNPolynomial::
operator=(const RNPolynomial& polynomial)
{
  // Empty this polynomial
  Empty();

  // Copy polynomial terms
  terms.Resize(polynomial.NTerms());
  for (int i = 0; i < polynomial.NTerms(); i++) {
    RNPolynomialTerm *polynomial_term = polynomial.Term(i);
    RNPolynomialTerm *term = new RNPolynomialTerm(*polynomial_term);
    terms.Insert(term); 
    term->polynomial = this; 
  }

  // Return this
  return *this;
}



void RNPolynomial::
AddTerm(RNScalar c, int n, const int *v, const RNScalar *e, 
  RNBoolean already_sorted, RNBoolean already_unique)
{
  // Check coefficient
  if (c == 0) return;

  // Check everything else
  assert((n == 0) || ((v != NULL) && (e != NULL)));

  // Check if term is constant (handle separately for efficiency)
  if (n == 0) {
    // Add constant term
    if (NTerms() > 0) {
      // Check for existing constant term
      RNPolynomialTerm *match = Term(0);
      if (match->IsConstant()) {
        // Add constant to existing constant term
        match->c += c;
        if (match->c == 0.0) {
          // Remove unneeded term
          terms.Remove(match);
          delete match;
        }
      }
      else {
        // Create new constant term (put first in list)
        RNPolynomialTerm *term = new RNPolynomialTerm(c, 0, NULL, NULL, TRUE, TRUE);
        terms.InsertHead(term);
        term->polynomial = this; 
      }
    }
    else {
      // Create new constant term 
      RNPolynomialTerm *term = new RNPolynomialTerm(c, 0, NULL, NULL, TRUE, TRUE);
      terms.InsertHead(term);
      term->polynomial = this; 
    }
  }
  else {
    // Add variable term
    RNPolynomialTerm *term = new RNPolynomialTerm(c, n, v, e, already_sorted, already_unique);
    RNPolynomialTerm *match = FindTermWithSameVariables(term);
    if (match) { 
      // Update existing term
      match->c += c;
      delete term; 
      if (match->c == 0.0) {
        // Remove unneeded term
        terms.Remove(match);
        delete match;
      }
    }
    else { 
      // Insert new term
      terms.Insert(term);
      term->polynomial = this; 
    }
  }
}



RNScalar RNPolynomial::
Evaluate(const RNScalar *x) const
{
  // Return sum of terms
  RNScalar sum = 0;
  for (int i = 0; i < NTerms(); i++) {
    RNPolynomialTerm *term = Term(i);
    sum += term->Evaluate(x);
  }
  return sum;
}



RNScalar RNPolynomial::
Evaluate(double const* const* x) const
{
  // Evaluate polynomial (in format suitable for CERES)
  RNScalar sum = 0;
  for (int i = 0; i < NTerms(); i++) {
    RNPolynomialTerm *term = Term(i);
    sum += term->Evaluate(x);
  }
  return sum;
}



RNScalar RNPolynomial::
PartialDerivative(const RNScalar *x, int variable) const
{
  // Return sum of term partial derivatives for variable
  RNScalar sum = 0;
  for (int i = 0; i < NTerms(); i++) {
    RNPolynomialTerm *term = Term(i);
    sum += term->PartialDerivative(x, variable);
  }
  return sum;
}



RNScalar RNPolynomial::
PartialDerivative(double const* const* x, int variable) const
{
  // Return sum of term partial derivatives for variable
  RNScalar sum = 0;
  for (int i = 0; i < NTerms(); i++) {
    RNPolynomialTerm *term = Term(i);
    sum += term->PartialDerivative(x, variable);
  }
  return sum;
}



void RNPolynomial::
Print(FILE *fp) const
{
  // Print polynomial
  printf("{ ");
  for (int i = 0; i < NTerms(); i++) {
    RNPolynomialTerm *term = Term(i);
    fprintf(fp, "  ");
    term->Print(fp);
  }
  printf(" }\n");
}



RNPolynomialTerm *RNPolynomial::
FindTermWithSameVariables(const RNPolynomialTerm *query) const
{
  // Check if any terms
  if (NTerms() == 0) return NULL;

  // Check if query has variables
  if (query->IsConstant()) {
    // Check for constant term (must be first)
    RNPolynomialTerm *term = Term(0);
    if (term->IsConstant()) return term;
    else return NULL;
  }
  else {
    // Find term with matching variables
    for (int i = 0; i < NTerms(); i++) {
      RNPolynomialTerm *term = Term(i);
      assert((i == 0) || (!term->IsConstant()));
      if (term->HasVariables(query->NVariables(), query->Variables(), query->Exponents())) return term;
    }
  }

  // Did not find matching term
  return NULL;
}
  


RNPolynomialTerm *RNPolynomial::
FindTermWithVariables(int n, int *v, RNScalar *e) const
{
  // Check if query has variables
  if (n == 0) {
    // Check for constant term (must be first)
    RNPolynomialTerm *term = Term(0);
    if (term->IsConstant()) return term;
    else return NULL;
  }
  else {
    // Find matching term
    for (int i = 0; i < NTerms(); i++) {
      RNPolynomialTerm *term = Term(i);
      assert((i == 0) || (!term->IsConstant()));
      if (term->HasVariables(n, v, e)) return term;
    }
  }

  // Did not find matching term
  return NULL;
}
  


void RNPolynomial::
UpdateVariableRange(int& min_v, int& max_v) const
{
  // Update variable index for each term
  for (int i = 0; i < NTerms(); i++) {
    RNPolynomialTerm *term = Term(i);
    term->UpdateVariableRange(min_v, max_v);
  }
}



void RNPolynomial::
UpdateVariableIndex(int max_variables, int& variable_count, 
  int *variable_marks, int current_mark, 
  int *index_to_variable, int *variable_to_index,
  RNBoolean remap_variables) const
{
  // Just checking
  assert(variable_to_index || !remap_variables);

  // Update variable index for each term
  for (int i = 0; i < NTerms(); i++) {
    RNPolynomialTerm *term = Term(i);
    term->UpdateVariableIndex(max_variables, variable_count, 
      variable_marks, current_mark,
      index_to_variable, variable_to_index,
      remap_variables);
  }
}

////////////////////////////////////////////////////////////////////////
// Polynomial term
////////////////////////////////////////////////////////////////////////

RNPolynomialTerm::
RNPolynomialTerm(RNScalar _c, int _n, const int *_v, const RNScalar *_e, 
  RNBoolean already_sorted, RNBoolean already_unique)
  : n(0),
    c(_c)
{
  // Check number of variables
  if (_n == 0) {
#ifndef RN_POLYNOMIAL_TERM_STATIC_MEMORY
    v = NULL;
    e = NULL;
#endif
  }
  else {
    // Copy term info
    n = 0;
#ifndef RN_POLYNOMIAL_TERM_STATIC_MEMORY
    v = new int [ _n ];
    e = new RNScalar [ _n ];
#else
    // Check number of variables
    if (_n > max_variables) {
      fprintf(stderr, "RNPolynomialTerm max variables exceeded\n");
      return;
    }
#endif
    for (int i = 0; i < _n; i++) {
      if (_e[i] != 0.0) {
        assert(n < _n);
        v[n] = _v[i];
        e[n] = _e[i];
        n++;
      }
    }

    // Sort variables
    if (!already_sorted) {
      for (int i = 1; i < n; i++) {
        for (int j = i; j > 0; j--) {
          if (v[j] < v[j-1]) {
            assert(j > 0);
            assert(j < _n);
            int vswap = v[j-1];
            v[j-1] = v[j];
            v[j] = vswap;
            double eswap = e[j-1];
            e[j-1] = e[j];
            e[j] = eswap;
          }
        }
      }
    }

    // Check for duplicate variables
    if (!already_unique) {
      for (int i = 0; i < n-1; i++) {
        assert(i >= 0);
        assert(n >= 0);
        assert(i < n-1);
        if (v[i] == v[i+1]) {
          assert(i < n-1);
          e[i] += e[i+1];
          if (RNIsZero(e[i])) {
            for (int j = i; j < n-2; j++) {
              assert(j >= 0);
              assert(j < n-2);
              v[j] = v[j+2];
              e[j] = e[j+2];
            }
            n -= 2;
            i -= 2;
          }
          else {
            for (int j = i+1; j < n-1; j++) {
              assert(j >= 0);
              assert(j < n-1);
              v[j] = v[j+1];
              e[j] = e[j+1];
            }
            n -= 1;
            i--;
          }
        }
      }
    }
  }
}



RNPolynomialTerm::
RNPolynomialTerm(const RNPolynomialTerm& term)
  : n(term.n),
    c(term.c)
{
  // Copy stuff from term
  if (n == 0) {
#ifndef RN_POLYNOMIAL_TERM_STATIC_MEMORY
    v = NULL;
    e = NULL;
#endif
  }
  else {
#ifndef RN_POLYNOMIAL_TERM_STATIC_MEMORY
    v = new int [ n ];
    e = new RNScalar [ n ];
#endif
    for (int i = 0; i < n; i++) {
      v[i] = term.v[i];
      e[i] = term.e[i];
    }
  }
}



RNPolynomialTerm::
~RNPolynomialTerm(void)
{
  // Delete stuff
#ifndef RN_POLYNOMIAL_TERM_STATIC_MEMORY
  if (v) delete [] v;
  if (e) delete [] e;
#endif
}




RNScalar RNPolynomialTerm::
Degree(void) const
{
  // Return degree
  RNScalar degree = 0;
  for (int i = 0; i < NVariables(); i++) degree += e[i];
  return degree;
}



void RNPolynomialTerm::
Empty(void)
{
  // Delete stuff
#ifndef RN_POLYNOMIAL_TERM_STATIC_MEMORY
  if (v) delete [] v;
  if (e) delete [] e;
#endif
  c = 0;
  n = 0;
}



RNBoolean RNPolynomialTerm::
IsZero(void) const
{
  // Return whether term is zero for all possible evaluations
  if (c == 0) return TRUE;
  return FALSE;
}



RNBoolean RNPolynomialTerm::
IsOne(void) const
{
  // Return whether term is one for all possible evaluations
  if ((c == 1) && (n == 0)) return TRUE;
  return FALSE;
}



RNBoolean RNPolynomialTerm::
IsConstant(void) const
{
  // Return whether term is a constant
  if (n == 0) return TRUE;
  return FALSE;
}



RNBoolean RNPolynomialTerm::
IsLinear(void) const
{
  // Return whether term is linear
  if (IsConstant()) return TRUE;
  if ((n == 1) && (e[0] == 1.0)) return TRUE;
  return FALSE;
}



RNBoolean RNPolynomialTerm::
IsQuadratic(void) const
{
  // Return whether term is quadratic
  if (IsLinear()) return TRUE;
  if ((n == 1) && (e[0] == 2.0)) return TRUE;
  if ((n == 2) && (e[0] == 1.0) && (e[1] == 1.0)) return TRUE;
  return FALSE;
}



RNBoolean RNPolynomialTerm::
HasVariable(int variable) const
{
  // Search for variable
  for (int i = 0; i < n; i++) {
    if (v[i] == variable) return TRUE;
  }

  // Not found
  return FALSE;
}



RNScalar RNPolynomialTerm::
Evaluate(const RNScalar *x) const
{
  // Evaluate the term
  RNScalar result = c;
  for (int i = 0; i < NVariables(); i++) {
    int k = Variable(i);
    if (e[i] == 1.0) result *= x[k];
    else if (e[i] == 2.0) result *= x[k] *x[k];
    else if ((e[i] < 0) && RNIsZero(x[k])) result *= RN_INFINITY;
    else result *= pow(x[k], e[i]);
  }
  return result;
}



RNScalar RNPolynomialTerm::
Evaluate(double const* const* x) const
{
  // Evaluate the term (in format suitable for CERES)
  RNScalar result = c;
  for (int i = 0; i < NVariables(); i++) {
    int k = Variable(i);
    if (e[i] == 1.0) result *= x[k][0];
    else if (e[i] == 2.0) result *= x[k][0] * x[k][0];
    else if ((e[i] < 0) && RNIsZero(x[k][0])) result *= RN_INFINITY;
    else result *= pow(x[k][0], e[i]);
  }
  return result;
}



RNScalar RNPolynomialTerm::
PartialDerivative(const RNScalar *x, int variable) const
{
  // Check if has variable
  if (!HasVariable(variable)) return 0.0;

  // Evaluate the derivative with respect to variable
  RNScalar result = c;
  for (int i = 0; i < NVariables(); i++) {
    int k = Variable(i);
    if (variable == k) {
      if (e[i] == 1.0) result *= 1.0;
      else if (e[i] == 2.0) result *= 2.0 * x[k];
      else if ((e[i] < 1.0) && RNIsZero(x[k])) result *= RN_INFINITY;
      else result *= e[i] * pow(x[k], e[i] - 1.0);
    }
    else {
      if (e[i] == 1.0) result *= x[k];
      else if (e[i] == 2.0) result *= x[k] * x[k];
      else if ((e[i] < 0) && RNIsZero(x[k])) result *= RN_INFINITY;
      else result *= pow(x[k], e[i]);
    }
  }
  return result;
}



RNScalar RNPolynomialTerm::
PartialDerivative(double const* const* x, int variable) const
{
  // Check if has variable
  if (!HasVariable(variable)) return 0.0;

  // Evaluate the derivative with respect to variable
  RNScalar result = c;
  for (int i = 0; i < NVariables(); i++) {
    int k = Variable(i);
    if (k == variable) {
      if (e[i] == 1.0) result *= 1.0;
      else if (e[i] == 2.0) result *= 2.0 * x[k][0];
      else if ((e[i] < 1.0) && RNIsZero(x[k][0])) result *= RN_INFINITY;
      else result *= e[i] * pow(x[k][0], e[i] - 1.0);
    }
    else {
      if (e[i] == 1.0) result *= x[k][0];
      else if (e[i] == 2.0) result *= x[k][0] * x[k][0];
      else if ((e[i] < 0) && RNIsZero(x[k][0])) result *= RN_INFINITY;
      else result *= pow(x[k][0], e[i]);
    }
  }
  return result;
}



void RNPolynomialTerm::
Print(FILE *fp) const
{
  // Print term
  fprintf(fp, "%g ", Coefficient());
  for (int j = 0; j < NVariables(); j++) {
    fprintf(fp, "(%d^%g) ", Variable(j), Exponent(j));
  }
}




RNBoolean RNPolynomialTerm::
HasVariables(int query_n, const int *query_v, const RNScalar *query_e) const
{
  // Check if query has same number of variables
  if (n != query_n) return FALSE;

  // Compare variables and exponents 
  for (int i = 0; i < n; i++) {
    // This assumes variables are sorted
    if (v[i] != query_v[i]) return FALSE;
    if (e[i] != query_e[i]) return FALSE;
  }

  // Passed all tests
  return TRUE;
}



void RNPolynomialTerm::
UpdateVariableRange(int& min_v, int& max_v) const
{
  // Update indices
  for (int i = 0; i < n; i++) {
    if (v[i] > max_v) max_v = v[i];
    if (v[i] < min_v) min_v = v[i];
  }
}


void RNPolynomialTerm::
UpdateVariableIndex(int max_variables, int& variable_count, 
  int *variable_marks, int current_mark, 
  int *index_to_variable, int *variable_to_index,
  RNBoolean remap_variables) const
{
  // Just checking
  assert(variable_to_index || !remap_variables);

  // Process each variable
  for (int i = 0; i < n; i++) {
    assert((v[i] >= 0) && (v[i] < max_variables));
    // Update index
    if (variable_marks[v[i]] != current_mark) {
      if (index_to_variable) index_to_variable[variable_count] = v[i];
      if (variable_to_index) variable_to_index[v[i]] = variable_count;
      variable_marks[v[i]] = current_mark;
      variable_count++;
    }

    // Remap variable
    if (remap_variables) {
      v[i] = variable_to_index[v[i]];
    }
  }
}

