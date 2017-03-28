// Source file for the dense matrix after LU decomposition



// Include files

#include "RNMath.h"



RNDenseLUMatrix::
RNDenseLUMatrix(void)
  : RNDenseMatrix(),
    pivots(NULL)
{
}



RNDenseLUMatrix::
RNDenseLUMatrix(int nrows, int ncols, RNScalar *values)
  : RNDenseMatrix(nrows, ncols, values),
    pivots(NULL)
{
  // Perform LU decomposition
  Decompose();
}



RNDenseLUMatrix::
RNDenseLUMatrix(const RNMatrix& matrix)
  : RNDenseMatrix(matrix),
    pivots(NULL)
{
  // Perform LU decomposition
  Decompose();
}



RNDenseLUMatrix::
RNDenseLUMatrix(const RNDenseMatrix& matrix)
  : RNDenseMatrix(matrix),
    pivots(NULL)
{
  // Perform LU decomposition
  Decompose();
}



RNDenseLUMatrix::
RNDenseLUMatrix(const RNDenseLUMatrix& matrix)
  : RNDenseMatrix(matrix)
{
  // Copy pivots
  pivots = new int [ NRows() ];
  for (int i = 0; i < NRows(); i++) {
    pivots[i] = matrix.pivots[i];
  }
}



RNDenseLUMatrix::
~RNDenseLUMatrix(void)
{
  // Delete pivots
  if (pivots) delete [] pivots;
}



RNScalar RNDenseLUMatrix::
Determinant(void) const
{
  // Compute determinant
  RNScalar det = 1.0;
  for (int i = 0; i < ncols - 1; i++) {
      det *= Value(i,i);
      if (pivots[i] != i) {
          det = -det;
      }
  }
  det *= Value(ncols-1, ncols-1);
  return det;
}



RNDenseMatrix RNDenseLUMatrix::
Inverse() const
{
  // Compute inverse
  RNDenseMatrix result(nrows, nrows);
  RNDenseMatrix b(nrows, nrows);
  for (int i = 0; i < nrows; i++) b.SetValue(i, i, 1);
  BackSubstitute(b, result);
  return result;
}



int RNDenseLUMatrix::
DecomposeLU(RNDenseMatrix& L, RNDenseMatrix& U) const
{
  // LU decomposition
  L.Reset(NRows(), NRows());

  // Fill lower triangle matrix
  for (int i = 0; i < NRows(); i++) 
      L.SetValue(i, i, 1);
  for (int i = 0; i < NRows(); i++) {
    for (int j = 0; j < i; j++) {
      L.SetValue(i, j, Value(i, j));
    }
  }

  // Fill upper triangle matrix
  U.Reset(NRows(), NRows());
  for (int i = 0; i < NRows(); i++) {
    for (int j = i; j < NRows(); j++) {
      U.SetValue(i, j, Value(i, j));
    }
  }

  // Return success
  return 1;
}



void RNDenseLUMatrix::
Decompose(void) 
{
  // Just checking
  assert(nrows == ncols);

  // More convenient variables
  RNDenseLUMatrix& a = *this;

  // (Re)Allocate pivots
  if (pivots) delete [] pivots;
  pivots = new int [ nrows ];

  // Decompose
  for (int j = 0; j < nrows - 1; j++) {

    // Find line of pivot
    pivots[j] = j;
    for (int i = j + 1; i < nrows; i++) {
      if (fabs(a[i][j]) > fabs(a[pivots[j]][j])) {
        pivots[j] = i;
      }
    }

    // Swap lines if necessary
    if (pivots[j] != j) {
      int i = pivots[j];
      for (int k = 0; k < nrows; k++) {
        RNScalar t = a[i][k];
        a[i][k] = a[j][k];
        a[j][k] = t;
      }
    }

    // Check if matrix is singular
    if (RNIsZero(a[j][j])) {
      fprintf(stderr, "Warning: singular matrix given to LU");
      break;
    }

    // Eliminate elements below diagonal
    for (int i = j + 1; i < nrows; i++) {
      a[i][j] = -a[i][j] / a[j][j];
      for (int k = j + 1; k < nrows; k++) {
        a[i][k] += a[i][j] * a[j][k];
      }
    }
  }
}



void RNDenseLUMatrix::
BackSubstitute(RNDenseMatrix &b, RNDenseMatrix &x) const
{
  // Just checking
  assert(b.NRows() == ncols);
  assert(x.NRows() == b.NRows());
  assert(x.NColumns() == b.NColumns());

  // More convenient variables
  const RNDenseMatrix& a = *this;
  int m = x.NRows();
  int n = x.NColumns();

  // Swap lines of b when necessary
  for (int i = 0; i < m - 1; i++) {
    if (pivots[i] != i) {
      for (int j = 0, k = pivots[i]; j < n; j++) {
        RNScalar t = b[i][j];
        b[i][j] = b[k][j];
        b[k][j] = t;
      }
    }
  }

  // Apply multipliers to b
  for (int j = 0; j < m - 1; j++) {
    for (int i = j + 1; i < m; i++) {
      for (int k = 0; k < n; k++) {
        b[i][k] += a[i][j] * b[j][k];
      }
    }
  }

  // Back substitution
  for (int k = 0; k < n; k++) {
    for (int i = m - 1; i >= 0; i--) {
      x[i][k] = b[i][k];
      for (int j = i + 1; j < m; j++) {
        x[i][k] -= a[i][j] * x[j][k];
      }
      assert(fabs(a[i][i]) >= DBL_EPSILON);
      x[i][k] /= a[i][i];
    }
  }
}




