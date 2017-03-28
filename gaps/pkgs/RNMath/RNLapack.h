// Include file for Lapack wrapper functions


#ifdef RN_NO_LAPACK
#undef RN_USE_LAPACK
#endif
#ifdef RN_USE_LAPACK




////////////////////////////////////////////////////////////////////////
// External include files
////////////////////////////////////////////////////////////////////////

#include<stdlib.h>
#include<stdio.h>
#include<math.h>



////////////////////////////////////////////////////////////////////////
// System of equations
////////////////////////////////////////////////////////////////////////

extern "C" void dgesv_(int *n, int *nrhs, 
  double *a, int *lda, int *ipiv, double *b, 
  int *ldb, int *info );

//  N       (input) INTEGER
//          The number of linear equations, i.e., the order of the
//          matrix A.  N >= 0.
//
//  NRHS    (input) INTEGER
//          The number of right hand sides, i.e., the number of columns
//          of the matrix B.  NRHS >= 0.
//
//  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
//          On entry, the N-by-N coefficient matrix A.
//          On exit, the factors L and U from the factorization
//          A = P//L//U; the unit diagonal elements of L are not stored.
//
//  LDA     (input) INTEGER
//          The leading dimension of the array A.  LDA >= max(1,N).
//
//  IPIV    (output) INTEGER array, dimension (N)
//          The pivot indices that define the permutation matrix P;
//          row i of the matrix was interchanged with row IPIV(i).
//
//  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
//          On entry, the N-by-NRHS matrix of right hand side matrix B.
//          On exit, if INFO = 0, the N-by-NRHS solution matrix X.
//
//  LDB     (input) INTEGER
//          The leading dimension of the array B.  LDB >= max(1,N).
//
//  INFO    (output) INTEGER
//          = 0:  successful exit
//          < 0:  if INFO = -i, the i-th argument had an illegal value
//          > 0:  if INFO = i, U(i,i) is exactly zero.  The factorization
//                has been completed, but the factor U is exactly
//                singular, so the solution could not be computed.

// Solve a system of equations Ax=b with n variables and n equations
// A is a n by n matrix 
// b has n right hand sides
// x has n variables -- this vector will be filled in with the answer
// nrhs has the number of right hand sides -- i.e., the nuumber of columns in b and x
inline int RNSolveLinearSystem(double *A, double *x, double *b, int n, int nrhs)
{
  // Allocate temporary memory
  int info; 
  int *ipiv = new int [ n ];
  double *a = new double [ n * n ];
  double *X = new double [ n * nrhs ];
    
  // Copy x/b into working buffer
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < nrhs; j++) {
      X[j*n+i] = b[i*nrhs+j];
    }
  }

  // Copy A into working buffer
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      a[j*n+i] = A[i*n+j];
    }
  }

  // Call LAPACK function
  dgesv_(&n, &nrhs, a, &n, ipiv, X, &n, &info);
  if (info != 0) {
    fprintf(stderr, "Error solving system of equations: %d\n", info);
  }

  // Copy answer into result
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < nrhs; j++) {
      x[i*nrhs+j] = X[j*n+i];
    }
  }

  // Delete temporary memory
  delete [] ipiv;
  delete [] a;
  delete [] X;

  // Return status
  return (info == 0) ? 1 : 0;
}


////////////////////////////////////////////////////////////////////////
// Least squares
////////////////////////////////////////////////////////////////////////

extern "C" void dgels_(char *trans, int *M, 
  int *N, int *nrhs, double *A, int *lda, double *b, 
  int *ldb, double *work, int *lwork, int *info);

//  TRANS   (input) CHARACTER*1
//          = 'N': the linear system involves A;
//          = 'T': the linear system involves A**T.
//
//  M       (input) INTEGER
//          The number of rows of the matrix A.  M >= 0.
//
//  N       (input) INTEGER
//          The number of columns of the matrix A.  N >= 0.
//
//  NRHS    (input) INTEGER
//          The number of right hand sides, i.e., the number of
//          columns of the matrices B and X. NRHS >=0.
//
//  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
//          On entry, the M-by-N matrix A.
//          On exit,
//            if M >= N, A is overwritten by details of its QR
//                       factorization as returned by DGEQRF;
//            if M <  N, A is overwritten by details of its LQ
//                       factorization as returned by DGELQF.
//
//  LDA     (input) INTEGER
//          The leading dimension of the array A.  LDA >= max(1,M).
//
//  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS)
//          On entry, the matrix B of right hand side vectors, stored
//          columnwise; B is M-by-NRHS if TRANS = 'N', or N-by-NRHS
//          if TRANS = 'T'.
//          On exit, if INFO = 0, B is overwritten by the solution
//          vectors, stored columnwise:
//          if TRANS = 'N' and m >= n, rows 1 to n of B contain the least
//          squares solution vectors; the residual sum of squares for the
//          solution in each column is given by the sum of squares of
//          elements N+1 to M in that column;
//          if TRANS = 'N' and m < n, rows 1 to N of B contain the
//          minimum norm solution vectors;
//          if TRANS = 'T' and m >= n, rows 1 to M of B contain the
//          minimum norm solution vectors;
//          if TRANS = 'T' and m < n, rows 1 to M of B contain the
//          least squares solution vectors; the residual sum of squares
//          for the solution in each column is given by the sum of
//          squares of elements M+1 to N in that column.
//
//  LDB     (input) INTEGER
//          The leading dimension of the array B. LDB >= MAX(1,M,N).
//
//  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
//          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
//
//  LWORK   (input) INTEGER
//          The dimension of the array WORK.
//          LWORK >= max( 1, MN + max( MN, NRHS ) ).
//          For optimal performance,
//          LWORK >= max( 1, MN + max( MN, NRHS )*NB ).
//          where MN = min(M,N) and NB is the optimum block size.
//
//          If LWORK = -1, then a workspace query is assumed; the routine
//          only calculates the optimal size of the WORK array, returns
//          this value as the first entry of the WORK array, and no error
//          message related to LWORK is issued by XERBLA.
//
//  INFO    (output) INTEGER
//          = 0:  successful exit
//          < 0:  if INFO = -i, the i-th argument had an illegal value
//          > 0:  if INFO =  i, the i-th diagonal element of the
//                triangular factor of A is zero, so that A does not have
//                full rank; the least squares solution could not be
//                computed.

// Solve an over-constrained system of equations Ax=b with m variables and n equations
// A is a m by n matrix 
// b has m right hand sides
// x has n variables -- this vector will be filled in with the answer
// nrhs has the number of right hand sides -- i.e., the nuumber of columns in b and x
inline int RNSolveLeastSquares(double *A, double *x, double *b, int m, int n, int nrhs)
{
  // Allocate temporary memory
  char trans = 'N';
  int info; 
  int lwork = 64 * m; // ???
  double *work = new double [ lwork ];
  double *a = new double [ m * n ];
  double *X = new double [ m * nrhs ];

  // Copy x/b into working buffer
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < nrhs; j++) {
      X[j*m+i] = b[i*nrhs+j];
    }
  }

  // Copy A into working buffer
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      a[j*m+i] = A[i*n+j];
    }
  }
    
  // Call LAPACK function to solve system of equations
  dgels_(&trans, &m, &n, &nrhs, a, &m, X, &m, work, &lwork, &info);
  if (info != 0) {
    fprintf(stderr, "Error solving system of equations: %d\n", info);
  }

  // Copy answer into result
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < nrhs; j++) {
      x[i*nrhs+j] = X[j*m+i];
    }
  }

  // Delete temporary memory
  delete [] work;
  delete [] a;
  delete [] X;

  // Return status
  return (info == 0) ? 1 : 0;
}



////////////////////////////////////////////////////////////////////////

extern "C" void dgesvd_(const char* jobu, const char* jobvt, int* M, int* N,
        double* A, int* lda, double* S, double* U, int* ldu,
        double* VT, int* ldvt, double* work, int* lwork, const int* info);

// Perform SVD on matrix a
inline int RNDecomposeSVD(const double *A, double *u, double *w, double *vt, int m, int n)
{
  // Allocate temporary memory
  int info; 
  int lwork = 10*m;
  double *work = new double [ lwork ];
  double *a = new double [ m * n ];
  double *t = new double [ m * m ];
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      a[j*m+i] = A[i*n+j];
    }
  }

  // Call the lapack function
  dgesvd_("All", "All", &m, &n, a, &m, w, t, &m, vt, &n, work, &lwork, &info);

  // Copy u
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      u[i*n+j] = t[j*m+i];
    }
  }

  // Transpose vt 
  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
      double swap = vt[j*n+i];
      vt[j*n+i] = vt[i*n+j];
      vt[i*n+j] = swap;
    }
  }

  // Delete memory
  delete [] work;
  delete [] t;
  delete [] a;

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////

extern "C" void dgeev_(const char* jobvl, const char* jobvr, int* N,
        double* A, int* lda, double* wr, double* wi, 
        double *vl, int *ldvl, double *vr, int *ldvr,
        double* work, int* lwork, const int* info);

// Compute eigenvalues and eigenvectors of a symmetric matrix a
inline int RNDecomposeEigen(const double *A, double *eigenvalues, double *eigenvectors, int n)
{
  // Allocate temporary memory
  int info; 
  int lwork = 100*n;
  double *work = new double [ lwork ];
  double *wr = new double [ n ];
  double *wi = new double [ n ];
  double *vl = new double [ 2 * n * n ];
  double *vr = new double [ 2 * n * n ];
  double *a = new double [ n * n ];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      a[j*n+i] = A[i*n+j];
    }
  }

  // Call the lapack function
  dgeev_("N", "V", &n, a, &n, wr, wi, vl, &n, vr, &n, work, &lwork, &info);

  // Copy eigenvalues
  for (int i = 0; i < n; i++) {
    eigenvalues[i] = wr[i];
  }

  // Copy eigenvectors
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      eigenvectors[j*n+i] = vr[i*n+j];
    }
  }

  // Delete memory
  delete [] work;
  delete [] wr;
  delete [] wi;
  delete [] vl;
  delete [] vr;
  delete [] a;

  // Return success (eigenvectors are in columns of result)
  return 1;
}



#else

inline int RNSolveLinearSystem(double *A, double *x, double *b, int n, int rhs)
{
  // Print error message
  fprintf(stderr, "Cannot execute RNSolveLinearSystem: LAPACK solver disabled during compile.\n");
  fprintf(stderr, "Enable it by adding -DRN_USE_LAPACK and -llapack to compilation and link commands.\n");
  return 0;
}



inline int RNSolveLeastSquares(double *A, double *x, double *b, int m, int n, int nrhs)
{
  // Print error message
  fprintf(stderr, "Cannot execute RNSolveLeastSquares: LAPACK solver disabled during compile.\n");
  fprintf(stderr, "Enable it by adding -DRN_USE_LAPACK and -llapack to compilation and link commands.\n");
  return 0;
}



inline int RNDecomposeSVD(const double *a, double *u, double *w, double *vt, int m, int n)
{
  // Print error message
  fprintf(stderr, "Cannot execute RNDecomposeSVD: LAPACK solver disabled during compile.\n");
  fprintf(stderr, "Enable it by adding -DRN_USE_LAPACK and -llapack to compilation and link commands.\n");
  return 0;
}



inline int RNDecomposeEigen(const double *a, double *eigenvalues, double *eigenvectors, int n)
{
  // Print error message
  fprintf(stderr, "Cannot execute RNDecomposeEigen: LAPACK solver disabled during compile.\n");
  fprintf(stderr, "Enable it by adding -DRN_USE_LAPACK and -llapack to compilation and link commands.\n");
  return 0;
}


#endif

