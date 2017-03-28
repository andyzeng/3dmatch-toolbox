// Source file for dense matrix class



// Include files

#include "RNMath.h"



RNDenseMatrix::
RNDenseMatrix(void)
  : values(NULL), nrows(0), ncols(0)
{
}



RNDenseMatrix::
RNDenseMatrix(int nrows, int ncols, RNScalar *values)
  : values(NULL), nrows(0), ncols(0)
{
  // Reset values
  Reset(nrows, ncols, values);
}



RNDenseMatrix::
RNDenseMatrix(const RNMatrix& matrix)
  : values(NULL), nrows(matrix.NRows()), ncols(matrix.NColumns())
{
  // Copy values
  if ((nrows > 0) && (ncols > 0)) {
    this->values = new RNScalar [ nrows * ncols ];
    for (int i = 0; i < nrows; i++) {
      for (int j = 0; j < ncols; j++) {
        SetValue(i, j, matrix.Value(i, j));
      }
    }
  }
}



RNDenseMatrix::
RNDenseMatrix(const RNDenseMatrix& matrix)
  : values(NULL), nrows(matrix.nrows), ncols(matrix.ncols) 
{
  // Copy values
  if ((nrows > 0) && (ncols > 0)) {
    this->values = new RNScalar [ nrows * ncols ];
    for (int i = 0; i < nrows * ncols; i++) {
      this->values[i] = matrix.values[i];
    }
  }
}



RNDenseMatrix::
~RNDenseMatrix(void)
{
  // Delete values
  if (values) delete [] values;
}



int RNDenseMatrix::
NRows(void) const
{
  // Return number of rows
  return nrows;
}



int RNDenseMatrix::
NColumns(void) const
{
  // Return number of columns
  return ncols;
}



RNScalar RNDenseMatrix::
Value(int i, int j) const
{
  // Return entry (i,j)
  return values[i*ncols+j];
}



void RNDenseMatrix::
SetValue(int i, int j, RNScalar value)
{
  // Set entry (i,j)
  values[i*ncols+j] = value;
}



RNBoolean RNDenseMatrix::
IsDense(void) const
{
  return TRUE;
}



RNBoolean RNDenseMatrix::
IsSparse(void) const
{
  return FALSE;
}



RNBoolean RNDenseMatrix::
IsZero(void) const
{
  // Return if all values are zero
  for (int i = 0; i < nrows * ncols; i++) 
    if (values[i] != 0) return FALSE;
  return TRUE;
}



RNBoolean RNDenseMatrix::
IsSymmetric(void) const
{
  // Return if matrix is symmetric
  if (nrows != ncols) return FALSE;
  for (int i = 0; i < nrows; i++) {
    for (int j = i+1; j < nrows; j++) {
      if (values[i*nrows+j] != values[j*nrows+i]) return FALSE;
    }
  }
  return TRUE;
}



RNScalar RNDenseMatrix::
Determinant(void) const
{
  // Compute determinant via LU decomposition
  RNDenseLUMatrix LU(*this);
  return LU.Determinant();
}



RNDenseMatrix RNDenseMatrix::
Transpose(void) const
{
  // Make array of transposed values
  RNScalar *transpose = new RNScalar [ nrows * ncols ];
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      transpose[j*nrows+i] = Value(i, j);
    }
  }

  // Return transpose of this matrix
  return RNDenseMatrix(ncols, nrows, transpose);
}



RNDenseMatrix RNDenseMatrix::
Inverse(void) const
{
  // Compute inverse via LU decomposition
  RNDenseLUMatrix LU(*this);
  return LU.Inverse();
}



RNDenseMatrix RNDenseMatrix::
Submatrix(int min_row, int max_row, int min_col, int max_col) const
{
  // Build submatrix
  RNDenseMatrix result(max_row - min_row + 1, max_col - min_col + 1);
  for (int i = 0; i < max_row - min_row + 1; i++) {
    for (int j = 0; j < max_col - min_col + 1; j++) {
      result.SetValue(i, j, Value(min_row+i, min_col+j));
    }
  }

  // Return submatrix
  return result;
}



RNBoolean RNDenseMatrix::
operator==(const RNDenseMatrix& matrix) const
{
  // Return if all values are same
  for (int i = 0; i < nrows * ncols; i++) 
    if (values[i] != matrix.values[i]) return FALSE;
  return TRUE;
}



RNBoolean RNDenseMatrix::
operator!=(const RNDenseMatrix& matrix) const
{
  return !(*this == matrix);
}



void RNDenseMatrix::
Negate(void)
{
  // Negate all values
  for (int i = 0; i < nrows * ncols; i++) 
    values[i] = -values[i];
}



void RNDenseMatrix::
Flip(void)
{
  // Replace this matrix with its transpose
  RNDenseMatrix copy(*this);
  int swap = nrows;
  nrows = ncols;
  ncols = swap;
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      values[i*nrows+j] = copy.Value(j, i);
    }
  }
}



void RNDenseMatrix::
Invert(void)
{
  // Compute inverse
  RNDenseLUMatrix LU(*this);
  *this = LU.Inverse();
}



void RNDenseMatrix::
Add(RNScalar a)
{
  // Add all values by a
  for (int i = 0; i < nrows * ncols; i++) 
    values[i] += a;
}



void RNDenseMatrix::
Subtract(RNScalar a)
{
  // Subtract all values by a
  for (int i = 0; i < nrows * ncols; i++) 
    values[i] -= a;
}



void RNDenseMatrix::
Divide(RNScalar a)
{
  // Divide all values by a
  if (a == 0) return;
  for (int i = 0; i < nrows * ncols; i++) 
    values[i] /= a;
}



void RNDenseMatrix::
Multiply(RNScalar a)
{
  // Multiply all values by a
  for (int i = 0; i < nrows * ncols; i++) 
    values[i] *= a;
}



void RNDenseMatrix::
Add(const RNDenseMatrix& matrix)
{
  // Add all values entry-by-entry
  for (int i = 0; i < nrows * ncols; i++) 
    values[i] += matrix.values[i];
}



void RNDenseMatrix::
Subtract(const RNDenseMatrix& matrix)
{
  // Subtract all values entry-by-entry
  for (int i = 0; i < nrows * ncols; i++) 
    values[i] -= matrix.values[i];
}



void RNDenseMatrix::
Multiply(const RNDenseMatrix& matrix)
{
  // Multiply by matrix
  *this = (*this) * matrix;
}



void RNDenseMatrix::
Reset(int nrows, int ncols, RNScalar *values)
{
  // Delete old values
  if (values) delete [] values;
  values = NULL;

  // Copy dimensions
  this->nrows = nrows;
  this->ncols = ncols;

  // Initialize values
  if ((nrows > 0) && (ncols > 0)) {
    this->values = new RNScalar [ nrows * ncols ];
    if (values) {
      for (int i = 0; i < nrows * ncols; i++) 
        this->values[i] = values[i];
    }
    else {
      for (int i = 0; i < nrows * ncols; i++) 
        this->values[i] = 0;
    }
  }
}



int RNDenseMatrix::
DecomposeSVD(RNDenseMatrix& U, RNVector& S, RNDenseMatrix& Vt) const
{
  // SVD
  U.Reset(NRows(), NRows());
  S.Reset(NRows());
  Vt.Reset(NRows(), NColumns());
  RNDecomposeSVD(values, U.values, S.values, Vt.values, nrows, ncols);

  // Return success
  return 1;
}



int RNDenseMatrix::
DecomposeLU(RNDenseMatrix& L, RNDenseMatrix& U) const
{
  // LU decomposition
  RNDenseLUMatrix LU(*this);
  return LU.DecomposeLU(L, U);
}



int RNDenseMatrix::
DecomposeEigen(RNVector& eigenvalues, RNDenseMatrix& eigenvectors)
{
  // Eigen decomposition
  assert(IsSymmetric());
  eigenvalues.Reset(NRows());
  eigenvectors.Reset(NRows(), NRows());
  return RNDecomposeEigen(values, eigenvalues.values, eigenvectors.values, nrows);
}



RNDenseMatrix& RNDenseMatrix::
operator=(const RNDenseMatrix& matrix)
{
  // Copy matrix
  if (values) delete [] values;
  nrows = matrix.NRows();
  ncols = matrix.NRows();
  values = new RNScalar [ nrows * ncols ];
  for (int i = 0; i < nrows * ncols; i++) {
    values[i] = matrix.values[i];
  }
  return *this;
}



RNDenseMatrix& RNDenseMatrix::
operator+=(const RNDenseMatrix& matrix)
{
  Add(matrix);
  return *this;
}



RNDenseMatrix& RNDenseMatrix::
operator-=(const RNDenseMatrix& matrix)
{
  Subtract(matrix);
  return *this;
}



RNDenseMatrix& RNDenseMatrix::
operator*=(const RNDenseMatrix& matrix)
{
  Multiply(matrix);
  return *this;
}



RNDenseMatrix& RNDenseMatrix::
operator*=(RNScalar a)
{
  Multiply(a);
  return *this;
}



RNDenseMatrix& RNDenseMatrix::
operator/=(RNScalar a)
{
  if (a != 0) Multiply(1/a);
  return *this;
}




RNDenseMatrix operator-(const RNDenseMatrix& matrix)
{
  RNDenseMatrix result(matrix);
  result.Negate();
  return result;
}



RNDenseMatrix operator+(const RNDenseMatrix& matrix1, const RNDenseMatrix& matrix2)
{
  RNDenseMatrix result(matrix1);
  result.Add(matrix2);
  return result;
}



RNDenseMatrix operator-(const RNDenseMatrix& matrix1, const RNDenseMatrix& matrix2)
{
  RNDenseMatrix result(matrix1);
  result.Subtract(matrix2);
  return result;
}



RNDenseMatrix operator*(RNScalar a, const RNDenseMatrix& matrix)
{
  RNDenseMatrix result(matrix);
  result.Multiply(a);
  return result;
}



RNDenseMatrix operator*(const RNDenseMatrix& matrix, RNScalar a)
{
  RNDenseMatrix result(matrix);
  result.Multiply(a);
  return result;
}



RNDenseMatrix operator*(const RNDenseMatrix& matrix1, const RNDenseMatrix& matrix2)
{
  // Multiply matrices 
  RNDenseMatrix result(matrix1.NRows(), matrix2.NColumns());
  for (int i = 0; i < result.NRows(); i++) {
    for (int j = 0; j < result.NColumns(); j++) {
      RNScalar value = 0.0;
      for (int k = 0; k < matrix1.NColumns(); k++) 
         value += matrix1[i][k] * matrix2[k][j];
      result.SetValue(i, j, value);
    }
  }
  return result;
}



RNDenseMatrix operator/(const RNDenseMatrix& matrix, RNScalar a)
{
  RNDenseMatrix result(matrix);
  if (a != 0) result.Multiply(1/a);
  return result;
}



RNVector operator*(const RNDenseMatrix& matrix, const RNVector& vector)
{
  // Multiply matrices 
  RNVector result(matrix.NRows());
  for (int i = 0; i < matrix.NRows(); i++) {
    RNScalar value = 0.0;
    for (int j = 0; j < matrix.NColumns(); j++) 
       value += matrix[i][j] * vector[j];
     result.SetValue(i, value);
  }
  return result;
}



int RNDenseMatrix::
ReadFile(const char *filename)
{
  // Parse input filename extension
  const char *extension;
  if (!(extension = strrchr(filename, '.'))) {
    printf("Filename %s has no extension (e.g., .ply)\n", filename);
    return 0;
  }

  // Read file of appropriate type
  int status = 0;
  if (!strncmp(extension, ".txt", 4)) 
    status = ReadASCIIFile(filename);
  else if (!strncmp(extension, ".matrix", 7)) 
    status = ReadSquareBinaryFile(filename);
  else {
    RNFail("Unable to read file %s (unrecognized extension: %s)\n", filename, extension);
    return 0;
  }

  // Return status
  return status;
}



int RNDenseMatrix::
WriteFile(const char *filename) const
{
  // Parse input filename extension
  const char *extension;
  if (!(extension = strrchr(filename, '.'))) {
    printf("Filename %s has no extension (e.g., .ply)\n", filename);
    return 0;
  }

  // Write file of appropriate type
  int status = 0;
  if (!strncmp(extension, ".txt", 4)) 
    status = WriteASCIIFile(filename);
  else if (!strncmp(extension, ".matrix", 7)) 
    status = WriteSquareBinaryFile(filename);
  else {
    RNFail("Unable to write file %s (unrecognized extension: %s)\n", filename, extension);
    return 0;
  }

  // Return status
  return status;
}



int RNDenseMatrix::
ReadASCIIFile(const char *filename)
{
  // Open file
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Unable to open %s\n", filename);
    return 0;
  }

  // Read number of rows and columns
  if (fscanf(fp, "%d%d", &nrows, &ncols) != (unsigned int) 2) {
    fprintf(stderr, "Unable to read header from %s\n", filename);
    return 0;
  }

  // Read values
  values = new RNScalar [ nrows * ncols ];
  for (int i = 0; i < nrows * ncols; i++) {
    if (fscanf(fp, "%lf", &values[i]) != (unsigned int) 1) {
      fprintf(stderr, "Unable to read values from %s\n", filename);
      return 0;
    }
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



int RNDenseMatrix::
WriteASCIIFile(const char *filename) const
{
  // Open file
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "Unable to open %s\n", filename);
    return 0;
  }

  // Write number of rows and columns
  fprintf(fp, "%d %d\n", nrows, ncols);

  // Write values
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) 
      fprintf(fp, "%g ", Value(i,j));
    fprintf(fp, "\n");
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



int RNDenseMatrix::
ReadSquareBinaryFile(const char *filename)
{
  // Open file
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open %s\n", filename);
    return 0;
  }

  // Read number of values
  float value;
  int nvalues = 0;
  while (fread(&value, sizeof(float), 1, fp) == (unsigned int) 1) {
    nvalues++;
  }

  // Rewind file
  fseek(fp, SEEK_SET, 0);

  // Determine number of rows and columns
  nrows = ncols = (int) (sqrt(nvalues) + 0.5);
  if (nrows * ncols != nvalues) {
    fprintf(stderr, "Unable to read binary matrix that is not square in %s\n", filename);
    return 0;
  }

  // Read values
  values = new RNScalar [ nrows * ncols ];
  for (int i = 0; i < nrows * ncols; i++) {
    float value;
    if (fread(&value, sizeof(float), 1, fp) != (unsigned int) 1) {
      fprintf(stderr, "Unable to read values from %s\n", filename);
      return 0;
    }
    values[i] = value;
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



int RNDenseMatrix::
WriteSquareBinaryFile(const char *filename) const
{
  // Open file
  FILE *fp = fopen(filename, "wb");
  if (!fp) {
    fprintf(stderr, "Unable to open %s\n", filename);
    return 0;
  }

  // Write values
  for (int i = 0; i < nrows * ncols; i++) {
    float value = values[i];
    if (fwrite(&value, sizeof(float), 1, fp) != (unsigned int) 1) {
      fprintf(stderr, "Unable to write values to %s\n", filename);
      return 0;
    }
  }

  // Close file
  fclose(fp);

  // Return success
  return 1;
}



