#include "linear_algebra.h"

#include <stdexcept>
#include <iostream>

SVD calculate_partial_svd(Mat A, unsigned subspace_dim, MPI_Comm communicator, unsigned maxit, double tolerance)
{
  SVD svd;
  PetscErrorCode ierr;
  // Create singular value solver context and solve
  ierr = SVDCreate(communicator, &svd);
  ierr = SVDSetOperator(svd, A);
  ierr = SVDSetWhichSingularTriplets(svd, SVD_LARGEST);
  ierr = SVDSetDimensions(svd, subspace_dim, PETSC_DEFAULT, PETSC_DEFAULT);
  ierr = SVDSetTolerances(svd, tolerance, maxit);
  ierr = SVDSetType(svd, SVDLANCZOS);
  ierr = SVDSetFromOptions(svd);
  ierr = SVDSolve(svd);

  if(ierr != 0)
    throw std::runtime_error("Partial SVD calculations failed.");

  return svd;
}

EPS calculate_partial_evd(Mat A, unsigned subspace_dim, MPI_Comm communicator, unsigned maxit, double tolerance)
{
  EPS eps;
  PetscErrorCode ierr;
  // Create eigenvalue solver context and solve
  ierr = EPSCreate(communicator,&eps);
  ierr = EPSSetOperators(eps, A, NULL);
  ierr = EPSSetProblemType(eps, EPS_HEP);
  ierr = EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE);
  ierr = EPSSetDimensions(eps, subspace_dim, PETSC_DEFAULT, PETSC_DEFAULT);
  ierr = EPSSetTolerances(eps, tolerance, maxit);
  ierr = EPSSetType(eps, EPSKRYLOVSCHUR);
  ierr = EPSSetFromOptions(eps);
  ierr = EPSSolve(eps);

  if(ierr != 0)
    throw std::runtime_error("Partial eigenvalue calculations failed.");

  return eps;
}

double estimate_largest_singular_value(Mat A, MPI_Comm communicator, unsigned maxit, double tolerance)
{
  SVD svd;
  PetscErrorCode ierr;
  // Create singular value solver context
  ierr = SVDCreate(communicator, &svd);
  ierr = SVDSetOperator(svd, A);
  ierr = SVDSetWhichSingularTriplets(svd, SVD_LARGEST);
  ierr = SVDSetDimensions(svd, 1, 3, PETSC_DEFAULT);
  ierr = SVDSetTolerances(svd, tolerance, maxit);
  ierr = SVDSetType(svd, SVDLANCZOS);
  ierr = SVDSetFromOptions(svd);
  ierr = SVDSolve(svd);

  if(ierr != 0)
    throw std::runtime_error("Partial SVD calculations failed.");

  PetscInt nconv;
  SVDGetConverged(svd, &nconv);

//  if(nconv == 0)
//    throw std::runtime_error("SVD did not converged, try changing tolerances and/or max # of iterations.");

  double sigma;
  SVDGetSingularTriplet (svd, 0, &sigma, PETSC_NULL, PETSC_NULL);
  SVDDestroy (&svd);

  return sigma;
}

Vec augment_vectors(Vec a, Vec b, MPI_Comm communicator)
{
  PetscInt size_a, size_b;
  VecGetSize(a, &size_a);
  VecGetSize(b, &size_b);

  Vec c;
  VecCreateMPI (communicator, PETSC_DECIDE, size_a + size_b, &c);
  augment_vectors(a, b, c);

  return c;
}

void augment_vectors(Vec a, Vec b, Vec out)
{
  PetscInt size_a;
  VecGetSize(a, &size_a);

  PetscInt istart, iend;
  VecGetOwnershipRange (a, &istart, &iend);

  PetscScalar* valarray;
  VecGetArray(a, &valarray);

  PetscInt idx = 0;
  for(PetscInt i = istart; i < iend; ++i)
    VecSetValue(out, i, valarray[idx++], INSERT_VALUES);

  VecRestoreArray (a, &valarray);

  VecGetOwnershipRange (b, &istart, &iend);
  VecGetArray (b, &valarray);
  idx = 0;
  for(PetscInt i = istart; i < iend; ++i)
    VecSetValue(out, i + size_a, valarray[idx++], INSERT_VALUES);

  VecRestoreArray (b, &valarray);

  VecAssemblyBegin(out);
  VecAssemblyEnd(out);
}

Mat augment_matrices(Mat A, Mat B, PetscReal scaling, MPI_Comm communicator)
{
  Mat C;

  PetscInt local_cols, cols, rowsA, rowsB;
  MatGetLocalSize(B, PETSC_NULL, &local_cols);
  MatGetSize(A, &rowsA, &cols);
  MatGetSize(B, &rowsB, PETSC_NULL);

  CompositeMatrixData* matrix_data = new CompositeMatrixData;
  MatCreateVecs(A, &matrix_data->vec_cols, &matrix_data->vec_rows);
  matrix_data->A = A;
  matrix_data->B = B;
  matrix_data->scaling_factor = scaling;
  matrix_data->communicator = communicator;

  MatCreateShell (communicator, PETSC_DECIDE, local_cols, rowsA + rowsB, cols, (void*)matrix_data, &C);
  MatShellSetOperation (C, MATOP_MULT, (void(*)(void))calculate_composite_op_action);
  MatShellSetOperation (C, MATOP_MULT_TRANSPOSE, (void(*)(void))calculate_composite_op_transposed_action);

  return C;
}

PetscErrorCode calculate_composite_op_action(Mat A, Vec x, Vec y)
{
  PetscErrorCode ierr;
  CompositeMatrixData* matrix_data;
  ierr = MatShellGetContext (A, &matrix_data);

  ierr = MatMult(matrix_data->A, x, matrix_data->vec_rows);
  ierr = MatMult(matrix_data->B, x, matrix_data->vec_cols);
  VecScale(matrix_data->vec_cols, matrix_data->scaling_factor);

  augment_vectors(matrix_data->vec_rows, matrix_data->vec_cols, y);

  return ierr;
}

PetscErrorCode calculate_composite_op_transposed_action(Mat A, Vec x, Vec y)
{
  PetscErrorCode ierr;
  CompositeMatrixData* matrix_data;
  ierr = MatShellGetContext (A, &matrix_data);

  PetscInt rows;
  MatGetSize (matrix_data->A, &rows, PETSC_NULL);

  // Separate input vector into data and model dependent parts
  PetscInt istart, iend;
  VecGetOwnershipRange (x, &istart, &iend);

  PetscScalar* valarray;
  VecGetArray (x, &valarray);

  PetscInt idx = 0;
  for(PetscInt i = istart; i < iend; ++i, ++idx)
  {
    if(i < rows)
      VecSetValue (matrix_data->vec_rows, i, valarray[idx], INSERT_VALUES);
    else
      VecSetValue (matrix_data->vec_cols, i - rows, valarray[idx], INSERT_VALUES);
  }

  VecRestoreArray (x, &valarray);

  VecAssemblyBegin(matrix_data->vec_rows);
  VecAssemblyEnd(matrix_data->vec_rows);
  VecAssemblyBegin(matrix_data->vec_cols);
  VecAssemblyEnd(matrix_data->vec_cols);

  Vec v;
  VecDuplicate(matrix_data->vec_cols, &v);

  ierr = MatMultTranspose (matrix_data->A, matrix_data->vec_rows, v);
  VecScale(matrix_data->vec_cols, matrix_data->scaling_factor);
  ierr = MatMultTransposeAdd (matrix_data->B, matrix_data->vec_cols, v, y);

  VecDestroy (&v);

  return ierr;
}

Mat create_matmattranspose_object (Mat A, MPI_Comm communicator)
{
  Mat C;

  PetscInt rows, cols, local_rows, local_cols;
  MatGetSize(A, &rows, &cols);
  MatGetLocalSize(A, &local_rows, &local_cols);

  CompositeMatrixData* matrix_data = new CompositeMatrixData;
  MatCreateVecs(A, &matrix_data->vec_cols, PETSC_NULL);
  matrix_data->A = A;

  MatCreateShell (communicator, local_rows, local_rows, rows, rows, (void*)matrix_data, &C);
  MatShellSetOperation (C, MATOP_MULT, (void(*)(void))calculate_matmattr_action);
  MatShellSetOperation (C, MATOP_MULT_TRANSPOSE, (void(*)(void))calculate_matmattr_transposed_action);

  MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY);

  return C;
}

Mat create_mattransposemat_object (Mat A, MPI_Comm communicator)
{
  Mat C;

  PetscInt rows, cols, local_rows, local_cols;
  MatGetSize(A, &rows, &cols);
  MatGetLocalSize(A, &local_rows, &local_cols);

  CompositeMatrixData* matrix_data = new CompositeMatrixData;
  MatCreateVecs(A, PETSC_NULL, &matrix_data->vec_rows);
  matrix_data->A = A;

  MatCreateShell (communicator, local_cols, local_cols, cols, cols, (void*)matrix_data, &C);
  MatShellSetOperation (C, MATOP_MULT, (void(*)(void))calculate_mattrmat_action);
  MatShellSetOperation (C, MATOP_MULT_TRANSPOSE, (void(*)(void))calculate_mattrmat_transposed_action);

  MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY);

  return C;
}

PetscErrorCode calculate_matmattr_action(Mat A, Vec x, Vec y)
{
  PetscErrorCode ierr;

  CompositeMatrixData* matrix_data;
  ierr = MatShellGetContext (A, &matrix_data);

  ierr = MatMultTranspose (matrix_data->A, x, matrix_data->vec_cols);
  ierr = MatMult (matrix_data->A, matrix_data->vec_cols, y);

  return ierr;
}

PetscErrorCode calculate_matmattr_transposed_action(Mat A, Vec x, Vec y)
{
  return calculate_matmattr_action(A, x, y);
}

PetscErrorCode calculate_mattrmat_action(Mat A, Vec x, Vec y)
{
  PetscErrorCode ierr;

  CompositeMatrixData* matrix_data;
  ierr = MatShellGetContext (A, &matrix_data);

  ierr = MatMult (matrix_data->A, x, matrix_data->vec_rows);
  ierr = MatMultTranspose (matrix_data->A, matrix_data->vec_rows, y);

  return ierr;
}

PetscErrorCode calculate_mattrmat_transposed_action(Mat A, Vec x, Vec y)
{
  return calculate_mattrmat_action(A, x, y);
}

PetscErrorCode calculate_hessian_action(Mat H, Vec x, Vec y)
{
  PetscErrorCode ierr;

  CompositeMatrixData* matrix_data;
  ierr = MatShellGetContext (H, &matrix_data);

  ierr = MatMult(matrix_data->A, x, matrix_data->vec_rows);
  ierr = MatMultTranspose(matrix_data->A, matrix_data->vec_rows, y);
  if(matrix_data->B != NULL)
  {
    ierr = MatMult(matrix_data->B, x, matrix_data->vec_cols);
    ierr = VecAXPY(y, matrix_data->scaling_factor, matrix_data->vec_cols);
  }

  return ierr;
}

PetscErrorCode calculate_hessian_transposed_action (Mat H, Vec x, Vec y)
{
  return calculate_hessian_action(H, x, y);
}
