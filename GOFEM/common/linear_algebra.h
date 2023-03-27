#ifndef LINALG_H
#define LINALG_H

#include <slepc.h>

/*
 * Calculates partial SVD of the matrix (using
 * SLEPc) and returns constructed context.
 */
SVD calculate_partial_svd (Mat A, unsigned subspace_dim, MPI_Comm communicator, unsigned maxit = 8, double tolerance = 1e-6);

/*
 * Calculates partial EVD of the matrix (using
 * SLEPc) and returns constructed context.
 */
EPS calculate_partial_evd (Mat A, unsigned subspace_dim, MPI_Comm communicator, unsigned maxit = 8, double tolerance = 1e-6);

double estimate_largest_singular_value (Mat A, MPI_Comm communicator, unsigned maxit = 8, double tolerance = 1e-6);

/*
 * Augments two vectors such that c = [a; b].
 * All vectors should be preallocated.
 */
Vec augment_vectors(Vec a, Vec b, MPI_Comm communicator);
void augment_vectors(Vec a, Vec b, Vec out);

/*
 * Augments two matrices such that C = [A; B].
 * Matrix C is just shell matrix implementing
 * matvec products with the original A and B.
 */
Mat augment_matrices (Mat A, Mat B, PetscReal scaling, MPI_Comm communicator);

/*
 * Create matrix shell object which given A implements A*A^T
 * matrix.
 */
Mat create_matmattranspose_object (Mat A, MPI_Comm communicator);

/*
 * Create matrix shell object which given A implements A^T*A
 * matrix.
 */
Mat create_mattransposemat_object (Mat A, MPI_Comm communicator);

struct CompositeMatrixData
{
  Vec vec_rows, vec_cols;
  Mat A, B;
  MPI_Comm communicator;
  PetscReal scaling_factor;
};

PetscErrorCode calculate_composite_op_action(Mat A, Vec x, Vec y);
PetscErrorCode calculate_composite_op_transposed_action(Mat A, Vec x, Vec y);

PetscErrorCode calculate_matmattr_action(Mat A, Vec x, Vec y);
PetscErrorCode calculate_matmattr_transposed_action(Mat A, Vec x, Vec y);

PetscErrorCode calculate_mattrmat_action(Mat A, Vec x, Vec y);
PetscErrorCode calculate_mattrmat_transposed_action(Mat A, Vec x, Vec y);

PetscErrorCode calculate_hessian_action(Mat H, Vec x, Vec y);
PetscErrorCode calculate_hessian_transposed_action(Mat H, Vec x, Vec y);

#endif // LINALG_H
