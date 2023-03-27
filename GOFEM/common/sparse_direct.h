#ifndef SPARSE_DIRECT_H
#define SPARSE_DIRECT_H

#include <vector>

#include <deal.II/lac/vector.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>

using namespace dealii;

#ifdef USE_MUMPS
#  include <dmumps_c.h>
#endif

/**
 * This class provides a serial interface to the sparse direct solver <a
 * href="http://mumps.enseeiht.fr">MUMPS</a>. MUMPS is direct method based on
 * a multifrontal approach, which performs a direct LU or LDLt factorizations. 
 * The matrix coming in may have either symmetric or nonsymmetric sparsity
 * pattern.
 *
 * @note This class is useable if and only if a working installation of <a
 * href="http://mumps.enseeiht.fr">MUMPS</a> exists on your system and was
 * detected during configuration of <code>deal.II</code>.
 */
class SparseDirectMUMPS
{
private:

#ifdef USE_MUMPS
  DMUMPS_STRUC_C id;
#endif

  std::vector<double> rhs;

  /**
   * Representation of a matrix
   */
  double *data;
  int *col, *row;
  int n, nz;

  /**
   * extracted local matrix
   */
  Mat *submatrix;

  /**
   * Representation of sparse right-hand side vectors
   */
  std::vector<double> rhs_sparse;
  std::vector<int> irhs_ptr, irhs_sparse;

  /**
   * This function initializes a MUMPS instance and hands over the system's
   * matrix <tt>matrix</tt>.
   */
  void initialize_matrix (const PETScWrappers::MPI::SparseMatrix &matrix);

  /**
   * Extract sparsity pattern and data arrays in MUMPS matrix format
   */
  void extract_matrix_data(const PETScWrappers::MPI::SparseMatrix &matrix);

  /**
   * Copy the computed solution into the solution vector.
   */
  void copy_solution (PETScWrappers::MPI::Vector &vector);

  void copy_solutions (std::vector<Vec> &vectors);

  /**
   * Copy right-hand side to the MUMPS structure.
   */
  void copy_rhs_to_mumps(const PETScWrappers::MPI::Vector &rhs);

  /**
   * Solve for multiple vectors
   */
  void vmult(std::vector<Vec> &dst, const std::vector<Vec> &src,
             unsigned nnz_per_rhs, bool transposed);

  /**
   * Flag stores whether the function initialize() has already
   * been called.
   */
  bool initialize_called;

  /**
   * If true uses LDLt, otherwise LU.
   */
  bool symmetric_mode;

  int convert_to_triples_seqaij_seqsbaij(Mat A, int shift, int *nnz, int **r, int **c, PetscScalar **v);
  int convert_to_triples_seqaij_seqaij(Mat A,int shift,int *nnz,int **r, int **c, PetscScalar **v);

public:

  /**
   * Constructor
   */
  SparseDirectMUMPS ();

  /**
   * Destructor
   */
  ~SparseDirectMUMPS ();

  /**
   * This function initializes a MUMPS instance and computes the factorization
   * of the system's matrix.
   */
  void initialize (const PETScWrappers::MPI::SparseMatrix &matrix);

  /**
   * A function in which the inverse of the matrix is applied to the input
   * vector src and the solution is written in the output vector dst.
   */
  void vmult (PETScWrappers::MPI::Vector &dst, const PETScWrappers::MPI::Vector &src, bool transposed);

  /**
   * Exploit sparsity structure of the right-hand sides when solving
   */
  void vmult_bunch(std::vector<PETScWrappers::MPI::BlockVector> &dst,
                   const std::vector<PETScWrappers::MPI::BlockVector> &src,
                   unsigned nnz_per_rhs, bool transposed);

  void vmult_bunch(std::vector<PETScWrappers::MPI::Vector> &dst,
                   const std::vector<PETScWrappers::MPI::Vector> &src,
                   unsigned nnz_per_rhs, bool transposed);

  /**
   * Switch to LDLt if matrix is symmetric
   */
  void set_symmetric_mode(bool f);

  /**
   * Clear all internal structures and call destroy routine of MUMPS
   */
  void clear();

  /**
   * Checks if the object has been initialized
   */
  bool is_initialized() const;
};

#endif
