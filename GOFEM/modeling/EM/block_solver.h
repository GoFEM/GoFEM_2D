#ifndef BLOCKSOLVER_H
#define BLOCKSOLVER_H

#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_block_vector.h>

using namespace dealii;

namespace LinearSolvers
{

/*
 * This class implements efficient block preconditioner for Maxwell
 * equations. It is designed specifically for the
 * real-valued formulation of the complex Maxwell's equations in
 * frequency domain. Therefore, it expects system matrix to be 2x2
 * block system (C -M; -M^T -C) with C being curl-curl and M conductivity
 * mass matrices, respectively.
 * Efficient preconditioner fot this system is the following matrix
 * P = ( C + M   0   )^-1
 *     (   0   C + M )
 * where product (C + M)^-1*v with arbitrary vector v can calculated
 * efficiently using e.g. hypre's AMS/AMG or LDL^T from MUMPS.
 *
 * For the extended system we have one more equation to satisfy, namely
 * a scalar laplace equation. This class recognizies if stabilized
 * formulation is used and solves then both.
 */
template <class PreconditionerVector, class PreconditionerScalar>
class BlockPreconditioner : public Subscriptor
{
public:
  BlockPreconditioner()
  {}

  BlockPreconditioner (const std::vector<std::shared_ptr<PETScWrappers::MPI::SparseMatrix>> &B,
                       const std::shared_ptr<PreconditionerVector> &preconditioner_curl,
                       const std::shared_ptr<PreconditionerScalar> &preconditioner_laplace):
    preconditioner_matrix(B),
    vector_preconditioner(preconditioner_curl),
    scalar_preconditioner(preconditioner_laplace)
  {}

  void vmult (PETScWrappers::MPI::BlockVector &dst, const PETScWrappers::MPI::BlockVector &src) const
  {
    vector_preconditioner->solve (*preconditioner_matrix[0], dst.block(0), src.block(0));
    vector_preconditioner->solve (*preconditioner_matrix[0], dst.block(1), src.block(1));

    if(scalar_preconditioner)
    {
      scalar_preconditioner->solve (*preconditioner_matrix[1], dst.block(2), src.block(2));
      scalar_preconditioner->solve (*preconditioner_matrix[1], dst.block(3), src.block(3));
    }
  }

private:
  std::vector<std::shared_ptr<PETScWrappers::MPI::SparseMatrix>> preconditioner_matrix; // B = (C + M)
  // initialized preconditioner that can efficiently solve problems of type Bx = b
  std::shared_ptr<PreconditionerVector> vector_preconditioner;
  std::shared_ptr<PreconditionerScalar> scalar_preconditioner;
};

}

#endif // BLOCKSOLVER_H
