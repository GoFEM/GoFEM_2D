#include "sparse_direct.h"

#ifdef USE_MUMPS

#include <numeric>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include "/home/ag/lib/petsc-3.18.1/src/mat/impls/aij/seq/aij.h"
#include "petscvec.h"
#include "petscmat.h"

SparseDirectMUMPS::SparseDirectMUMPS ()
  : initialize_called (false), symmetric_mode(false)
{}

SparseDirectMUMPS::~SparseDirectMUMPS ()
{
  clear();
}

void SparseDirectMUMPS::initialize_matrix (const PETScWrappers::MPI::SparseMatrix &matrix)
{
  if(matrix.n() != matrix.m())
    throw std::runtime_error("Matrix needs to be square.");

  // Check we haven't been here before:
  if (initialize_called == true)
    throw std::runtime_error("Cannot call initialize two times.");

  // Initialize MUMPS instance:
  id.job = -1;
  id.par =  1;
  id.sym = (symmetric_mode ? 2 : 0);
  id.comm_fortran = MPI_Comm_c2f(MPI_COMM_SELF);
  dmumps_c (&id);

  // Extract data from the matrix to the MUMPS structures
  extract_matrix_data(matrix);

  id.n   = n;
  id.nz  = nz;
  id.irn = row;
  id.jcn = col;
  id.a   = data;

  // Output control
  id.icntl[3] = 0;

  // Ordering: use AMF
  id.icntl[6] = 2;

  // Exit by setting this flag:
  initialize_called = true;
}

void SparseDirectMUMPS::extract_matrix_data(const PETScWrappers::MPI::SparseMatrix &matrix)
{
  n = matrix.n();

  std::vector<PetscInt> indices(n);
  std::iota(indices.begin(), indices.end(), 0);

  IS is;
  ISCreateGeneral(PETSC_COMM_SELF, n, indices.data(), PETSC_COPY_VALUES, &is);

  MatCreateSubMatrices((Mat)matrix, 1, &is, &is, MAT_INITIAL_MATRIX, &submatrix);

  ISDestroy(&is);

  if(symmetric_mode)
    convert_to_triples_seqaij_seqsbaij(*submatrix, 1, &nz, &row, &col, &data);
  else
    convert_to_triples_seqaij_seqaij(*submatrix, 1, &nz, &row, &col, &data);
}

void SparseDirectMUMPS::copy_rhs_to_mumps (const PETScWrappers::MPI::Vector &new_rhs)
{
  Assert(unsigned(n) == new_rhs.size(), ExcMessage("Matrix size and rhs length must be equal."));

  PetscScalar *data;
  VecGetArray((Vec)new_rhs,&data);

  memcpy(rhs.data(), data, rhs.size() * sizeof(PetscScalar));
  id.rhs = &rhs[0];

  VecRestoreArray((Vec)new_rhs,&data);
}

void SparseDirectMUMPS::copy_solution (PETScWrappers::MPI::Vector &vector)
{
  Assert(unsigned(n) == vector.size(), ExcMessage("Matrix size and solution vector length must be equal."));
  Assert(unsigned(n) == rhs.size(), ExcMessage("Class not initialized with a rhs vector."));

  // Copy solution into the given vector
  PetscScalar *data;
  VecGetArray((Vec)vector,&data);

  memcpy(data, rhs.data(), rhs.size() * sizeof(PetscScalar));

  VecRestoreArray((Vec)vector,&data);
}

void SparseDirectMUMPS::copy_solutions (std::vector<Vec> &vectors)
{
  // Copy solution into the given vector
  for (size_t i = 0; i < vectors.size(); ++i)
  {
    PetscScalar *data;
    VecGetArray(vectors[i],&data);
    memcpy(data, rhs.data() + i*n, n * sizeof(PetscScalar));
    VecRestoreArray(vectors[i],&data);
  }
}

void SparseDirectMUMPS::initialize (const PETScWrappers::MPI::SparseMatrix &matrix)
{
  if(initialize_called)
    throw std::runtime_error("The object has already been initialized. Clear first.");

  // Initialize MUMPS instance:
  initialize_matrix (matrix);
  // Start factorization
  id.job = 4;
  dmumps_c (&id);
}

void SparseDirectMUMPS::vmult (PETScWrappers::MPI::Vector &dst, const PETScWrappers::MPI::Vector &src, bool transposed)
{
  // Check that the solver has been initialized by the routine above:
  Assert (initialize_called == true, ExcNotInitialized());

  // and that the matrix has at least one nonzero element:
  Assert (nz != 0, ExcNotInitialized());

  Assert(unsigned(n) == dst.size(), ExcMessage("Destination vector has the wrong size."));
  Assert(unsigned(n) == src.size(), ExcMessage("Source vector has the wrong size."));

  rhs.resize(n);

  // Hand over right-hand side
  copy_rhs_to_mumps(src);

  // Start solver
  id.icntl[19] = 0;
  id.icntl[8] = transposed ? 2 : 1;
  id.job = 3;
  dmumps_c (&id);
  copy_solution (dst);
}

void SparseDirectMUMPS::vmult(std::vector<Vec> &dst, const std::vector<Vec> &src,
                              unsigned nnz_per_rhs, bool transposed)
{
  rhs.resize(dst.size() * n, 0.);
  irhs_ptr.resize(dst.size() + 1, 0);
  rhs_sparse.clear();
  rhs_sparse.reserve(nnz_per_rhs * dst.size() * 2);
  irhs_sparse.clear();
  irhs_sparse.reserve(nnz_per_rhs * dst.size() * 2);

  unsigned nnz = 0;
  for(unsigned rhs_idx = 0; rhs_idx < src.size(); ++rhs_idx)
  {
    irhs_ptr[rhs_idx] = nnz + 1;
    PetscScalar *data;
    VecGetArray(src[rhs_idx],&data);

    for(int i = 0; i < n; ++i)
      if(fabs(data[i]) > 10.*std::numeric_limits<double>::min())
      {
        ++nnz;
        rhs_sparse.push_back(data[i]);
        irhs_sparse.push_back(i + 1);
      }

    VecRestoreArray(src[rhs_idx],&data);
  }

  irhs_ptr[dst.size()] = nnz + 1;

  // Set mumps parameters
  id.icntl[19] = 1;
  id.icntl[8] = transposed ? 2 : 1;

  id.nrhs = src.size();
  id.rhs = rhs.data();
  id.lrhs = n;
  id.rhs_sparse = rhs_sparse.data();
  id.irhs_sparse = irhs_sparse.data();
  id.irhs_ptr = irhs_ptr.data();
  id.nz_rhs = nnz;

  id.job = 3;
  dmumps_c (&id);

  copy_solutions (dst);
}

int SparseDirectMUMPS::convert_to_triples_seqaij_seqsbaij(Mat A, int shift,
                                                          int *nnz, int **r, int **c,
                                                          PetscScalar **v)
{
  const PetscInt    *ai,*aj,*ajj,*adiag,M=A->rmap->n;
  PetscInt          nz,rnz,i,j;
  PetscScalar *av,*v1;
  PetscScalar       *val;
  PetscErrorCode    ierr;
  PetscInt          *row,*col;
  Mat_SeqAIJ        *aa=(Mat_SeqAIJ*)A->data;

  MatSeqAIJGetArray(A,&av);
  ai = aa->i; aj=aa->j;
  adiag=aa->diag;
  {
    /* count nz in the uppper triangular part of A */
    nz = 0;
    for (i=0; i<M; i++)
      nz += ai[i+1] - adiag[i];
    *nnz = nz;

    ierr = PetscMalloc((2*nz*sizeof(PetscInt)+nz*sizeof(PetscScalar)), &row);CHKERRQ(ierr);
    col  = row + nz;
    val  = (PetscScalar*)(col + nz);

    nz = 0;
    for (i=0; i<M; i++)
    {
      rnz = ai[i+1] - adiag[i];
      ajj = aj + adiag[i];
      v1  = av + adiag[i];
      for (j=0; j<rnz; j++)
      {
        row[nz] = i+shift;
        col[nz] = ajj[j] + shift;
        val[nz++] = v1[j];
      }
    }
    *r = row; *c = col; *v = val;
  }

  MatSeqAIJRestoreArray(A,&av);

  return 0;
}

int SparseDirectMUMPS::convert_to_triples_seqaij_seqaij(Mat A, int shift, int *nnz,
                                                        int **r, int **c, PetscScalar **v)
{
  const PetscInt *ai,*aj,*ajj,M=A->rmap->n;
  PetscInt       nz,rnz,i,j;
  PetscErrorCode ierr;
  PetscInt       *row,*col;
  PetscScalar    *av;
  Mat_SeqAIJ     *aa=(Mat_SeqAIJ*)A->data;

  MatSeqAIJGetArray(A,&av);
  *v   = (PetscScalar*)av;

  nz   = aa->nz;
  ai   = aa->i;
  aj   = aa->j;
  *nnz = nz;
  ierr = PetscMalloc1(2*nz, &row);CHKERRQ(ierr);
  col  = row + nz;

  nz = 0;
  for (i=0; i<M; i++)
  {
    rnz = ai[i+1] - ai[i];
    ajj = aj + ai[i];
    for (j=0; j<rnz; j++)
    {
      row[nz] = i+shift;
      col[nz++] = ajj[j] + shift;
    }
  }
  *r = row; *c = col;

  MatSeqAIJRestoreArray(A,&av);

  return 0;
}

void SparseDirectMUMPS::vmult_bunch(std::vector<PETScWrappers::MPI::BlockVector> &dst,
                                    const std::vector<PETScWrappers::MPI::BlockVector> &src,
                                    unsigned nnz_per_rhs, bool transposed)
{
  Assert(unsigned(n) == dst[0].size(), ExcDimensionMismatch(n, dst[0].size()));
  Assert(unsigned(n) == src[0].size(), ExcDimensionMismatch(n, src[0].size()));

  std::vector<Vec> dst_vecs(dst.size()), src_vecs(src.size());
  for(unsigned rhs_idx = 0; rhs_idx < src.size(); ++rhs_idx)
  {
    dst_vecs[rhs_idx] = (Vec)dst[rhs_idx].block(0);
    src_vecs[rhs_idx] = (Vec)src[rhs_idx].block(0);
  }

  vmult(dst_vecs, src_vecs, nnz_per_rhs, transposed);
}

void SparseDirectMUMPS::vmult_bunch(std::vector<PETScWrappers::MPI::Vector> &dst,
                                    const std::vector<PETScWrappers::MPI::Vector> &src,
                                    unsigned nnz_per_rhs, bool transposed)
{
  Assert(unsigned(n) == dst[0].size(), ExcDimensionMismatch(n, dst[0].size()));
  Assert(unsigned(n) == src[0].size(), ExcDimensionMismatch(n, src[0].size()));

  std::vector<Vec> dst_vecs(dst.size()), src_vecs(src.size());
  for(unsigned rhs_idx = 0; rhs_idx < src.size(); ++rhs_idx)
  {
    dst_vecs[rhs_idx] = (Vec)dst[rhs_idx];
    src_vecs[rhs_idx] = (Vec)src[rhs_idx];
  }

  vmult(dst_vecs, src_vecs, nnz_per_rhs, transposed);
}

void SparseDirectMUMPS::set_symmetric_mode(bool f)
{
  symmetric_mode = f;
}

void SparseDirectMUMPS::clear()
{
  if(initialize_called)
  {
    id.job = -2;
    dmumps_c (&id);

    MatDestroyMatrices(1, &submatrix);

    PetscFree(row);
    row = nullptr;
    col = nullptr;
    data = nullptr;
    rhs.clear();
    irhs_ptr.clear();
    rhs_sparse.clear();
    irhs_sparse.clear();

    initialize_called = false;
  }
}

bool SparseDirectMUMPS::is_initialized() const
{
  return initialize_called;
}

#endif // USE_MUMPS

