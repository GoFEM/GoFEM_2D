#ifndef EM2DFEM_H
#define EM2DFEM_H

#include "../emfem.h"
#include "common/sparse_direct.h"

/*
 * Abstract class that provides interface to implement
 * specific 2D time-harmonic Maxwell solver
 * Uses i*omega*t convention
 */
class EM2DFEM: public EMFEM<2>
{
public:
  EM2DFEM(MPI_Comm comm,
          const unsigned int order,
          const unsigned int mapping_order,
          const PhysicalModelPtr<2> &model,
          const BackgroundModel<2> &bg_model);
  EM2DFEM(MPI_Comm comm,
          const unsigned int order,
          const unsigned int mapping_order,
          const PhysicalModelPtr<2> &model);
  virtual ~EM2DFEM();

  // This callback is called after solution for a wavenumber was calculated
  void set_adjoint_solver_callback(std::function<void()> &callback);

  void run();

protected:
  void clear();

  // System matrix setup and assembly
  void assemble_system_matrix ();
  void local_assemble_system (const DoFHandler<2>::active_cell_iterator& cell,
                              Assembly::Scratch::MaxwellSystem<2>& scratch,
                              Assembly::CopyData::MaxwellSystem& data);
  void setup_system (const unsigned n_rhs);

  // RHS vector assembly
  virtual void assemble_dual_rhs_vector (const std::vector<Point<2>>& delta_positions,
                                         const AffineConstraints<double> &constraints,
                                         PETScWrappers::MPI::BlockVector& rhs_vector);

  // Constructs RHS for dipole source.
  void assemble_dipole_rhs_vector(const DipoleSource &phys_source,
                                  const AffineConstraints<double> &constraints,
                                  PETScWrappers::MPI::BlockVector &rhs_vector);

  /*
   * This method estimates numerical error of the solution for every cell.
   */
  void estimate_error ();

  void setup_preconditioner ();

  void solve (std::vector<PETScWrappers::MPI::BlockVector> &solution_vectors,
              std::vector<PETScWrappers::MPI::BlockVector>& rhs_vectors,
              const std::vector<unsigned> &constraints_indices,
              bool adjoint = false, bool verbose = true, unsigned start_index = 0);

  // Solves system using direct solver via PETSc interface
  void solve_using_direct_solver (std::vector<PETScWrappers::MPI::BlockVector>& solution_vectors,
                                  std::vector<PETScWrappers::MPI::BlockVector>& rhs_vectors,
                                  const std::vector<unsigned> &constraints_indices,
                                  bool adjoint, bool verbose, unsigned start_index = 0);
  // Solves system using GMRES preconditioned with CG and AMG
  void solve_using_cgamg_solver (std::vector<PETScWrappers::MPI::BlockVector>& solution_vectors,
                                 std::vector<PETScWrappers::MPI::BlockVector>& rhs_vectors,
                                 const std::vector<unsigned> &constraints_indices,
                                 bool adjoint, bool verbose, unsigned start_index = 0);
  // Solves system using my wrapper for the direct solver MUMPS
  void solve_using_mumps_solver (std::vector<PETScWrappers::MPI::BlockVector>& solution_vectors,
                                 std::vector<PETScWrappers::MPI::BlockVector>& rhs_vectors,
                                 const std::vector<unsigned> &constraints_indices,
                                 bool adjoint, bool verbose, unsigned start_index = 0);

  /*
   * Returns electric and magnetic field at some point for all sources
   * E[0] - Ex src1; E[1] - Ex src2; ... E[n-1] - Ex srcn;
   * E[n] - Ey src1; E[n+1] - Ey src2; ... E[2*n-1] - Ey srcn;
   * E[2*n] - Ez src1; E[2n+1] - Ez src2; ... E[3*n-1] - Ez srcn;
   * and the same for H field.
   * NOTE: point p is given in real coordinates and not local cell coordinates.
   * NOTE2: for CSEM modeling returns fields in wavedomain
   */
  virtual void field_at_point (const DoFHandler<2>::active_cell_iterator& cell,
                               const Point<2>& p, cvector& E, cvector& H) const;

  double get_wavenumber() const;

  void output_specific_information(std::vector<std::shared_ptr<DataPostprocessor<2> > > &data,
                                   DataOut<2> &data_out) const;

  SparseDirectMUMPS solver_mumps;

  double wavenumber;
  Tensor<2, 2> R; // rotation matrix

  std::function<void()> run_adjoint_solver;
};

#endif // EM2DFEM_H

