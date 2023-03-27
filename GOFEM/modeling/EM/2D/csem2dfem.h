#ifndef CSEM2DFEM_H
#define CSEM2DFEM_H

#include "em2dfem.h"

/*
 * Class that provides interface to implement
 * 2D CSEM solver for a given wavenumber
 */
class CSEM2DFEM: public EM2DFEM
{
public:
  CSEM2DFEM(MPI_Comm comm,
            const unsigned int order,
            const unsigned int mapping_order,
            const PhysicalModelPtr<2> &model,
            const BackgroundModel<2> &bg_model);
  CSEM2DFEM(MPI_Comm comm,
            const unsigned int order,
            const unsigned int mapping_order,
            const PhysicalModelPtr<2> &model);
  virtual ~CSEM2DFEM();

  void set_wavenumber(double kx);
  virtual void set_dipole_sources(const std::vector<DipoleSource> &sources);

  virtual unsigned n_data_at_point () const;

protected:
  void copy_local_to_global_system (const Assembly::CopyData::MaxwellSystem& data);

  unsigned get_number_of_constraint_matrices() const;
  void set_boundary_values ();

  // RHS vector assembly
  void assemble_problem_rhs ();

  // Builds rhs vector using secondary-field formulation and assuming homogeneous space of air
  void assemble_function_rhs_vector (const DipoleSource &phys_source,
                                     const AffineConstraints<double> &constraints,
                                     PETScWrappers::MPI::BlockVector& rhs_vector);

  virtual std::string data_header () const;

  cvector calculate_data_at_receiver (const std::vector<cvector>& E,
                                      const std::vector<cvector>& H) const;

  // File with digital filter stored
  std::string filter_file;
};

#endif // CSEM2DFEM_H

