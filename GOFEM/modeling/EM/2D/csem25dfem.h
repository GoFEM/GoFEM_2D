#ifndef CSEM25DFEM_H
#define CSEM25DFEM_H

#include <functional>

#include "csem2dfem.h"
#include "common/sincos_transform.h"

/*
 * Class that implements 2.5D CSEM solver
 */
class CSEM25DFEM: public CSEM2DFEM
{
public:
  CSEM25DFEM (MPI_Comm comm,
              const unsigned int fe_order,
              const unsigned int mapping_order,
              const PhysicalModelPtr<2> &model);
  CSEM25DFEM (MPI_Comm comm,
              const unsigned int fe_order,
              const unsigned int mapping_order,
              const PhysicalModelPtr<2> &model,
              const BackgroundModel<2> &bg_model);
  virtual ~CSEM25DFEM () {}

  void run();
  void clear();

  // Since we solve 2.5D problem where we need to integrate along strike direction x,
  // the position of the receivers along this direction need to be taken into account
  void set_receivers (const std::vector<Receiver>& recvs);
  void set_dipole_sources(const std::vector<DipoleSource> &sources);
  //void set_physical_sources (const CurrentFunctionList<2>::type& sources);

  void set_wavenumber_range(double kmin, double kmax, unsigned nk);
  void set_digital_filter_file(const std::string file);

  // Return calculated data according to the source-receiver map
  void get_survey_data(ModelledData<dcomplex> &modelled_data) const;

  dvector get_wavenumbers() const;
  dcomplex transform_field_to_space_domain(const TransformationType &trans_type, const cvector &fkx,
                                           const double &strike_coord) const;
  void get_symmetry_map(const DipoleSource &src, std::vector<TransformationType> &symmetry) const;

  const Point<3> get_receiver_position(const std::string &recname) const;

  const std::vector<cvector> &get_final_data() const;

protected:
  void init();
  std::vector<cvector> transform_symmetric_fields_to_space_domain(const std::vector<std::vector<cvector>> &wavedomain_fields);
  std::vector<cvector> transform_fields_to_space_domain(const std::vector<std::vector<cvector>> &wavedomain_fields);
  void transform_fields_to_wave_domain();

  void update_receiver_positions_along_strike();

  void add_primary_fields(std::vector<cvector> &secondary_fields);
  std::vector<cvector> combine_principal_sources(std::vector<cvector> &data);

private:
  std::vector<double> positive_wavenumbers, log_wavenumbers, negative_wavenumbers;   // values of wave numbers in spectral strike domain
  std::vector<double> receiver_positions_along_strike;

  // Each element stores a vector with indices to principal sources comprising
  // an arbitrarily oriented dipole
  // (for a dipole with non-zerp x and yz moments this amounts to two independent sources)
  std::vector<std::vector<unsigned>> source_mask;
  std::vector<DipoleSource> input_sources;

  // Final data for input sources. This vector is organized as follows:
  // There are as many sub-vectors stored in this vector as many receivers
  // Every sub-vector has n_sources*n_data_components elements sorted with
  // respect to the sources.
  std::vector<cvector> combined_data;

  std::vector<std::shared_ptr<PETScWrappers::MPI::SparseMatrix>> system_matrices;

  unsigned current_wavenumber_index;

  mutable SinCosTransform<dcomplex> transform;
};

#endif // CSEM25DFEM_H

