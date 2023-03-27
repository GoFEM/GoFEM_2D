#include "csem25dfem.h"

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_tools.h>

#include "functions/exact_solution.h"

CSEM25DFEM::CSEM25DFEM (MPI_Comm comm,
                        const unsigned int fe_order,
                        const unsigned int mapping_order,
                        const PhysicalModelPtr<2> &model):
  CSEM2DFEM(comm, fe_order, mapping_order, model)
{
  init();
}

CSEM25DFEM::CSEM25DFEM (MPI_Comm comm,
                        const unsigned int fe_order,
                        const unsigned int mapping_order,
                        const PhysicalModelPtr<2> &model,
                        const BackgroundModel<2> &bg_model):
  CSEM2DFEM(comm, fe_order, mapping_order, model, bg_model)
{
  init();
}

void CSEM25DFEM::init()
{
  fill_array_logscale(1e-5, 1e-1, 30, positive_wavenumbers);

  log_wavenumbers.resize(positive_wavenumbers.size());
  for(size_t i = 0; i < positive_wavenumbers.size(); ++i)
    log_wavenumbers[i] = log10(positive_wavenumbers[i]);

  estimate_error_on_last_cycle = false;
}

void CSEM25DFEM::run()
{
  //transform_fields_to_wave_domain();
  //return;

  //std::ifstream ifs("wavesolutions.txt");
//  std::ofstream ofs("wavesolutions.txt");
//  ofs << std::setprecision(7);

  Timer t;

  // In case of no refinement we can reuse multiple structures for different wavenumbers
  if(refinement_steps == 1)
    reuse_data_structures = true;

  std::vector<std::vector<cvector>> wavedomain_fields (positive_wavenumbers.size() + negative_wavenumbers.size(), std::vector<cvector>(n_physical_sources));
  for(unsigned i = 0; i < positive_wavenumbers.size(); ++i)
  {
    current_wavenumber_index = i;
    std::cout << "  Solving for wavenumber: " << positive_wavenumbers[i] << "\n";
    set_wavenumber(positive_wavenumbers[i]);
    CSEM2DFEM::run();
    wavedomain_fields[i] = data_at_receivers ();

    if(!reuse_data_structures)
      clear();

//        ofs << std::setw(14) << std::left << positive_wavenumbers[i] << "\t"
//            << std::setw(14) << std::left << wavedomain_fields[i][0][0].real() << "\t" << std::setw(14) << std::left << wavedomain_fields[i][0][0].imag() << "\t"
//            << std::setw(14) << std::left << wavedomain_fields[i][0][1].real() << "\t" << std::setw(14) << std::left << wavedomain_fields[i][0][1].imag() << "\t"
//            << std::setw(14) << std::left << wavedomain_fields[i][0][2].real() << "\t" << std::setw(14) << std::left << wavedomain_fields[i][0][2].imag() << "\t"
//            << std::setw(14) << std::left << wavedomain_fields[i][0][3].real() << "\t" << std::setw(14) << std::left << wavedomain_fields[i][0][3].imag() << "\t"
//            << std::setw(14) << std::left << wavedomain_fields[i][0][4].real() << "\t" << std::setw(14) << std::left << wavedomain_fields[i][0][4].imag() << "\t"
//            << std::setw(14) << std::left << wavedomain_fields[i][0][5].real() << "\t" << std::setw(14) << std::left << wavedomain_fields[i][0][5].imag() << "\n";

    //    wavedomain_fields[i][0].resize(6);

    //    double kx;
    //    ifs >> kx >> wavedomain_fields[i][0][0] >> wavedomain_fields[i][0][1] >> wavedomain_fields[i][0][2]
    //        >> wavedomain_fields[i][0][3] >> wavedomain_fields[i][0][4] >> wavedomain_fields[i][0][5];
  }

  for(unsigned i = 0; i < negative_wavenumbers.size(); ++i)
  {
    current_wavenumber_index = i;
    std::cout << "  Solving for wavenumber: " << negative_wavenumbers[i] << "\n";
    set_wavenumber(negative_wavenumbers[i]);
    CSEM2DFEM::run();
    unsigned k = i + positive_wavenumbers.size();
    wavedomain_fields[k] = data_at_receivers ();

    if(!reuse_data_structures)
      clear();

    //    ofs << negative_wavenumbers[i] << "\t"
    //        << wavedomain_fields[k][0][0] << "\t"
    //        << wavedomain_fields[k][0][1] << "\t"
    //        << wavedomain_fields[k][0][2] << "\t"
    //        << wavedomain_fields[k][0][3] << "\t"
    //        << wavedomain_fields[k][0][4] << "\t"
    //        << wavedomain_fields[k][0][5] << "\n";
  }

//  ofs.close();

  pcout << "  Solution for all " << positive_wavenumbers.size() + negative_wavenumbers.size() << " wavenumbers took: " << t.wall_time() << "\n";
  t.restart();

  std::vector<cvector> data;
  if(negative_wavenumbers.empty())
    data = transform_symmetric_fields_to_space_domain (wavedomain_fields);
  else
    data = transform_fields_to_space_domain (wavedomain_fields);
  pcout << "  Inverse fourier transformation took: " << t.wall_time() << "\n";

  // Add primary field if secondary field approach has been used
  if(background_model.n_layers() > 0)
    add_primary_fields(data);

  // Arbitrary oriented dipoles were split in x and yz components to preserve symmetry
  // We need to sum them up again to get input dipole responses
  combined_data = combine_principal_sources(data);

  // Output calculated at receiver positions fields in space domain
  if(output_type.find("point") != std::string::npos)
  {
    CSEM2DFEM::set_dipole_sources(input_sources);
    output_point_data(combined_data, 0);
    CSEM25DFEM::set_dipole_sources(input_sources);
  }
}

void CSEM25DFEM::add_primary_fields(std::vector<cvector> &secondary_fields)
{
  dvector sigma_air = background_model.conductivities(),
        epsilon_air = background_model.permittivities();

  if(sigma_air.size() != 1)
  {
    throw std::runtime_error("Specified background model seems to have more than one layer. "
                             "Currently the background model has to be a homogeneous space.");
  }

  pcout << "  Calculate and add primary fields.\n";
  if(this_mpi_process == 0)
  {
    for(size_t sidx = 0; sidx < n_physical_sources; ++sidx)
    {
      const DipoleSource& source = dynamic_cast<const DipoleSource&>(*physical_sources[sidx]);

      ExactSolutionCSEMSpace space_solution(source, sigma_air[0], epsilon_air[0], frequency, PhaseLag);

      for(size_t ridx = 0; ridx < receivers.size(); ++ridx)
      {
        Point<3> position = receivers[ridx].position<Point<3>>(0);

        Vector<double> Ep(6), Hp(6);
        space_solution.set_field(EField);
        space_solution.vector_value(position, Ep);
        space_solution.set_field(HField);
        space_solution.vector_value(position, Hp);

        for(size_t component = 0; component < 6; ++component)
        {
          if(component < 3)
            secondary_fields[ridx][6*sidx + component] += dcomplex(Ep[component], Ep[component + 3]);
          else
            secondary_fields[ridx][6*sidx + component] += dcomplex(Hp[component - 3], Hp[component]);
        }

        //        std::cout << sources[i].get_name() << "\t" << receivers[j].get_name() << "\t"
        //                  << dcomplex(Ep[0], Ep[3]) << "\t" << dcomplex(Ep[1], Ep[4]) << "\t" << dcomplex(Ep[2], Ep[5]) << "\t"
        //                  << dcomplex(Hp[0], Hp[3]) << "\t" << dcomplex(Hp[1], Hp[4]) << "\t" << dcomplex(Hp[2], Hp[5]) << "\n";
      }
    }
  }
}

std::vector<cvector> CSEM25DFEM::combine_principal_sources(std::vector<cvector> &data)
{
  std::vector<cvector> combined_data(receivers.size(), cvector(source_mask.size() * 6, 0.));

  // Combine fields from principal component dipoles
  for(size_t i = 0; i < source_mask.size(); ++i)
  {
    for(size_t sidx = 0; sidx < source_mask[i].size(); ++sidx)
    {
      for(size_t ridx = 0; ridx < receivers.size(); ++ridx)
      {
        for(size_t component = 0; component < 6; ++component)
          combined_data[ridx][6*i + component] += data[ridx][6*source_mask[i][sidx] + component];
      }
    }
  }

  return combined_data;
}

void CSEM25DFEM::update_receiver_positions_along_strike()
{
  receiver_positions_along_strike.clear();
  for (size_t i = 0; i < receivers.size (); ++i)
  {
    Point<3> p = receivers[i].position<Point<3>>(0);
    receiver_positions_along_strike.push_back(p[0]);
  }
}

void CSEM25DFEM::set_dipole_sources(const std::vector<DipoleSource> &sources)
{
  std::vector<PhysicalSourcePtr> srcs;

  source_mask.clear();
  source_mask.resize(sources.size());

  input_sources = sources;

  bool complex_sources_exist = false;

  for(size_t i = 0; i < sources.size(); ++i)
  {
    if(sources[i].n_dipole_elements() == 1)
    {
      // In 2.5D modeling, in order to preserve solution symmetry w.r.t. strike dimension
      // and avoid solving for twice as many wave numbers, we split elementary dipole sources
      // into x and yz principal contributions. The routine extracting solution at receivers
      // need to sum up solution from principal components to get response of an arbitrarily
      // oriented dipole.
      dvec3d extent = sources[i].dipole_extent();

      if(fabs(extent[0]) > 1e-6 && (fabs(extent[1]) > 1e-6 || fabs(extent[2]) > 1e-6))
      {
        DipoleSource source_x(sources[i]), source_yz(sources[i]);

        source_yz.set_name(sources[i].get_name() + "_yz");
        source_x.set_name(sources[i].get_name() + "_x");

        extent[1] = 0.;
        extent[2] = 0.;
        source_x.set_dipole_extent(extent);

        extent = sources[i].dipole_extent();
        extent[0] = 0.;
        source_yz.set_dipole_extent(extent);

        srcs.push_back(PhysicalSourcePtr(new DipoleSource(source_x)));
        srcs.push_back(PhysicalSourcePtr(new DipoleSource(source_yz)));

        source_mask[i].push_back(srcs.size() - 2);
        source_mask[i].push_back(srcs.size() - 1);
      }
      else
      {
        srcs.push_back(PhysicalSourcePtr(new DipoleSource(sources[i])));
        source_mask[i].push_back(srcs.size() - 1);
      }
    }
    else
    {
      throw std::runtime_error("Complex dipoles are currently not supported in 2.5D modeling.");

      complex_sources_exist = true;
      srcs.push_back(PhysicalSourcePtr(new DipoleSource(sources[i])));
      source_mask[i].push_back(srcs.size() - 1);
    }
  }

  // Dismiss any symmetry because of wire sources
  if(complex_sources_exist)
  {
    negative_wavenumbers.clear();
    for(double v: positive_wavenumbers)
      negative_wavenumbers.push_back(-v);
  }

  EM2DFEM::set_physical_sources(srcs);
}

void CSEM25DFEM::set_wavenumber_range(double kmin, double kmax, unsigned nk)
{
  fill_array_logscale(kmin, kmax, nk, positive_wavenumbers);

  log_wavenumbers.clear();
  for(double k: positive_wavenumbers)
    log_wavenumbers.push_back(log10(k));
}

void CSEM25DFEM::set_digital_filter_file(const std::string file)
{
  filter_file = file;
  transform.reinit(filter_file);
}

void CSEM25DFEM::get_survey_data(ModelledData<dcomplex> &modelled_data) const
{
  construct_survey_data(combined_data, modelled_data);
}

dvector CSEM25DFEM::get_wavenumbers() const
{
  if(negative_wavenumbers.size() != 0)
    throw std::runtime_error("You rely on incomplete implementation, cannot proceed...");

  return positive_wavenumbers;
}

void CSEM25DFEM::clear()
{
  system_matrices.clear();
  EMFEM<2>::clear();
}

void CSEM25DFEM::set_receivers(const std::vector<Receiver> &recvs)
{
  for(auto &rec: receivers)
  {
    if(rec.n_electrodes() != 1)
      throw std::runtime_error("2D modeling does not support multi-electrode receivers.");
  }

  CSEM2DFEM::set_receivers(recvs);
  update_receiver_positions_along_strike();
}

dcomplex CSEM25DFEM::transform_field_to_space_domain(const TransformationType &trans_type, const cvector &fkx, const double &strike_coord) const
{
  return transform.integrate(positive_wavenumbers, log_wavenumbers, fkx,
                             strike_coord, trans_type, InverseTransform, 1.);
}

void CSEM25DFEM::get_symmetry_map(const DipoleSource &source, std::vector<TransformationType> &symmetry) const
{
  TransformationType i0, i1;

  // Get dipole type
  if ( source.get_type() == ElectricDipole ) // electric dipole
  {
    i0 = CosineTransform; // 0 is for even cosine transform
    i1 = SineTransform;   // 1 is for odd sine transform
  }
  else // magnetic dipole
  {
    i0 = SineTransform;
    i1 = CosineTransform;
  }

  bool xaligned = std::fabs(source.dipole_extent()[0]) > 0.,
       yz_aligned = std::fabs(source.dipole_extent()[1]) > 0. ||
       std::fabs(source.dipole_extent()[2]) > 0.;

  // Set symmetry variables based on dipole type (e or b) and direction of dipole (x or yz)
  TransformationType sym1, sym2;
  if ( xaligned && !yz_aligned )
  {
    sym1 = i0;
    sym2 = i1;
  }
  else if( !xaligned && yz_aligned ) // yz plane
  {
    sym1 = i1;
    sym2 = i0;
  }
  else
    throw std::runtime_error("This is a wire source which ruins any symmetry in the spectral domain. Use full integration domain.");

  symmetry = {sym1, sym2, sym2, sym2, sym1, sym1};
}

const Point<3> CSEM25DFEM::get_receiver_position(const std::string &recname) const
{
  const unsigned idx = get_receiver_index(recname);
  return receivers[idx].position<Point<3>>(0);
}

const std::vector<cvector> &CSEM25DFEM::get_final_data() const
{
  return combined_data;
}

std::vector<cvector> CSEM25DFEM::transform_symmetric_fields_to_space_domain(const std::vector<std::vector<cvector>> &wavedomain_fields)
{
  // Data in space domain
  std::vector<cvector> data(receivers.size(), cvector(n_physical_sources * n_data_at_point(), 0.));

  pcout << "  Perform inverse fourier transformation.\n";
  if(this_mpi_process == 0)
  {
    // Perform inverse fourier transformation for all sources and receivers
    for(size_t sidx = 0; sidx < n_physical_sources; ++sidx)
    {
      const DipoleSource& source = dynamic_cast<const DipoleSource&>(*physical_sources[sidx]);

      std::vector<TransformationType> symmetry;
      get_symmetry_map(source, symmetry);

      for(size_t ridx = 0; ridx < receivers.size(); ++ridx)
      {
        // Distance between receiver and source along strike direction x
        double xr = receiver_positions_along_strike[ridx] - source.position_along_strike();
        for(size_t component = 0; component < 6; ++component)
        {
          cvector fkx(positive_wavenumbers.size());
          for(size_t k = 0; k < positive_wavenumbers.size(); ++k)
            fkx[k] = wavedomain_fields[k][ridx][6*sidx + component];

          data[ridx][6*sidx + component] = transform_field_to_space_domain(symmetry[component], fkx, xr);
        }
      }
    }
  }

  return data;
}

std::vector<cvector> CSEM25DFEM::transform_fields_to_space_domain(const std::vector<std::vector<cvector>> &wavedomain_fields)
{
  // Data in space domain
  std::vector<cvector> data(receivers.size(), cvector(n_physical_sources * n_data_at_point(), 0.));

  unsigned nkx = positive_wavenumbers.size();

  pcout << "  Perform inverse fourier transformation.\n";
  if(this_mpi_process == 0)
  {
    // Perform inverse fourier transformation for all sources and receivers
    for(size_t sidx = 0; sidx < n_physical_sources; ++sidx)
    {
      const DipoleSource& source = dynamic_cast<const DipoleSource&>(*physical_sources[sidx]);

      for(size_t ridx = 0; ridx < receivers.size(); ++ridx)
      {
        // Distance between receiver and source along strike direction x
        double xr = receiver_positions_along_strike[ridx];

        // If the source is dipole and total field approach is used,
        // we need to take its position into account
        if(source.n_dipole_elements() == 1 && background_model.n_layers() == 0)
          xr -= source.position_along_strike();

        if(fabs(xr) < std::numeric_limits<double>::epsilon())
          xr = 0.;

        for(size_t component = 0; component < 6; ++component)
        {
          cvector fkx_even(nkx), fkx_odd(nkx);
          for(size_t k = 0; k < nkx; ++k)
          {
            fkx_even[k] = 0.5 * (wavedomain_fields[k][ridx][6*sidx + component] + wavedomain_fields[k + nkx][ridx][6*sidx + component]);
            fkx_odd[k]  = 0.5 * (wavedomain_fields[k][ridx][6*sidx + component] - wavedomain_fields[k + nkx][ridx][6*sidx + component]);
          }

          data[ridx][6*sidx + component] = transform.integrate(positive_wavenumbers, log_wavenumbers, fkx_even, xr, CosineTransform, InverseTransform, 1.)
              + transform.integrate(positive_wavenumbers, log_wavenumbers, fkx_odd, xr, SineTransform, InverseTransform, 1.);
        }
      }
    }
  }

  return data;
}

void CSEM25DFEM::transform_fields_to_wave_domain()
{
  const DipoleSource& source = dynamic_cast<const DipoleSource&>(*physical_sources[0]);

  dvector sigma = background_model.conductivities(),
          epsilon = background_model.permittivities();

  std::cout << std::setprecision(7);

  ExactSolutionCSEMSpaceKx<2> solution_kx(source, sigma[0], epsilon[0],
                                          phys_model->model_extension(),
                                          filter_file, PhaseLag);

  solution_kx.set_frequency(frequency);

  const Point<3> p = receivers[0].position<Point<3>>(0);
  const Point<2> p2d(p[1], p[2]);

  for(size_t k = 0; k < positive_wavenumbers.size(); ++k)
  {
    std::cout << std::setw(14) << std::left << positive_wavenumbers[k] << " ";

    solution_kx.set_wavenumber(positive_wavenumbers[k]);

    Vector<double> E(6);
    solution_kx.set_field(EField);
    solution_kx.vector_value(p2d, E);
    std::cout << std::setw(14) << std::left << E[0] << " "
              << std::setw(14) << std::left << E[3] << " "
              << std::setw(14) << std::left << E[1] << " "
              << std::setw(14) << std::left << E[4] << " "
              << std::setw(14) << std::left << E[2] << " "
              << std::setw(14) << std::left << E[5] << " ";
    solution_kx.set_field(HField);
    solution_kx.vector_value(p2d, E);
    std::cout << std::setw(14) << std::left << E[0] << " "
              << std::setw(14) << std::left << E[3] << " "
              << std::setw(14) << std::left << E[1] << " "
              << std::setw(14) << std::left << E[4] << " "
              << std::setw(14) << std::left << E[2] << " "
              << std::setw(14) << std::left << E[5] << "\n";
  }

  for(size_t k = 0; k < negative_wavenumbers.size(); ++k)
  {
    std::cout << negative_wavenumbers[k] << "\t";

    solution_kx.set_wavenumber(negative_wavenumbers[k]);

    Vector<double> E(6);
    solution_kx.set_field(EField);
    solution_kx.vector_value(p2d, E);
    std::cout << std::setw(14) << std::left << E[0] << " "
              << std::setw(14) << std::left << E[3] << " "
              << std::setw(14) << std::left << E[1] << " "
              << std::setw(14) << std::left << E[4] << " "
              << std::setw(14) << std::left << E[2] << " "
              << std::setw(14) << std::left << E[5] << " ";
    solution_kx.set_field(HField);
    solution_kx.vector_value(p2d, E);
    std::cout << std::setw(14) << std::left << E[0] << " "
              << std::setw(14) << std::left << E[3] << " "
              << std::setw(14) << std::left << E[1] << " "
              << std::setw(14) << std::left << E[4] << " "
              << std::setw(14) << std::left << E[2] << " "
              << std::setw(14) << std::left << E[5] << "\n";
  }
}
