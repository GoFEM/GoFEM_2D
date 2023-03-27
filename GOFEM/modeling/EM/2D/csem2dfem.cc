#include "csem2dfem.h"

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/dofs/dof_tools.h>

#include "functions/exact_solution.h"

CSEM2DFEM::CSEM2DFEM (MPI_Comm comm,
                      const unsigned int order,
                      const unsigned int mapping_order,
                      const PhysicalModelPtr<2> &model,
                      const BackgroundModel<2> &bg_model):
  EM2DFEM(comm, order, mapping_order, model, bg_model)
{}

CSEM2DFEM::CSEM2DFEM (MPI_Comm comm,
                      const unsigned int order,
                      const unsigned int mapping_order,
                      const PhysicalModelPtr<2> &model):
  EM2DFEM(comm, order, mapping_order, model)
{}

CSEM2DFEM::~CSEM2DFEM ()
{
  clear ();
}

void CSEM2DFEM::set_wavenumber(double kx)
{
  wavenumber = kx;
}

void CSEM2DFEM::copy_local_to_global_system(const Assembly::CopyData::MaxwellSystem &data)
{
  constraints[0].distribute_local_to_global (data.local_matrix, data.local_dof_indices, system_matrix);
}

unsigned CSEM2DFEM::get_number_of_constraint_matrices() const
{
  return 1;
}

void CSEM2DFEM::set_boundary_values ()
{
  switch (boundary_conditions)
  {
  case Dirichlet:
  {
    pcout << "  Impose Dirichlet boundary conditions\n";

    Functions::ZeroFunction<2> zero_function(fe.n_blocks());
    for(types::boundary_id bid: all_boundaries)
      for(auto& cm: constraints)
        VectorTools::interpolate_boundary_values (mapping, dof_handler, bid, zero_function, cm);

    break;
  }
  case Neumann:
    pcout << "  Impose Neumann boundary conditions\n";
    break;

  default:
    throw std::runtime_error("Unsupported boundary conditions.");
  }

  solution_constraints_indices.resize(n_physical_sources + 1, 0);
}

void CSEM2DFEM::assemble_function_rhs_vector(const DipoleSource &phys_source,
                                             const AffineConstraints<double> &constraints,
                                             PETScWrappers::MPI::BlockVector &rhs_vector)
{
  dvector sigma_air = background_model.conductivities(),
          epsilon_air = background_model.permittivities();

  if(sigma_air.size() != 1)
    throw std::runtime_error("Specified background model seems to have more or less than one layer. "
                             "Currently the background model has to be a homogeneous space.");

  DipoleSource source = phys_source;
  // For secondary field calculations and dipole source we assume a source is located at x = 0
  if(source.get_type() == ElectricDipole || source.get_type() == MagneticDipole)
    source.set_positions_along_strike(0);
  ExactSolutionCSEMSpaceKx<2> solution_kx(source, sigma_air[0], epsilon_air[0],
      phys_model->model_extension(), filter_file, PhaseLag);

  solution_kx.set_frequency(frequency);
  solution_kx.set_wavenumber(wavenumber);

  const FEValuesExtractors::Scalar e_re (0);
  const FEValuesExtractors::Scalar e_im (1);
  const FEValuesExtractors::Scalar h_re (2);
  const FEValuesExtractors::Scalar h_im (3);

  QGauss<2> quadrature (fe.degree + 1);
  UpdateFlags update_flags = update_JxW_values | update_quadrature_points | update_values | update_gradients;
  FEValues<2> fe_values (mapping, fe, quadrature, update_flags);

  Vector<double> Ep (6); // Primary field in spectral domain
  std::vector<double> sigma_values (quadrature.size()), epsilon_values(quadrature.size());//,mu_values(quadrature.size());
  Vector<double> local_rhs (fe.dofs_per_cell);
  std::vector<types::global_dof_index>	dof_indices (fe.dofs_per_cell);

  // work with angular frequency
  const double omega = 2.0 * M_PI * frequency;
  dcomplex lambda, a, ey_b = 0, ey_c = 0, ez_b = 0, ez_c = 0;
  double ex_re, ex_im, hx_re, hx_im;

  for (DoFHandler<2>::active_cell_iterator cell = dof_handler.begin_active (); cell != dof_handler.end (); ++cell)
    if (cell->is_locally_owned ())
    {
      fe_values.reinit (cell);
      local_rhs = 0.0;

      const std::vector<double>& JxW_values = fe_values.get_JxW_values ();
      const std::vector<Point<2>>& quadrature_points = fe_values.get_quadrature_points ();

      //phys_model->permeability_list (cell, quadrature_points, mu_values);
      phys_model->conductivity_list (cell, quadrature_points, sigma_values);
      phys_model->permittivity_list (cell, quadrature_points, epsilon_values);

      for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points; ++q_point)
      {
        dcomplex sigma = sigma_values[q_point] - II*omega*epsilon_values[q_point],
                 sigma_bg = background_model.conductivity_at(cell, quadrature_points[q_point])
                 - II*omega*background_model.permittivity_at(cell, quadrature_points[q_point]);

        dcomplex sigma_diff = sigma - sigma_bg;

        if(abs(sigma_diff) < std::numeric_limits<float>::epsilon())
          continue; // no secondary sources at this point

        solution_kx.vector_value(quadrature_points[q_point], Ep);

        const double omega_mu = omega * mu0;/*scratch.mu_values[q_point]*/

        lambda = 1.0 / (wavenumber*wavenumber - II*omega_mu*sigma);
        a = sigma_diff * dcomplex(Ep[0], Ep[3]);
        ey_b =  II * omega_mu * lambda * sigma_diff * dcomplex(Ep[1], Ep[4]);
        ez_b =  II * omega_mu * lambda * sigma_diff * dcomplex(Ep[2], Ep[5]);
        ey_c = -II * wavenumber * lambda * sigma_diff * dcomplex(Ep[1], Ep[4]);
        ez_c = -II * wavenumber * lambda * sigma_diff * dcomplex(Ep[2], Ep[5]);

        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
        {
          ex_re = - fe_values[e_re].value (i, q_point) * a.real()
                  + fe_values[e_re].gradient (i, q_point)[0] * ey_c.real() + fe_values[e_re].gradient (i, q_point)[1] * ez_c.real();
          ex_im = - fe_values[e_im].value (i, q_point) * a.imag()
                  + fe_values[e_im].gradient (i, q_point)[0] * ey_c.imag() + fe_values[e_im].gradient (i, q_point)[1] * ez_c.imag();

          hx_re = -fe_values[h_re].gradient (i, q_point)[0] * ez_b.real() + fe_values[h_re].gradient (i, q_point)[1] * ey_b.real();
          hx_im = -fe_values[h_im].gradient (i, q_point)[0] * ez_b.imag() + fe_values[h_im].gradient (i, q_point)[1] * ey_b.imag();

          // Two steps are done below to support symmetry of the system:
          // First: the H term has negative sign
          // Second: imaginary parts are multiplied by -1
          local_rhs(i) += JxW_values[q_point] * (ex_re - ex_im - hx_re + hx_im);
        }
      }

      cell->get_dof_indices (dof_indices);
      // This call assumes that we have homogeneous constraints.
      // If non-homogeneous constraints are required (e.g. due to more complex BC)
      // one has to pass additionally local matrix to this routine. See doc ConstraintMatrix
      constraints.distribute_local_to_global (local_rhs, dof_indices, rhs_vector);
    }
}

void CSEM2DFEM::assemble_problem_rhs()
{
  for(unsigned i = 0; i < n_physical_sources; ++i)
  {
    if(background_model.n_layers() > 0)
      assemble_function_rhs_vector(dynamic_cast<const DipoleSource&>(*physical_sources[i]),
                                   constraints[0], system_rhs[i]); // secondary field
    else
      assemble_dipole_rhs_vector(dynamic_cast<const DipoleSource&>(*physical_sources[i]),
                                 constraints[0], system_rhs[i]); // primary field
  }

  if(adaptivity_type == GoalOriented)
    assemble_dual_rhs_vector(unique_point_receiver_positions, constraints[0], system_rhs[n_physical_sources]);

  for(size_t i = 0; i < system_rhs.size(); ++i)
    system_rhs[i].compress (VectorOperation::add);
}

std::string CSEM2DFEM::data_header() const
{
  return "Ex\tEy\tEz\tHx\tHy\tHz";
}

unsigned CSEM2DFEM::n_data_at_point() const
{
  return 6;
}

cvector CSEM2DFEM::calculate_data_at_receiver(const std::vector<cvector> &E,
                                              const std::vector<cvector> &H) const
{
  if(E.size() > 1 || H.size() > 1)
    throw std::runtime_error("CSEM modelling cannot deal with multi-electrode receivers");

  cvector TF(n_data_at_point()*n_physical_sources);

  for(size_t i = 0; i < n_physical_sources; ++i)
  {
    TF[6*i + 0] = E[0][i];
    TF[6*i + 1] = E[0][i + n_physical_sources];
    TF[6*i + 2] = E[0][i + 2*n_physical_sources];
    TF[6*i + 3] = H[0][i];
    TF[6*i + 4] = H[0][i + n_physical_sources];
    TF[6*i + 5] = H[0][i + 2*n_physical_sources];
  }

  return TF;
}

void CSEM2DFEM::set_dipole_sources(const std::vector<DipoleSource> &sources)
{
  std::vector<PhysicalSourcePtr> srcs(sources.size());
  for(size_t i = 0; i < sources.size(); ++i)
    srcs[i].reset(new PhysicalSource(sources[i]));

  EM2DFEM::set_physical_sources(srcs);
}
