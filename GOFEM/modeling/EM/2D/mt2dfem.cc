#include "mt2dfem.h"

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/dofs/dof_tools.h>

#include "functions/exact_solution.h"

MT2DFEM::MT2DFEM (MPI_Comm comm,
                  const unsigned int order,
                  const unsigned int mapping_order,
                  const PhysicalModelPtr<2> &model):
  EM2DFEM(comm, order, mapping_order, model)
{
  n_physical_sources = 1;
  error_estimates.resize (n_physical_sources);

  model_corner_points.resize(2);

  current_functions.resize(1);
  physical_sources.resize(1);
  current_functions[0].reset(new CurrentFunction<2>("Plane_wave_function"));
  physical_sources[0].reset(new PhysicalSource("Plane_wave"));
}

MT2DFEM::~MT2DFEM ()
{
  clear ();
}

void MT2DFEM::copy_local_to_global_system(const Assembly::CopyData::MaxwellSystem &data)
{
  constraints[0].distribute_local_to_global (data.local_matrix, data.local_rhs,
                                             data.local_dof_indices, system_matrix,
                                             system_rhs[0]);
}

void MT2DFEM::create_grid (const unsigned int cycle)
{
  EMFEM<2>::create_grid(cycle);

  if (cycle == 0)
  {
    // Find bounding box the domain
    const std::vector<Point<2>> &vertices = triangulation.get_vertices ();
    Point<2> begin(1e10, 1e10), end(-1e10, -1e10);

    for (size_t i = 0; i < vertices.size(); ++i)
    {
      if(begin(0) > vertices[i](0))
        begin(0) = vertices[i](0);
      if(begin(1) > vertices[i](1))
        begin(1) = vertices[i](1);

      if(end(0) < vertices[i](0))
        end(0) = vertices[i](0);
      if(end(1) < vertices[i](1))
        end(1) = vertices[i](1);
    }

    model_corner_points[0] = begin;
    model_corner_points[1] = end;
  }
}

unsigned MT2DFEM::get_number_of_constraint_matrices() const
{
  return 2;
}

void MT2DFEM::set_boundary_values ()
{
  switch (boundary_conditions)
  {
  case Dirichlet:
  {
    pcout << "  Impose Dirichlet boundary conditions\n";

//    {
//      ExactSolutionMT1D<2> exact_solution(get_boundary_model(left_boundary), frequency);
//      VectorTools::interpolate_boundary_values (mapping, dof_handler, left_boundary, exact_solution, constraints);
//      VectorTools::interpolate_boundary_values (mapping, dof_handler, top_boundary, exact_solution, constraints);
//      VectorTools::interpolate_boundary_values (mapping, dof_handler, bottom_boundary, exact_solution, constraints);
//    }

#ifdef SHARED_TRIANGULATION
    auto west_boundary_model = get_boundary_model(phys_model, triangulation, west_boundary_id);
    auto east_boundary_model = get_boundary_model(phys_model, triangulation, east_boundary_id);
#else
    setup_initial_triangulation(local_copy_model, local_copy_triangulation);
    auto west_boundary_model = get_boundary_model(local_copy_model, local_copy_triangulation, west_boundary_id);
    auto east_boundary_model = get_boundary_model(local_copy_model, local_copy_triangulation, east_boundary_id);
#endif

    ExactSolutionMT1DTapered<2> exact_solution(west_boundary_model, east_boundary_model,
                                               model_corner_points, frequency, PhaseLead);
//    ExactSolutionMT1D<2> exact_solution(west_boundary_model, frequency, EW, PhaseLead);

    Functions::ZeroFunction<2> zero_function(fe.n_blocks());

    for(types::boundary_id bid: all_boundaries)
    {
      VectorTools::interpolate_boundary_values (mapping, dof_handler, bid, exact_solution, constraints[0]);
      VectorTools::interpolate_boundary_values (mapping, dof_handler, bid, zero_function, constraints[1]);
    }

    break;
  }
  case Neumann:
    pcout << "  Impose Neumann boundary conditions\n";
    throw std::runtime_error ("Neumann BCs for 2D MT modeling are not yet implemented.");
    break;

  default:
    throw std::runtime_error("Unsupported boundary conditions.");
  }

  solution_constraints_indices.push_back(0);
  solution_constraints_indices.push_back(1);
}

void MT2DFEM::assemble_problem_rhs()
{
  if(adaptivity_type == GoalOriented)
    assemble_dual_rhs_vector(unique_point_receiver_positions, constraints[1],
                             system_rhs[n_physical_sources]);

  for(size_t i = 0; i < system_rhs.size(); ++i)
    system_rhs[i].compress (VectorOperation::add);
}

std::string MT2DFEM::data_header() const
{
  return "Zxy\tZyx\tTzy\trho_xy\trho_yx\tphi_xy\tphi_yx\tEx\tEy\tEz\tHx\tHy\tHz";
}

unsigned MT2DFEM::n_data_at_point() const
{
  return 13;
}

void MT2DFEM::tangential_fields_at(const std::vector<Point<2> > &points, std::vector<cvector> &fields,
                                   FieldFormulation field_type) const
{
  // Angular frequency
  const double omega = 2.0 * numbers::PI * frequency;

  std::vector<std::vector<Tensor<1, 2>>> grads (fe.n_blocks(), std::vector<Tensor<1, 2>> (1));
  std::vector<std::complex<double>> sigma_values(1);
  std::vector<double> epsilon_values(1);

  // Initialize FEValues with given cell
  const FEValuesExtractors::Scalar e_re (0);
  const FEValuesExtractors::Scalar e_im (1);
  const FEValuesExtractors::Scalar h_re (2);
  const FEValuesExtractors::Scalar h_im (3);

  std::pair<DoFHandler<2>::active_cell_iterator, Point<2>> cell_point;

  try
  {
    cell_point = GridTools::find_active_cell_around_point (mapping, dof_handler, points[0]);
  }
  catch(std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }

  for(unsigned i = 0; i < points.size(); ++i)
  {
    if(!cell_point.first->point_inside(points[i]))
    {
      try
      {
        cell_point = GridTools::find_active_cell_around_point (mapping, dof_handler, points[i]);
      }
      catch(std::exception& e)
      {
        std::cout << e.what() << std::endl;
      }
    }

    if(cell_point.first->is_locally_owned())
    {
      Point<2> p_unit = mapping.transform_real_to_unit_cell(cell_point.first, points[i]);
      const Quadrature<2> quadrature (GeometryInfo<2>::project_to_unit_cell (p_unit));
      FEValues<2> fe_values (mapping, fe, quadrature, update_gradients);
      fe_values.reinit (cell_point.first);

      if(field_type == EField)
      {
        phys_model->permittivity_list (cell_point.first, quadrature.get_points (), epsilon_values);
        phys_model->complex_conductivity_list (cell_point.first, quadrature.get_points (), sigma_values, frequency);

        const dcomplex sigma = sigma_values[0] - II*omega*epsilon_values[0];

        fe_values[h_re].get_function_gradients (solutions[0], grads[2]);
        fe_values[h_im].get_function_gradients (solutions[0], grads[3]);

        // Rotated gradient, i.e. multiplied with [0 -1; 1 0]
        fields[i][1] = dcomplex(grads[2][0][1], grads[3][0][1]) / sigma;
        fields[i][2] = -dcomplex(grads[2][0][0], grads[3][0][0]) / sigma;
      }
      else // assume HField
      {
        const dcomplex i_omega_mu = II*omega*mu0;

        fe_values[e_re].get_function_gradients (solutions[0], grads[0]);
        fe_values[e_im].get_function_gradients (solutions[0], grads[1]);

        // Rotated gradient, i.e. multiplied with [0 -1; 1 0]
        fields[i][1] = dcomplex(grads[0][0][1], grads[1][0][1]) / i_omega_mu;
        fields[i][2] = -dcomplex(grads[0][0][0], grads[1][0][0]) / i_omega_mu;
      }
    }
  }
}

cvector MT2DFEM::calculate_data_at_receiver(const std::vector<cvector> &E,
                                            const std::vector<cvector> &H) const
{
  if(E.size() > 2 || H.size() > 2)
    throw std::runtime_error("MT modelling cannot deal with multi-electrode receivers");

  // Electrode index
  unsigned eidx = 0, hidx = 0;
  if(E.size() == 2 || H.size() == 2) // Intersite measurements?
    hidx = 1; // Take magnetic fields from different location

  cvector TF(n_data_at_point());

  TF[0] = E[eidx][0] / H[hidx][1]; // Zxy
  TF[1] = E[eidx][1] / H[hidx][0]; // Zyx
  TF[2] = H[eidx][2] / H[hidx][1]; // Tzy

  //std::cout << TF[0] << "\t" << TF[1] << "\t" << TF[2] << std::endl;

  const double omega = 2.0 * M_PI * frequency;
  TF[3] = 1. / (mu0 * omega) * pow(abs(TF[0]), 2.); // rho_xy
  TF[4] = 1. / (mu0 * omega) * pow(abs(TF[1]), 2.); // rho_yx
  TF[5] = std::arg(TF[0]) * 180. / M_PI; // phi_xy
  TF[6] = std::arg(TF[1]) * 180. / M_PI; // phi_yx

  TF[7] = E[eidx][0]; TF[8] = E[eidx][1]; TF[9] = E[eidx][2];
  TF[10] = H[hidx][0]; TF[11] = H[hidx][1]; TF[12] = H[hidx][2];

  return TF;
}
