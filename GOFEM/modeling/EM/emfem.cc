#include "emfem.h"

#include <fstream>
#include <iostream>
#include <functional>
#include <chrono>

#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_nedelec_sz.h>

#ifdef SHARED_TRIANGULATION
#include <deal.II/grid/grid_refinement.h>
#endif

#include "mpi/mpi_error.h"
#include "functions/exact_solution.h"
#include "mesh_tools/mesh_tools.h"
#include "survey/dipole_source.h"

void create_petsc_sparse_matrix(PETScWrappers::MPI::SparseMatrix &matrix,
                                const MPI_Comm &communicator,
                                const types::global_dof_index  m,
                                const types::global_dof_index  n,
                                const unsigned int  local_rows,
                                const unsigned int  local_columns,
                                const unsigned int  n_nonzero_per_row,
                                const bool          is_symmetric,
                                const unsigned int  n_offdiag_nonzero_per_row)
{
  Assert(local_rows <= m, ExcLocalRowsTooLarge(local_rows, m));

  Mat& petsc_matrix = matrix.petsc_matrix();

  // use the call sequence indicating only
  // a maximal number of elements per row
  // for all rows globally
  const PetscErrorCode ierr = MatCreateAIJ(communicator,
                                           local_rows,
                                           local_columns,
                                           m,
                                           n,
                                           n_nonzero_per_row,
                                           nullptr,
                                           n_offdiag_nonzero_per_row,
                                           nullptr,
                                           &petsc_matrix);
  PETScWrappers::set_matrix_option(petsc_matrix, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);
  AssertThrow(ierr == 0, ExcPETScError(ierr));

  // set symmetric flag, if so requested
  if (is_symmetric == true)
  {
    PETScWrappers::set_matrix_option(petsc_matrix, MAT_SYMMETRIC, PETSC_TRUE);
  }
}

template <int dim>
std::vector<const FiniteElement<dim>*> create_fe_list (const unsigned int order,
                                                       const FieldFormulation f,
                                                       const bool standard_faces)
{
  std::vector<const FiniteElement<dim>*> fe_list;

  if(dim == 3)
  {
    if(standard_faces)
      fe_list.push_back(new FE_Nedelec<dim> (order));
    else
      fe_list.push_back(new FE_NedelecSZ<dim> (order));

    if(f == EFieldStabilized || f == HFieldStabilized)
      fe_list.push_back(new FE_Q<dim> (order + 1));
  }
  else if(dim == 2)
  {
    // Ex_re, Ex_im, Hx_re, Hx_im
    fe_list.push_back(new FE_Q<dim>(order));
    fe_list.push_back(new FE_Q<dim>(order));
  }
  else
    throw std::runtime_error("Dimensions other than 2 and 3 are not supported.");

  return fe_list;
}

template <int dim>
std::vector<unsigned int> create_fe_multiplicities (const FieldFormulation f)
{
  std::vector<unsigned int> multiplicities;

  if(dim == 3)
  {
    if(f == EFieldStabilized || f == HFieldStabilized)
    {
      multiplicities.push_back(2);
      multiplicities.push_back(2);
    }
    else
      multiplicities.push_back(2);
  }
  else if(dim == 2)
  {
    // Ex_re, Ex_im, Hx_re, Hx_im
    multiplicities.push_back(2);
    multiplicities.push_back(2);
  }
  else
    throw std::runtime_error("Dimensions other than 2 and 3 are not supported.");

  return multiplicities;
}

template<int dim>
EMFEM<dim>::EMFEM (MPI_Comm comm,
                   const unsigned int order,
                   const unsigned int mapping_order,
                   const PhysicalModelPtr<dim> &model,
                   const BackgroundModel<dim> &bg_model,
                   const FieldFormulation formulation,
                   bool face_orientation):
  EMFEM<dim>(comm, order, mapping_order, model, formulation, face_orientation)
{
  background_model = bg_model;
  approach_type = ScatteredField;
}

template<int dim>
EMFEM<dim>::EMFEM (MPI_Comm comm, const unsigned int order,
                   const unsigned int mapping_order,
                   const PhysicalModelPtr<dim> &model,
                   const FieldFormulation formulation,
                   bool face_orientation):
  mpi_communicator (comm),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  #ifdef SHARED_TRIANGULATION
  triangulation (mpi_communicator,
                 Triangulation<dim>::limit_level_difference_at_vertices, true,
                 parallel::shared::Triangulation<dim>::Settings::partition_metis),
  #else
  triangulation (mpi_communicator),
  #endif
  dof_handler(triangulation),
  fe(create_fe_list<dim>(order, formulation, face_orientation),
     create_fe_multiplicities<dim>(formulation)),
  mapping(mapping_order),
  phys_model(model),
  formulation(formulation),
  pcout(std::cout, this_mpi_process == 0)
{
  adaptivity_type = ResidualBased;
  refinement_strategy = Number;
  boundary_conditions = Dirichlet;
  initial_refinements = 0;
  refinement_steps = 3;
  refine_around_receivers = 0;
  max_iterations = 60;
  solver_residual = 1e-8;
  adjoint_residual = 1e-8;
  inner_solver_residual = 1e-2;
  preconditioner_type = Direct;
  ensure_no_coarser_neighbors = false;
  estimate_error_on_last_cycle = true;
  direct_solver_dofs_threshold = 800000;
  n_maximum_dofs = 10000000;
  quadrature_order = fe.degree + 1;
  reuse_data_structures = false;
  fit_mesh_to_topography = false;
  output_format = "vtu";
  mesh_order = 1;
  approach_type = TotalField;
}

template<int dim>
EMFEM<dim>::~EMFEM ()
{
  clear ();
}

template<int dim>
void EMFEM<dim>::clear ()
{
  dof_handler.clear ();
  triangulation.clear ();
  system_rhs.clear();
  solutions.clear();
  system_matrix.clear();
#ifndef SHARED_TRIANGULATION
  local_copy_triangulation.clear();
#endif
}

template<int dim>
void EMFEM<dim>::setup_preconditioner()
{
  //
}

template<int dim>
void EMFEM<dim>::zero_matrices_and_vectors()
{
  system_matrix = 0;

  for(size_t i = 0; i < system_rhs.size(); ++i)
    system_rhs[i] = 0;
  for(size_t i = 0; i < solutions.size(); ++i)
    solutions[i] = 0;
}

template<int dim>
void EMFEM<dim>::assemble_dipole_rhs_vector(const DipoleSource &/*dipole*/,
                                            const AffineConstraints<double> &/*constraints*/,
                                            PETScWrappers::MPI::BlockVector &/*rhs_vector*/)
{
  throw std::runtime_error("assemble_dipole_rhs_vector: not implemented.");
}

template<int dim>
void EMFEM<dim>::post_solve()
{
  return;
}

template<int dim>
void EMFEM<dim>::output_specific_information(std::vector<std::shared_ptr<DataPostprocessor<dim>>> &/*data*/,
                                             DataOut<dim> &/*data_out*/) const
{
  return;
}

template<int dim>
void EMFEM<dim>::output_surface_data(unsigned /*cycle*/) const
{
  return;
}

template<int dim>
void EMFEM<dim>::create_grid (const unsigned int cycle)
{
  if (cycle == 0)
  {
    if(triangulation.n_levels() == 0)
      setup_initial_triangulation(phys_model, triangulation);
    else // mesh has been loaded externally (e.g. when inversion is done)
      set_boundary_indicators(triangulation);

    pcout << "  Refine grid ..." << std::endl;
    triangulation.refine_global (initial_refinements);

    if (refine_around_receivers > 0)
    {
#ifdef SHARED_TRIANGULATION
      pcout << "  Refine grid around receivers "
            << refine_around_receivers
            << " times" << std::endl;

      std::vector<Point<dim>> points = unique_point_receiver_positions;
      for(size_t i = 0; i < physical_sources.size(); ++i)
      {
        Point<dim> p;
        if(DipoleSource* src = dynamic_cast<DipoleSource*>(physical_sources[i].get()))
        {
          for(unsigned n = 0; n < src->n_dipole_elements(); ++n)
          {
            src->position(p, dim, n);
            points.push_back(p);
          }
        }
      }

      for (unsigned i = 0; i < refine_around_receivers; ++i)
        MyGridTools::refine_grid_around_points_cache (triangulation, mapping, points);
#else
      throw std::runtime_error("Refinement around receivers for the fully distributed meshes is not supported.");
#endif
    }
  }
  else
  {
    pcout << "  Refine grid ..." << std::endl;

    switch (adaptivity_type)
    {
    case ResidualBased:
    case GoalOriented:
    {
      if(theta < 1.)
      {
        for(size_t n = 0; n < n_physical_sources; ++n)
        {
#ifdef SHARED_TRIANGULATION
          const unsigned int n_local_cells = triangulation.n_locally_owned_active_cells ();
          PETScWrappers::MPI::Vector distributed_error_per_cell (mpi_communicator,
                                                                 triangulation.n_active_cells(),
                                                                 n_local_cells);

          for (unsigned int i = 0; i < estimated_cell_errors[n].size(); ++i)
            if (estimated_cell_errors[n](i) != 0)
              distributed_error_per_cell(i) = estimated_cell_errors[n](i);

          distributed_error_per_cell.compress (VectorOperation::insert);
          estimated_cell_errors[n] = distributed_error_per_cell;

          if(refinement_strategy == Fraction)
            GridRefinement::refine_and_coarsen_fixed_fraction (triangulation, estimated_cell_errors[n], theta, 0);
          else if(refinement_strategy == Number)
            GridRefinement::refine_and_coarsen_fixed_number (triangulation, estimated_cell_errors[n], theta, 0);
#else
          throw std::runtime_error("Adaptive refinement for the fully distributed meshes is not supported.");
#endif
        }

        triangulation.prepare_coarsening_and_refinement ();
        triangulation.execute_coarsening_and_refinement ();
      }
      else
      {
        triangulation.refine_global (1);
      }
      break;
    }
    case Global:
    default:
      triangulation.refine_global (1);
    }
  }

  if(ensure_no_coarser_neighbors)
  {
#ifdef SHARED_TRIANGULATION
    bool coarser_neighbors_exist;
    unsigned count = 0;

    pcout << "  Make sure there are no coarser neighbors around receiver cells.\n  Cycle #" << std::endl;
    // we really need to make sure that there are no coarser neighbors hence the loop
    do
    {
      coarser_neighbors_exist =
          MyGridTools::refine_coarser_neighbours_around_receivers_serial_cache
          (triangulation, mapping, unique_point_receiver_positions);

      pcout << count++ << "  ";

      if(count > 16)
        break;
    }
    while (coarser_neighbors_exist);

    pcout << std::endl;
#else
    throw std::runtime_error("Refinement around receivers for the fully distributed meshes is not supported.");
#endif
  }

  // Set physical properties from parents to the newly created children cells
  phys_model->set_cell_properties_to_children (triangulation);

  pcout << "  Active cells: " << triangulation.n_global_active_cells () << std::endl;

  double local_min_cell_diameter = GridTools::minimal_cell_diameter(triangulation);
  pcout << "  Minimal cell diameter: "
        << Utilities::MPI::min(local_min_cell_diameter, mpi_communicator)
        << " m" << std::endl;
}


template<int dim>
void EMFEM<dim>::setup_initial_triangulation(PhysicalModelPtr<dim> &model,
                                             Triangulation<dim> &tria)
{
  pcout << "  Create grid ..." << std::endl;
  model->create_triangulation (tria);

  set_boundary_indicators(tria);
}

template<int dim>
void EMFEM<dim>::set_boundary_indicators(Triangulation<dim> &tria)
{
  // Go through all cells, not only active!
  for (typename Triangulation<dim>::cell_iterator cell = tria.begin();
       cell != tria.end(); ++cell)
  {
    for(unsigned f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
    {
      auto face = cell->face(f);

      if ((face->at_boundary() && face->boundary_id() == numbers::invalid_boundary_id) ||
          face->boundary_id() == 0)
      {
        types::boundary_id bid = numbers::invalid_boundary_id;

        if(dim == 3)
        {
          if(f == 0)
            bid = south_boundary_id;
          else if(f == 1)
            bid = north_boundary_id;
          else if(f == 2)
            bid = west_boundary_id;
          else if(f == 3)
            bid = east_boundary_id;
          else if(f == 4)
            bid = top_boundary_id;
          else if(f == 5)
            bid = bottom_boundary_id;
        }
        else if(dim == 2)
        {
          if(f == 0)
            bid = west_boundary_id;
          else if(f == 1)
            bid = east_boundary_id;
          else if(f == 2)
            bid = top_boundary_id;
          else if(f == 3)
            bid = bottom_boundary_id;
        }

        // Set boundary indicator to the face
        face->set_boundary_id (bid);
      }
    }
  }
}

template<int dim>
double EMFEM<dim>::get_wavenumber() const
{
  throw std::runtime_error("This class has no wave number.");
}

template<int dim>
void EMFEM<dim>::run ()
{
  pcout << fe.get_name() << std::endl;
  pcout << "Mapping degree = " << mapping.get_degree() << std::endl;

  if(physical_sources.size() == 0 || (adaptivity_type == GoalOriented && physical_sources.size() == 1))
    throw std::runtime_error("No physical sources were specified.");

  error_estimates = std::vector<std::vector<double> > (n_physical_sources);

  unsigned cycle = 0;
  std::vector<double> error_reduction(n_physical_sources, 0.);

  Timer timer_all;
  timer_all.start ();

  while (cycle <= refinement_steps)
  {
    pcout << "Running cycle #" << cycle << std::endl;

    Timer timer;
    timer.start ();

    // !!!!! Note this may not work for inversion!
    if(dof_handler.n_dofs() == 0 || reuse_data_structures == false)
    {
      create_grid (cycle);
      pcout << "  Grid refinement time: " << timer.wall_time () << std::endl;
      timer.restart ();

      if(dof_handler.n_dofs() >= n_maximum_dofs)
      {
        pcout << "  Current number of DoFs " << dof_handler.n_dofs()
              << " is larger than the allocated DoFs budget " << n_maximum_dofs
              << "\n";
        break;
      }

      setup_system (adaptivity_type == GoalOriented ? n_physical_sources + 1 : n_physical_sources);
      pcout << "  System setup time: " << timer.wall_time () << std::endl;
      timer.restart ();
    }
    else
      zero_matrices_and_vectors();

    assemble_system_matrix ();
    pcout << "  System matrix assembling time: " << timer.wall_time () << std::endl;
    timer.restart ();

    assemble_problem_rhs ();
    pcout << "  Right-hand side vectors assembling time: " << timer.wall_time () << std::endl;
    timer.restart ();

    setup_preconditioner ();
    pcout << "  Setting up preconditioner time: " << timer.wall_time () << std::endl;
    timer.restart ();

    solve (solutions, system_rhs, solution_constraints_indices);
    pcout << "  Solution time: " << timer.wall_time () << std::endl;
    timer.restart ();

    post_solve();

    // Possibly skip error estimation at the last reinement iteration to save time
    if(/*cycle < refinement_steps ||*/
       estimate_error_on_last_cycle == true)
    {
      estimate_error ();
      pcout << "  Error estimation time: " << timer.wall_time () << std::endl;
      timer.restart ();
    }

    output_results (cycle);
    pcout << "  Output time: " << timer.wall_time () << std::endl;
    timer.restart ();

    bool has_converged = false;

    // Check whether errors for all sources dropped sufficiently
    if(cycle > 0)
    {
      has_converged = true;
      for(unsigned i = 0; i < n_physical_sources; ++i)
      {
        error_reduction[i] = error_estimates[i][0] / error_estimates[i].back();
        if(error_reduction[i] < target_error_reduction)
          has_converged = false;
      }
    }
    else
    {
      has_converged = true;
      for(unsigned i = 0; i < n_physical_sources; ++i)
      {
        if(error_estimates.size() > 0 &&
           error_estimates[i].size() > 0 &&
           error_estimates[i][0] > 1e-10)
          has_converged = false;
      }
    }

    pcout << "Done cycle: " << cycle++ << std::endl;

    if(has_converged)
      break;
  }

  pcout << "Total modeling time: " << timer_all.wall_time() << std::endl;
}

template<int dim>
BackgroundModel<dim> EMFEM<dim>::get_boundary_model(const PhysicalModelPtr<dim> &model,
                                                    const Triangulation<dim> &tria,
                                                    const types::boundary_id bid) const
{
  std::map<double, typename Triangulation<dim>::active_cell_iterator> cells_at_boundary;

  for (const auto &cell: tria.active_cell_iterators())
  {
    for(unsigned f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
    {
      if (cell->face(f)->boundary_id() == bid)
      {
        double depth = std::min(cell->face(f)->vertex(0)[dim-1],
            cell->face(f)->vertex(1)[dim-1]);
        cells_at_boundary.insert(std::make_pair(depth, cell));
      }
    }
  }

  dvector depths;
  std::vector<Material> materials;

  for(const auto& p: cells_at_boundary)
  {
    depths.push_back(p.first);
    materials.push_back(model->cell_material(p.second));
  }

  // Make sure the last layer goes beyond the model
  depths[depths.size() - 1] = depths.back()*2.;

  return BackgroundModel<dim>(depths, materials);
}

template<int dim>
unsigned EMFEM<dim>::get_source_index(const std::string &name) const
{
  unsigned sidx = std::numeric_limits<unsigned>::max();

  for(unsigned i = 0; i < physical_sources.size(); ++i)
  {
    bool found = physical_sources[i]->get_name() == name;

    if(found)
    {
      sidx = i;
      break;
    }
  }

  if(sidx == std::numeric_limits<unsigned>::max())
    throw std::runtime_error("Could not find source " + name);

  return sidx;
}

template<int dim>
unsigned EMFEM<dim>::get_receiver_index(const std::string &name) const
{
  unsigned ridx = std::numeric_limits<unsigned>::max();

  for(unsigned i = 0; i < receivers.size(); ++i)
  {
    bool found = receivers[i].get_name() == name;

    if(found)
    {
      ridx = i;
      break;
    }
  }

  if(ridx == std::numeric_limits<unsigned>::max())
    throw std::runtime_error("Could not find receiver " + name);

  return ridx;
}

template<int dim>
void EMFEM<dim>::print_memory_consumption() const
{
  size_t tria_memory = triangulation.memory_consumption();
  size_t dh_memory = dof_handler.memory_consumption();

  size_t vectors_memory = 0;
  for(auto &v: solutions)
    vectors_memory += v.memory_consumption();

  for(auto &v: system_rhs)
    vectors_memory += v.memory_consumption();

  vectors_memory += completely_distributed_solution.memory_consumption();

  size_t constraints_memory = 0;
  for(auto &cm: constraints)
    constraints_memory += cm.memory_consumption();

  size_t matrix_memory = 0;
  for(unsigned i = 0; i < system_matrix.n_block_rows(); ++i)
    for(unsigned j = 0; j < system_matrix.n_block_cols(); ++j)
      matrix_memory += system_matrix.block(i, j).memory_consumption();

  auto matrix_memories = Utilities::MPI::all_gather(mpi_communicator, matrix_memory);
  auto tria_memories = Utilities::MPI::all_gather(mpi_communicator, tria_memory);
  auto dh_memories = Utilities::MPI::all_gather(mpi_communicator, dh_memory);
  auto vec_memories = Utilities::MPI::all_gather(mpi_communicator, vectors_memory);
  auto cm_memories = Utilities::MPI::all_gather(mpi_communicator, constraints_memory);

  if(this_mpi_process == 0)
  {
    unsigned n_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_communicator);
    std::ofstream ofs(output_data_file + "_memory_p=" + std::to_string(n_mpi_processes));
    ofs << "  Memory consumption in bytes:\n  #proc\tTria\tDH\tVectors\tConstraints\tMatrix\n";
    for(unsigned i = 0; i < n_mpi_processes; ++i)
      ofs << "  " << i << "\t"
          << tria_memories[i] << "\t"
          << dh_memories[i] << "\t"
          << vec_memories[i] << "\t"
          << cm_memories[i] << "\t"
          << matrix_memories[i] << std::endl;
  }
}

// ------------------ SET PARAMETERS -------------------------- //
template<int dim>
void EMFEM<dim>::set_frequency (double f)
{
  frequency = f;
}

template<int dim>
void EMFEM<dim>::set_no_coarser_neighbors_around_receivers(bool f)
{
  ensure_no_coarser_neighbors = f;
}

template<int dim>
void EMFEM<dim>::set_estimate_error_on_last_cycle(bool f)
{
  estimate_error_on_last_cycle = f;
}

template<int dim>
void EMFEM<dim>::set_boundary_conditions_id (const std::string& id)
{
  if (id == "Dirichlet")
    boundary_conditions = Dirichlet;

  else if (id == "Neumann")
    boundary_conditions = Neumann;

  else if (id == "Silver-Mueller")
    boundary_conditions = SilverMueller;

  else if (id == "Dirichlet2D")
    boundary_conditions = Dirichlet2D;

  else
    ExcNotImplemented ();
}

template<int dim>
void EMFEM<dim>::set_adaptivity_type_id (const std::string& id)
{
  if (id == "global")
    adaptivity_type = Global;

  else if (id == "residual")
    adaptivity_type = ResidualBased;

  else if (id == "goal-oriented")
  {
    adaptivity_type = GoalOriented;

    if(receivers.size() > 0)
      throw std::runtime_error("You set receivers before adaptivity strategy, this is not allowed.");

    /*
     * Goal-oriented dual source is a sum of delta-functions
     * at receivers' locations
     */
    if (physical_sources.size() == n_physical_sources)
      physical_sources.push_back(PhysicalSourcePtr(new DipoleSource("Dual source", ElectricDipole)));
    else
      throw std::runtime_error("set_adaptivity_type_id(): Something went wrong with source structures.");
  }
  else
    throw std::runtime_error("Unknown adaptivity strategy.");

  direct_solver_dofs_threshold = (adaptivity_type == GoalOriented) ? 1000000 : 800000;
}

template<int dim>
void EMFEM<dim>::set_refinement_strategy_id(const std::string &id)
{
  if (id == "Number")
    refinement_strategy = Number;
  else if (id == "Fraction")
    refinement_strategy = Fraction;
  else
    ExcNotImplemented ();
}

template<int dim>
void EMFEM<dim>::set_theta (const double t)
{
  theta = t;
}

template<int dim>
void EMFEM<dim>::set_initial_refinements (const unsigned n)
{
  initial_refinements = n;
}

template<int dim>
void EMFEM<dim>::set_refinement_steps (const unsigned n)
{
  refinement_steps = n;

  if(refinement_steps == 0)
    estimate_error_on_last_cycle = false;
}

template<int dim>
void EMFEM<dim>::set_target_error_reduction (const double n)
{
  target_error_reduction = n;
}

template<int dim>
void EMFEM<dim>::set_dofs_budget(const unsigned n)
{
  n_maximum_dofs = n;
}

template<int dim>
void EMFEM<dim>::set_output_type (const std::string type)
{
  output_type = type;
  boost::algorithm::to_lower(output_type);
}

template<int dim>
void EMFEM<dim>::set_output_format (const std::string format)
{
  if(boost::iequals(format, "vtu") ||
     boost::iequals(format, "vtk"))
    output_format = format;
  else
    throw std::runtime_error("Only VTU and VTK formats are supported.");
}

template<int dim>
void EMFEM<dim>::set_output_mesh_order(const unsigned order)
{
  mesh_order = order;
}

template<int dim>
void EMFEM<dim>::set_parallel_output(bool f)
{
  parallel_output = f;
}

template<int dim>
void EMFEM<dim>::set_receivers (const std::vector<Receiver> &recs)
{
  receivers = recs;

  const size_t shift = dim == 2 ? 1 : 0;

  cvec3d unit_current = {1., 1., 1.};

  std::vector<Point<dim>> positions;
  for(auto &r: receivers)
  {
    for(unsigned i = 0; i < r.n_electrodes(); ++i)
    {
      dvec3d p3d = r.position<dvec3d>(i);

      Point<dim> p;
      for(unsigned d = 0; d < dim; ++d)
        p[d] = p3d[d + shift];

      positions.push_back(p);

      if(adaptivity_type == GoalOriented)
      {
        DipoleSource &dual_dipole = dynamic_cast<DipoleSource&>(*physical_sources.back());
        dual_dipole.add_dipole_element(p3d, {1., 1., 1.}, unit_current);
      }
    }
  }

  get_unique_electrodes(positions, unique_point_receiver_positions, 1.);

  if(unique_point_receiver_positions.size() < receivers.size())
    throw std::runtime_error("Number of unique measurement positions is smaller than number of receivers. Duplicated receivers?");
}

template<int dim>
void EMFEM<dim>::set_physical_sources(const std::vector<PhysicalSourcePtr> &sources)
{
  if(sources.size() == 0)
    return;

  n_physical_sources = sources.size();
  error_estimates.resize (n_physical_sources);
  physical_sources = sources;

  if(adaptivity_type == GoalOriented)
    physical_sources.push_back(PhysicalSourcePtr(new DipoleSource("DualSource", ElectricDipole)));
}

template<int dim>
void EMFEM<dim>::set_data_map(const SrcRecMap &data_mapping)
{
  if(physical_sources.size() == 0)
  {
    throw std::runtime_error("You set source-receiver map to an object without sources. "
                             "Make sure you set sources first.");
  }

  if(data_map.size() > 0)
  {
    // Check that all sources and receivers are present
    data_map = data_mapping;

    for(auto &p: data_map)
    {
      const std::string sname = p.first;

      const std::vector<std::string> &rnames = p.second;
      for(const std::string &rname: rnames)
      {
        get_source_index(sname);
        get_receiver_index(rname);
      }
    }
  }
  else
  {
    std::vector<std::string> receiver_names;
    for(auto &rec: receivers)
      receiver_names.push_back(rec.get_name());

    // In case no source-receiver file was specified, get data at all receivers
    for(unsigned i = 0; i < n_physical_sources; ++i)
      data_map[physical_sources[i]->get_name()] = receiver_names;
  }
}

template<int dim>
void EMFEM<dim>::set_output_file (const std::string fname)
{
  output_data_file = fname;
}

template<int dim>
void EMFEM<dim>::set_refinement_around_receivers (const unsigned cycles)
{
  refine_around_receivers = cycles;
}

template<int dim>
void EMFEM<dim>::set_preconditioner_type (const std::string preconditioner)
{
  auto it = solver_type_conversion.find(preconditioner);
  if(it == solver_type_conversion.end())
    throw std::runtime_error("Unknown solver type " + preconditioner);

  preconditioner_type = it->second;
}

template<int dim>
void EMFEM<dim>::set_maximum_solver_iterations (const unsigned max_it)
{
  max_iterations = max_it;
}

template<int dim>
void EMFEM<dim>::set_solver_residual (const double residual)
{
  solver_residual = residual;
}

template<int dim>
void EMFEM<dim>::set_adjoint_solver_residual(double residual)
{
  adjoint_residual = residual;
}

template<int dim>
void EMFEM<dim>::set_inner_solver_residual (const double residual)
{
  inner_solver_residual = residual;
}

template<int dim>
void EMFEM<dim>::set_inner_solver_maximum_iterations(const unsigned max_it)
{
  max_inner_iterations = max_it;
}

template<int dim>
void EMFEM<dim>::set_perform_topography_adjustment(bool f)
{
  fit_mesh_to_topography = f;
}

template<int dim>
const SrcRecMap &EMFEM<dim>::get_data_map() const
{
  return data_map;
}

template<int dim>
void EMFEM<dim>::get_survey_data(ModelledData<dcomplex> &modelled_data) const
{
  std::vector<cvector> data = data_at_receivers();
  construct_survey_data(data, modelled_data);
}

template<int dim>
PhysicalSourcePtr EMFEM<dim>::get_physical_source(const std::string &srcname)
{
  const unsigned idx = get_source_index(srcname);
  return physical_sources[idx];
}

template<int dim>
void EMFEM<dim>::set_verbosity(bool flag)
{
  if(flag)
    pcout.set_condition(this_mpi_process == 0);
  else
    pcout.set_condition(false);
}

template<int dim>
void EMFEM<dim>::set_reuse_data_structures(bool f)
{
  reuse_data_structures = f;
}

#ifndef SHARED_TRIANGULATION
template<int dim>
void EMFEM<dim>::set_local_model(const PhysicalModelPtr<dim> &model)
{
  local_copy_model = model;
}
#endif

template <int dim>
void EMFEM<dim>::get_unique_electrodes(const std::vector<Point<dim>> &positions,
                                       std::vector<Point<dim>> &unique_positions,
                                       double min_distance)
{
  unique_positions.clear();

  for(auto &p1: positions)
  {
    bool exists = false;

    for(auto &p2: unique_positions)
      if(p2.distance(p1) < min_distance)
      {
        exists = true;
        break;
      }

    if(!exists)
      unique_positions.push_back(p1);
    //    else
    //      std::cout << p1 << std::endl;
  }
}

template<int dim>
std::vector<cvector> EMFEM<dim>::data_at_receivers () const
{
  std::vector<cvector> derived_data_values;

  if (receivers.size () == 0)
    return derived_data_values;

  std::map<std::string, std::vector<cvector>> E, H;
  collect_fields_on_root_cache(E, H);
  //collect_fields_on_root(E, H);

  if (this_mpi_process == 0)
  {
    derived_data_values.resize(receivers.size());

    for(unsigned i = 0; i < receivers.size(); ++i)
    {
      derived_data_values[i] = calculate_data_at_receiver(E[receivers[i].get_name()],
          H[receivers[i].get_name()]);
    }
  }

  return derived_data_values;
}

template<int dim>
void EMFEM<dim>::collect_fields_on_root_cache(StringVectorMap &E, StringVectorMap &H) const
{
  const size_t shift = dim == 2 ? 1 : 0;

  // Electric and magnetic fields calculated locally at a single location
  cvector E_local(n_physical_sources * 3), H_local(n_physical_sources * 3);

  std::vector<std::vector<Point<dim>>> receiver_locations(receivers.size());
  for (size_t i = 0; i < receivers.size(); ++i)
  {
    const auto& receiver = receivers[i];

    std::vector<Point<dim>> positions(receiver.n_electrodes());
    for(unsigned j = 0; j < receiver.n_electrodes(); ++j)
    {
      dvec3d p3d = receiver.position<dvec3d>(j);
      for(unsigned d = 0; d < dim; ++d)
        positions[j][d] = p3d[d + shift];
    }

    receiver_locations[i] = positions;
  }
  auto receiver_cells = find_all_receiver_positions_cells(receiver_locations);

  // Calculate data at specified locations
  for (size_t i = 0; i < receivers.size(); ++i)
  {
    for(unsigned j = 0; j < receivers[i].n_electrodes(); ++j)
    {
      if (receiver_cells[i][j]->state() != IteratorState::invalid) // Are there locally owned cells for this point?
      {
        auto cell_point = receiver_cells[i][j];

        //        std::cout << this_mpi_process << "\t" << rname << "\t" << j << "\t" << positions[j] << std::endl;

        typename DoFHandler<dim>::active_cell_iterator fe_cell(&triangulation,
                                                               cell_point->level(),
                                                               cell_point->index(),
                                                               &dof_handler);

        field_at_point(fe_cell, receiver_locations[i][j], E_local, H_local);
        if(this_mpi_process == 0)
        {
          E[receivers[i].get_name()].push_back(E_local);
          H[receivers[i].get_name()].push_back(H_local);
        }
        else
        {
          MPI_Ssend (&i, 1, MPI_UNSIGNED_LONG, 0, 444, mpi_communicator);
          MPI_Ssend (&E_local[0], E_local.size() * 2, MPI_DOUBLE_PRECISION, 0, 555, mpi_communicator);
          MPI_Ssend (&H_local[0], H_local.size() * 2, MPI_DOUBLE_PRECISION, 0, 555, mpi_communicator);
        }
      }
      else
      {
        if(this_mpi_process == 0)
        {
          MPI_Status status;
          size_t idx;
          MPI_Recv (&idx, 1, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE, 444, mpi_communicator, &status);
          MPI_Recv (&E_local[0], E_local.size() * 2, MPI_DOUBLE_PRECISION, MPI_ANY_SOURCE, 555, mpi_communicator, &status);
          MPI_Recv (&H_local[0], H_local.size() * 2, MPI_DOUBLE_PRECISION, MPI_ANY_SOURCE, 555, mpi_communicator, &status);
          E[receivers[idx].get_name()].push_back(E_local);
          H[receivers[idx].get_name()].push_back(H_local);

          //std::cout << "Received data with #" << idx+1 << " for " << receiver_names[i] << std::endl;
        }
      }
    }
  }
}

template<int dim>
std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator> >
EMFEM<dim>::find_all_receiver_positions_cells(const std::vector<std::vector<Point<dim> > > &receiver_locations) const
{
  std::vector<Point<dim>> all_receiver_points;

  // Collect all points into a single vector
  for (const auto &rlocations: receiver_locations)
    std::copy(rlocations.begin(), rlocations.end(),
              std::back_inserter(all_receiver_points));

  // Find all points using an efficient algorithm
  GridTools::Cache<dim> cache(triangulation, mapping);
  IteratorFilters::LocallyOwnedCell locally_owned_cell_predicate;
  std::vector<BoundingBox<dim>>     local_bbox =
      GridTools::compute_mesh_predicate_bounding_box(
        cache.get_triangulation(),
        std::function<bool(
          const typename Triangulation<dim>::active_cell_iterator &)>(
          locally_owned_cell_predicate));

  // Obtaining the global mesh description through an all to all communication
  auto global_bboxes = GridTools::exchange_local_bounding_boxes(local_bbox,
                                                                triangulation.get_communicator());

  // Using the distributed version of compute point location
  auto output_tuple =
      GridTools::distributed_compute_point_locations(cache, all_receiver_points, global_bboxes);

  const auto &local_cell_iterators = std::get<0>(output_tuple);
  const auto &local_points = std::get<3>(output_tuple);

  const double dist_threshold = 1e-2;

  unsigned n_points_found = 0;

  // Fill in receiver location cells
  std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator> > receiver_cells(receiver_locations.size());
  for (size_t ridx = 0; ridx < receiver_locations.size(); ++ridx)
  {
    std::vector<typename Triangulation<dim>::active_cell_iterator> cell_iterators(receiver_locations[ridx].size());

    // Loop over receiver points (there can be more than one point per receiver -- e.g. when inter-site TFs are needed)
    for(size_t eidx = 0; eidx < receiver_locations[ridx].size(); ++eidx)
    {
      // Search for a cell which contains the current point
      for(size_t pidx = 0; pidx < local_points.size(); ++pidx)
      {
        for(size_t j = 0; j < local_points[pidx].size(); ++j)
        {
          if(receiver_locations[ridx][eidx].distance(local_points[pidx][j]) < dist_threshold)
          {
            cell_iterators[eidx] = local_cell_iterators[pidx];
            ++n_points_found;
            break;
          }
        }
      }
    }

    receiver_cells[ridx] = cell_iterators;
  }

  const unsigned n_points_found_all = Utilities::MPI::sum<unsigned>(n_points_found, mpi_communicator);
  if(n_points_found_all != all_receiver_points.size())
    throw std::runtime_error("Not all cells for receiver points were found.");

  return receiver_cells;
}

template<int dim>
void EMFEM<dim>::collect_fields_on_root(StringVectorMap &E, StringVectorMap &H) const
{
  const size_t shift = dim == 2 ? 1 : 0;

  // Electric and magnetic fields calculated locally at a single location
  cvector E_local(n_physical_sources * 3), H_local(n_physical_sources * 3);

  // Calculate data at specified locations
  for (size_t i = 0; i < receivers.size(); ++i)
  {
    const auto& receiver = receivers[i];

    std::vector<Point<dim>> positions(receiver.n_electrodes());
    for(unsigned j = 0; j < receiver.n_electrodes(); ++j)
    {
      dvec3d p3d = receiver.position<dvec3d>(j);
      for(unsigned d = 0; d < dim; ++d)
        positions[j][d] = p3d[d + shift];
    }

    for(unsigned j = 0; j < positions.size(); ++j)
    {
      // In parallel, it is possible that some processes will not find
      // a cell, but the one that owns it locally should find it correctly
      bool cell_found = false;
      auto cell_point = GridTools::find_active_cell_around_point (mapping, dof_handler, positions[j]);
      if (cell_point.first != dof_handler.end())
        cell_found = true;

      // In some rare situations find_active_cell_around_point can
      // return a cell that is not locally owned on any of processes
      // or more than one process owns point (usually because receiver
      // is located exactly at the vertex/edge).
      unsigned locally_owned = (cell_found ? cell_point.first->is_locally_owned () : 0);
      auto locally_owned_global = Utilities::MPI::sum(locally_owned, mpi_communicator);

      if(locally_owned_global > 1)
      {
        throw std::runtime_error("Cell containing receiver " + receiver.get_name() +
                                 " is not uniquely determined. This may happen if a receiver is"
                                 " located exactly at the vertex/edge of a cell.");
      }
      else if(locally_owned_global == 0)
      {
        pcout << "No active cell containing receiver " << receiver.get_name() << " with coordinates " << positions[j] << " was found." << std::endl;

        std::fill(E_local.begin(), E_local.end(), 0);
        std::fill(H_local.begin(), H_local.end(), 0);

        if(this_mpi_process == 0)
        {
          E[receiver.get_name()].push_back(E_local);
          H[receiver.get_name()].push_back(H_local);
        }
        continue;
      }

      if (cell_found && cell_point.first->is_locally_owned ())
      {
        //        std::cout << this_mpi_process << "\t" << rname << "\t" << j << "\t" << positions[j] << std::endl;

        field_at_point(cell_point.first, positions[j], E_local, H_local);
        if(this_mpi_process == 0)
        {
          E[receiver.get_name()].push_back(E_local);
          H[receiver.get_name()].push_back(H_local);
        }
        else
        {
          MPI_Ssend (&i, 1, MPI_UNSIGNED_LONG, 0, 444, mpi_communicator);
          MPI_Ssend (&E_local[0], E_local.size() * 2, MPI_DOUBLE_PRECISION, 0, 555, mpi_communicator);
          MPI_Ssend (&H_local[0], H_local.size() * 2, MPI_DOUBLE_PRECISION, 0, 555, mpi_communicator);
        }
      }
      else
      {
        if(this_mpi_process == 0)
        {
          MPI_Status status;
          size_t idx;
          MPI_Recv (&idx, 1, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE, 444, mpi_communicator, &status);
          MPI_Recv (&E_local[0], E_local.size() * 2, MPI_DOUBLE_PRECISION, MPI_ANY_SOURCE, 555, mpi_communicator, &status);
          MPI_Recv (&H_local[0], H_local.size() * 2, MPI_DOUBLE_PRECISION, MPI_ANY_SOURCE, 555, mpi_communicator, &status);
          E[receiver.get_name()].push_back(E_local);
          H[receiver.get_name()].push_back(H_local);

          //std::cout << "Received data with #" << idx+1 << " for " << receiver_names[i] << std::endl;
        }
      }
    }
  }
}

template<int dim>
void EMFEM<dim>::construct_survey_data(const std::vector<cvector> &data, ModelledData<dcomplex> &modelled_data) const
{
  unsigned data_components = n_data_at_point();
  cvector receiver_data(data_components);

  for(auto &p: data_map)
  {
    const std::string source_name = p.first;
    const unsigned sidx = get_source_index(source_name);

    for(const std::string receiver_name: p.second)
    {
      unsigned ridx = get_receiver_index(receiver_name);

      for(unsigned component = 0; component < data_components; ++component)
        receiver_data[component] = data[ridx][data_components*sidx + component];

      modelled_data.set(source_name, receiver_name, receiver_data);
    }
  }
}

template<int dim>
void EMFEM<dim>::output_point_data (const unsigned int cycle) const
{
  output_point_data(data_at_receivers (), cycle);
}

template<int dim>
void EMFEM<dim>::output_point_data (const std::vector<cvector> &data, const unsigned int cycle) const
{
  // Only master process outputs data
  if (this_mpi_process != 0)
    return;

  std::ofstream ofs;
  ofs << std::setprecision(10);

  std::vector<char> final_name(output_data_file.size() + 150);
  sprintf(final_name.data(), "%s_c=%i_f=%.8e.txt", output_data_file.c_str(), cycle, frequency);

  ofs.open(final_name.data());
  ofs << "Source\tReceiver\t" << data_header () << "\n";

  if(data.size() == 0)
    return;

  unsigned ndata = n_data_at_point();

  for (size_t j = 0; j < n_physical_sources; ++j)
  {
    // For 3D MT: Skip one of the source polarization since
    // the data is recorded for the other one
    if(physical_sources[j]->get_name() == "Plane_wave_EW" ||
       physical_sources[j]->get_name() == "Uniform_P10")
      continue;

    for (size_t i = 0; i < data.size (); ++i)
    {
      ofs << physical_sources[j]->get_name() << "\t" << receivers[i].get_name();

      cvector::const_iterator it = data[i].begin () + j*ndata;
      for(unsigned n = 0; n < ndata; ++n, ++it)
      {
        const dcomplex val = *it;
        ofs << "\t" << val;
      }
      ofs << "\n";
    }
  }

  ofs.close ();
}

template<int dim>
void EMFEM<dim>::output_results (const unsigned int cycle) const
{
  if(output_type.find("volume") != std::string::npos)
    output_volume_data(cycle);

  if(output_type.find("surface") != std::string::npos)
    output_surface_data(cycle);

  if(output_type.find("point") != std::string::npos)
    output_point_data (cycle);
}

template<int dim>
void EMFEM<dim>::output_volume_data(const unsigned int cycle) const
{
  auto vector_part = DataComponentInterpretation::component_is_part_of_vector;
  auto scalar_part = DataComponentInterpretation::component_is_scalar;
  std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation;

  if(dim == 3)
    data_component_interpretation.resize(dim*2, vector_part);
  else
    data_component_interpretation.resize(dim*2, scalar_part);

  std::vector<std::shared_ptr<DataPostprocessor<dim>>> vectors;

  DataOutBase::VtkFlags flags;
  if(mesh_order > 1)
    flags.write_higher_order_cells = true;
  else
    flags.write_higher_order_cells = false;

  DataOut<dim> data_out;
  data_out.set_flags(flags);
  data_out.attach_dof_handler (dof_handler);

  // Solution and error estimates for both polarizations
  for (unsigned int i = 0; i < n_physical_sources; ++i)
  {
    std::vector<std::string> solution_names;

    if(dim == 3)
    {
      solution_names.resize(dim, physical_sources[i]->get_name() + "_real");
      for(unsigned d = 0; d < dim; ++d)
        solution_names.push_back (physical_sources[i]->get_name() + "_imag");
    }
    else if(dim == 2)
    {
      auto src = physical_sources[i]->get_name();
      solution_names = {src + "_Ex_real", src + "_Ex_imag",
                        src + "_Hx_real", src + "_Hx_imag"};
    }

    data_out.add_data_vector (solutions[i], solution_names,
                              DataOut<dim>::type_dof_data,
                              data_component_interpretation);

    if(estimated_cell_errors.size() > i && estimated_cell_errors[i].size() == triangulation.n_active_cells())
      data_out.add_data_vector (estimated_cell_errors[i], "error_" + physical_sources[i]->get_name());

    ghosted_rhs_vectors[i] = system_rhs[i];
    data_out.add_data_vector(ghosted_rhs_vectors[i], "rhs_" + physical_sources[i]->get_name()); // works for serial runs only
  }

  if(adaptivity_type == GoalOriented)
  {
    std::vector<std::string> solution_names;

    if(dim == 3)
    {
      solution_names.resize(dim, "Dual_real");
      for(unsigned d = 0; d < dim; ++d)
        solution_names.push_back ("Dual_imag");
    }
    else if(dim == 2)
    {
      solution_names = {"Dual_Ex_real", "Dual_Ex_imag",
                        "Dual_Hx_real", "Dual_Hx_imag"};
    }

    data_out.add_data_vector (solutions[n_physical_sources], solution_names,
                              DataOut<dim>::type_dof_data,
                              data_component_interpretation);

    if(estimated_cell_errors.size() > n_physical_sources)
      data_out.add_data_vector (estimated_cell_errors[n_physical_sources], "error_dual");
  }

  output_specific_information(vectors, data_out);

  // Domain id
  Vector<float> subdomain;
  if(Utilities::MPI::n_mpi_processes(mpi_communicator) > 1)
  {
    subdomain.reinit (triangulation.n_active_cells ());
    for (unsigned int i=0; i < subdomain.size (); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain ();
    data_out.add_data_vector (subdomain, "subdomain");
  }

  // Conductivity
  Vector<float> conductivity (triangulation.n_active_cells ());
  unsigned int cell_index = 0;
  for (const auto& cell: dof_handler.active_cell_iterators ())
  {
    if (cell->is_locally_owned ())
      conductivity(cell_index) = phys_model->conductivity_at (cell, cell->center(true));
    ++cell_index;
  }
  data_out.add_data_vector (conductivity, "conductivity");

  if(mesh_order == 1)
    data_out.build_patches (mapping, fe.degree);
  else
  {
    data_out.build_patches (mapping,
                            mesh_order,
                            DataOut<dim>::curved_inner_cells);
  }

  std::ostringstream filename;
  filename << output_data_file << "-p="
           << Utilities::int_to_string (fe.degree) << ".f="
           << frequency;

  if(Utilities::MPI::n_mpi_processes(mpi_communicator) > 1)
  {
    data_out.write_vtu_with_pvtu_record("./", filename.str(), cycle,
                                        mpi_communicator, 2,
                                        parallel_output ? 0 : 1);
  }
  else
  {
    filename << "." << cycle << ".vtu";
    std::ofstream ofs(filename.str());
    data_out.write_vtu(ofs);
  }
}

template class EMFEM<2>;
template class EMFEM<3>;
