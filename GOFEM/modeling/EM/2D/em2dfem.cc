#include "em2dfem.h"

#include <fstream>
#include <iostream>

#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>

class TangentialFieldPostprocessor : public DataPostprocessorVector<2>
{
  std::string prefix;
  PhysicalModelPtr<2> phys_model;
  double omega, k;

  mutable Tensor<1, 2, dcomplex> Et, Ht, gradEx, gradHx;
  Tensor<2, 2, dcomplex> Rc;

public:
  TangentialFieldPostprocessor (const PhysicalModelPtr<2>& model, double frequency,
                                double wavenumber, std::string name):
    DataPostprocessorVector<2> (name, update_quadrature_points | update_gradients | update_values),
    prefix(name), phys_model(model),
    omega(2.0 * numbers::PI * frequency),
    k(wavenumber)
  {
    Rc[0][0] = 0.; Rc[0][1] = -1.;
    Rc[1][0] = 1.; Rc[1][1] = 0.;
  }

  std::vector<std::string> get_names () const
  {
    std::vector<std::string> names;

    names.push_back (prefix + "_Et_real");
    names.push_back (prefix + "_Et_real");
    names.push_back (prefix + "_Et_imag");
    names.push_back (prefix + "_Et_imag");
    names.push_back (prefix + "_Ht_real");
    names.push_back (prefix + "_Ht_real");
    names.push_back (prefix + "_Ht_imag");
    names.push_back (prefix + "_Ht_imag");

    return names;
  }

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  get_data_component_interpretation () const
  {
    auto vector_part = DataComponentInterpretation::component_is_part_of_vector;
    std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation;
    data_component_interpretation.resize(8, vector_part);

    return data_component_interpretation;
  }

  virtual void
  evaluate_vector_field (const DataPostprocessorInputs::Vector<2> &input_data,
                         std::vector< Vector< double > >          &computed_quantities) const
  {
    const auto cell = input_data.get_cell<DoFHandler<2>>();

    const double omega_mu = omega * mu0;//mu_values[0];

    for (unsigned int p = 0; p < input_data.evaluation_points.size(); ++p)
    {
      const Point<2> &point = input_data.evaluation_points[p];

      const double epsilon  = phys_model->permittivity_at (cell, point);
      const dcomplex sigma0 = phys_model->complex_conductivity_at (cell, omega / 2. * M_PI, point);

      const dcomplex sigma = sigma0 - II*omega*epsilon;
      const dcomplex lambda = 1.0 / (k*k - II*omega_mu*sigma);

      gradEx[0] = dcomplex(input_data.solution_gradients[p][0][0],
                           input_data.solution_gradients[p][1][0]);
      gradEx[1] = dcomplex(input_data.solution_gradients[p][0][1],
                           input_data.solution_gradients[p][1][1]);

      gradHx[0] = dcomplex(input_data.solution_gradients[p][2][0],
                           input_data.solution_gradients[p][3][0]);
      gradHx[1] = dcomplex(input_data.solution_gradients[p][2][1],
                           input_data.solution_gradients[p][3][1]);

      Et = -II*k*lambda*gradEx - II*omega_mu*lambda*(Rc*gradHx);
      Ht = -II*k*lambda*gradHx + sigma*lambda*(Rc*gradEx);

      computed_quantities[p][0] = Et[0].real();
      computed_quantities[p][1] = Et[1].real();
      computed_quantities[p][2] = Et[0].imag();
      computed_quantities[p][3] = Et[1].imag();
      computed_quantities[p][4] = Ht[0].real();
      computed_quantities[p][5] = Ht[1].real();
      computed_quantities[p][6] = Ht[0].imag();
      computed_quantities[p][7] = Ht[1].imag();
    }
  }
};

EM2DFEM::EM2DFEM (MPI_Comm comm,
                  const unsigned int order,
                  const unsigned int mapping_order,
                  const PhysicalModelPtr<2> &model,
                  const BackgroundModel<2> &bg_model):
  EMFEM<2>(comm, order, mapping_order,
           model, bg_model, EHField, true),
  wavenumber(0.)
{
  R[0][0] = 0; R[0][1] = -1;
  R[1][0] = 1; R[1][1] = 0;
}

EM2DFEM::EM2DFEM (MPI_Comm comm,
                  const unsigned int order,
                  const unsigned int mapping_order,
                  const PhysicalModelPtr<2> &model):
 EM2DFEM(comm, order, mapping_order,
         model, BackgroundModel<2>())
{}


EM2DFEM::~EM2DFEM ()
{
  clear ();
}

void EM2DFEM::set_adjoint_solver_callback(std::function<void ()> &callback)
{
  run_adjoint_solver = callback;
}

void EM2DFEM::run()
{
  EMFEM::run();

  if(run_adjoint_solver)
    run_adjoint_solver();
}

void EM2DFEM::clear()
{
  solver_mumps.clear();
  EMFEM<2>::clear();
}

void EM2DFEM::assemble_system_matrix ()
{
  pcout << "  Quadrature points per cell: " << QGauss<2> (fe.degree + 1).size() << "\n";

  WorkStream::run (FilteredIterator<DoFHandler<2>::active_cell_iterator> (IteratorFilters::LocallyOwnedCell (), dof_handler.begin_active ()),
                   FilteredIterator<DoFHandler<2>::active_cell_iterator> (IteratorFilters::LocallyOwnedCell (), dof_handler.end ()),
                   std::bind (&EM2DFEM::local_assemble_system, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                   std::bind (&EM2DFEM::copy_local_to_global_system, this, std::placeholders::_1),
                   Assembly::Scratch::MaxwellSystem<2> (fe, mapping, QGauss<2> (fe.degree + 1), update_gradients | update_JxW_values | update_quadrature_points | update_values,
                                                        QGauss<1> (fe.degree + 1), update_JxW_values | update_quadrature_points | update_values | update_normal_vectors),
                   Assembly::CopyData::MaxwellSystem (fe.dofs_per_cell));

  system_matrix.compress (VectorOperation::add);

  // Assembling new system matrix means we have to destroy solver context
  // and refactorize it upon next solve
  solver_mumps.clear();

//  PetscViewer viewer;
//  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"/home/agrayver/Desktop/r&d/A.dat",FILE_MODE_WRITE,&viewer);
//  MatView((const Mat&)system_matrix.block(0,0),viewer);
//  PetscViewerDestroy(&viewer);
}

void EM2DFEM::local_assemble_system (const DoFHandler<2>::active_cell_iterator& cell,
                                     Assembly::Scratch::MaxwellSystem<2>& scratch,
                                     Assembly::CopyData::MaxwellSystem& data)
{
  const FEValuesExtractors::Scalar e_re (0);
  const FEValuesExtractors::Scalar e_im (1);
  const FEValuesExtractors::Scalar h_re (2);
  const FEValuesExtractors::Scalar h_im (3);

  scratch.fe_values.reinit (cell);

  const std::vector<double>& JxW_values = scratch.fe_values.get_JxW_values ();
  const std::vector<Point<2>>& quadrature_points = scratch.fe_values.get_quadrature_points ();

  phys_model->complex_conductivity_list (cell, quadrature_points, scratch.complex_sigma_values, frequency);
  //phys_model->conductivity_list (cell, quadrature_points, scratch.sigma_values);
  phys_model->permittivity_list (cell, quadrature_points, scratch.epsilon_values);
  //phys_model->permeability_list (cell, quadrature_points, scratch.mu_values);
  data.local_matrix = 0.;
  data.local_rhs = 0.;

  for (unsigned int q_point = 0; q_point < scratch.fe_values.n_quadrature_points; ++q_point)
  {
    for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
    {
      scratch.shape_grads[0][i] = scratch.fe_values[e_re].gradient (i, q_point);
      scratch.shape_grads[1][i] = scratch.fe_values[e_im].gradient (i, q_point);
      scratch.shape_grads[2][i] = scratch.fe_values[h_re].gradient (i, q_point);
      scratch.shape_grads[3][i] = scratch.fe_values[h_im].gradient (i, q_point);
      scratch.shape_grads_rotated[0][i] = R * scratch.shape_grads[0][i];
      scratch.shape_grads_rotated[1][i] = R * scratch.shape_grads[1][i];
      scratch.shape_grads_rotated[2][i] = R * scratch.shape_grads[2][i];
      scratch.shape_grads_rotated[3][i] = R * scratch.shape_grads[3][i];
      scratch.shape_scalar_values[0][i] = scratch.fe_values[e_re].value (i, q_point);
      scratch.shape_scalar_values[1][i] = scratch.fe_values[e_im].value (i, q_point);
      scratch.shape_scalar_values[2][i] = scratch.fe_values[h_re].value (i, q_point);
      scratch.shape_scalar_values[3][i] = scratch.fe_values[h_im].value (i, q_point);

//      std::cout << i << "\t" << fe.system_to_base_index(i).second << "\t"
//                << fe.system_to_base_index(i).first.second << "\n";
    }

    // Work with angular frequency
    const double omega = 2.0 * numbers::PI * frequency;
    double omega_mu = omega * mu0/*scratch.mu_values[q_point]*/;
    dcomplex sigma = scratch.complex_sigma_values[q_point] + II*omega*scratch.epsilon_values[q_point];
    dcomplex omega_mu_sigma = omega_mu * sigma;
    //double sigma = scratch.sigma_values[q_point];
    //double omega_mu_sigma = omega_mu * sigma;

    dcomplex lambda, a, b, c;
    lambda = 1.0 / (wavenumber*wavenumber + II*omega_mu_sigma);
    a =  sigma * lambda;
    b = -II*omega_mu*lambda;
    c =  II*wavenumber*lambda;

    // Run inner loop only till index i, because local matrix is symmetric
    for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
    {
      for (unsigned int j = 0; j <= i; ++j)
//      for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
      {
        double ex_re = 0, ex_im = 0, hx_re = 0, hx_im = 0;

        // Note: lines of the system corresponding to imaginary DoFs have been multiplied by
        // -1 to make resulting system symmetric
//        data.local_matrix(i, j) += JxW_values[q_point] * (// E-polarization
//                                                          scratch.shape_grads[0][i] * scratch.shape_grads[0][j]
//                                                         +scratch.shape_grads[1][i] * scratch.shape_grads[1][j]
//                                                         +omega_mu_sigma * scratch.shape_scalar_values[0][i] * scratch.shape_scalar_values[1][j]
//                                                         -omega_mu_sigma * scratch.shape_scalar_values[1][i] * scratch.shape_scalar_values[0][j]
//                                                          // H-polarization
//                                                         +1./sigma * scratch.shape_grads[2][i] * scratch.shape_grads[2][j]
//                                                         +1./sigma * scratch.shape_grads[3][i] * scratch.shape_grads[3][j]
//                                                         +omega_mu * scratch.shape_scalar_values[2][i] * scratch.shape_scalar_values[3][j]
//                                                         -omega_mu * scratch.shape_scalar_values[3][i] * scratch.shape_scalar_values[2][j]);

        ex_re =  a.real() * (scratch.shape_grads[0][i] * scratch.shape_grads[0][j])
               - a.imag() * (scratch.shape_grads[0][i] * scratch.shape_grads[1][j])
               + sigma.real() * scratch.shape_scalar_values[0][i] * scratch.shape_scalar_values[0][j]
               - sigma.imag() * scratch.shape_scalar_values[0][i] * scratch.shape_scalar_values[1][j];
        ex_im =  a.imag() * (scratch.shape_grads[1][i] * scratch.shape_grads[0][j])
               + a.real() * (scratch.shape_grads[1][i] * scratch.shape_grads[1][j])
               + sigma.real() * scratch.shape_scalar_values[1][i] * scratch.shape_scalar_values[1][j]
               + sigma.imag() * scratch.shape_scalar_values[1][i] * scratch.shape_scalar_values[0][j];
        hx_re =  b.real() * (scratch.shape_grads[2][i] * scratch.shape_grads[2][j])
               - b.imag() * (scratch.shape_grads[2][i] * scratch.shape_grads[3][j])
               + omega_mu * scratch.shape_scalar_values[2][i] * scratch.shape_scalar_values[3][j];
        hx_im =  b.imag() * (scratch.shape_grads[3][i] * scratch.shape_grads[2][j])
               + b.real() * (scratch.shape_grads[3][i] * scratch.shape_grads[3][j])
               - omega_mu * scratch.shape_scalar_values[3][i] * scratch.shape_scalar_values[2][j];

        if(wavenumber != 0.) // add CSEM part which couples E-H fields
        {
//          ex_re -= c.real() * scratch.shape_grads_rotated[0][i] * scratch.shape_grads[2][j]
//                  -c.imag() * scratch.shape_grads_rotated[0][i] * scratch.shape_grads[3][j];
//          ex_im -= c.imag() * scratch.shape_grads_rotated[1][i] * scratch.shape_grads[2][j]
//                  +c.real() * scratch.shape_grads_rotated[1][i] * scratch.shape_grads[3][j];
//          hx_re -= c.real() * scratch.shape_grads_rotated[2][i] * scratch.shape_grads[0][j]
//                  -c.imag() * scratch.shape_grads_rotated[2][i] * scratch.shape_grads[1][j];
//          hx_im -= c.imag() * scratch.shape_grads_rotated[3][i] * scratch.shape_grads[0][j]
//                  +c.real() * scratch.shape_grads_rotated[3][i] * scratch.shape_grads[1][j];

          ex_re += c.real() * scratch.shape_grads_rotated[2][j] * scratch.shape_grads[0][i]
                  -c.imag() * scratch.shape_grads_rotated[3][j] * scratch.shape_grads[0][i];
          ex_im += c.imag() * scratch.shape_grads_rotated[2][j] * scratch.shape_grads[1][i]
                  +c.real() * scratch.shape_grads_rotated[3][j] * scratch.shape_grads[1][i];
          hx_re += c.real() * scratch.shape_grads_rotated[0][j] * scratch.shape_grads[2][i]
                  -c.imag() * scratch.shape_grads_rotated[1][j] * scratch.shape_grads[2][i];
          hx_im += c.imag() * scratch.shape_grads_rotated[0][j] * scratch.shape_grads[3][i]
                  +c.real() * scratch.shape_grads_rotated[1][j] * scratch.shape_grads[3][i];
        }

        // Two steps are done below to support symmetry of the system:
        // First: the H term has negative sign
        // Second: imaginary parts are multiplied by -1
        data.local_matrix(i, j) += JxW_values[q_point] * (ex_re - ex_im - hx_re + hx_im);
//        data.local_matrix(i, j) += JxW_values[q_point] * (ex_re + ex_im + hx_re + hx_im);
      }
    }
  }

  // Fill rest of the matrix
  for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
    for (unsigned int j = i + 1; j < fe.dofs_per_cell; ++j)
      data.local_matrix(i, j) = data.local_matrix(j, i);

  cell->get_dof_indices (data.local_dof_indices);
}

void EM2DFEM::assemble_dipole_rhs_vector(const DipoleSource &phys_source,
                                         const AffineConstraints<double> &constraints,
                                         PETScWrappers::MPI::BlockVector &rhs_vector)
{
  if(phys_source.n_dipole_elements() != 1)
    throw std::runtime_error("Wire sources are not supported with the total field approach."
                             "Use secondary field approach with complex sources.");

  const FEValuesExtractors::Scalar e_re (0);
  const FEValuesExtractors::Scalar e_im (1);
  const FEValuesExtractors::Scalar h_re (2);
  const FEValuesExtractors::Scalar h_im (3);

  Point<2> rec_position;
  phys_source.position(rec_position, 2);
  std::pair<DoFHandler<2>::active_cell_iterator, Point<2> >
      cell_point = GridTools::find_active_cell_around_point (mapping, dof_handler, rec_position);

  if (cell_point.first->is_locally_owned ())
  {
    Vector<double> local_rhs (fe.dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices (fe.dofs_per_cell);

    UpdateFlags update_flags = update_values | update_gradients;

    const Quadrature<2> quadrature(GeometryInfo<2>::project_to_unit_cell(cell_point.second));
    FEValues<2> fe_values (mapping, fe, quadrature, update_flags);

    // Currently only real currents are supported
    double current = 1;
    const Tensor<1, 3> re_moment = current * array2vec<Tensor<1, 3>>(phys_source.dipole_extent());

    Tensor<1, 2> St; // transverse to strike component
    St[0] = re_moment[1];
    St[1] = re_moment[2];

    fe_values.reinit (cell_point.first);
    local_rhs = 0.0;

    std::vector<dcomplex> sigma_values(quadrature.size());
    std::vector<double> epsilon_values(quadrature.size());//,mu_values(quadrature.size());
    //phys_model->permeability_list (cell_point.first, quadrature.get_points (), mu_values);
    phys_model->complex_conductivity_list (cell_point.first, quadrature.get_points (), sigma_values, frequency);
    phys_model->permittivity_list (cell_point.first, quadrature.get_points (), epsilon_values);

    const double omega = 2.0 * numbers::PI * frequency;
    const double omega_mu = omega * mu0/*scratch.mu_values[q_point]*/;
    const dcomplex sigma = sigma_values[0] + II*omega*epsilon_values[0];

    const dcomplex lambda = 1.0 / (wavenumber*wavenumber + II * omega_mu * sigma);
    const dcomplex a =  II * omega_mu * sigma * lambda;
    const dcomplex b = -II * omega_mu * lambda;
    // For CSEM:
    const dcomplex c = -II * wavenumber * lambda;
    const dcomplex d =  II * omega_mu * c;

    double ex_re = 0, ex_im = 0, hx_re = 0, hx_im = 0;

    for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
    {
      if(phys_source.get_type() == ElectricDipole)
      {
        ex_re = -fe_values[e_re].value (i, 0) * re_moment[0] +
                 c.real() * fe_values[e_re].gradient (i, 0) * St;
        ex_im = c.imag() * fe_values[e_im].gradient (i, 0) * St;

        hx_re = (b.real() * R * St) * fe_values[h_re].gradient (i, 0);
        hx_im = (b.imag() * R * St) * fe_values[h_im].gradient (i, 0);
      }
      else if(phys_source.get_type() == MagneticDipole)
      {
        ex_re = (a.real() * R * St) * fe_values[e_re].gradient (i, 0);
        ex_im = (a.imag() * R * St) * fe_values[e_im].gradient (i, 0);

        hx_re = d.real() * fe_values[h_re].gradient (i, 0) * St;
        hx_im = -omega_mu * fe_values[h_im].value (i, 0) * re_moment[0] +
//        hx_im = omega_mu * fe_values[h_im].value (i, 0) * re_moment[0] +
                d.imag() * fe_values[h_im].gradient (i, 0) * St;
      }
      else
        ExcNotImplemented();

//      if(fabs(ex_re) > 0 || fabs(ex_im) > 0 || fabs(hx_re) > 0 || fabs(hx_im) > 0)
//        std::cout << phys_source.get_type() << "\t" << re_moment << "\t" << ex_re
//                  << "\t" << ex_im << "\t" << hx_re << "\t" << hx_im << "\n";

      // Two steps are done below to support symmetry of the system:
      // First: the H term has negative sign
      // Second: imaginary parts are multiplied by -1
      local_rhs(i) += ex_re - ex_im - hx_re + hx_im;
//      local_rhs(i) += ex_re + ex_im + hx_re + hx_im;

      //std::cout << i << "\t" << fe.system_to_component_index(i).first << "\t" << fe.system_to_component_index(i).first << "\n";
    }

    cell_point.first->get_dof_indices (dof_indices);
    constraints.distribute_local_to_global (local_rhs, dof_indices, rhs_vector);
  }
}

void EM2DFEM::assemble_dual_rhs_vector(const std::vector<Point<2> > &delta_positions,
                                       const AffineConstraints<double> &constraints,
                                       PETScWrappers::MPI::BlockVector &rhs_vector)
{
  const FEValuesExtractors::Scalar e_re (0);
  const FEValuesExtractors::Scalar e_im (1);
  const FEValuesExtractors::Scalar h_re (2);
  const FEValuesExtractors::Scalar h_im (3);

  Vector<double> local_rhs (fe.dofs_per_cell);
  UpdateFlags update_flags = update_JxW_values | update_quadrature_points | update_values;
  QGauss<2> quadrature (fe.degree + 2);
  FEValues<2> fe_values (mapping, fe, quadrature, update_flags);
  std::vector<types::global_dof_index> dof_indices (fe.dofs_per_cell);

  std::vector<double> shape_value(4);

  for(size_t k = 0; k < delta_positions.size(); ++k)
  {
    std::pair<DoFHandler<2>::active_cell_iterator, Point<2> >
        cell_point = GridTools::find_active_cell_around_point (mapping, dof_handler, delta_positions[k]);

    if (cell_point.first->is_locally_owned ())
    {
      fe_values.reinit (cell_point.first);
      local_rhs = 0.0;

//      const std::vector<double>& JxW_values = fe_values.get_JxW_values ();

      for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points; ++q_point)
      {
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
        {
          shape_value[0] = fe_values[e_re].value (i, q_point);
          shape_value[1] = fe_values[e_im].value (i, q_point);
          shape_value[2] = fe_values[h_re].value (i, q_point);
          shape_value[3] = fe_values[h_im].value (i, q_point);

//          local_rhs (i) += JxW_values[q_point] * (shape_value[0] + shape_value[1] + shape_value[2] + shape_value[3]);
          local_rhs (i) += (shape_value[0] + shape_value[1] + shape_value[2] + shape_value[3]);
        }
      }

      cell_point.first->get_dof_indices (dof_indices);
      constraints.distribute_local_to_global (local_rhs, dof_indices, rhs_vector);
    }
  }
}

void EM2DFEM::setup_system (const unsigned n_rhs)
{
  dof_handler.distribute_dofs (fe);

  /*
   * Order dofs w.r.t to real and imaginary parts of our system
   * to obtain block system matrix in the end.
   * Note: when performing this ordering for non-block matrices in
   * parallel then contignuity of the indices gets lost making PETSc
   * fail.
   */
  if(Utilities::MPI::n_mpi_processes(mpi_communicator) == 1)
    DoFRenumbering::block_wise (dof_handler);

  locally_owned_partitioning.clear ();
  locally_relevant_partitioning.clear ();

  const IndexSet& locally_owned_dofs = dof_handler.locally_owned_dofs();
  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs (dof_handler, locally_relevant_dofs);

  locally_owned_partitioning.push_back(locally_owned_dofs);
  locally_relevant_partitioning.push_back(locally_relevant_dofs);

  // Setup constraints
  constraints.resize (get_number_of_constraint_matrices());
  for(auto& cm: constraints)
  {
    cm.clear ();
    cm.reinit (locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints (dof_handler, cm);
  }

  set_boundary_values ();

  for(auto& cm: constraints)
    cm.close();

  // Allocate solution and rhs vectors
  system_rhs.clear ();
  solutions.clear ();

  system_rhs.resize (n_rhs);
  solutions.resize (n_rhs);
  estimated_cell_errors.resize(n_rhs);
  ghosted_rhs_vectors.resize(n_rhs);

  for(size_t i = 0; i < solutions.size(); ++i)
  {
    solutions[i].reinit (locally_owned_partitioning, locally_relevant_partitioning, mpi_communicator);
    system_rhs[i].reinit (locally_owned_partitioning, mpi_communicator);
    estimated_cell_errors[i].reinit(triangulation.n_active_cells());
    ghosted_rhs_vectors[i].reinit(locally_owned_partitioning, locally_relevant_partitioning, mpi_communicator);
  }

  completely_distributed_solution.reinit (locally_owned_partitioning, mpi_communicator);

  // Initialize system matrix
  BlockDynamicSparsityPattern bdsp (locally_relevant_partitioning);
  DoFTools::make_sparsity_pattern (dof_handler, bdsp, constraints[0], false);
  SparsityTools::distribute_sparsity_pattern (bdsp, dof_handler.locally_owned_dofs (),
                                              mpi_communicator, locally_relevant_dofs);
  system_matrix.reinit (locally_owned_partitioning, bdsp, mpi_communicator);

  pcout << "  Degrees of freedom: " << dof_handler.n_dofs () << "  ";
  pcout << "  (DoFs per cell: " << fe.dofs_per_cell << ")" << std::endl;
  pcout << "  Non-zero elements: " << system_matrix.block(0, 0).n_nonzero_elements() << std::endl;
}

void EM2DFEM::setup_preconditioner()
{
  //
}

void EM2DFEM::solve (std::vector<PETScWrappers::MPI::BlockVector>& solution_vectors,
                     std::vector<PETScWrappers::MPI::BlockVector>& rhs_vectors,
                     const std::vector<unsigned> &constraints_indices,
                     bool adjoint, bool verbose, unsigned start_index)
{
  if(preconditioner_type == Direct)
  {
    if(Utilities::MPI::n_mpi_processes(mpi_communicator) == 1)
      solve_using_mumps_solver (solution_vectors, rhs_vectors, constraints_indices,
                                adjoint, verbose, start_index);
    else
      solve_using_direct_solver (solution_vectors, rhs_vectors, constraints_indices,
                                 adjoint, verbose, start_index);
  }
  else if(preconditioner_type == AMG)
  {
    throw std::runtime_error("AMG solver is not implemented. Use direct solver.");
  }
  else if(preconditioner_type == AutoSelection)
  {
    //if(dof_handler.n_dofs() > n_dofs_threshold)
    //  solve_using_cgamg_solver (solution_vectors, rhs_vectors, constraints_indices,
    //                            adjoint, verbose, start_index);
    //else
      solve_using_direct_solver (solution_vectors, rhs_vectors, constraints_indices,
                                 adjoint, verbose, start_index);
  }
  else
    Assert(false, ExcNotImplemented ());
}

void EM2DFEM::solve_using_direct_solver (std::vector<PETScWrappers::MPI::BlockVector>& solution_vectors,
                                         std::vector<PETScWrappers::MPI::BlockVector>& rhs_vectors,
                                         const std::vector<unsigned> &constraints_indices,
                                         bool /*adjoint*/, bool verbose, unsigned start_index)
{
  Assert (rhs_vectors.size() == solution_vectors.size(), ExcDimensionMismatch (rhs_vectors.size(), solution_vectors.size()));

  if(verbose) pcout << "  Solve system using direct solver MUMPS." << std::endl;

  SolverControl solver_control (max_iterations);
  //solver_control.enable_history_data ();

  // Create solver and solve for all RHS vectors consecutively
  PETScWrappers::SparseDirectMUMPS solver (solver_control);
  solver.set_symmetric_mode(true);
  //PETScWrappers::SolverGMRES solver (solver_control);
  //PETScWrappers::PreconditionJacobi pre(system_matrix.block(0, 0));

  for(unsigned i = start_index; i < rhs_vectors.size(); ++i)
  {
    //solver_control.set_tolerance (solver_residual * rhs_vectors[i].l2_norm ());

    completely_distributed_solution = 0.0;

    Timer timer;
    timer.start ();

    try
    {
      solver.solve (system_matrix.block(0, 0), completely_distributed_solution.block(0), rhs_vectors[i].block(0));

      //PETScWrappers::MPI::Vector r(completely_distributed_solution.block(0));
      //system_matrix.block(0, 0).residual(r, completely_distributed_solution.block(0), rhs_vectors[i].block(0));
      //if(verbose) pcout << "  Relative residual " << r.l2_norm() / rhs_vectors[i].block(0).l2_norm() << " for rhs #" << i << " has been reached.\n";
    }
    catch (SolverControl::NoConvergence)
    {
      if(verbose) pcout << "Couldn't convergence to the desired tolerance.\n";
    }

    if(verbose) pcout << "    Total solution time for rhs #" << i << ": " << timer.wall_time () << std::endl;

    constraints[constraints_indices[i]].distribute (completely_distributed_solution);
    solution_vectors[i] = completely_distributed_solution;

    //pcout << "mean rhs,sol: " << rhs_vectors[i].mean_value() << "\t" << solution_vectors[i].mean_value() << "\t"
    //      << "norm rhs,sol: " << rhs_vectors[i].l2_norm() << "\t" << solution_vectors[i].l2_norm() << std::endl;
  }
}

void EM2DFEM::solve_using_mumps_solver (std::vector<PETScWrappers::MPI::BlockVector>& solution_vectors,
                                        std::vector<PETScWrappers::MPI::BlockVector>& rhs_vectors,
                                        const std::vector<unsigned> &constraints_indices,
                                        bool /*adjoint*/, bool verbose, unsigned start_index)
{
  Assert (rhs_vectors.size() == solution_vectors.size(), ExcDimensionMismatch (rhs_vectors.size(), solution_vectors.size()));

  if(verbose) pcout << "  Solve system using direct solver MUMPS." << std::endl;

  if(!solver_mumps.is_initialized())
  {
    solver_mumps.set_symmetric_mode(true);

    Timer timer;
    timer.start();
    solver_mumps.initialize(system_matrix.block(0, 0));
    if(verbose) pcout << "  Time for factorization " << timer.wall_time () << std::endl;
  }

  if(Utilities::MPI::n_mpi_processes(mpi_communicator) > 1)
  {
    std::vector<PETScWrappers::MPI::BlockVector> temporary_solutions(solution_vectors.size());
    for(unsigned i = 0; i < temporary_solutions.size(); ++i)
      temporary_solutions[i].reinit(locally_owned_partitioning, mpi_communicator);

    solver_mumps.vmult_bunch(temporary_solutions, rhs_vectors, fe.dofs_per_cell, false);

    for(unsigned i = start_index; i < rhs_vectors.size(); ++i)
    {
      temporary_solutions[i].compress(VectorOperation::insert);
      constraints[constraints_indices[i]].distribute (temporary_solutions[i]);
      solution_vectors[i] = temporary_solutions[i];
    }
  }
  else
  {
    solver_mumps.vmult_bunch(rhs_vectors, rhs_vectors, fe.dofs_per_cell, false);

    for(unsigned i = start_index; i < rhs_vectors.size(); ++i)
    {
      rhs_vectors[i].compress(VectorOperation::insert);
      constraints[constraints_indices[i]].distribute (rhs_vectors[i]);
      solution_vectors[i] = rhs_vectors[i];
    }
  }
}

void EM2DFEM::estimate_error ()
{
  error_estimates.push_back(std::vector<double>());

  for(size_t i = 0; i < solutions.size(); ++i)
  {
    Vector<float> kelly_estimates(triangulation.n_active_cells());
    KellyErrorEstimator<2>::estimate (mapping, dof_handler,
                                      QGauss<1>(fe.degree + 2),
                                      std::map<types::boundary_id, const Function<2>*>(),
                                      solutions[i],
                                      kelly_estimates,
                                      ComponentMask(),
                                      nullptr,
                                      numbers::invalid_unsigned_int,
                                      triangulation.locally_owned_subdomain());

    error_estimates[i].push_back(kelly_estimates.l2_norm());

    std::copy(kelly_estimates.begin(), kelly_estimates.end(), estimated_cell_errors[i].begin());
  }

  if(adaptivity_type == GoalOriented)
  {
    for(size_t i = 0; i < n_physical_sources; ++i)
    {
      for(size_t j = 0; j < estimated_cell_errors[i].size(); ++j)
        estimated_cell_errors[i][j] *= estimated_cell_errors[n_physical_sources][j];
    }
  }

  for(size_t i = 0; i < solutions.size(); ++i)
    pcout << "  Error for source " << physical_sources[i]->get_name() << ": " << error_estimates[i].back() << "\n";
}

void EM2DFEM::field_at_point(const DoFHandler<2>::active_cell_iterator& cell, const Point<2>& p, cvector& E, cvector& H) const
{
  // Angular frequency
  double omega = 2.0 * numbers::PI * frequency;

  // Extract electric and magnetic fields using this instances
  std::vector<std::vector<double >> values (fe.n_blocks(), std::vector<double> (1));
  std::vector<std::vector<Tensor<1, 2>>> grads (fe.n_blocks(), std::vector<Tensor<1, 2>> (1));

  // Initialize FEValues with given cell
  const FEValuesExtractors::Scalar e_re (0);
  const FEValuesExtractors::Scalar e_im (1);
  const FEValuesExtractors::Scalar h_re (2);
  const FEValuesExtractors::Scalar h_im (3);

  Point<2> p_unit = mapping.transform_real_to_unit_cell(cell, p);
  const Quadrature<2> quadrature (GeometryInfo<2>::project_to_unit_cell (p_unit));
  FEValues<2> fe_values (mapping, fe, quadrature, update_values | update_gradients);
  fe_values.reinit (cell);

  std::vector<std::complex<double>> sigma_values(quadrature.size());
  //std::vector<double> sigma_values(quadrature.size());
  std::vector<double> epsilon_values(quadrature.size());//, mu_values(quadrature.size());
  //phys_model->permeability_list (cell, quadrature.get_points (), mu_values);
  phys_model->permittivity_list (cell, quadrature.get_points (), epsilon_values);
  phys_model->complex_conductivity_list (cell, quadrature.get_points (), sigma_values, frequency);
  //phys_model->conductivity_list (cell, quadrature.get_points (), sigma_values);

  const double omega_mu = omega * mu0;//mu_values[0];
  const dcomplex sigma = sigma_values[0] + II*omega*epsilon_values[0];
  const dcomplex lambda = 1.0 / (wavenumber*wavenumber + II*omega_mu*sigma);

  Tensor<2, 2, dcomplex> Rc;
  Rc[0][0] = dcomplex(0., 0.); Rc[0][1] = dcomplex(-1., 0.);
  Rc[1][0] = dcomplex(1., 0.); Rc[1][1] = dcomplex(0., 0.);

  // Extract electric and magnetic fields for both polarizations
  for (unsigned int j = 0; j < n_physical_sources; ++j)
  {
    fe_values[e_re].get_function_values (solutions[j], values[0]);
    fe_values[e_im].get_function_values (solutions[j], values[1]);
    fe_values[h_re].get_function_values (solutions[j], values[2]);
    fe_values[h_im].get_function_values (solutions[j], values[3]);

    fe_values[e_re].get_function_gradients (solutions[j], grads[0]);
    fe_values[e_im].get_function_gradients (solutions[j], grads[1]);
    fe_values[h_re].get_function_gradients (solutions[j], grads[2]);
    fe_values[h_im].get_function_gradients (solutions[j], grads[3]);

    E[j] = dcomplex(values[0][0], values[1][0]); // Ex
//    E[j + n_physical_sources] = dcomplex(grads[2][0][1]/sigma_values[0].real(), grads[3][0][1]/sigma_values[0].real()); // Ey = dHx/dz*1/sigma
//    E[j + 2*n_physical_sources] = -dcomplex(grads[2][0][0]/sigma_values[0].real(), grads[3][0][0]/sigma_values[0].real()); // Ez = -dHx/dy*1/sigma

    H[j] = dcomplex(values[2][0], values[3][0]); // Hx
//    H[j + n_physical_sources] = -dcomplex(grads[1][0][1]/omega_mu, grads[0][0][1]/omega_mu); // Hy = -dEx/dz*(iwmu)^-1
//    H[j + 2*n_physical_sources] = dcomplex(grads[1][0][0]/omega_mu, grads[0][0][0]/omega_mu); // Hz = dEx/dy*(iwmu)^-1

    Tensor<1, 2, dcomplex> Et, Ht, gradEx, gradHx;

    gradEx[0] = dcomplex(grads[0][0][0], grads[1][0][0]);
    gradEx[1] = dcomplex(grads[0][0][1], grads[1][0][1]);

    gradHx[0] = dcomplex(grads[2][0][0], grads[3][0][0]);
    gradHx[1] = dcomplex(grads[2][0][1], grads[3][0][1]);
    //pcout << gradEx << " " << gradHx << "\n";

    Et = -II*wavenumber*lambda*gradEx - II*omega_mu*lambda*(Rc*gradHx);
    Ht = -II*wavenumber*lambda*gradHx + sigma*lambda*(Rc*gradEx);

    E[j + n_physical_sources] = Et[0];
    E[j + 2*n_physical_sources] = Et[1];

    H[j + n_physical_sources] = Ht[0];
    H[j + 2*n_physical_sources] = Ht[1];
  }
}

double EM2DFEM::get_wavenumber() const
{
  return wavenumber;
}

void EM2DFEM::output_specific_information(std::vector<std::shared_ptr<DataPostprocessor<2>>> &data,
                                          DataOut<2> &data_out) const
{
  data.resize(n_physical_sources);

  for(unsigned i = 0; i < n_physical_sources; ++i)
  {
    auto name = physical_sources[i]->get_name();
    data[i].reset(new TangentialFieldPostprocessor(phys_model, frequency,
                                                   wavenumber, name));
    data_out.add_data_vector(solutions[i], *data[i]);
  }    
}
