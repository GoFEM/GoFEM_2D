#ifndef _ERRORESTIMATOR_
#define _ERRORESTIMATOR_

#include <map>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/base/quadrature_lib.h>

#include <physical_model/physical_model.h>
#include <functions/current_function.h>

using namespace dealii;

class ResidualEstimatorMaxwell
{
public:
  ResidualEstimatorMaxwell(const FESystem<3, 3>& fe, const Mapping<3>& mapping,
                           const PhysicalModelPtr<3>& physical_model, double frequency);

  template<typename VECTOR>
  void residual_term(const typename DoFHandler<3>::active_cell_iterator& cell,
                     const std::vector<VECTOR> &solution_vectors,
                     const std::vector<VECTOR> &rhs_vectors,
                     const std::vector<CurrentFunctionPtr<3>>& sources,
                     std::vector<double>& estimated_errors);

  template<typename VECTOR>
  void jump_term(const typename DoFHandler<3>::active_cell_iterator& cell,
                 const std::vector<VECTOR> &solution_vectors,
                 const std::vector<CurrentFunctionPtr<3>>& sources,
                 std::vector<double>& estimated_errors);

  template<typename VECTOR>
  void residual_error_at_dofs(const DoFHandler<3>::active_cell_iterator &cell,
                              const VECTOR &solution_vectors, const VECTOR &rhs_vectors,
                              const CurrentFunctionPtr<3> &sources, Vector<double> &estimated_errors);

  template<typename VECTOR>
  void jump_error_at_dofs(const typename DoFHandler<3>::active_cell_iterator& cell,
                          const VECTOR &solution_vectors,
                          const CurrentFunctionPtr<3>& sources,
                          Vector<double> &estimated_errors);

private:
  const FEValuesExtractors::Vector re;
  const FEValuesExtractors::Vector im;

  const unsigned fe_degree;
  const double omega;

  const QGauss<3> quadrature;
  const QGauss<2> face_quadrature;
  const unsigned int n_q_points;
  const unsigned int n_face_q_points;

  FEFaceValues<3> fe_face_neighbor_values;
  FEFaceValues<3> fe_face_values;
  FESubfaceValues<3> fe_subface_neighbor_values;
  FESubfaceValues<3> fe_subface_values;
  FEValues<3> fe_values;

  std::vector<double> epsilon_face_values;
  std::vector<double> epsilon_neighbor_values;
  std::vector<double> epsilon_values;
  std::vector<double> sigma_face_values;
  std::vector<double> sigma_neighbor_values;
  std::vector<double> sigma_values;
  std::vector<double> mu_face_values;
  std::vector<double> mu_neighbor_values;
  std::vector<double> mu_values;

  // Use std::unordered_map instead of std::map once we transfer to C++11
  std::vector<std::map<DoFHandler<3>::face_iterator, double> > face_contributions;

  //std::vector<std::vector<double> > divergences (2, std::vector<double> (n_q_points));
  std::vector<std::vector<Tensor<1, 3> > > curls;
  std::vector<std::vector<Tensor<1, 3> > > curls_neighbor;
  std::vector<std::vector<Tensor<1, 3> > > values;
  std::vector<std::vector<Tensor<1, 3> > > rhs_values;
  std::vector<std::vector<Tensor<1, 3> > > values_face;
  std::vector<std::vector<Tensor<1, 3> > > values_neighbor;
  std::vector<std::vector<Tensor<3, 3> > > hessians;
  std::vector<Tensor<1, 3> > jumps, jump_curl;

  std::vector<Vector<double> > current_density_values;
  //std::vector<Vector<double> > current_density_divergences (n_q_points, Vector<double> (2));
  std::vector<Vector<double> > current_density_neighbor_values;
  std::vector<Vector<double> > right_hand_side_values;

  Vector<double> jump_values;
  Vector<double> residuals;

  const PhysicalModelPtr<3>& phys_model;
};


template<typename VECTOR>
void ResidualEstimatorMaxwell::residual_term(const typename DoFHandler<3>::active_cell_iterator& cell,
                                             const std::vector<VECTOR> &solution_vectors,
                                             const std::vector<VECTOR> &/*rhs_vectors*/,
                                             const std::vector<CurrentFunctionPtr<3> > &sources,
                                             std::vector<double>& estimated_errors)
{
  Assert(estimated_errors.size() == solution_vectors.size(), ExcDimensionMismatch(estimated_errors.size(), solution_vectors.size()));

  size_t n_solutions = solution_vectors.size ();
  std::fill(estimated_errors.begin(), estimated_errors.end(), 0.);

  face_contributions.resize(n_solutions);

  // Here we start with the residual-based term.
  fe_values.reinit (cell);

  const double weight = cell->diameter () * cell->diameter () / (fe_degree * fe_degree);
  const std::vector<double>& JxW_values = fe_values.get_JxW_values ();
  const std::vector<Point<3> >& quadrature_points = fe_values.get_quadrature_points ();

  phys_model->conductivity_list (cell, quadrature_points, sigma_values);
  phys_model->permittivity_list (cell, quadrature_points, epsilon_values);
  phys_model->permeability_list (cell, quadrature_points, mu_values);

  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    epsilon_values[q_point] *= omega;

  for (unsigned int i = 0; i < n_solutions; ++i)
  {
    //fe_values[re].get_function_divergences (solution_vectors[i], divergences[0]);
    //fe_values[im].get_function_divergences (solution_vectors[i], divergences[1]);
    fe_values[re].get_function_hessians (solution_vectors[i], hessians[0]);
    fe_values[im].get_function_hessians (solution_vectors[i], hessians[1]);
    fe_values[re].get_function_values (solution_vectors[i], values[0]);
    fe_values[im].get_function_values (solution_vectors[i], values[1]);

    sources[i]->vector_value_list (cell, quadrature_points, right_hand_side_values);
    //sources[i]->divergence_list (cell, quadrature_points, current_density_divergences);

//    fe_values[re].get_function_values (rhs_vectors[i], rhs_values[0]);
//    fe_values[im].get_function_values (rhs_vectors[i], rhs_values[1]);

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    {
//      if(rhs_values[1][q_point].norm() > 0)
//        std::cout << rhs_values[1][q_point];

//      right_hand_side_values[q_point](0) = -rhs_values[0][q_point][0];
//      right_hand_side_values[q_point](1) = -rhs_values[0][q_point][1];
//      right_hand_side_values[q_point](2) = -rhs_values[0][q_point][2];
//      right_hand_side_values[q_point](3) = -rhs_values[1][q_point][0];
//      right_hand_side_values[q_point](4) = -rhs_values[1][q_point][1];
//      right_hand_side_values[q_point](5) = -rhs_values[1][q_point][2];

      // The residual is computed.
      residuals (0) = (hessians[0][q_point][0][1][1] + hessians[0][q_point][0][2][2] - hessians[0][q_point][1][0][1] - hessians[0][q_point][2][0][2]) / mu_values[q_point] + omega * (sigma_values[q_point] * values[1][q_point][0] + epsilon_values[q_point] * values[0][q_point][0]) + right_hand_side_values[q_point] (0);
      residuals (1) = (hessians[0][q_point][1][0][0] + hessians[0][q_point][1][2][2] - hessians[0][q_point][0][0][1] - hessians[0][q_point][2][1][2]) / mu_values[q_point] + omega * (sigma_values[q_point] * values[1][q_point][1] + epsilon_values[q_point] * values[0][q_point][1]) + right_hand_side_values[q_point] (1);
      residuals (2) = (hessians[0][q_point][2][0][0] + hessians[0][q_point][2][1][1] - hessians[0][q_point][0][0][2] - hessians[0][q_point][1][1][2]) / mu_values[q_point] + omega * (sigma_values[q_point] * values[1][q_point][2] + epsilon_values[q_point] * values[0][q_point][2]) + right_hand_side_values[q_point] (2);
      residuals (3) = (hessians[1][q_point][0][1][1] + hessians[1][q_point][0][2][2] - hessians[1][q_point][1][0][1] - hessians[1][q_point][2][0][2]) / mu_values[q_point] + omega * (epsilon_values[q_point] * values[1][q_point][0] - sigma_values[q_point] * values[0][q_point][0]) + right_hand_side_values[q_point] (3);
      residuals (4) = (hessians[1][q_point][1][0][0] + hessians[1][q_point][1][2][2] - hessians[1][q_point][0][0][1] - hessians[1][q_point][2][1][2]) / mu_values[q_point] + omega * (epsilon_values[q_point] * values[1][q_point][1] - sigma_values[q_point] * values[0][q_point][1]) + right_hand_side_values[q_point] (4);
      residuals (5) = (hessians[1][q_point][2][0][0] + hessians[1][q_point][2][1][1] - hessians[1][q_point][0][0][2] - hessians[1][q_point][1][1][2]) / mu_values[q_point] + omega * (epsilon_values[q_point] * values[1][q_point][2] - sigma_values[q_point] * values[0][q_point][2]) + right_hand_side_values[q_point] (5);
      // The divergence is computed.
      //residuals (6) = current_density_divergences[q_point] (0) + sigma_values[q_point] * divergences[0][q_point] - epsilon_values[q_point] * divergences[1][q_point];
      //residuals (7) = current_density_divergences[q_point] (1) + sigma_values[q_point] * divergences[1][q_point] + epsilon_values[q_point] * divergences[0][q_point];
      // The cell term is computed.
      estimated_errors[i] += JxW_values[q_point] * residuals.norm_sqr ();
    }

    estimated_errors[i] *= weight;
  }
}

template<typename VECTOR>
void ResidualEstimatorMaxwell::residual_error_at_dofs(const DoFHandler<3>::active_cell_iterator &cell,
                                                      const VECTOR &solution, const VECTOR &rhs_vector,
                                                      const CurrentFunctionPtr<3> &sources,
                                                      Vector<double> &estimated_errors)
{
  // Here we start with the residual-based term.
  fe_values.reinit (cell);

  unsigned dofs_per_cell = fe_values.get_fe().dofs_per_cell;

  const std::vector<double>& JxW_values = fe_values.get_JxW_values ();
  const std::vector<Point<3> >& quadrature_points = fe_values.get_quadrature_points ();

  phys_model->conductivity_list (cell, quadrature_points, sigma_values);
  phys_model->permittivity_list (cell, quadrature_points, epsilon_values);
  phys_model->permeability_list (cell, quadrature_points, mu_values);

  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    epsilon_values[q_point] *= omega;

  std::vector<Tensor<1, 3> > shape_values(2);
  estimated_errors = 0.;

  fe_values[re].get_function_hessians (solution, hessians[0]);
  fe_values[im].get_function_hessians (solution, hessians[1]);
  fe_values[re].get_function_values (solution, values[0]);
  fe_values[im].get_function_values (solution, values[1]);

  sources->vector_value_list (cell, quadrature_points, right_hand_side_values);

  for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points; ++q_point)
  {
    residuals (0) = (hessians[0][q_point][0][1][1] + hessians[0][q_point][0][2][2] - hessians[0][q_point][1][0][1] - hessians[0][q_point][2][0][2]) / mu_values[q_point] + omega * (sigma_values[q_point] * values[1][q_point][0] + epsilon_values[q_point] * values[0][q_point][0]) + right_hand_side_values[q_point] (0);
    residuals (1) = (hessians[0][q_point][1][0][0] + hessians[0][q_point][1][2][2] - hessians[0][q_point][0][0][1] - hessians[0][q_point][2][1][2]) / mu_values[q_point] + omega * (sigma_values[q_point] * values[1][q_point][1] + epsilon_values[q_point] * values[0][q_point][1]) + right_hand_side_values[q_point] (1);
    residuals (2) = (hessians[0][q_point][2][0][0] + hessians[0][q_point][2][1][1] - hessians[0][q_point][0][0][2] - hessians[0][q_point][1][1][2]) / mu_values[q_point] + omega * (sigma_values[q_point] * values[1][q_point][2] + epsilon_values[q_point] * values[0][q_point][2]) + right_hand_side_values[q_point] (2);
    residuals (3) = (hessians[1][q_point][0][1][1] + hessians[1][q_point][0][2][2] - hessians[1][q_point][1][0][1] - hessians[1][q_point][2][0][2]) / mu_values[q_point] + omega * (epsilon_values[q_point] * values[1][q_point][0] - sigma_values[q_point] * values[0][q_point][0]) + right_hand_side_values[q_point] (3);
    residuals (4) = (hessians[1][q_point][1][0][0] + hessians[1][q_point][1][2][2] - hessians[1][q_point][0][0][1] - hessians[1][q_point][2][1][2]) / mu_values[q_point] + omega * (epsilon_values[q_point] * values[1][q_point][1] - sigma_values[q_point] * values[0][q_point][1]) + right_hand_side_values[q_point] (4);
    residuals (5) = (hessians[1][q_point][2][0][0] + hessians[1][q_point][2][1][1] - hessians[1][q_point][0][0][2] - hessians[1][q_point][1][1][2]) / mu_values[q_point] + omega * (epsilon_values[q_point] * values[1][q_point][2] - sigma_values[q_point] * values[0][q_point][2]) + right_hand_side_values[q_point] (5);

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      shape_values[0] = fe_values[re].value (i, q_point);
      shape_values[1] = fe_values[im].value (i, q_point);

      for (unsigned int d = 0; d < 3; ++d)
      {
        estimated_errors (i) += JxW_values[q_point] * (shape_values[0][d] * residuals (d)
            + shape_values[1][d] * residuals (d + 3));
      }
    }
  }
}

template<typename VECTOR>
void ResidualEstimatorMaxwell::jump_term(const typename DoFHandler<3>::active_cell_iterator& cell,
                                         const std::vector<VECTOR> &solution_vectors,
                                         const std::vector<CurrentFunctionPtr<3> > &sources,
                                         std::vector<double>& estimated_errors)
{
  Assert(estimated_errors.size() == solution_vectors.size(), ExcDimensionMismatch(estimated_errors.size(), solution_vectors.size()));

  size_t n_solutions = solution_vectors.size ();
  std::fill(estimated_errors.begin(), estimated_errors.end(), 0.);

  face_contributions.resize(n_solutions);

  // Here we compute the jump-based term. We have to consider the following two cases: (i) faces which have children and (ii) others.
  for (unsigned int face = 0; face < GeometryInfo<3>::faces_per_cell; ++face)
    if (!(cell->face (face)->at_boundary ()))
    {
      // First, case (i).
      if (cell->face (face)->has_children ())
        for (unsigned int subface = 0; subface < cell->face (face)->n_children (); ++subface)
        {
          bool new_face = true;

          for (size_t i = 0; i < face_contributions.size (); ++i)
          {
            std::map<DoFHandler<3>::face_iterator, double>::const_iterator it = face_contributions[i].find (cell->face (face)->child (subface));
            if (it != face_contributions[i].end ())
            {
              estimated_errors[i] += it->second;
              new_face = false;
            }
          }

          if (new_face)
          {
            const double weight = 0.5 * cell->face (face)->child (subface)->diameter () / fe_degree;

            fe_subface_values.reinit (cell, face, subface);

            const std::vector<double>& JxW_values = fe_subface_values.get_JxW_values ();
            const std::vector<Point<3> >& quadrature_points = fe_subface_values.get_quadrature_points ();
            const std::vector<Tensor<1, 3> >& normals = fe_subface_values.get_normal_vectors ();

            phys_model->conductivity_list (cell, quadrature_points, sigma_face_values);
            phys_model->permittivity_list (cell, quadrature_points, epsilon_face_values);
            phys_model->permeability_list (cell, quadrature_points, mu_face_values);

            const DoFHandler<3>::active_cell_iterator& neighbor = cell->neighbor_child_on_subface (face, subface);

            phys_model->conductivity_list (neighbor, quadrature_points, sigma_neighbor_values);
            phys_model->permittivity_list (neighbor, quadrature_points, epsilon_neighbor_values);
            phys_model->permeability_list (neighbor, quadrature_points, mu_neighbor_values);
            fe_face_neighbor_values.reinit (neighbor, cell->neighbor_of_neighbor (face));

            for (unsigned int i = 0; i < n_solutions; ++i)
            {
              fe_face_neighbor_values[re].get_function_curls (solution_vectors[i], curls_neighbor[0]);
              fe_face_neighbor_values[im].get_function_curls (solution_vectors[i], curls_neighbor[1]);
              fe_face_neighbor_values[re].get_function_values (solution_vectors[i], values_neighbor[0]);
              fe_face_neighbor_values[im].get_function_values (solution_vectors[i], values_neighbor[1]);
              fe_subface_values[re].get_function_curls (solution_vectors[i], curls[0]);
              fe_subface_values[im].get_function_curls (solution_vectors[i], curls[1]);
              fe_subface_values[re].get_function_values (solution_vectors[i], values_face[0]);
              fe_subface_values[im].get_function_values (solution_vectors[i], values_face[1]);

              sources[i]->current_density_value_list (cell, quadrature_points, current_density_values);
              sources[i]->current_density_value_list (neighbor, quadrature_points, current_density_neighbor_values);

              double error_contribution = 0.0;

              for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
              {
                // We compute the jump of the curl across the face.
                for (unsigned int j = 0; j < 2; ++j)
                {
                  curls[j][q_point] = curls[j][q_point] / mu_face_values[q_point];
                  curls_neighbor[j][q_point] = curls_neighbor[j][q_point] / mu_neighbor_values[q_point];

                  jump_curl[j] = cross_product_3d (normals[q_point], curls[j][q_point] - curls_neighbor[j][q_point]);
                }

                // And the jump of the FE solution itself.
                for (unsigned int d = 0; d < 3; ++d)
                {
                  jumps[0][d] = current_density_values[q_point] (d) + sigma_face_values[q_point] * values_face[0][q_point][d] + omega * (epsilon_neighbor_values[q_point] * values_neighbor[1][q_point][d] - epsilon_face_values[q_point] * values_face[1][q_point][d]) - current_density_neighbor_values[q_point] (d) - sigma_neighbor_values[q_point] * values_neighbor[0][q_point][d];
                  jumps[1][d] = current_density_values[q_point] (d + 3) + sigma_face_values[q_point] * values_face[1][q_point][d] + omega * (epsilon_face_values[q_point] * values_face[0][q_point][d] - epsilon_neighbor_values[q_point] * values_neighbor[0][q_point][d]) - current_density_neighbor_values[q_point] (d + 3) - sigma_neighbor_values[q_point] * values_neighbor[1][q_point][d];
                }

                jump_values (0) = normals[q_point] * jumps[0];
                jump_values (1) = normals[q_point] * jumps[1];
                error_contribution += JxW_values[q_point] * (jump_curl[0].norm_square () + jump_curl[1].norm_square () + jump_values.norm_sqr ());
              }

              error_contribution *= weight;
              estimated_errors[i] += error_contribution;
              face_contributions[i].insert (std::make_pair (cell->face (face)->child (subface), error_contribution));
            }
          }
        }
      // Case (ii)
      else
      {
        bool new_face = true;

        for (size_t i = 0; i < face_contributions.size (); ++i)
        {
          std::map<DoFHandler<3>::face_iterator, double>::const_iterator it = face_contributions[i].find (cell->face (face));
          if (it != face_contributions[i].end ())
          {
            estimated_errors[i] += it->second;
            new_face = false;
          }
        }

        if (new_face)
        {
          const bool neighbor_is_coarser = cell->neighbor_is_coarser (face);
          const double weight = 0.5 * cell->face (face)->diameter () / fe_degree;

          fe_face_values.reinit (cell, face);

          const std::vector<double>& JxW_values = fe_face_values.get_JxW_values ();
          const std::vector<Point<3> >& quadrature_points = fe_face_values.get_quadrature_points ();
          const std::vector<Tensor<1, 3> >& normals = fe_face_values.get_normal_vectors ();

          phys_model->conductivity_list (cell, quadrature_points, sigma_face_values);
          phys_model->permittivity_list (cell, quadrature_points, epsilon_face_values);
          phys_model->permeability_list (cell, quadrature_points, mu_face_values);

          for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
            epsilon_face_values[q_point] *= omega;

          const DoFHandler<3>::active_cell_iterator& neighbor = cell->neighbor (face);

          phys_model->conductivity_list (neighbor, quadrature_points, sigma_neighbor_values);
          phys_model->permittivity_list (neighbor, quadrature_points, epsilon_neighbor_values);
          phys_model->permeability_list (neighbor, quadrature_points, mu_neighbor_values);

          if (neighbor_is_coarser)
          {
            const std::pair<unsigned int, unsigned int>& neighbor_of_coarser_neighbor = cell->neighbor_of_coarser_neighbor (face);

            fe_subface_neighbor_values.reinit (neighbor, neighbor_of_coarser_neighbor.first, neighbor_of_coarser_neighbor.second);
          }

          else
            fe_face_neighbor_values.reinit (cell->neighbor (face), cell->neighbor_of_neighbor (face));

          for (unsigned int i = 0; i < n_solutions; ++i)
          {
            if (neighbor_is_coarser)
            {
              fe_subface_neighbor_values[re].get_function_curls (solution_vectors[i], curls_neighbor[0]);
              fe_subface_neighbor_values[im].get_function_curls (solution_vectors[i], curls_neighbor[1]);
              fe_subface_neighbor_values[re].get_function_values (solution_vectors[i], values_neighbor[0]);
              fe_subface_neighbor_values[im].get_function_values (solution_vectors[i], values_neighbor[1]);
            }

            else
            {
              fe_face_neighbor_values[re].get_function_curls (solution_vectors[i], curls_neighbor[0]);
              fe_face_neighbor_values[im].get_function_curls (solution_vectors[i], curls_neighbor[1]);
              fe_face_neighbor_values[re].get_function_values (solution_vectors[i], values_neighbor[0]);
              fe_face_neighbor_values[im].get_function_values (solution_vectors[i], values_neighbor[1]);
            }

            fe_face_values[re].get_function_curls (solution_vectors[i], curls[0]);
            fe_face_values[im].get_function_curls (solution_vectors[i], curls[1]);
            fe_face_values[re].get_function_values (solution_vectors[i], values_face[0]);
            fe_face_values[im].get_function_values (solution_vectors[i], values_face[1]);

            sources[i]->current_density_value_list (cell, quadrature_points, current_density_values);
            sources[i]->current_density_value_list (neighbor, quadrature_points, current_density_neighbor_values);

            double error_contribution = 0.0;

            for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
            {
              // We compute the jump of the curl across the face.
              for (unsigned int j = 0; j < 2; ++j)
              {
                curls[j][q_point] = curls[j][q_point] / mu_face_values[q_point];
                curls_neighbor[j][q_point] = curls_neighbor[j][q_point] / mu_neighbor_values[q_point];

                jump_curl[j] = cross_product_3d (normals[q_point], curls[j][q_point] - curls_neighbor[j][q_point]);
              }

              // And the jump of the FE solution itself.
              for (unsigned int d = 0; d < 3; ++d)
              {
                jumps[0][d] = current_density_values[q_point] (d) + sigma_face_values[q_point] * values_face[0][q_point][d] + omega * (epsilon_neighbor_values[q_point] * values_neighbor[1][q_point][d] - epsilon_face_values[q_point] * values_face[1][q_point][d]) - current_density_neighbor_values[q_point] (d) - sigma_neighbor_values[q_point] * values_neighbor[0][q_point][d];
                jumps[1][d] = current_density_values[q_point] (d + 3) + sigma_face_values[q_point] * values_face[1][q_point][d] + omega * (epsilon_face_values[q_point] * values_face[0][q_point][d] - epsilon_neighbor_values[q_point] * values_neighbor[0][q_point][d]) - current_density_neighbor_values[q_point] (d + 3) - sigma_neighbor_values[q_point] * values_neighbor[1][q_point][d];
              }

              jump_values (0) = normals[q_point] * jumps[0];
              jump_values (1) = normals[q_point] * jumps[1];
              error_contribution += JxW_values[q_point] * (jump_curl[0].norm_square () + jump_curl[1].norm_square () + jump_values.norm_sqr ());
            }

            error_contribution *= weight;
            estimated_errors[i] += error_contribution;
            face_contributions[i].insert (std::make_pair (cell->face (face), error_contribution));
          }
        }
      }
    }
}

template<typename VECTOR>
void ResidualEstimatorMaxwell::jump_error_at_dofs(const DoFHandler<3>::active_cell_iterator &cell,
                                                  const VECTOR &solution_vector,
                                                  const CurrentFunctionPtr<3> &sources,
                                                  Vector<double> &estimated_errors)
{
  unsigned dofs_per_cell = fe_values.get_fe().dofs_per_quad;
  std::vector<Tensor<1, 3> > shape_values(2);

  // Here we compute the jump-based term. We have to consider the following two cases: (i) faces which have children and (ii) others.
  for (unsigned int face = 0; face < GeometryInfo<3>::faces_per_cell; ++face)
    if (!(cell->face (face)->at_boundary ()))
    {
      // First, case (i).
      if (cell->face (face)->has_children ())
        for (unsigned int subface = 0; subface < cell->face (face)->n_children (); ++subface)
        {
          bool new_face = true;

          if (new_face)
          {
            fe_subface_values.reinit (cell, face, subface);

            const std::vector<double>& JxW_values = fe_subface_values.get_JxW_values ();
            const std::vector<Point<3> >& quadrature_points = fe_subface_values.get_quadrature_points ();
            const std::vector<Point<3> >& normals = fe_subface_values.get_normal_vectors ();

            phys_model->conductivity_list (cell, quadrature_points, sigma_face_values);
            phys_model->permittivity_list (cell, quadrature_points, epsilon_face_values);
            phys_model->permeability_list (cell, quadrature_points, mu_face_values);

            const DoFHandler<3>::active_cell_iterator& neighbor = cell->neighbor_child_on_subface (face, subface);

            phys_model->conductivity_list (neighbor, quadrature_points, sigma_neighbor_values);
            phys_model->permittivity_list (neighbor, quadrature_points, epsilon_neighbor_values);
            phys_model->permeability_list (neighbor, quadrature_points, mu_neighbor_values);
            fe_face_neighbor_values.reinit (neighbor, cell->neighbor_of_neighbor (face));

            fe_face_neighbor_values[re].get_function_curls (solution_vector, curls_neighbor[0]);
            fe_face_neighbor_values[im].get_function_curls (solution_vector, curls_neighbor[1]);
            fe_face_neighbor_values[re].get_function_values (solution_vector, values_neighbor[0]);
            fe_face_neighbor_values[im].get_function_values (solution_vector, values_neighbor[1]);
            fe_subface_values[re].get_function_curls (solution_vector, curls[0]);
            fe_subface_values[im].get_function_curls (solution_vector, curls[1]);
            fe_subface_values[re].get_function_values (solution_vector, values_face[0]);
            fe_subface_values[im].get_function_values (solution_vector, values_face[1]);

            sources->current_density_value_list (cell, quadrature_points, current_density_values);
            sources->current_density_value_list (neighbor, quadrature_points, current_density_neighbor_values);

            for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
            {
              // We compute the jump of the curl across the face.
              for (unsigned int j = 0; j < 2; ++j)
              {
                curls[j][q_point] = curls[j][q_point] / mu_face_values[q_point];
                curls_neighbor[j][q_point] = curls_neighbor[j][q_point] / mu_neighbor_values[q_point];

                jump_curl[j] = cross_product_3d (normals[q_point], curls[j][q_point] - curls_neighbor[j][q_point]);
              }

              // And the jump of the FE solution itself.
              for (unsigned int d = 0; d < 3; ++d)
              {
                jumps[0][d] = current_density_values[q_point] (d) + sigma_face_values[q_point] * values_face[0][q_point][d] + omega * (epsilon_neighbor_values[q_point] * values_neighbor[1][q_point][d] - epsilon_face_values[q_point] * values_face[1][q_point][d]) - current_density_neighbor_values[q_point] (d) - sigma_neighbor_values[q_point] * values_neighbor[0][q_point][d];
                jumps[1][d] = current_density_values[q_point] (d + 3) + sigma_face_values[q_point] * values_face[1][q_point][d] + omega * (epsilon_face_values[q_point] * values_face[0][q_point][d] - epsilon_neighbor_values[q_point] * values_neighbor[0][q_point][d]) - current_density_neighbor_values[q_point] (d + 3) - sigma_neighbor_values[q_point] * values_neighbor[1][q_point][d];
              }

              jump_values (0) = normals[q_point] * jumps[0];
              jump_values (1) = normals[q_point] * jumps[1];

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                shape_values[0] = fe_subface_values[re].value (j, q_point);
                shape_values[1] = fe_subface_values[im].value (j, q_point);

                for (unsigned int d = 0; d < 3; ++d)
                {
                  estimated_errors (j) += JxW_values[q_point] *
                      (shape_values[0][d] * jump_curl[0][d]
                      +shape_values[1][d] * jump_curl[1][d]);
                }

              }
            }
          }
        }
      // Case (ii)
      else
      {
        bool new_face = true;

        if (new_face)
        {
          const bool neighbor_is_coarser = cell->neighbor_is_coarser (face);

          fe_face_values.reinit (cell, face);

          const std::vector<double>& JxW_values = fe_face_values.get_JxW_values ();
          const std::vector<Point<3> >& quadrature_points = fe_face_values.get_quadrature_points ();
          const std::vector<Point<3> >& normals = fe_face_values.get_normal_vectors ();

          phys_model->conductivity_list (cell, quadrature_points, sigma_face_values);
          phys_model->permittivity_list (cell, quadrature_points, epsilon_face_values);
          phys_model->permeability_list (cell, quadrature_points, mu_face_values);

          for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
            epsilon_face_values[q_point] *= omega;

          const DoFHandler<3>::active_cell_iterator& neighbor = cell->neighbor (face);

          phys_model->conductivity_list (neighbor, quadrature_points, sigma_neighbor_values);
          phys_model->permittivity_list (neighbor, quadrature_points, epsilon_neighbor_values);
          phys_model->permeability_list (neighbor, quadrature_points, mu_neighbor_values);

          if (neighbor_is_coarser)
          {
            const std::pair<unsigned int, unsigned int>& neighbor_of_coarser_neighbor = cell->neighbor_of_coarser_neighbor (face);

            fe_subface_neighbor_values.reinit (neighbor, neighbor_of_coarser_neighbor.first, neighbor_of_coarser_neighbor.second);
          }

          else
            fe_face_neighbor_values.reinit (cell->neighbor (face), cell->neighbor_of_neighbor (face));

          if (neighbor_is_coarser)
          {
            fe_subface_neighbor_values[re].get_function_curls (solution_vector, curls_neighbor[0]);
            fe_subface_neighbor_values[im].get_function_curls (solution_vector, curls_neighbor[1]);
            fe_subface_neighbor_values[re].get_function_values (solution_vector, values_neighbor[0]);
            fe_subface_neighbor_values[im].get_function_values (solution_vector, values_neighbor[1]);
          }

          else
          {
            fe_face_neighbor_values[re].get_function_curls (solution_vector, curls_neighbor[0]);
            fe_face_neighbor_values[im].get_function_curls (solution_vector, curls_neighbor[1]);
            fe_face_neighbor_values[re].get_function_values (solution_vector, values_neighbor[0]);
            fe_face_neighbor_values[im].get_function_values (solution_vector, values_neighbor[1]);
          }

          fe_face_values[re].get_function_curls (solution_vector, curls[0]);
          fe_face_values[im].get_function_curls (solution_vector, curls[1]);
          fe_face_values[re].get_function_values (solution_vector, values_face[0]);
          fe_face_values[im].get_function_values (solution_vector, values_face[1]);

          sources->current_density_value_list (cell, quadrature_points, current_density_values);
          sources->current_density_value_list (neighbor, quadrature_points, current_density_neighbor_values);

          for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
          {
            // We compute the jump of the curl across the face.
            for (unsigned int j = 0; j < 2; ++j)
            {
              curls[j][q_point] = curls[j][q_point] / mu_face_values[q_point];
              curls_neighbor[j][q_point] = curls_neighbor[j][q_point] / mu_neighbor_values[q_point];

              jump_curl[j] = cross_product_3d (normals[q_point], curls[j][q_point] - curls_neighbor[j][q_point]);
            }

            // And the jump of the FE solution itself.
            for (unsigned int d = 0; d < 3; ++d)
            {
              jumps[0][d] = current_density_values[q_point] (d) + sigma_face_values[q_point] * values_face[0][q_point][d] + omega * (epsilon_neighbor_values[q_point] * values_neighbor[1][q_point][d] - epsilon_face_values[q_point] * values_face[1][q_point][d]) - current_density_neighbor_values[q_point] (d) - sigma_neighbor_values[q_point] * values_neighbor[0][q_point][d];
              jumps[1][d] = current_density_values[q_point] (d + 3) + sigma_face_values[q_point] * values_face[1][q_point][d] + omega * (epsilon_face_values[q_point] * values_face[0][q_point][d] - epsilon_neighbor_values[q_point] * values_neighbor[0][q_point][d]) - current_density_neighbor_values[q_point] (d + 3) - sigma_neighbor_values[q_point] * values_neighbor[1][q_point][d];
            }

            jump_values (0) = normals[q_point] * jumps[0];
            jump_values (1) = normals[q_point] * jumps[1];

            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              shape_values[0] = fe_face_values[re].value (j, q_point);
              shape_values[1] = fe_face_values[im].value (j, q_point);

              for (unsigned int d = 0; d < 3; ++d)
              {
                estimated_errors (j) += JxW_values[q_point] *
                    (shape_values[0][d] * jump_curl[0][d]
                    +shape_values[1][d] * jump_curl[1][d]);
              }
            }
          }
        }
      }
    }
}

#endif
