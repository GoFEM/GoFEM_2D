#include "error_estimator.h"

ResidualEstimatorMaxwell::ResidualEstimatorMaxwell(const FESystem<3> &fe, const Mapping<3> &mapping,
                                                   const PhysicalModelPtr<3> &physical_model, double frequency):
  re(0), im(3), fe_degree(fe.degree),
  omega(2. * numbers::PI * frequency),
  quadrature(fe.degree + 1), face_quadrature(fe.degree + 1),
  n_q_points(quadrature.size()), n_face_q_points(face_quadrature.size()),
  fe_face_neighbor_values (mapping, fe, face_quadrature, update_gradients | update_values),
  fe_face_values (mapping, fe, face_quadrature, update_gradients | update_JxW_values | update_normal_vectors | update_quadrature_points | update_values),
  fe_subface_neighbor_values (mapping, fe, face_quadrature, update_gradients | update_values),
  fe_subface_values (mapping, fe, face_quadrature, update_gradients | update_JxW_values | update_normal_vectors | update_quadrature_points | update_values),
  fe_values (mapping, fe, quadrature, update_gradients | update_hessians | update_JxW_values | update_quadrature_points | update_values),
  jumps (2), jump_curl (2), jump_values (2), residuals (8),
  phys_model(physical_model)
{
  epsilon_face_values.resize  (n_face_q_points);
  epsilon_neighbor_values.resize  (n_face_q_points);
  epsilon_values.resize  (n_q_points);
  sigma_face_values.resize  (n_face_q_points);
  sigma_neighbor_values.resize  (n_face_q_points);
  sigma_values.resize  (n_q_points);
  mu_face_values.resize  (n_face_q_points);
  mu_neighbor_values.resize  (n_face_q_points);
  mu_values.resize  (n_q_points);

  //std::vector<std::vector<double> > divergences (2, std::vector<double> (n_q_points));
  curls.resize (2, std::vector<Tensor<1, 3> > (n_face_q_points));
  curls_neighbor.resize (2, std::vector<Tensor<1, 3> > (n_face_q_points));
  values.resize (2, std::vector<Tensor<1, 3> > (n_q_points));
  rhs_values.resize(2, std::vector<Tensor<1, 3> > (n_q_points));
  values_face.resize (2, std::vector<Tensor<1, 3> > (n_face_q_points));
  values_neighbor.resize (2, std::vector<Tensor<1, 3> > (n_face_q_points));
  hessians.resize (2, std::vector<Tensor<3, 3> > (n_q_points));

  current_density_values.resize (n_face_q_points, Vector<double> (6));
  current_density_neighbor_values.resize (n_face_q_points, Vector<double> (6));
  right_hand_side_values.resize (n_q_points, Vector<double> (6));
}
