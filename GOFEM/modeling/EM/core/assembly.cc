#include "assembly.h"

namespace Assembly
{
namespace Scratch
{

template<int dim>
MaxwellSystem<dim>::MaxwellSystem (const FiniteElement<dim>& fe, const Mapping<dim>& mapping,
                              const Quadrature<dim>& quadrature, const UpdateFlags update_flags,
                              const Quadrature<dim-1>& face_quadrature, const UpdateFlags face_update_flags):
  fe_face_values (mapping, fe, face_quadrature, face_update_flags),
  fe_values (mapping, fe, quadrature, update_flags),
  epsilon_values (quadrature.size ()), epsilon_values_face (face_quadrature.size ()),
  sigma_values (quadrature.size ()), complex_sigma_values(quadrature.size()),
  sigma_values_face (face_quadrature.size ()),
  mu_values (quadrature.size ()), mu_values_face (face_quadrature.size ()),
  shape_curls (fe.n_blocks(), std::vector<Tensor<1, dim> > (fe.dofs_per_cell)),
  shape_grads (fe.n_blocks(), std::vector<Tensor<1, dim> > (fe.dofs_per_cell)),
  shape_grads_rotated (fe.n_blocks(), std::vector<Tensor<1, dim> > (fe.dofs_per_cell)),
  shape_values (fe.n_blocks(), std::vector<Tensor<1, dim> > (fe.dofs_per_cell)),
  shape_scalar_values (fe.n_blocks(), std::vector<double> (fe.dofs_per_cell)),
  right_hand_side_values (quadrature.size (), Vector<double> (2*dim))
{}

template<int dim>
MaxwellSystem<dim>::MaxwellSystem (const MaxwellSystem<dim>& scratch):
  fe_face_values (scratch.fe_face_values.get_mapping (), scratch.fe_face_values.get_fe (),
                  scratch.fe_face_values.get_quadrature (), scratch.fe_face_values.get_update_flags ()),
  fe_values (scratch.fe_values.get_mapping (), scratch.fe_values.get_fe (),
             scratch.fe_values.get_quadrature (), scratch.fe_values.get_update_flags ()),
  epsilon_values (scratch.epsilon_values),
  epsilon_values_face (scratch.epsilon_values_face),
  sigma_values (scratch.sigma_values),
  complex_sigma_values(scratch.complex_sigma_values),
  sigma_values_face (scratch.sigma_values_face),
  mu_values (scratch.mu_values),
  mu_values_face (scratch.mu_values_face),
  shape_curls (scratch.shape_curls),
  shape_grads (scratch.shape_grads),
  shape_grads_rotated (scratch.shape_grads_rotated),
  shape_values (scratch.shape_values),
  shape_scalar_values (scratch.shape_scalar_values),
  right_hand_side_values (scratch.right_hand_side_values)
{}

template struct MaxwellSystem<2>;
template struct MaxwellSystem<3>;
}

namespace CopyData
{
MaxwellSystem::MaxwellSystem (const unsigned int dofs_per_cell):
  local_matrix (dofs_per_cell, dofs_per_cell),
  local_dof_indices (dofs_per_cell),
  local_rhs (Vector<double> (dofs_per_cell))
{}

MaxwellSystem::MaxwellSystem (const MaxwellSystem& data):
  local_matrix (data.local_matrix),
  local_dof_indices (data.local_dof_indices),
  local_rhs (data.local_rhs)
{}
}
}
