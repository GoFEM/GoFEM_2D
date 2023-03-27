#ifndef ASSEMBLY_H
#define ASSEMBLY_H

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>

using namespace dealii;

namespace Assembly
{
  namespace Scratch
  {
    template<int dim>
    struct MaxwellSystem
    {
      FEFaceValues<dim> fe_face_values;
      FEValues<dim> fe_values;
      MaxwellSystem (const FiniteElement<dim>& fe, const Mapping<dim>& mapping,
                     const Quadrature<dim>& quadrature,const UpdateFlags update_flags,
                     const Quadrature<dim-1>& face_quadrature, const UpdateFlags face_update_flags);
      MaxwellSystem (const MaxwellSystem<dim> &data);
      std::vector<double> epsilon_values;
      std::vector<double> epsilon_values_face;
      std::vector<double> sigma_values;
      std::vector<std::complex<double>> complex_sigma_values; // for applications with induced polarization
      std::vector<double> sigma_values_face;
      std::vector<double> mu_values;
      std::vector<double> mu_values_face;
      std::vector<std::vector<Tensor<1, dim> > > shape_curls;
      std::vector<std::vector<Tensor<1, dim> > > shape_grads;
      std::vector<std::vector<Tensor<1, dim> > > shape_grads_rotated;
      std::vector<std::vector<Tensor<1, dim> > > shape_values;
      std::vector<std::vector<double> > shape_scalar_values;
      std::vector<Vector<double> > right_hand_side_values;
    };
    
  }

  namespace CopyData
  {
    struct MaxwellSystem
    {
      FullMatrix<double> local_matrix;
      MaxwellSystem (const unsigned int dofs_per_cell);
      MaxwellSystem (const MaxwellSystem& data);
      std::vector<types::global_dof_index> local_dof_indices;
      Vector<double> local_rhs;
    };
  }
}

#endif // ASSEMBLY_H
