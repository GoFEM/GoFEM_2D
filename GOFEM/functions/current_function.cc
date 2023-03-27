#include "current_function.h"

template<int dim>
CurrentFunction<dim>::CurrentFunction (const std::string source_name):
  Function<dim> (dim*2), name(source_name)
{}

template<int dim>
std::string CurrentFunction<dim>::get_name() const
{
  return name;
}

template<int dim>
void CurrentFunction<dim>::set_name(const std::string source_name)
{
  name = source_name;
}

template<int dim>
void CurrentFunction<dim>::current_density_value_list (const typename DoFHandler<dim>::active_cell_iterator& cell, const std::vector<Point<dim> > &points, std::vector<Vector<double> > &vectors) const
{
  Assert (vectors.size () == points.size (), ExcDimensionMismatch (vectors.size (), points.size ()));

  for (size_t i = 0; i < points.size (); ++i)
    current_density_value (cell, points[i], vectors[i]);
}

template<int dim>
void CurrentFunction<dim>::divergence_list (const typename DoFHandler<dim>::active_cell_iterator& cell, const std::vector<Point<dim> > &points, std::vector<Vector<double> > &vectors) const
{
  Assert (vectors.size () == points.size (), ExcDimensionMismatch (vectors.size (), points.size ()));

  for (size_t i = 0; i < points.size (); ++i)
    divergence (cell, points[i], vectors[i]);
}

template<int dim>
void CurrentFunction<dim>::vector_value_list(const typename DoFHandler<dim>::active_cell_iterator &cell, const std::vector<Point<dim> > &points, std::vector<Vector<double> > &vectors) const
{
  Assert (vectors.size () == points.size (), ExcDimensionMismatch (vectors.size (), points.size ()));

  for (size_t i = 0; i < points.size (); ++i)
    vector_value (cell, points[i], vectors[i]);
}

template<int dim>
void CurrentFunction<dim>::current_density_value(const typename DoFHandler<dim>::active_cell_iterator &/*cell*/,
                                                const Point<dim> &/*p*/, Vector<double> &values) const
{
  values = 0;
}

template<int dim>
void CurrentFunction<dim>::vector_gradients (const typename DoFHandler<dim>::active_cell_iterator& cell,
                                            const Point<dim> &p, std::vector<Tensor<1, dim> > &gradients) const
{
  Assert (gradients.size () == dim*2, ExcDimensionMismatch (gradients.size (), dim*2));

  const double h = 1e-7;
  Point<dim> tmp;
  std::vector<Vector<double> > values (2, Vector<double> (dim*2));

  for (unsigned int d = 0; d < dim; ++d)
  {
    tmp = p;
    tmp (d) += h;
    current_density_value (cell, tmp, values[0]);
    tmp (d) -= 2.0 * h;
    current_density_value (cell, tmp, values[1]);

    for (unsigned int i = 0; i < gradients.size(); ++i)
      gradients[i][d] = 0.5 * (values[0] (i) - values[1] (i)) / h;
  }
}

template<int dim>
void CurrentFunction<dim>::divergence(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                     const Point<dim> &p, Vector<double> &divergence) const
{
  Assert (divergence.size () == 2, ExcDimensionMismatch (divergence.size (), 2));

  std::vector<Tensor<1, dim> > gradients (dim*2);

  divergence *= 0.;

  vector_gradients (cell, p, gradients);
  for (unsigned int d = 0; d < dim; ++d)
  {
    divergence (0) = gradients[d][d] + gradients[d][d] + gradients[d][d];
    divergence (1) = gradients[d+3][d] + gradients[d+3][d] + gradients[d+3][d];
  }
}

template<int dim>
void CurrentFunction<dim>::vector_value(const typename DoFHandler<dim>::active_cell_iterator& /*cell*/,
                                       const Point<dim> &/*p*/, Vector<double> &values) const
{
  values = 0;
}

template class CurrentFunction<2>;
template class CurrentFunction<3>;
