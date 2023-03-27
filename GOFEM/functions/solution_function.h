#ifndef SOLUTION_FUNCTION_H
#define SOLUTION_FUNCTION_H

#include <deal.II/base/function.h>
#include "common.h"

using namespace dealii;

template<int dim>
class SolutionFunction: public Function<dim>
{
public:
  SolutionFunction(unsigned n_components):
    Function<dim>(n_components)
  {}

  virtual SolutionFunction<dim>* clone() const = 0;

  virtual void set_field(FieldFormulation ftype) = 0;
  virtual void set_frequency(double frequency) = 0;

  /*
   * Returns gradient tensor where gradient[i][d] denotes
   * partial derivative of i-th component w.r.t. d-th dimension
   */
  using Function<dim>::vector_gradients; // Silence -Woverloaded-virtual
  void vector_gradients(const Point<dim> &p,
                        std::vector<Tensor<1, dim> > &gradients) const;

  void curl(const Point<dim> &p, Vector<double> &curl_value) const;
  void curl_list(const std::vector<Point<dim>> &points,
                 std::vector<Vector<double>> &curl_values) const;
};

template<int dim>
using SolutionFunctionPtr = std::shared_ptr<SolutionFunction<dim>>;

template<int dim>
void SolutionFunction<dim>::vector_gradients (const Point<dim> &p, std::vector<Tensor<1, dim> > &gradients) const
{
  Assert (gradients.size () == 2*dim, ExcDimensionMismatch (gradients.size (), 2*dim));

  const double h = 1e-7;
  Point<dim> tmp;
  std::vector<Vector<double> > values (2, Vector<double> (2*dim));

  for (unsigned int d = 0; d < dim; ++d)
  {
    tmp = p;
    tmp (d) += h;
    this->vector_value (tmp, values[0]);
    tmp (d) -= 2.0 * h;
    this->vector_value (tmp, values[1]);

    for (unsigned int i = 0; i < 2*dim; ++i)
      gradients[i][d] = 0.5 * (values[0] (i) - values[1] (i)) / h;
  }
}

template<int dim>
void SolutionFunction<dim>::curl(const Point<dim> &p, Vector<double> &curl_value) const
{
  const unsigned curl_size = (dim == 2) ? 2 : 6;
  (void)curl_size;
  Assert (curl_value.size () == curl_size, ExcDimensionMismatch (curl_value.size (), curl_size));

  std::vector<Tensor<1, dim> > gradients(dim*2);
  vector_gradients(p, gradients);

  if(dim == 3)
  {
    curl_value[0] = gradients[2][1] - gradients[1][2];
    curl_value[1] = gradients[0][2] - gradients[2][0];
    curl_value[2] = gradients[1][0] - gradients[0][1];

    curl_value[3] = gradients[5][1] - gradients[4][2];
    curl_value[4] = gradients[3][2] - gradients[5][0];
    curl_value[5] = gradients[4][0] - gradients[3][1];
  }
  else if(dim == 2)
  {
    curl_value[0] = gradients[1][0] - gradients[0][1];
    curl_value[1] = gradients[3][0] - gradients[2][1];
  }
}

template<int dim>
void SolutionFunction<dim>::curl_list(const std::vector<Point<dim> > &points,
                                      std::vector<Vector<double> > &curl_values) const
{
  for(unsigned i = 0; i < points.size(); ++i)
    curl(points[i], curl_values[i]);
}

#endif // SOLUTION_FUNCTION_H
