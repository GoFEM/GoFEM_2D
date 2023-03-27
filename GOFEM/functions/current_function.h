#ifndef CURRENT_FUNCTION_H
#define CURRENT_FUNCTION_H

#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/dofs/dof_handler.h>

using namespace dealii;

/*
 * Abstract current function class.
 */
template <int dim>
class CurrentFunction: public Function<dim>
{
public:
  CurrentFunction(const std::string source_name);

  std::string get_name() const;
  void set_name(const std::string source_name);

  virtual void current_density_value (const typename DoFHandler<dim>::active_cell_iterator& cell, const Point<dim> &p, Vector<double> &values) const;
  virtual void current_density_value_list (const typename DoFHandler<dim>::active_cell_iterator& cell, const std::vector<Point<dim> > &points, std::vector<Vector<double> > &vectors) const;
  virtual void divergence (const typename DoFHandler<dim>::active_cell_iterator& cell, const Point<dim> &p, Vector<double> &divergence) const;
  virtual void divergence_list (const typename DoFHandler<dim>::active_cell_iterator& cell, const std::vector<Point<dim> > & points, std::vector<Vector<double> > &vectors) const;
  virtual void vector_value_list (const typename DoFHandler<dim>::active_cell_iterator& cell, const std::vector<Point<dim> > & points, std::vector<Vector<double> > &vectors) const;
  virtual void vector_value (const typename DoFHandler<dim>::active_cell_iterator& cell, const Point<dim> &p, Vector<double> &values) const;

protected:
  virtual void vector_gradients (const typename DoFHandler<dim>::active_cell_iterator& cell,
                                 const Point<dim> &p, std::vector<Tensor<1, dim> > &gradients) const;

  // Prevent overload hidden methods (suppresses compiler warning)
  using Function<dim>::vector_gradients;
  using Function<dim>::vector_value;
  using Function<dim>::vector_value_list;

  CurrentFunction(const CurrentFunction<dim>&):
    Function<dim>()
  {}

protected:
  std::string name;
};

template<int dim>
using CurrentFunctionPtr = std::shared_ptr<CurrentFunction<dim>>;

#endif // CURRENT_FUNCTION_H
