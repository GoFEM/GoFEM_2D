#ifndef SECONDARY_SOURCE_H
#define SECONDARY_SOURCE_H

#include "functions/current_function.h"
#include "functions/solution_function.h"

#include "physical_model/physical_model.h"

/*
 * Implements deal.II-like interface for calculation of secondary
 * sources due to specified 3D and 1D conductivity models.
 * Assumes +i*omega*t convention.
 */
class SecondarySourceEM: public CurrentFunction<3>
{
public:
  SecondarySourceEM (const BackgroundModel<3> &bgmodel,
                     const PhysicalModelPtr<3> &phys_model,
                     const SolutionFunctionPtr<3> &solution,
                     const double frequency,
                     const std::string &name = "");

  SecondarySourceEM* clone() const;

  virtual void current_density_value (const typename DoFHandler<3>::active_cell_iterator& cell,
                                      const Point<3> &p, Vector<double> &values) const;
  virtual void vector_value (const typename DoFHandler<3>::active_cell_iterator& cell,
                             const Point<3> &p, Vector<double> &values) const;

  void reinit (const double frequency,
               const BackgroundModel<3> &bgmodel,
               const PhysicalModelPtr<3> &phys_model);

  void set_formulation (FieldFormulation f);

  // Prevent hidden overload methods
  using Function<3>::vector_value;

private:
  void current_at (const typename DoFHandler<3>::active_cell_iterator& cell,
                   const Point<3> &p, Tensor<1, 3, dcomplex> &J) const;

private:
  const BackgroundModel<3>* background_model;
  PhysicalModelPtr<3> model;
  std::shared_ptr<SolutionFunction<3>> analytical_solution;
  FieldFormulation formulation;
  double omega;

  mutable Tensor<1, 3, dcomplex> F;
  mutable Vector<double> field_values;
};

#endif // SECONDARY_SOURCE_H
