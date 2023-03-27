#include "secondary_source.h"

SecondarySourceEM::SecondarySourceEM (const BackgroundModel<3> &bgmodel,
                                      const PhysicalModelPtr<3> &phys_model,
                                      const SolutionFunctionPtr<3> &solution,
                                      const double frequency, const std::string &name):
  CurrentFunction<3>(name), analytical_solution(solution), field_values(6)
{
  reinit (frequency, bgmodel, phys_model);
}

SecondarySourceEM *SecondarySourceEM::clone() const
{
  return new SecondarySourceEM(*this);
}

void SecondarySourceEM::current_at (const typename DoFHandler<3>::active_cell_iterator& cell,
                                    const Point<3> &p, Tensor<1, 3, dcomplex> &J) const
{
  const double sigma_bg = background_model->conductivity_at (cell, p);
  const double eps_bg = background_model->permittivity_at (cell, p);
  const dcomplex sbg = sigma_bg + II * omega * eps_bg;

  const double sigma = model->conductivity_at (cell, p);
  const double eps = model->permittivity_at (cell, p);
  const dcomplex s = sigma + II * omega * eps;

  if(fabs(s - sbg) < std::numeric_limits<double>::epsilon())
  {
    J *= 0.;
  }
  else
  {
    analytical_solution->set_field(EField);
    analytical_solution->vector_value(p, field_values);
    analytical_solution->set_field(formulation);
    J[0] = dcomplex(field_values[0], field_values[3]);
    J[1] = dcomplex(field_values[1], field_values[4]);
    J[2] = dcomplex(field_values[2], field_values[5]);

    J *= (s - sbg);
  }
}

inline
void SecondarySourceEM::current_density_value (const typename DoFHandler<3>::active_cell_iterator& cell,
                                              const Point<3> &p, Vector<double> &values) const
{
  Assert (values.size () == 6, ExcDimensionMismatch (values.size(), 6));

  current_at (cell, p, F);

  values(0) = F[0].real ();
  values(3) = F[0].imag ();
  values(1) = F[1].real ();
  values(4) = F[1].imag ();
  values(2) = F[2].real ();
  values(5) = F[2].imag ();
}

inline
void SecondarySourceEM::vector_value (const typename DoFHandler<3>::active_cell_iterator& cell,
                                     const Point<3> &p, Vector<double> &values) const
{
  Assert (values.size () == 6, ExcDimensionMismatch (values.size(), 6));

  current_at (cell, p, F);

  if(formulation == EField || formulation == EFieldStabilized)
  {
    F *= -II*omega;
  }
  else if(formulation == HField || formulation == HFieldStabilized)
  {
    const double sigma = model->conductivity_at (cell, p);
    const double eps = model->permittivity_at (cell, p);
    const dcomplex s = sigma + II * omega * eps;

//    const double sigma_bg = background_model->conductivity_at (cell, p);
//    const double eps_bg = background_model->permittivity_at (cell, p);
//    const dcomplex sbg = sigma_bg + II * omega * eps_bg;
//    F *= 1. / (s - sbg);
//    F *= (1./sbg - 1./s)*s;
    F *= 1. / s;
  }

  values(0) = F[0].real ();
  values(3) = F[0].imag ();
  values(1) = F[1].real ();
  values(4) = F[1].imag ();
  values(2) = F[2].real ();
  values(5) = F[2].imag ();
}

void SecondarySourceEM::reinit (const double frequency,
                                const BackgroundModel<3> &bgmodel,
                                const PhysicalModelPtr<3> &phys_model)
{
  background_model = &bgmodel;
  model = phys_model;
  omega = 2.0 * numbers::PI * frequency;

  analytical_solution->set_frequency(frequency);
}

void SecondarySourceEM::set_formulation(FieldFormulation f)
{
  formulation = f;
}
