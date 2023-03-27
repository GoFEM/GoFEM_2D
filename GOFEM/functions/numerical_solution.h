#ifndef NUMERICAL_SOLUTION_H
#define NUMERICAL_SOLUTION_H

#include <deal.II/numerics/fe_field_function.h>

#include "modeling/EM/2D/mt2dfem.h"
#include "solution_function.h"

class NumericalSolutionMT2D: public SolutionFunction<3>
{
public:
  NumericalSolutionMT2D(const PhysicalModelPtr<2> &model, double frequency,
                        unsigned fe_order, unsigned mapping_order,
                        PlaneWavePolarization polarization,
                        PhaseConvention phase,
                        FieldFormulation field_type,
                        const std::string function_name);

  NumericalSolutionMT2D* clone() const;

  void set_field(FieldFormulation ftype);
  void set_frequency(double frequency);
  void curl_list(const std::vector<Point<3> > &points, std::vector<Vector<double>> &curl_values) const;

  void vector_value (const Point<3> &p, Vector<double> &values) const;

  // Note it's assumed all points are within one cell
  void vector_value_list (const std::vector<Point<3>> &points, std::vector<Vector<double>> &values) const;

private:
  PlaneWavePolarization _polarization;
  PhaseConvention _phase;
  FieldFormulation _field_type;
  std::shared_ptr<MT2DFEM> mt2d;
  double _frequency;

  mutable std::vector<cvector> _fields;
  mutable std::vector<Point<2>> _points;
};


#endif // NUMERICAL_SOLUTION_H
