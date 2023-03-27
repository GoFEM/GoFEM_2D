#include "numerical_solution.h"

#include "physical_model/physical_model.h"

NumericalSolutionMT2D::NumericalSolutionMT2D(const PhysicalModelPtr<2> &model,
                                             double frequency,
                                             unsigned fe_order,
                                             unsigned mapping_order,
                                             PlaneWavePolarization polarization,
                                             PhaseConvention phase,
                                             FieldFormulation field_type,
                                             const std::string function_name):
  SolutionFunction<3> (6), _polarization(polarization),
  _phase(phase), _field_type(field_type)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  mt2d.reset(new MT2DFEM(MPI_COMM_SELF, fe_order + 1, mapping_order, model));
  mt2d->set_frequency(frequency);
  mt2d->set_estimate_error_on_last_cycle(false);
  mt2d->set_output_type("");
  mt2d->set_verbosity(false);
  mt2d->set_output_file(function_name + "_" + std::to_string(rank));
  mt2d->set_refinement_steps(0);
#ifndef SHARED_TRIANGULATION
  mt2d->set_local_model(model);
#endif
//  mt2d->set_receivers({Receiver({0., 0.,1.}, "R1")});
  mt2d->run();
}

NumericalSolutionMT2D *NumericalSolutionMT2D::clone() const
{
  return new NumericalSolutionMT2D(*this);
}

void NumericalSolutionMT2D::set_field(FieldFormulation ftype)
{
  _field_type = ftype;
}

void NumericalSolutionMT2D::set_frequency(double frequency)
{
  mt2d->set_frequency(frequency);
  mt2d->run();
}

void NumericalSolutionMT2D::curl_list(const std::vector<Point<3> > &/*points*/,
                                      std::vector<Vector<double> > &/*curl_values*/) const
{
  throw std::runtime_error("NumericalSolutionMT2D::curl_list is not implemented.");
}

void NumericalSolutionMT2D::vector_value(const Point<3> &/*p*/, Vector<double> &/*values*/) const
{
  throw std::runtime_error("NumericalSolutionMT2D::vector_value not implemented");
}

void NumericalSolutionMT2D::vector_value_list(const std::vector<Point<3> > &points,
                                              std::vector<Vector<double> > &values) const
{
  _points.resize(points.size());
  _fields.resize(values.size());

  for(unsigned i = 0; i < points.size(); ++i)
  {
    if(_polarization == EW)
      _points[i][0] = points[i][1];
    else if(_polarization == NS)
      _points[i][0] = points[i][0];

    _points[i][1] = points[i][2];
    _fields[i].resize(3);
  }

  // Get fields
  mt2d->tangential_fields_at(_points, _fields, _field_type);

  for(unsigned i = 0; i < values.size(); ++i)
  {
    if(_polarization == NS)
    {
      values[i][0] = -_fields[i][1].real();
      values[i][1] = 0;
      values[i][2] = -_fields[i][2].real();
      values[i][3] = -_fields[i][1].imag();
      values[i][4] = 0;
      values[i][5] = -_fields[i][2].imag();
    }
    else // EW
    {
      values[i][0] = 0;
      values[i][1] = _fields[i][1].real();
      values[i][2] = _fields[i][2].real();
      values[i][3] = 0;
      values[i][4] = _fields[i][1].imag();
      values[i][5] = _fields[i][2].imag();
    }

    if(_phase == PhaseLag)
    {
      values[i][3] *= -1.0;
      values[i][4] *= -1.0;
      values[i][5] *= -1.0;
    }
  }
}
