#ifndef EXACT_SOLUTION_H
#define EXACT_SOLUTION_H

#include <deal.II/lac/vector.h>
#include <deal.II/dofs/dof_handler.h>

#include "physical_model/physical_model.h"
#include "survey/dipole_source.h"
#include "functions/solution_function.h"
#include "analytical/mt1d.h"
#include "analytical/mt1d_normalized.h"
#include "common/sincos_transform.h"

using namespace dealii;

/*
 * 1D MT function for E/H field.
 * Used for 2D MT boundary conditions
 * Uses -i_omega_mu convention if PhaseLag, and +i_omega_mu if PhaseLead
 */
template<int dim>
class ExactSolutionMT1D: public SolutionFunction<dim>
{
public:
  ExactSolutionMT1D(const BackgroundModel<dim> &bgmodel, double frequency,
                    PlaneWavePolarization polarization,
                    PhaseConvention phase = PhaseLag,
                    FieldFormulation formulation = EField);

  ExactSolutionMT1D<dim> *clone() const;

  void set_frequency(double frequency);

  // Returns array of [E_re E_im H_re H_im] for the given 1d model at the point p
  virtual void vector_value (const Point<dim> &p, Vector<double> &values) const;

  void electric_field(const Point<dim> &p, Tensor<1, dim, dcomplex> &E) const;
  void magnetic_field(const Point<dim> &p, Tensor<1, dim, dcomplex> &H) const;

  void set_polarization(PlaneWavePolarization polarization);
  void set_field(FieldFormulation ftype);

private:
  //MT1DNormalized mt1d;
  MT1D _mt1d;
  BackgroundModel<dim> _bgmodel;
  PlaneWavePolarization _polarization;
  PhaseConvention _phase;
  FieldFormulation _field_type;
};

/*
 * 1D MT function for E/H field with two boundary models specified.
 * In between two models field values are interpolated using cosine taper.
 * Used for 2D MT boundary conditions. Particularly efficient when left and
 * right boundaries differ.
 * Uses -i_omega_mu convention if PhaseLag, and +i_omega_mu if PhaseLead
 */
template<int dim>
class ExactSolutionMT1DTapered: public Function<dim>
{
public:
  ExactSolutionMT1DTapered(const BackgroundModel<dim> &left_model, const BackgroundModel<dim> &right_model,
                           const std::vector<Point<dim>> &model_corner_points, double frequency,
                           PhaseConvention phase);
  virtual ~ExactSolutionMT1DTapered() {}

  // Returns array of [E_re E_im H_re H_im] for the given 1d models at the point p
  virtual void vector_value (const Point<dim> &p, Vector<double> &values) const;

private:
  MT1D mt1d_left, mt1d_right;
//  MT1DNormalized mt1d_left, mt1d_right;
  double ymin, ymax;

  PhaseConvention _phase;
};

/*
 * Analytical solution for an arbitrary electric/magnetic dipole in a homogeneous space
 * Used for 3D CSEM testing.
 * For an rotated and tilted source decomposes it into principal dipoles along each coordinate
 * and calculates the resulting field as a superposiion of these dipoles' responses
 * Implements equations 2.40, 2.42, 2.56, 2.57 of Ward and Hohmann, 1988
 * Uses -i*omega*t convention if PhaseLag, and +i*omega*t if PhaseLead
 * Note: this class takes displacement currents into account by using permittivity of the vacuum
 */
class ExactSolutionCSEMSpace: public SolutionFunction<3>
{
public:
  ExactSolutionCSEMSpace(const DipoleSource &source, double sigma,
                         double epsilon, double frequency, PhaseConvention phase);

  ExactSolutionCSEMSpace* clone() const;

  // Returns array [Fx_re Fy_re Fz_re Fx_im Fy_im Fz_im] for the homogeneous space model
  // where F is either E or H depending on flag "field" (by default F = E)
  virtual void vector_value (const Point<3> &p, Vector<double> &values) const;

  void set_field(FieldFormulation ftype);
  void set_frequency(double f);

  FieldFormulation get_field_type() const;

private:
  // assumes that X is east-west
  void calculate_field(const Point<3> &p, const dcomplex &moment, const std::vector<unsigned> &index,
                       const std::vector<double> &sign, const Point<3> &source_location,
                       std::vector<dcomplex> &fields) const;

  void get_rotation_properties(char orientation, std::vector<unsigned> &index, std::vector<double> &sign);

  double omega;
  dcomplex conductivity;
  double permittivity;
  dcomplex wavenumber;
  mutable std::vector<dcomplex> F;
  DipoleSource source_;
  FieldFormulation field_type;
  std::vector<Point<3>> dipole_locations;
  std::vector<std::vector<unsigned>> indices;
  std::vector<std::vector<double>> signs;
  std::vector<double> dipole_extent;
  std::vector<dcomplex> dipole_current;
  double strike_position;
  PhaseConvention phase_convention;
};

/*
 * Solution for an arbitrary electric/magnetic dipole in a
 * spectral domain (kx,y,z) for a homogeneous space.
 * Used for 2.5D CSEM modeling with secondary field formulation.
 * Uses -i_omega_mu convention if PhaseLag, and +i_omega_mu if PhaseLead
 */
template<int dim>
class ExactSolutionCSEMSpaceKx: public Function<dim>
{
public:
  ExactSolutionCSEMSpaceKx(const DipoleSource &source, double conductivity, double permittivity,
                           const Point<dim> &model_size, const std::string &filter_file, PhaseConvention phase);
  virtual ~ExactSolutionCSEMSpaceKx() {}

  void set_frequency(double f);
  void set_wavenumber(double kx);
  void set_field(FieldFormulation field_type);

  virtual void vector_value_list (const std::vector<Point<dim> > & points, std::vector<Vector<double> > &vectors) const;
  virtual void vector_value (const Point<dim> &p, Vector<double> &values) const;

private:
  cvec3d calculate_field_in_spectral_domain(const Point<dim> &p) const;
  void init_filter();

  std::vector<double> positive_integration_points, log_positive_integration_points, negative_integration_points;
  DipoleSource physical_source;
  double wavenumber;
  ExactSolutionCSEMSpace space_solution;
  std::vector<TransformationType> symmetry, e_symmetry, h_symmetry;
  mutable SinCosTransform<dcomplex> transform;
  mutable std::vector<cvector> positive_strike_fields, negative_strike_fields, even_strike_fields, odd_strike_fields;
  std::vector<dcomplex> E;
};

template<int dim>
class ZeroSolutionFunction: public SolutionFunction<dim>
{
public:
  ZeroSolutionFunction():
    SolutionFunction<dim>(2*dim)
  {}

  virtual ZeroSolutionFunction<dim>* clone() const
  {
    return new ZeroSolutionFunction(*this);
  }

  void set_field(FieldFormulation /*ftype*/) {}
  void set_frequency(double /*frequency*/) {}

  void vector_value (const Point<dim> &/*p*/, Vector<double> &values) const
  {
      values = 0.;
  }

  void value (const Point<dim> &/*p*/, double &value) const
  {
      value = 0.;
  }

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

#endif // EXACT_SOLUTION_H
