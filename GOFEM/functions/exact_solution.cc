#include "exact_solution.h"

template<int dim>
ExactSolutionMT1D<dim>::ExactSolutionMT1D (const BackgroundModel<dim> &bgmodel, double frequency,
                                           PlaneWavePolarization polarization,
                                           PhaseConvention phase,
                                           FieldFormulation formulation):
  SolutionFunction<dim> ((formulation == EFieldStabilized ||
                          formulation == HFieldStabilized) ? (2*dim+2) : 2*dim),
  _bgmodel(bgmodel), _polarization(polarization), _phase(phase),
  _field_type(formulation)
{
  _mt1d.reinit (bgmodel.conductivities (),
                bgmodel.permittivities (),
                bgmodel.permeabilities (),
                bgmodel.layer_depths (),
                frequency);
}

template<int dim>
ExactSolutionMT1D<dim> *ExactSolutionMT1D<dim>::clone() const
{
  return new ExactSolutionMT1D<dim>(*this);
}

template<int dim>
void ExactSolutionMT1D<dim>::set_frequency(double frequency)
{
  _mt1d.reinit (_bgmodel.conductivities (),
                _bgmodel.permittivities (),
                _bgmodel.permeabilities (),
                _bgmodel.layer_depths (),
                frequency);
}

template<int dim>
void ExactSolutionMT1D<dim>::electric_field(const Point<dim> &/*p*/, Tensor<1, dim, dcomplex> &/*E*/) const
{
  throw std::runtime_error("Not implemented.");
}

template<>
void ExactSolutionMT1D<3>::electric_field(const Point<3> &p, Tensor<1, 3, dcomplex> &E) const
{
  E *= 0.;

  dcomplex efield = _mt1d.electric_field(p[2]);

  if(_phase == PhaseLag)
    efield = std::conj(efield);

  if(_polarization == NS) // NS polarization
    E[0] = efield;
  else if(_polarization == EW) // EW polarization
    E[1] = -efield;
}

template<int dim>
void ExactSolutionMT1D<dim>::magnetic_field(const Point<dim> &/*p*/, Tensor<1, dim, dcomplex> &/*H*/) const
{
  throw std::runtime_error("Not implemented.");
}

template<>
void ExactSolutionMT1D<3>::magnetic_field(const Point<3> &p, Tensor<1, 3, dcomplex> &H) const
{
  H *= 0.;

  dcomplex hfield = _mt1d.magnetic_field(p[2]);

  if(_phase == PhaseLag)
    hfield = std::conj(hfield);

  if(_polarization == NS) // NS polarization
    H[1] = hfield;
  else if(_polarization == EW) // EW polarization
    H[0] = hfield;
}

template<int dim>
void ExactSolutionMT1D<dim>::set_polarization(PlaneWavePolarization polarization)
{
  _polarization = polarization;
}

template<int dim>
void ExactSolutionMT1D<dim>::set_field(FieldFormulation ftype)
{
  _field_type = ftype;
}

template<>
void ExactSolutionMT1D<2>::vector_value (const Point<2> &p, Vector<double> &values) const
{
  Assert (values.size () == 4, ExcDimensionMismatch (values.size(), 4));

  dcomplex E = _mt1d.electric_field(p[1]);

  if(_phase == PhaseLag)
    E = std::conj(E);

  values[0] = E.real();
  values[1] = E.imag();

  dcomplex H = _mt1d.magnetic_field(p[1]);

  if(_phase == PhaseLag)
    H = std::conj(H);

  values[2] = H.real();
  values[3] = H.imag();
}

template<>
void ExactSolutionMT1D<3>::vector_value (const Point<3> &p, Vector<double> &values) const
{
//  if(_field_type == EField || _field_type == HField)
//  {
//    Assert (values.size () == 6, ExcDimensionMismatch (values.size(), 6));
//  }
//  else
//  {
//    Assert (values.size () == 8, ExcDimensionMismatch (values.size(), 8));
//  }

  Tensor<1, 3, dcomplex> F;

  if(_field_type == EField || _field_type == EFieldStabilized)
    electric_field(p, F);
  else if(_field_type == HField || _field_type == HFieldStabilized)
    magnetic_field(p, F);

  values(0) = F[0].real();
  values(1) = F[1].real();
  values(2) = F[2].real();
  values(3) = F[0].imag();
  values(4) = F[1].imag();
  values(5) = F[2].imag();

  if(_field_type == EFieldStabilized || _field_type == HFieldStabilized)
  {
    values(6) = 0;
    values(7) = 0;
  }
}

template<int dim>
ExactSolutionMT1DTapered<dim>::ExactSolutionMT1DTapered(const BackgroundModel<dim> &left_model,
                                                        const BackgroundModel<dim> &right_model,
                                                        const std::vector<Point<dim>> &model_corner_points,
                                                        double frequency, PhaseConvention phase):
  Function<dim> (2*dim), _phase(phase)
{
  mt1d_left.reinit (left_model.conductivities (),
               left_model.permittivities (),
               left_model.permeabilities (),
               left_model.layer_depths (),
               frequency);

  mt1d_right.reinit (right_model.conductivities (),
               right_model.permittivities (),
               right_model.permeabilities (),
               right_model.layer_depths (),
               frequency);

  ymin = model_corner_points[0][0];
  ymax = model_corner_points[1][0];
}

template<int dim>
void ExactSolutionMT1DTapered<dim>::vector_value (const Point<dim> &/*p*/, Vector<double> &/*values*/) const
{
  throw std::runtime_error("ExactSolutionMT1DTapered<dim>::vector_value not implemented.");
}

template<>
void ExactSolutionMT1DTapered<2>::vector_value (const Point<2> &p, Vector<double> &values) const
{
  Assert (values.size () == 4, ExcDimensionMismatch (values.size(), 4));

  // Apply cosine taper between left and right boundaries
  dcomplex El = mt1d_left.electric_field(p[1]);
  dcomplex Er = mt1d_right.electric_field(p[1]);

  if(_phase == PhaseLag)
  {
    El = std::conj(El);
    Er = std::conj(Er);
  }

  double eta = (p[0] - ymin) / (ymax - ymin);
  double ediff = (1.0 - cos(M_PI * eta)) / 2.;
  dcomplex E = (Er - El) * ediff + El;

  values[0] = E.real();
  values[1] = E.imag();

  dcomplex Hl = mt1d_left.magnetic_field(p[1]);
  dcomplex Hr = mt1d_right.magnetic_field(p[1]);

  if(_phase == PhaseLag)
  {
    Hl = std::conj(Hl);
    Hr = std::conj(Hr);
  }

  dcomplex H = (Hr - Hl) * ediff + Hl;
  values[2] = H.real();
  values[3] = H.imag();
}

ExactSolutionCSEMSpace::ExactSolutionCSEMSpace(const DipoleSource &source, double sigma, double epsilon,
                                               double frequency, PhaseConvention phase):
  SolutionFunction<3> (6), F(3), source_(source), field_type(EField),
  strike_position(0.), phase_convention(phase)
{
  if(source.n_dipole_elements() != 1)
    throw std::runtime_error("ExactSolutionCSEMSpace: no support for complex dipole");

  omega = 2.0 * M_PI * frequency;
  permittivity = epsilon;
  conductivity = sigma + II * omega * permittivity;
  wavenumber = sqrt(-II * omega * mu0 * conductivity);

  std::vector<char> source_orientation = {'x', 'y', 'z'};

  for(unsigned n = 0; n < source.n_dipole_elements(); ++n)
  {
    // Decompose sources into dipoles for principal axes
    // The resulting field is a superposition of the fields
    // from these dipoles
    const dvec3d& extent = source.dipole_extent(n);
    Point<3> source_location;
    source_.position(source_location, 3, n);

    for(unsigned i = 0; i < 3; ++i)
    {
      if(std::fabs(extent[i]) > 0.)
      {
        dipole_extent.push_back(extent[i]);
        dipole_locations.push_back(source_location);
        dipole_current.push_back(source_.current(n)[i]);
        std::vector<double> sign(3, 1.);
        std::vector<unsigned> index(3);
        get_rotation_properties(source_orientation[i], index, sign);
        indices.push_back(index);
        signs.push_back(sign);
      }
    }
  }
}

ExactSolutionCSEMSpace *ExactSolutionCSEMSpace::clone() const
{
  return new ExactSolutionCSEMSpace(*this);
}

void ExactSolutionCSEMSpace::vector_value(const Point<3> &p, Vector<double> &values) const
{
  Assert (values.size () == 6, ExcDimensionMismatch (values.size(), 6));

  // Calculate and sum fields for each principal dipole
  values = 0.;
  for(unsigned i = 0; i < indices.size(); ++i)
  {
    //std::cout << dipole_locations[i] << " " << dipole_moment[i] << "\n";
    calculate_field(p, dipole_extent[i] * dipole_current[i], indices[i], signs[i], dipole_locations[i], F);

    values[0] += F[0].real();
    values[1] += F[1].real();
    values[2] += F[2].real();
    values[3] += F[0].imag();
    values[4] += F[1].imag();
    values[5] += F[2].imag();
  }

  if(PhaseLag)
  {
    values[3] *= -1.;
    values[4] *= -1.;
    values[5] *= -1.;
  }
}

void ExactSolutionCSEMSpace::set_field(FieldFormulation ftype)
{
  field_type = ftype;
}

void ExactSolutionCSEMSpace::set_frequency(double f)
{
  omega = 2.0 * M_PI * f;
  conductivity = dcomplex(conductivity.real(), omega * permittivity);
  wavenumber = std::sqrt(-II*omega*mu0*conductivity);
}

FieldFormulation ExactSolutionCSEMSpace::get_field_type() const
{
  return field_type;
}

void ExactSolutionCSEMSpace::calculate_field(const Point<3> &p, const dcomplex &moment,
                                             const std::vector<unsigned> &index,
                                             const std::vector<double> &sign,
                                             const Point<3> &source_location,
                                             std::vector<dcomplex> &fields) const
{
  Point<3> receiver_pos;
  for(unsigned d = 0; d < 3; ++d)
  receiver_pos[d] = p[d];

  double R = receiver_pos.distance(source_location);

  if(source_.get_type() == ElectricDipole)
  {
    if(field_type == EField)
    {
      dcomplex f1 =  moment / (4.0 * M_PI * conductivity * R * R * R) * exp(-II * wavenumber * R);
      dcomplex f2 = -wavenumber*wavenumber*R*R + 3.0*II*wavenumber*R + 3.0;
      dcomplex f3 =  wavenumber*wavenumber*R*R - II*wavenumber*R - 1.0;

      fields[index[0]] = f1 * ( f2*(receiver_pos(index[0]) - source_location(index[0]))*(receiver_pos(index[0]) - source_location(index[0])) / (R * R) + f3);
      fields[index[1]] = f1 *   f2*(receiver_pos(index[0]) - source_location(index[0]))*(receiver_pos(index[1]) - source_location(index[1])) / (R * R);
      fields[index[2]] = f1 *   f2*(receiver_pos(index[0]) - source_location(index[0]))*(receiver_pos(index[2]) - source_location(index[2])) / (R * R);
    }
    else if(field_type == HField)
    {
      dcomplex f1 = moment / (4.0 * M_PI * R * R) * exp(-II * wavenumber * R);
      dcomplex f2 = II*wavenumber*R + 1.0;

      fields[index[0]] = 0;
      fields[index[1]] = f1 * f2 * sign[index[1]]*(receiver_pos(index[2]) - source_location(index[2])) / R;
      fields[index[2]] = f1 * f2 * sign[index[2]]*(receiver_pos(index[1]) - source_location(index[1])) / R;
    }
  }
  else if(source_.get_type() == MagneticDipole)
  {
    if(field_type == EField)
    {
      dcomplex f1 = (II*omega*mu0*moment) / (4.0 * M_PI * R * R) * exp(-II * wavenumber * R);
      dcomplex f2 =  II*wavenumber*R + 1.0;

      fields[index[0]] = 0;
      fields[index[1]] = f1 * f2 * sign[index[1]]*(receiver_pos(index[2]) - source_location(index[2])) / R;
      fields[index[2]] = f1 * f2 * sign[index[2]]*(receiver_pos(index[1]) - source_location(index[1])) / R;
    }
    else if(field_type == HField)
    {
      // note minus sign for f1 is not from Ward and Hohmann, but required to get correct sign (when compared with EM1D by R. Streich)
      dcomplex f1 = -moment / (4.0 * M_PI * R * R * R) * exp(-II * wavenumber * R);
      dcomplex f2 = -wavenumber*wavenumber*R*R + 3.0*II*wavenumber*R + 3.0;
      dcomplex f3 =  wavenumber*wavenumber*R*R - II*wavenumber*R - 1.0;

      fields[index[0]] = f1 * ( f2*(receiver_pos(index[0]) - source_location(index[0]))*(receiver_pos(index[0]) - source_location(index[0])) / (R * R) + f3);
      fields[index[1]] = f1 *   f2*(receiver_pos(index[0]) - source_location(index[0]))*(receiver_pos(index[1]) - source_location(index[1])) / (R * R);
      fields[index[2]] = f1 *   f2*(receiver_pos(index[0]) - source_location(index[0]))*(receiver_pos(index[2]) - source_location(index[2])) / (R * R);
    }
  }
}

void ExactSolutionCSEMSpace::get_rotation_properties(char orientation, std::vector<unsigned> &index, std::vector<double> &sign)
{
  // Permute dimensions depending on the dipole orientation
  if(orientation == 'x')
  {
    index[0] = 0;
    index[1] = 1;
    index[2] = 2;
  }
  else if(orientation == 'y')
  {
    index[0] = 1;
    index[1] = 0;
    index[2] = 2;
  }
  else if(orientation == 'z')
  {
    index[0] = 2;
    index[1] = 1;
    index[2] = 0;
  }
  else
    throw std::runtime_error("Unknown orientation of the source.");

  sign = std::vector<double>(3, 1.);
  if(orientation == 'x')
    sign[1] = -1.;
  else if(orientation == 'y')
    sign[2] = -1.;
  else if(orientation == 'z')
    sign[0] = -1.;
}

template<int dim>
ExactSolutionCSEMSpaceKx<dim>::ExactSolutionCSEMSpaceKx(const DipoleSource &source, double conductivity,
                                                        double permittivity, const Point<dim> &model_size,
                                                        const std::string &filter_file, PhaseConvention phase):
  physical_source(source), space_solution(source, conductivity, permittivity, 0, phase),
  transform(filter_file), E(3)
{
  double xmin = 0.01, xmax = model_size[0] * 3;
  fill_array_logscale(xmin, xmax, 512, positive_integration_points);

  log_positive_integration_points.resize(positive_integration_points.size());
  for(size_t i = 0; i < positive_integration_points.size(); ++i)
    log_positive_integration_points[i] = log10(positive_integration_points[i]);

  positive_strike_fields.resize(3, cvector(positive_integration_points.size()));

  // For sources other than an elementary dipole symmetry is generally not preserved.
  // Therefore, we need to consider both symmetric and anti-symmetric parts of the field
  if(source.n_dipole_elements() > 1)
  {
    for(double v: positive_integration_points)
      negative_integration_points.push_back(-v);

    negative_strike_fields.resize(3, cvector(negative_integration_points.size()));
    even_strike_fields.resize(3, cvector(negative_integration_points.size()));
    odd_strike_fields.resize(3, cvector(negative_integration_points.size()));
  }
  else
    init_filter();

  space_solution.set_field(EField);
}

template<int dim>
void ExactSolutionCSEMSpaceKx<dim>::set_frequency(double f)
{
  space_solution.set_frequency(f);
}

template<int dim>
void ExactSolutionCSEMSpaceKx<dim>::set_wavenumber(double kx)
{
  wavenumber = kx;
}

template<int dim>
void ExactSolutionCSEMSpaceKx<dim>::set_field(FieldFormulation field_type)
{
  space_solution.set_field(field_type);

  if(field_type == EField)
    symmetry = e_symmetry;
  else
    symmetry = h_symmetry;
}

template<int dim>
void ExactSolutionCSEMSpaceKx<dim>::vector_value_list(const std::vector<Point<dim> > &points, std::vector<Vector<double> > &vectors) const
{
  Assert (vectors.size () == points.size (), ExcDimensionMismatch (vectors.size (), points.size ()));

  for (size_t i = 0; i < points.size (); ++i)
    vector_value (points[i], vectors[i]);
}

template<int dim>
void ExactSolutionCSEMSpaceKx<dim>::vector_value (const Point<dim> &p, Vector<double> &values) const
{
  Assert (values.size () == 6, ExcDimensionMismatch (values.size(), 6));

  // Calculate and sum fields for each principal dipole
  cvec3d E = calculate_field_in_spectral_domain(p);

  values[0] = E[0].real();
  values[1] = E[1].real();
  values[2] = E[2].real();
  values[3] = E[0].imag();
  values[4] = E[1].imag();
  values[5] = E[2].imag();
}

template<int dim>
cvec3d ExactSolutionCSEMSpaceKx<dim>::calculate_field_in_spectral_domain(const Point<dim> &p) const
{
  cvec3d E;

  Vector<double> Es(6);

  // Fields from 0 to +Inf
  for(unsigned i = 0; i < positive_integration_points.size(); ++i)
  {
    Point<3> p3d(positive_integration_points[i], p[0], p[1]);
    space_solution.vector_value(p3d, Es);

    for(unsigned c = 0; c < 3; ++c)
      positive_strike_fields[c][i] = dcomplex(Es[c], Es[c+3]);
  }

  if(physical_source.get_type() == ElectricDipole ||
     physical_source.get_type() == MagneticDipole)
  {
    E[0] = transform.integrate(positive_integration_points, log_positive_integration_points, positive_strike_fields[0], wavenumber, symmetry[0], ForwardTransform, 1e-7);
    E[1] = transform.integrate(positive_integration_points, log_positive_integration_points, positive_strike_fields[1], wavenumber, symmetry[1], ForwardTransform, 1e-7);
    E[2] = transform.integrate(positive_integration_points, log_positive_integration_points, positive_strike_fields[2], wavenumber, symmetry[2], ForwardTransform, 1e-7);
  }
  else
  {
    // Fields from -Inf to 0
    for(unsigned i = 0; i < negative_integration_points.size(); ++i)
    {
      Point<3> p3d(negative_integration_points[i], p[0], p[1]);
      space_solution.vector_value(p3d, Es);

      for(unsigned c = 0; c < 3; ++c)
        negative_strike_fields[c][i] = dcomplex(Es[c], Es[c+3]);
    }

    // Split function into symmetric (even) and anti-symmetric (odd) parts and treat them separately
    for(unsigned i = 0; i < negative_integration_points.size(); ++i)
    {
      for(unsigned c = 0; c < 3; ++c)
      {
        even_strike_fields[c][i] = 0.5 * (positive_strike_fields[c][i] + negative_strike_fields[c][i]);
        odd_strike_fields[c][i] = 0.5 * (positive_strike_fields[c][i] - negative_strike_fields[c][i]);
      }
    }

    E[0] = transform.integrate(positive_integration_points, log_positive_integration_points, even_strike_fields[0], wavenumber, CosineTransform, ForwardTransform, 1e-7)
         + transform.integrate(positive_integration_points, log_positive_integration_points, odd_strike_fields[0], wavenumber, SineTransform, ForwardTransform, 1e-7);
    E[1] = transform.integrate(positive_integration_points, log_positive_integration_points, even_strike_fields[1], wavenumber, CosineTransform, ForwardTransform, 1e-7)
         + transform.integrate(positive_integration_points, log_positive_integration_points, odd_strike_fields[1], wavenumber, SineTransform, ForwardTransform, 1e-7);
    E[2] = transform.integrate(positive_integration_points, log_positive_integration_points, even_strike_fields[2], wavenumber, CosineTransform, ForwardTransform, 1e-7)
         + transform.integrate(positive_integration_points, log_positive_integration_points, odd_strike_fields[2], wavenumber, SineTransform, ForwardTransform, 1e-7);
  }

  return E;
}

template<int dim>
void ExactSolutionCSEMSpaceKx<dim>::init_filter()
{
  TransformationType i0, i1;

  // Get dipole type
  if ( physical_source.get_type() == ElectricDipole) // electric dipole
  {
    i0 = CosineTransform; // 0 is for even cosine transform
    i1 = SineTransform;   // 1 is for odd sine transform
  }
  else // magnetic dipole
  {
    i0 = SineTransform;
    i1 = CosineTransform;
  }

  TransformationType sym1, sym2;

  // Set symmetry variables based on dipole type (e or b) and direction of dipole (x or yz)
  if ( std::fabs(physical_source.dipole_extent()[0]) > 0. )
  {
    sym1 = i0;
    sym2 = i1;
  }
  else if( std::fabs(physical_source.dipole_extent()[1]) > 0.
        || std::fabs(physical_source.dipole_extent()[2]) > 0. ) // yz plane
  {
    sym1 = i1;
    sym2 = i0;
  }
  else
  {
    throw std::runtime_error("This is a wire source which ruins any symmetry in the spectral domain. Use full integration domain.");
  }

  e_symmetry = {sym1, sym2, sym2};
  h_symmetry = {sym2, sym1, sym1};

  symmetry = e_symmetry;
}

template class ExactSolutionMT1D<2>;
template class ExactSolutionMT1D<3>;
template class ExactSolutionMT1DTapered<2>;
template class ExactSolutionCSEMSpaceKx<2>;
