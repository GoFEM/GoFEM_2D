#ifndef _COMMON_
#define _COMMON_

#include <complex>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <limits>
#include <array>

typedef std::complex<double> dcomplex;
typedef std::vector<double> dvector;
typedef std::vector<dcomplex> cvector;
typedef std::array<double, 3> dvec3d;
typedef std::array<double, 2> dvec2d;
typedef std::array<dcomplex, 3> cvec3d;

// Fundamental constants
static const double c0 = 299792458.0;
static const double mu0 = 4. * M_PI * 1e-7;
static const double eps0 = 1.0 / (c0*c0*mu0);
static const dcomplex II = dcomplex(0.0,1.0);
// Earth's mean radius
static const double earth_radius = 6371000;

// Maximum thickness of the layer in meters for spherical code
// (to mitigate effect of the gradient conductivity)
const static double maximum_thickness = 50000;
const static double minimum_thickness = 130.;

// PhaseLag = -i*omega*t; PhaseLead = i*omega*t
enum PhaseConvention {PhaseLag, PhaseLead};

enum CoordinateSystem {Cartesian, Spherical};

enum ModelParameterization {Model_Layered, Model_Spherical,
                            Model_XYZ, Model_Thermodynamic,
                            Model_Layered_Blocks};

// GOFEM specific
enum BoundaryConditions {Dirichlet, Neumann, SilverMueller, Dirichlet2D};
enum AdaptivityStrategy {Global, ResidualBased, GoalOriented};
enum RefinementStrategy {Number, Fraction};
enum PlaneWavePolarization {NS, EW};
enum PreconditionerType {Direct, CGAMS, AMS, AMG, LOAMG, DirectSolver, AutoSelection};
enum FieldFormulation {EField, HField, EFieldStabilized, HFieldStabilized, EHField};
enum ApproachType {TotalField, ScatteredField};

// Data specific. Note that survey method is different from the
// forward calculator type in a sense that several codes can be
// used to model the same method. For example, depending on the
// model dimension, MT can be calculator with 1D analytical code
// or numerically using GoFEM, therefore both method and forward
// operator type need to be specified
enum SurveyMethod {Geoelectric, CSEM, MT, TEM, GlobalEM, MTSphere, ReceiverFunction};
enum ForwardType {MT3D_GOFEM, MT2D_GOFEM, MTLayered, C1D, CSEMLayered, X3DG, GEM1D, RF_AMMON, RF_REFLECTIVITY};
enum SourceType {PlaneWave = 0, Dipole,
                 SphericalHarmonic, DCDipole,
                 RadialSheet, CurrentShell,
                 ExternalSourceFile, PointEarthquake};

// Inversion specific
enum RegularizationOperator {Gradient, Diagonal, Roughness};

static std::map<std::string, ModelParameterization> model_types_conversion =
{
  {"LAYERED", Model_Layered},
  {"SPHERICAL", Model_Spherical},
  {"XYZ", Model_XYZ},
  {"THERMODYNAMIC", Model_Thermodynamic},
  {"LAYERED_BLOCKS", Model_Layered_Blocks}
};

static std::map<std::string, FieldFormulation> formulation_types_conversion =
{
  {"E", EField},
  {"H", HField},
  {"EStabilized", EFieldStabilized},
  {"HStabilized", HFieldStabilized}
};

static std::map<ModelParameterization, std::string> types_model_conversion =
{
  {Model_Layered, "LAYERED"},
  {Model_Layered_Blocks, "LAYERED_BLOCKS"},
  {Model_Spherical, "SPHERICAL"},
  {Model_XYZ, "XYZ"},
  {Model_Thermodynamic, "THERMODYNAMIC"}
};

static std::map<std::string, SurveyMethod> method_type_conversion =
{
  {"CSEM", CSEM},
  {"MT", MT},
  {"TEM", TEM},
  {"Geoelectric", Geoelectric},
  {"GlobalEM", GlobalEM},
  {"MTSphere", MTSphere},
  {"ReceiverFunction", ReceiverFunction}
};

static std::map<SurveyMethod, std::string> type_method_conversion =
{
  {CSEM, "CSEM"},
  {MT, "MT"},
  {TEM, "TEM"},
  {Geoelectric, "Geoelectric"},
  {GlobalEM, "GlobalEM"},
  {MTSphere, "MTSphere"},
  {ReceiverFunction, "ReceiverFunction"}
};

static std::map<ForwardType, std::string> type_calculator_conversion =
{
  {MT3D_GOFEM, "MT3D_GOFEM"},
  {MT2D_GOFEM, "MT2D_GOFEM"},
  {MTLayered, "MT1D"},
  {CSEMLayered, "CSEM1D"},
  {C1D, "C1D"},
  {GEM1D, "GEM1D"},
  {X3DG, "X3DG"},
  {RF_AMMON, "RF_AMMON"},
  {RF_REFLECTIVITY, "RF_REFLECTIVITY"}
};

static std::map<std::string, ForwardType> calculator_type_conversion =
{
  {"MT3D_GOFEM", MT3D_GOFEM},
  {"MT2D_GOFEM", MT2D_GOFEM},
  {"MT1D", MTLayered},
  {"CSEM1D", CSEMLayered},
  {"C1D", C1D},
  {"GEM1D", GEM1D},
  {"X3DG", X3DG},
  {"RF_AMMON", RF_AMMON},
  {"RF_REFLECTIVITY", RF_REFLECTIVITY}
};

static std::map<std::string, SourceType> source_type_conversion =
{
  {"PlaneWave", PlaneWave},
  {"Dipole", Dipole},
  {"SH", SphericalHarmonic},
  {"DCDipole", DCDipole},
  {"Sheet", RadialSheet},
  {"CurrentShell", CurrentShell},
  {"ExtFile", ExternalSourceFile},
  {"PointEarthquake", PointEarthquake}
};

static std::map<SourceType, std::string> string_to_source_type =
{
  {PlaneWave, "PlaneWave"},
  {Dipole, "Dipole"},
  {SphericalHarmonic, "SH"},
  {DCDipole, "DCDipole"},
  {RadialSheet, "Sheet"},
  {CurrentShell, "CurrentShell"},
  {ExternalSourceFile, "ExtFile"},
  {PointEarthquake, "PointEarthquake"}
};

static std::map<std::string, PreconditionerType> solver_type_conversion =
{
  {"Direct", Direct},
  {"CGAMS", CGAMS},
  {"AMS", AMS},
  {"AMG", AMG},
  {"DirectSolver", DirectSolver},
  {"AutoSelection", AutoSelection}
};

const unsigned west_boundary_id = 33;
const unsigned east_boundary_id = 44;
const unsigned top_boundary_id = 55;
const unsigned bottom_boundary_id = 66;
const unsigned north_boundary_id = 77;
const unsigned south_boundary_id = 88;

const std::vector<unsigned> all_boundaries = {west_boundary_id, east_boundary_id,
                                              top_boundary_id, bottom_boundary_id,
                                              north_boundary_id, south_boundary_id};

static std::map<unsigned, std::string> boundary_id_name_table =
{
  {top_boundary_id, "top"},
  {bottom_boundary_id, "bottom"},
  {east_boundary_id, "east"},
  {west_boundary_id, "west"},
  {north_boundary_id, "north"},
  {south_boundary_id, "south"}
};

// Some very general functions
bool isIndeterminate (const double v);

// Fills array with n values in the range [minv,maxv] distributed uniformly on log scale
void fill_array_logscale(double minv, double maxv, unsigned n, std::vector<double> &array);

// Case-insensitive string comparison
bool istrcompare(const std::string& a, const std::string& b);
bool replace(std::string& str, const std::string& from, const std::string& to);
std::string trim(const std::string& str, const std::string& whitespace = " \t\r");
void split(const std::string &s, char delim, std::vector<std::string> &elems);

// Transformations between geographic and geomagnetic (dipole) co-ordinates and components
// forward = true : gg -> gm
// forward = false: gm -> gg
// All in/out quantities are in degrees / metres
// Rewritten from gg2gm MATLAB function by Nils Olsen
dvec3d gg2gm(const dvec3d &pole_coords, const dvec3d &point, const bool forward);
cvec3d gg2gm(const dvec3d &pole_coords, const dvec3d &point,
             const cvec3d &field, const bool forward, dvec3d &point_transformed);

// Do mapping [0 180] -> [90 -90]
double colat2lat(const double &colat);
// Do mapping [0 360] -> [-180 180]
double colon2lon(const double &colon);

template<class T>
dvec3d vec2array(const T& p)
{
  assert(p.dimension == 3);
  return {p[0], p[1], p[2]};
}

template<class T>
T array2vec(const dvec3d& p)
{
  T t;
  for(unsigned d = 0; d < t.dimension; ++d)
    t[d] = p[d];
  return t;
}

/*
 * Given the lower and upper limits of integration x1 and x2, this routine
 * returns arrays x[0..n-1] and w[0..n-1] of length n, containing the
 * abscissas and weights of the Gauss-Legendre n-point quadrature formula.
 *
 * Taken from Numerical Recipies
 */
void gauleg(const double &x1,
            const double &x2,
            std::vector<double> &x,
            std::vector<double> &w,
            const unsigned n);
/*
 * Same as above, but for the two dimensional quadrature formula.
 * It is constructed by applying tensor product to the one dimensional
 * case.
 */
void gauleg2d(const std::array<double, 2> x,
              const std::array<double, 2> y,
              std::vector<std::array<double, 2>> &points,
              std::vector<double> &weights,
              const unsigned n);

/*
 * Given alpha, the parameter of the Laguerre polynomials, this routine
 * returns arrays x[0..n-1] and w[0..n-1] containing the abscissas and
 * weights of the n-point Gauss-Laguerre quadrature formula. The smallest
 * abscissa is returned in x[0], the largest in x[n-1].
 *
 * Taken from Numerical Recipies
 */
void gaulag(dvector &x, dvector &w, const double &alpha, const unsigned n);

/*
 * Executes command in the shell and
 * returns its terminal output
 */
std::string exec(const char* cmd, int &status);

/*
 * Convert point to spherical coordinate system
 * p_cartesian[0:2] = [x y z]
 * p_spherical[0:2] = [phi (0..2pi) theta (0..pi) r]
 */
template<class T>
T point_to_spherical(const T &p_cartesian)
{
  T p_spherical;

  // radius
  p_spherical[2] = sqrt(p_cartesian[0] * p_cartesian[0] +
                        p_cartesian[1] * p_cartesian[1] +
                        p_cartesian[2] * p_cartesian[2]);
  // azimuth angle
  p_spherical[0] = std::atan2(p_cartesian[1], p_cartesian[0]);
  // correct to [0,2*pi)
  if (p_spherical[0] < 0.0)
    p_spherical[0] += 2.0 * M_PI;

  // polar angle
  // acos returns the angle in the range [0,\pi]
  if (p_spherical[2] > std::numeric_limits<double>::min())
    p_spherical[1] = std::acos(p_cartesian[2] / p_spherical[2]);
  else
    p_spherical[1] = 0.0;

  return p_spherical;
}

/*
 * Convert point to cartesian coordinate system
 * p_cartesian[0:2] = [x y z]
 * p_spherical[0:2] = [phi (0..2pi) theta (0..pi) r]
 */
template<class T>
T point_to_cartesian(const T &p_spherical)
{
  T p_cartesian;
#ifdef DEBUG
  if (p_spherical[2] < 0.)
    throw std::runtime_error("Radial component is less than zero.");

  if (p_spherical[0] > 2. * M_PI)
    throw std::runtime_error("Phi component is greater than 2*pi.");
#endif

  p_cartesian[0] = p_spherical[2] * std::sin(p_spherical[1]) * std::cos(p_spherical[0]);
  p_cartesian[1] = p_spherical[2] * std::sin(p_spherical[1]) * std::sin(p_spherical[0]);
  p_cartesian[2] = p_spherical[2] * std::cos(p_spherical[1]);

  return p_cartesian;
}

template<class T, class U>
U vector_to_cartesian(const T &p_spherical, const U &v_spherical)
{
  U v_cartesian;

  /*
   *            vr                    vtheta            vphi
    vx = [ cos(Phi)*sin(Theta), cos(Phi)*cos(Theta), -sin(Phi)]
    vy = [ sin(Phi)*sin(Theta), cos(Theta)*sin(Phi),  cos(Phi)]
    vz = [          cos(Theta),         -sin(Theta),         0]
   */
  v_cartesian[0] = std::cos(p_spherical[0])*std::sin(p_spherical[1])*v_spherical[2]
                  +std::cos(p_spherical[0])*std::cos(p_spherical[1])*v_spherical[1]
                  -std::sin(p_spherical[0])*v_spherical[0];

  v_cartesian[1] = std::sin(p_spherical[0])*std::sin(p_spherical[1])*v_spherical[2]
                  +std::sin(p_spherical[0])*std::cos(p_spherical[1])*v_spherical[1]
                  +std::cos(p_spherical[0])*v_spherical[0];

  v_cartesian[2] =  std::cos(p_spherical[1])*v_spherical[2]
                   -std::sin(p_spherical[1])*v_spherical[1];

#ifdef DEBUG
  for(unsigned i = 0; i < 3; ++i)
    if(!std::isfinite(std::abs(v_cartesian[i])) || !std::isfinite(std::abs(v_cartesian[i])))
      throw std::runtime_error("Not a finite number in vector_to_cartesian");
#endif

  return v_cartesian;
}

template<class T, class U>
U vector_to_spherical(const T &p_cartesian, const U &v_cartesian)
{
  U v_spherical;
  const T p_spherical = point_to_spherical<T>(p_cartesian);

  /*
   vr     = [ cos(Psi)*sin(Theta), sin(Psi)*sin(Theta),  cos(Theta)]
   vtheta = [ cos(Psi)*cos(Theta), cos(Theta)*sin(Psi), -sin(Theta)]
   vphi   = [           -sin(Psi),            cos(Psi),           0]
   */
  v_spherical[0] =-std::sin(p_spherical[0])*v_cartesian[0]
                  +std::cos(p_spherical[0])*v_cartesian[1];
  v_spherical[1] = std::cos(p_spherical[0])*std::cos(p_spherical[1])*v_cartesian[0]
                  +std::cos(p_spherical[1])*std::sin(p_spherical[0])*v_cartesian[1]
                  -std::sin(p_spherical[1])*v_cartesian[2];
  v_spherical[2] = std::cos(p_spherical[0])*std::sin(p_spherical[1])*v_cartesian[0]
                  +std::sin(p_spherical[0])*std::sin(p_spherical[1])*v_cartesian[1]
                  +std::cos(p_spherical[1])*v_cartesian[2];

#ifdef DEBUG
  for(unsigned i = 0; i < 3; ++i)
    if(!std::isfinite(std::abs(v_spherical[i])) || !std::isfinite(std::abs(v_spherical[i])))
      throw std::runtime_error("Not a finite number in vector_to_spherical");
#endif

  return v_spherical;
}

#endif // _COMMON_
