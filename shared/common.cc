#include "common.h"

#include <cmath>

bool isIndeterminate (const double v)
{
  return (v != v);
}

void fill_array_logscale(double minv, double maxv, unsigned n, std::vector<double> &array)
{
  array.resize(n);
  double delta = fabs(log10(maxv) - log10(minv)) / (n - 1);

  for(unsigned i = 0; i < n; ++i)
    array[i] = pow(10., std::min(log10(minv), log10(maxv)) + i * delta);
}

bool icompare_pred(unsigned char a, unsigned char b)
{
  return std::tolower(a) == std::tolower(b);
}

bool istrcompare(const std::string& a, const std::string& b)
{
  if(a.length() != b.length())
    return false;
  else
    return std::equal(a.begin(), a.end(), b.begin(), icompare_pred);
}

bool replace(std::string& str, const std::string& from, const std::string& to)
{
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

std::string trim(const std::string& str, const std::string& whitespace)
{
    const auto strBegin = str.find_first_not_of(whitespace);
    if (strBegin == std::string::npos)
        return ""; // no content

    const auto strEnd = str.find_last_not_of(whitespace);
    const auto strRange = strEnd - strBegin + 1;

    if(strBegin+strRange > str.length())
      throw std::runtime_error("trim: out of range");

    return str.substr(strBegin, strRange);
}

void split(const std::string &s, char delim, std::vector<std::string> &elems)
{
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        elems.push_back(item);
    }
}

double colat2lat(const double &colat)
{
  return M_PI/2. - colat;
}

double colon2lon(const double &colon)
{
  if(colon <= M_PI)
    return colon;
  else
    return colon - 2.*M_PI;
}

void gauleg(const double &x1, const double &x2, dvector &x, dvector &w, const unsigned n)
{
  const double eps = 1.0e-8;

  double z1, pp, p3, p2, p1;

  if(x.size() != n || w.size() != n)
  {
    x.resize(n, 0.);
    w.resize(n, 0.);
  }

  const unsigned m = (n + 1) / 2;

  const double xm = 0.5 * (x2 + x1);
  const double xl = 0.5 * (x2 - x1);

  for (unsigned i = 0; i < m; ++i)
  {
    double z = std::cos(M_PI * (i + 0.75) / (n + 0.5));

    do
    {
      p1=1.0;
      p2=0.0;
      for (unsigned j = 0; j < n; ++j)
      {
        p3 = p2;
        p2 = p1;
        p1 = ((2. * j + 1.) * z * p2 - j * p3) / (j + 1);
      }
      pp = n * (z * p1 - p2) / (z * z - 1.);
      z1 = z;
      z = z - p1 / pp;
    }
    while (std::fabs(z - z1) > eps);

    x[i] = xm - xl*z;
    x[n-1-i] = xm + xl*z;
    w[i] = 2. * xl / ((1. - z * z) * pp * pp);
    w[n-1-i] = w[i];
  }
}

void gauleg2d(const std::array<double, 2> x,
              const std::array<double, 2> y,
              std::vector<std::array<double, 2>> &points,
              std::vector<double> &weights,
              const unsigned n)
{
  std::vector<double> pointsx, weightsx, pointsy, weightsy;
  gauleg(x[0], x[1], pointsx, weightsx, n);
  gauleg(y[0], y[1], pointsy, weightsy, n);

  points.resize(n*n);
  weights.resize(n*n);

  unsigned int present_index = 0;
  for (unsigned int i2 = 0; i2 < weightsy.size(); ++i2)
    for (unsigned int i1 = 0; i1 < weightsx.size(); ++i1)
    {
      // compose coordinates of
      // new quadrature point by tensor
      // product in the last component
      points[present_index][0] = pointsx[i1];
      points[present_index][1] = pointsy[i2];
      weights[present_index] = weightsx[i1] * weightsy[i2];
      ++present_index;
    };
}

double gammln(const double &xx)
{
  if (xx <= 0)
    throw std::runtime_error("Bad arg in gammln");

  static const double cof[14]={57.1562356658629235,-59.5979603554754912,
  14.1360979747417471,-0.491913816097620199,.339946499848118887e-4,
  .465236289270485756e-4,-.983744753048795646e-4,.158088703224912494e-3,
  -.210264441724104883e-3,.217439618115212643e-3,-.164318106536763890e-3,
  .844182239838527433e-4,-.261908384015814087e-4,.368991826595316234e-5};

  double y = xx, x = xx;

  double tmp = x + 5.24218750000000000;
  tmp = (x + 0.5) * log(tmp) - tmp;
  double ser = 0.999999999999997092;
  for (unsigned j = 0; j < 14; ++j)
    ser += cof[j] / ++y;
  return tmp + log(2.5066282746310005 * ser / x);
}

void gaulag(dvector &x, dvector &w, const double &alpha, const unsigned n)
{
  if(x.size() != n || w.size() != n)
  {
    x.resize(n, 0.);
    w.resize(n, 0.);
  }

  const unsigned maxit = 32;
  const double eps = 1.0e-11;
  unsigned its;

  double ai, p1, p2, p3, pp, z = 0, z1 = 0;

  for (unsigned i = 0; i < n; ++i)
  {
    if (i == 0)
    {
      z=(1.0+alpha)*(3.0+0.92*alpha)/(1.0+2.4*n+1.8*alpha);
    }
    else if (i == 1)
    {
      z += (15.0+6.25*alpha)/(1.0+0.9*alpha+2.5*n);
    }
    else
    {
      ai=i-1;
      z += ((1.0+2.55*ai)/(1.9*ai)+1.26*ai*alpha/
            (1.0+3.5*ai))*(z-x[i-2])/(1.0+0.3*alpha);
    }

    for (its = 0; its < maxit; ++its)
    {
      p1 = 1.0;
      p2 = 0.0;
      for (unsigned j = 0; j < n; ++j)
      {
        p3 = p2;
        p2 = p1;
        p1 = ((2 * j + 1 + alpha - z) * p2 - (j + alpha) * p3) / (j + 1);
      }
      pp = (n * p1 - (n + alpha) * p2) / z;
      z1 = z;
      z = z1 - p1 / pp;
      if (std::fabs(z - z1) <= eps)
        break;
    }
    if (its >= maxit)
      throw std::runtime_error("Too many iterations in gaulag");
    x[i] = z;
    w[i] = -exp(gammln(alpha + n) - gammln(double(n))) / (pp * n * p2);
  }
}


dvec3d gg2gm(const dvec3d &pole_coords, const dvec3d &point, const bool forward)
{
  const double rad = M_PI / 180.;
  const double s_p_b = sin(pole_coords[0]*rad), c_p_b = cos(pole_coords[0]*rad);
  const double c_t_b = cos(pole_coords[1]*rad), s_t_b = sin(pole_coords[1]*rad);

  double A[3][3] = {{+c_t_b*c_p_b, +c_t_b*s_p_b, -s_t_b},
                    {      -s_p_b,       +c_p_b,      0},
                    {+s_t_b*c_p_b, +s_t_b*s_p_b, +c_t_b}};

  if(!forward) // gm -> gg
  {
    std::swap(A[0][1], A[1][0]);
    std::swap(A[2][1], A[1][2]);
    std::swap(A[2][0], A[0][2]);
  }

  const double c_t = cos(point[1]*rad), s_t = sin(point[1]*rad);
  const double c_p = cos(point[0]*rad), s_p = sin(point[0]*rad);

  const double z = c_t;
  const double x = s_t * c_p;
  const double y = s_t * s_p;

  const double x_gm = A[0][0]*x + A[0][1]*y + A[0][2]*z;
  const double y_gm = A[1][0]*x + A[1][1]*y + A[1][2]*z;
  const double z_gm = A[2][0]*x + A[2][1]*y + A[2][2]*z;

  dvec3d p;
  p[1] = 90 - std::atan2(z_gm, sqrt(x_gm*x_gm + y_gm*y_gm))/rad;
  // mod(a, m) = a - m*floor(a/m);
  const double a = std::atan2(y_gm, x_gm)/rad;
  p[0] = a - 360. * std::floor(a / 360.);

  return p;
}

cvec3d gg2gm(const dvec3d &pole_coords, const dvec3d &point,
             const cvec3d &field, const bool forward,
             dvec3d &point_transformed)
{
  const double rad = M_PI / 180.;
  const double s_p_b = sin(pole_coords[0]*rad), c_p_b = cos(pole_coords[0]*rad);
  const double c_t_b = cos(pole_coords[1]*rad), s_t_b = sin(pole_coords[1]*rad);

  double A[3][3] = {{+c_t_b*c_p_b, +c_t_b*s_p_b, -s_t_b},
                    {      -s_p_b,       +c_p_b,      0},
                    {+s_t_b*c_p_b, +s_t_b*s_p_b, +c_t_b}};

  if(!forward) // gm -> gg
  {
    std::swap(A[0][1], A[1][0]);
    std::swap(A[2][1], A[1][2]);
    std::swap(A[2][0], A[0][2]);
  }

  double c_t = cos(point[1]*rad), s_t = sin(point[1]*rad);
  double c_p = cos(point[0]*rad), s_p = sin(point[0]*rad);

  const double z = c_t;
  const double x = s_t * c_p;
  const double y = s_t * s_p;

  const double x_gm = A[0][0]*x + A[0][1]*y + A[0][2]*z;
  const double y_gm = A[1][0]*x + A[1][1]*y + A[1][2]*z;
  const double z_gm = A[2][0]*x + A[2][1]*y + A[2][2]*z;

  point_transformed[2] = point[2];
  point_transformed[1] = 90 - std::atan2(z_gm, sqrt(x_gm*x_gm + y_gm*y_gm))/rad;
  // mod(a, m) = a - m*floor(a/m);
  const double a = std::atan2(y_gm, x_gm)/rad;
  point_transformed[0] = a - 360. * std::floor(a / 360.);

  // Now transform field
  const dcomplex B_theta = field[1],
                 B_phi   = field[0],
                 BE = B_theta * c_t,
                 Bx = BE * c_p - B_phi * s_p,
                 By = BE *s_p + B_phi *c_p,
                 Bz = -B_theta * s_t;

  const dcomplex Bx_gm = A[0][0]*Bx + A[0][1]*By + A[0][2]*Bz;
  const dcomplex By_gm = A[1][0]*Bx + A[1][1]*By + A[1][2]*Bz;
  const dcomplex Bz_gm = A[2][0]*Bx + A[2][1]*By + A[2][2]*Bz;

  c_t = cos(point_transformed[1]*rad);
  s_t = sin(point_transformed[1]*rad);
  c_p = cos(point_transformed[0]*rad);
  s_p = sin(point_transformed[0]*rad);

  cvec3d B_gm;
  const dcomplex BE1 = Bx_gm * c_p + By_gm * s_p;
  B_gm[1] = BE1 * c_t - Bz_gm * s_t; // B_theta
  B_gm[0] = By_gm * c_p - Bx_gm * s_p; // B_phi
  B_gm[2] = field[2]; // B_r

  return B_gm;
}

std::string exec(const char* cmd, int &status)
{
  std::array<char, 128> buffer;
  std::string result;
  FILE* pipe = popen(cmd, "r");
  if (!pipe)
    throw std::runtime_error("popen() failed!");

  while (!feof(pipe))
  {
    if (fgets(buffer.data(), 128, pipe) != NULL)
      result += buffer.data();
  }

  status = pclose(pipe);
  return result;
}
