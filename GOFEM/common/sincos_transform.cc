#include "sincos_transform.h"

#include <sstream>

template<class T>
SinCosTransform<T>::SinCosTransform(const std::string &filter_file)
{
  reinit(filter_file);
}

template<class T>
void SinCosTransform<T>::reinit(const std::string &filter_file)
{
  read_filter(filter_file);
}

template<class T>
void SinCosTransform<T>::read_filter(const std::string &filter_file)
{
  std::ifstream ifs (filter_file.c_str ());

  if (!ifs.is_open ())
    throw std::ios_base::failure (std::string ("Can not open file " + filter_file).c_str ());

  std::string line;

  // read in all data skipping comments and empty lines
  while (!ifs.eof())
  {
    std::getline (ifs, line);

    if (((line.length() < 2) && line[0] < '0') || (line[0] == '#'))
      continue;

    std::stringstream ss(line);

    double b, c, s;
    ss >> b >> c >> s;

    base.push_back(b);
    cosf.push_back(c);
    sinf.push_back(s);
  }
}

template<class T>
T SinCosTransform<T>::integrate(const dvector &x, const dvector &logx, const std::vector<T> &fx, const double &xi,
                                TransformationType type, TransformationDirection dir, const double &minx)
{
  double sign = 1; // no sign reversal for even (cos) integral
  double* filter = &cosf[0]; // even function, use cos integral

  if (type == SineTransform) // odd function, use sine integral
  {
    filter = &sinf[0];
    if (xi < 0)
      sign = -1; // sign reversal for odd integral in negative domain
  }

  calculate_spline_derivative(logx, fx);

  // limit small ranges
  const double r = std::max(minx, std::fabs(xi));
//  if(dir == ForwardTransform)
//    r = std::max(1e-8, std::fabs(xi));
//  else if(dir == InverseTransform)
//    r = std::max(1., std::fabs(xi));

  // Perform the cosine (or sine) transform using the digital filter method
  T sumf = 0.;
  for(unsigned i = 0; i < base.size(); ++i)
  {
    // Get x value from base
    const double x_base = base[i] / r;

    // Get interpolated field in wavenumber domain and apply digital filter:
    sumf += interpolate(x, logx, fx, x_base, log10(x_base)) * filter[i];
  }

  // Normalize by r
  sumf /= r;

  // Apply coefficients and return
  const T c = get_transformation_constant(type, dir);
  return sign * c * sumf;
}

template<>
void SinCosTransform<double>::calculate_spline_derivative(const dvector &logx, const std::vector<double> &fx)
{
  if(spline_derivative.size() != fx.size())
  {
    spline_derivative.resize(fx.size());
    y2real.resize(fx.size());
    ureal.resize(fx.size());
  }

  double sig, dx1, dx2, dx3, dyr1, dyr2;
  double preal;

  for(unsigned i = 1; i < fx.size() - 1; ++i)
  {
    dx1 = logx[i+1] - logx[i];
    dx2 = logx[i]   - logx[i-1];
    dx3 = logx[i+1] - logx[i-1];
    sig = dx2 / dx3;

    preal = sig*y2real[i-1] + 2.;
    y2real[i] = (sig - 1.)/preal;

    dyr1 = fx[i+1] - fx[i];
    dyr2 = fx[i]   - fx[i-1];

    ureal[i] = (6. * (dyr1/dx1 - dyr2/dx2) / dx3 - sig*ureal[i-1]) / preal;
  }

  for(unsigned k = fx.size() - 2; k > 0; --k)
    y2real[k] = y2real[k] * y2real[k+1] + ureal[k];

  for(unsigned i = 0; i < fx.size(); ++i)
    spline_derivative[i] = y2real[i];
}

template<>
void SinCosTransform<dcomplex>::calculate_spline_derivative(const dvector &logx, const std::vector<dcomplex> &fx)
{
  if(spline_derivative.size() != fx.size())
  {
    spline_derivative.resize(fx.size());
    y2real.resize(fx.size());
    y2imag.resize(fx.size());
    ureal.resize(fx.size());
    uimag.resize(fx.size());
  }

  for(unsigned i = 1; i < fx.size() - 1; ++i)
  {
    dx1 = logx[i+1] - logx[i];
    dx2 = logx[i]   - logx[i-1];
    dx3 = logx[i+1] - logx[i-1];
    sig = dx2 / dx3;

    preal = sig*y2real[i-1] + 2.;
    pimag = sig*y2imag[i-1] + 2.;

    y2real[i] = (sig - 1.)/preal;
    y2imag[i] = (sig - 1.)/pimag;

    dyr1 = fx[i+1].real() - fx[i].real();
    dyr2 = fx[i].real()   - fx[i-1].real();
    dyi1 = fx[i+1].imag() - fx[i].imag();
    dyi2 = fx[i].imag()   - fx[i-1].imag();

    ureal[i] = (6. * (dyr1/dx1 - dyr2/dx2) / dx3 - sig*ureal[i-1]) / preal;
    uimag[i] = (6. * (dyi1/dx1 - dyi2/dx2) / dx3 - sig*uimag[i-1]) / pimag;
  }

  for(unsigned k = fx.size() - 2; k > 0; --k)
  {
    y2real[k] = y2real[k] * y2real[k+1] + ureal[k];
    y2imag[k] = y2imag[k] * y2imag[k+1] + uimag[k];
  }

  for(unsigned i = 0; i < fx.size(); ++i)
    spline_derivative[i] = std::complex<double>(y2real[i], y2imag[i]);
}

template<class T>
T SinCosTransform<T>::interpolate(const dvector &x, const dvector &logx, const std::vector<T> &fx, double xi, double logxi)
{
  // Identify interval we are in
  // x larger than maximum tabulated value
  if(logxi >= logx[x.size() - 1]) // return zero for higher wavenumbers
  {
    //std::cout << "1  " << xlog << "  " << 0. << "\n";
    return 0.;
  }
  else if(logxi >= logx[0]) // x in tabulated range
  {
    int klo = 1, khi = x.size();

    while ((khi-klo) > 1)
    {
      int k = (khi + klo) / 2;
      if (logx[k - 1] > logxi)
        khi = k;
      else
        klo = k;
    }

    --klo;
    --khi;

    const double h2 = logx[khi] - logx[klo];
    const double a  = ( logx[khi] - logxi ) / h2;
    const double b  = ( logxi - logx[klo] ) / h2;
    //std::cout << "2  " << xlog << "  " << a * fkx[klo] + b * fkx[khi] + ((pow(a, 3.) - a) * spline_derivative[klo] + (pow(b, 3.) - b) * spline_derivative[khi]) * (h2*h2) / 6. << "\n";
    return a * fx[klo] + b * fx[khi] + ((pow(a, 3.) - a) * spline_derivative[klo] + (pow(b, 3.) - b) * spline_derivative[khi]) * (h2*h2) / 6.;
  }
  else // x is less than the minimum tabulated value
  {
    // Use trapezoidal extrapolation, this helps a lot for CSEM responses at long offsets
    const T dydx = (fx[1] - fx[0]) / (x[1] - x[0]);
    //std::cout << "3  " << xlog << "  " << fkx[0] + dydx*(x - kx[0]) << "\n";
    return fx[0] + dydx*(xi - x[0]);
  }
}

template<>
double SinCosTransform<double>::get_transformation_constant(TransformationType type, TransformationDirection dir) const
{
  double c = 1.;
  if (type == CosineTransform) // even function, use cos integral
  {
    if(dir == ForwardTransform)
      c = 2.;
    else if(dir == InverseTransform)
      c = 2. / M_PI;
  }
  else if (type == SineTransform) // odd function, use sine integral
  {
    if(dir == ForwardTransform)
      c = 2.;
    else if(dir == InverseTransform)
      c = 2. / M_PI;
  }

  return c;
}

template<>
std::complex<double> SinCosTransform<std::complex<double>>::get_transformation_constant(TransformationType type, TransformationDirection dir) const
{
  std::complex<double> c;
  if (type == CosineTransform) // even function, use cos integral
  {
    if(dir == ForwardTransform)
      c = dcomplex(2., 0.);
    else if(dir == InverseTransform)
      c = dcomplex(1.0 / M_PI, 0.);
  }
  else if (type == SineTransform) // odd function, use sine integral
  {
    if(dir == ForwardTransform)
      c = dcomplex(0., -2.);
    else if(dir == InverseTransform)
      c = dcomplex(0., 1.0 / M_PI);
  }

  return c;
}

template class SinCosTransform<double>;
template class SinCosTransform<std::complex<double>>;
