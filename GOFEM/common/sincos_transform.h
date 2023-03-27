#ifndef SINCOS_TRANSFORM_H
#define SINCOS_TRANSFORM_H

#include <string>
#include <fstream>

#include "common.h"

enum TransformationType {CosineTransform, SineTransform};
enum TransformationDirection {ForwardTransform, InverseTransform};

// Calculates sine or cosine Fourier transform using digital 601-point filter
// This class is ised to transform wave-domain fields to spatial fields andvice versa
template<class T>
class SinCosTransform
{
public:
  SinCosTransform() {}
  SinCosTransform(const std::string &filter_file);

  void reinit(const std::string &filter_file);

  /*
   Calculates the integral of const*f(x)*w(xi*x) from
   x = 0 to infinity where w() is cos() or sin() dedepnding on
   requested transformation type.
   if is_forward = true then:
   const is 1/pi or sqrt(-1)/pi for cosine and sine transforms.
   if is_forward = false then:
   const is 2*pi or sqrt(-1)*2*pi for cosine and sine transforms.
   This routine employs spline interpolation to fill in f(x) at
   values between discrete solutions. Spline is built for log10(x)

   Input function can be any E/H field component at discrete x values.
   Note: this methods assumes that x are in ascending order
  */
  T integrate(const dvector &x, const dvector &logx, const std::vector<T> &fx,
              const double &xi, TransformationType type, TransformationDirection dir,
              const double &minx);

private:
  void read_filter(const std::string &filter_file);

  // Subroutine to generate second derivative of spline interpolating function
  // Modified from Numerical Recipes section 3.3.
  void calculate_spline_derivative(const dvector &logx, const std::vector<T> &fx);

  // Performs spline interpolation on set of fkx at wavenumbers kx
  T interpolate(const dvector &x, const dvector &logx,
                const std::vector<T> &fx, double xi, double logxi);

  T get_transformation_constant(TransformationType type, TransformationDirection dir) const;

  // Base, cosine and sine filter coefficients
  std::vector<double> base, cosf, sinf;
  std::vector<T> spline_derivative;

  std::vector<double> y2real, y2imag, ureal, uimag;
  double sig, dx1, dx2, dx3, dyr1, dyr2, dyi1, dyi2;
  double preal, pimag;
};

#endif
