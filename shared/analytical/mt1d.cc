/**
 * CMAINV, Geophysical data inversion using Covariance Matrix Adaptation
 * Copyright (c) 2015 Alexander Grayver <agrayver@erdw.ethz.ch>
 *
 * This file is part of CMAINV.
 *
 * CMAINV is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CMAINV is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CMAINV. If not, see <http://www.gnu.org/licenses/>.
 */
#include "mt1d.h"
#include <cerrno>
#include <cfenv>

MT1D::MT1D(const dvector &sigma_, const dvector &eps_, const dvector &mu_,
           const dvector &depths_, const double freq)
{
  reinit (sigma_, eps_, mu_, depths_, freq);
}

void MT1D::reinit (const dvector& sigma_, const dvector &eps_, const dvector& mu_,
                   const dvector& depths_, const double freq)
{
  reinit(sigma_, eps_, mu_, depths_);
  precompute(freq);
}

dcomplex MT1D::electric_field (const double z) const
{
  dcomplex E = 0.0;
  for (size_t i = 1; i < depths.size(); ++i)
  {
    if (z >= depths[i - 1] && z < depths[i])
    {
      std::feclearexcept(FE_ALL_EXCEPT);

      const dcomplex A = a[i - 1] * exp( II * k[i - 1] * (z - depths[i - 1])),
                     B = b[i - 1] * exp(-II * k[i - 1] * (z - depths[i - 1]));

      if(std::fetestexcept(FE_OVERFLOW))
        E = 0.;
      else
        E = Z[i - 1] * (A + B);

      break;
    }
  }

  return std::conj(E);
}

dcomplex MT1D::magnetic_field (const double z) const
{
  dcomplex H = 0.0;
  for (size_t i = 1; i < depths.size(); ++i)
  {
    if (z >= depths[i - 1] && z < depths[i])
    {
      std::feclearexcept(FE_ALL_EXCEPT);

      const dcomplex A = a[i - 1] * exp( II * k[i - 1] * (z - depths[i - 1])),
                     B = b[i - 1] * exp(-II * k[i - 1] * (z - depths[i - 1]));

      if(std::fetestexcept(FE_OVERFLOW))
        H = 0.;
      else
        H = A - B;

      break;
    }
  }

  return std::conj(H);
}

void MT1D::precompute(double frequency)
{
  size_t n = sigma.size();

  // Preallocate memory
  k.resize (n);
  Z.resize (n);
  r.resize (n);
  a.resize (n);
  b.resize (n);
  R.clear();
  h.clear();

  omega = 2.0 * M_PI * frequency;

  for (size_t i = 0; i < n; ++i)
  {
    if(epsilon[i] < std::numeric_limits<double>::min())
      k[i] = sqrt (II * omega * mu[i] * sigma[i]); // neglect displacement currents
    else
      k[i] = omega * sqrt(0.5 * mu[i] * epsilon[i]) * (sqrt(sqrt(1. + pow((sigma[i]/(omega*epsilon[i])), 2.0)) + 1.) +
                                                       sqrt(sqrt(1. + pow((sigma[i]/(omega*epsilon[i])), 2.0)) - 1.) * II);

    Z[i] = (omega * mu[i]) / k[i];

    if (i > 0)
      h.push_back (depths[i] - depths[i - 1]);

    if (i > 0)
      R.push_back ((Z[i] - Z[i - 1]) / (Z[i] + Z[i - 1]));
  }

  r[n - 1] = 0.0;
  r[n - 2] = R[n - 2] * exp(2.0 * II * k[n - 2] * h[n - 2]);
  for (int i = n - 3; i > 0; --i)
  {
    r[i] = (R[i] + r[i + 1]) / (1.0 + R[i]*r[i + 1]) * exp(2.0 * II * k[i] * h[i]);
  }
  r[0] = (R[0] + r[1]) / (1.0 + R[0]*r[1]);

  a[0] = 1.0 / (1.0 - r[0]);
  a[1] = a[0] * ((1.0 - R[0]) / (1.0 + R[0]*r[1]));
  for (size_t i = 2; i < n - 1; ++i)
  {
    a[i] = a[i - 1] * ((1.0 - R[i - 1]) / (1.0 + R[i - 1] * r[i])) * exp(II * k[i - 1] * h[i - 1]);
  }
  a[n - 1] = a[n - 2] * (1.0 - R[n - 2]) * exp(II * k[n - 2] * h[n - 2]);

  for (size_t i = 0; i < n; ++i)
    b[i] = a[i]*r[i];
}

dcomplex MT1D::impedance (const double z) const
{
  return electric_field (z) / magnetic_field (z);
}

void MT1D::reinit(const dvector &sigma_, const dvector &depths_, const double freq)
{
  // Permetivitty and permeability of free space
  dvector eps(sigma_.size(), eps0);
  dvector mu(sigma_.size(), mu0);

  reinit(sigma_, eps, mu, depths_, freq);
}

void MT1D::reinit(const dvector &sigma_, const dvector &depths_)
{
  // Permetivitty and permeability of free space
  dvector eps_(sigma_.size(), eps0);
  dvector mu_(sigma_.size(), mu0);

  reinit(sigma_, eps_, mu_, depths_);
}

void MT1D::reinit(const dvector &sigma_, const dvector &eps_, const dvector &mu_, const dvector &depths_)
{
  size_t n = sigma_.size() + 1;

  depths.resize (n);
  sigma.resize (n);
  epsilon.resize (n);
  mu.resize (n);

  // Create artificial zero meters thick layer
  depths[0] = depths_[0];
  std::copy (depths_.begin (), depths_.end (), depths.begin () + 1);
  sigma[0] = sigma_[0];
  std::copy (sigma_.begin (), sigma_.end (), sigma.begin () + 1);
  epsilon[0] = eps_[0];
  std::copy (eps_.begin (), eps_.end (), epsilon.begin () + 1);
  mu[0] = mu_[0];
  std::copy (mu_.begin (), mu_.end (), mu.begin () + 1);

  depths.push_back(depths.back()*2);
  sigma.push_back(sigma.back());
  epsilon.push_back(epsilon.back());
  mu.push_back(mu.back());
}

void MT1D::rho_phase (const double z, double &rho, double &phase) const
{
  const dcomplex Z = impedance(z);

  for (size_t i = 1; i < depths.size(); ++i)
  {
    if (z >= depths[i - 1] && z < depths[i])
    {
      rho = 1.0 / (mu[i - 1] * omega) * (Z.real () * Z.real () + Z.imag () * Z.imag ());
      phase = atan (Z.imag () / Z.real ()) * 180.0 / M_PI;
      break;
    }
  }
}

dcomplex MT1D::impedance(const double z, double frequency)
{
  precompute(frequency);

  return electric_field (z) / magnetic_field (z);
}

void MT1D::rho_phase(const double z, double &rho, double &phase, double frequency)
{
  precompute(frequency);
  dcomplex Z = impedance(z);

  for (size_t i = 1; i < depths.size(); ++i)
  {
    if (z >= depths[i - 1] && z < depths[i])
    {
      rho = 1.0 / (mu[i - 1] * omega) * (Z.real () * Z.real () + Z.imag () * Z.imag ());
      phase = atan2 (Z.imag (), Z.real ()) * 180.0 / M_PI;
      break;
    }
  }
}

void MT1D::z_rho_phi(const double z, double frequency, dcomplex &imp, double &rho, double &phase)
{
  precompute(frequency);
  imp = impedance(z);

  for (size_t i = 1; i < depths.size(); ++i)
  {
    if (z >= depths[i - 1] && z < depths[i])
    {
      rho = 1.0 / (mu[i - 1] * omega) * (imp.real () * imp.real () + imp.imag () * imp.imag ());
      phase = atan2 (imp.imag (), imp.real ()) * 180.0 / M_PI;
      break;
    }
  }
}
