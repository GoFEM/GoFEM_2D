#include "mt1d_normalized.h"
#include <cerrno>
#include <cfenv>

dcomplex MT1DNormalized::electric_field (const double z) const
{
  size_t ilay = 0;
  for (size_t i = 0; i < depths.size(); ++i)
  {
    if (z >= depths[i])
      ilay = i;
  }

  const dcomplex kk = sqrt(-II*omega*mu[ilay]*sigma[ilay]);
  const dcomplex expm = exp(-kk * (z - depths[ilay]));

  dcomplex expp = 0;
  if (ilay != depths.size() - 1)
    expp = exp(kk * (z - depths[ilay + 1]));

  //std::cout << z << "\t" << expp << "\t" << expm << "\n";

  dcomplex E = a[ilay + 1] * expp + b[ilay + 1] * expm;

  // +-1.#IND check
  if (isIndeterminate (E.real ()))
    E = dcomplex(0.0, E.imag ());

  if (isIndeterminate (E.imag ()))
    E = dcomplex(E.real (), 0.0);

  return std::conj(E);
}

dcomplex MT1DNormalized::magnetic_field (const double z) const
{
  size_t ilay = 0;
  for (size_t i = 0; i < depths.size(); ++i)
  {
    if (z >= depths[i])
      ilay = i;
  }

  const dcomplex kk = sqrt(-II*omega*mu[ilay]*sigma[ilay]);
  const dcomplex expm = exp(-kk * (z - depths[ilay]));

  dcomplex expp = 0;
  if (ilay != depths.size() - 1)
    expp = exp(kk * (z - depths[ilay + 1]));

  dcomplex H = c[ilay + 1] * expp + d[ilay + 1] * expm;

  // +-1.#IND check
  if (isIndeterminate(H.real ()))
    H = dcomplex(0.0, H.imag ());

  if (isIndeterminate(H.imag ()))
    H = dcomplex(H.real (), 0.0);

  return std::conj(H);
}

void MT1DNormalized::reinit (const dvector& sigma_, const dvector &eps_,
                             const dvector& mu_, const dvector& depths_,
                             const double freq)
{
  omega = 2.0 * M_PI * freq;

  depths = depths_;
  sigma = sigma_;
  mu = mu_;
  epsilon = eps_;

  size_t n = sigma_.size();

  //std::cout << n << std::endl;

  cvector kk(n + 1, 0.), Rp(n + 1, 0.), expmgh(n + 1, 0.);

  kk[0] = sqrt(-II*omega*mu[0]*sigma_[0]);
  //std::cout << "kk[0] = " << kk[0] << std::endl;

  for (size_t i = 0; i < n; ++i)
  {
    kk[i + 1] = sqrt(-II*omega*mu[i]*sigma_[i]);
    //std::cout << "kk[" << i + 1 << "] = " << kk[i + 1] << std::endl;
  }

  for (size_t i = 1; i <= n - 1; ++i)
  {
    expmgh[i] = exp(-kk[i]*( depths_[i] - depths_[i - 1]));
    //std::cout << "expmgh[" << i << "] = " << expmgh[i] << std::endl;
  }

  for (int i = n - 1; i > 0; --i)
  {
    const dcomplex gmogp = (-II*omega*mu[i]*(sigma_[i - 1] - sigma_[i] )) / pow((kk[i] + kk[i + 1]), 2.);
    const dcomplex rjexp = Rp[i + 1] * expmgh[i + 1];
    Rp[i] = (gmogp + rjexp) / (1.0 + gmogp*rjexp) * expmgh[i];
    //std::cout << "Rp[" << i << "] = " << Rp[i] << std::endl;
  }

  // Preallocate memory
  a.resize (n + 1, 0.);
  b.resize (n + 1, 0.);
  c.resize (n + 1, 0.);
  d.resize (n + 1, 0.);

  a[0] = 1;
  c[0] = 1;
  
  for (size_t i = 1; i <= n; ++i)
  {
    // E coefficients
    dcomplex Atop = (a[i-1] + b[i-1]*expmgh[i-1]);
    b[i] = Atop / (1.0 + Rp[i]*expmgh[i]);
    a[i] = b[i]*Rp[i];

    // H coefficients:
    Atop = (c[i-1] + d[i-1]*expmgh[i-1] );   // H uses same recursion coeff, but with reversed sign (-):
    d[i] = Atop / (1.0 - Rp[i]*expmgh[i]);
    c[i] = -d[i]*Rp[i];

//    std::cout << "a[" << i << "] = " << a[i] << " "
//              << "b[" << i << "] = " << b[i] << " "
//              << "c[" << i << "] = " << c[i] << " "
//              << "d[" << i << "] = " << d[i]
//              << std::endl;
  }

}

