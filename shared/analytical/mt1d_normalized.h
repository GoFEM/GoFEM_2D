#ifndef _MT1D_NORMALIZED_
#define _MT1D_NORMALIZED_

#include "common.h"

/*
 * Forward MT problem for horizontally layered medium
 * Uses +i_omega_mu convention
 */
class MT1DNormalized
{
  dvector sigma;      // conductivities
  dvector epsilon;    // permittivities
  dvector mu;         // magnetic permeabilities
  dvector depths;     // depths of the layer boundaries
  double omega;       // angular frequency

  cvector a, b, c, d;    // some constant coefficients independent of z

public:
  MT1DNormalized () {}
  MT1DNormalized (const dvector& sigma_, const dvector& eps_, const dvector& mu_, const dvector& depths_, const double freq)
  {
    reinit (sigma_, eps_, mu_, depths_, freq);
  }

  dcomplex electric_field (const double z) const;
  dcomplex magnetic_field (const double z) const;
  void reinit (const dvector& sigma_, const dvector& eps_,
               const dvector& mu_, const dvector& depths_, const double freq);
};

#endif
