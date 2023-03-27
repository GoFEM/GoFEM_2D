#ifndef _MT1D_
#define _MT1D_

#include "common.h"

/*
 * Forward MT problem for horizontally stratified medium
 * Does not neglect displacement currents.
 * Can be used for RMT.
 * Uses +i*omega*t convention.
 */
class MT1D
{
  dvector sigma;      // conductivities
  dvector epsilon;    // permittivities
  dvector mu;         // magnetic permeabilities
  dvector depths;     // depths of the layer boundaries
  dvector h;          // thickness for each layer
  cvector k;          // wave numbers for each layer
  cvector Z;          // impedances
  cvector R;          // reflectivity coefficients
  double omega;       // angular frequency

  cvector r, a, b;    // some constant coefficients independent of z

public:
  MT1D () {}
  MT1D (const dvector& sigma_, const dvector& eps_,
        const dvector& mu_, const dvector& depths_, const double freq);

  void reinit (const dvector& sigma_, const dvector& eps_,
               const dvector& mu_, const dvector& depths_, const double freq);
  void reinit (const dvector& sigma_, const dvector& depths_, const double freq);

  void reinit (const dvector& sigma_, const dvector& depths_);
  void reinit (const dvector& sigma_, const dvector& eps_,
               const dvector& mu_, const dvector& depths_);

  dcomplex impedance (const double z) const;
  void rho_phase (const double z, double &rho, double& phase) const;

  dcomplex impedance (const double z, double frequency);
  void rho_phase (const double z, double &rho, double& phase, double frequency);

  void z_rho_phi(const double z, double frequency, dcomplex &imp, double &rho, double &phase);

  dcomplex electric_field (const double z) const;
  dcomplex magnetic_field (const double z) const;

private:
  void precompute(double frequency);
};

#endif // _MT1D_
