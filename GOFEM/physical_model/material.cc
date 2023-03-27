#include "material.h"

#include <sstream>
#include <stdexcept>
#include <limits>
#include <cmath>

#include <common.h>

std::istream& operator>> (std::istream &in, Material &r)
{
  r.reset();

  if(r.type == IsotropicPolarized)
  {
    in >> r.id >> r.name >> r.conductivity >> r.permittivity
       >> r.chargeability >> r.relaxation_time >> r.exponent;
  }
  else if(r.type == Isotropic)
  {
    in >> r.id >> r.name >> r.conductivity >> r.permittivity;
  }
  else
    throw std::runtime_error("Unknown or not supported material type.");

  return in;
}

std::ostream& operator<< (std::ostream &out, const Material &r)
{
  if(r.type == IsotropicPolarized)
  {
    out << r.id << "\t" << r.name << "\t" << r.conductivity << "\t"
        << r.permittivity << "\t" << r.chargeability << "\t"
        << r.relaxation_time << "\t" << r.exponent;
  }
  else if(r.type == Isotropic)
  {
    out << r.id << "\t" << r.name << "\t"
        << r.conductivity << "\t" << r.permittivity;
  }
  else
    throw std::runtime_error("Unknown or not supported material type.");

  return out;
}

Material::Material(MaterialType t)
{
  reset();
  type = t;
  name = "Material";
}

Material::Material(double conductivity,
                   double permittivity):
  permittivity(permittivity),
  conductivity(conductivity),
  type(Isotropic),
  name("Material")
{
}

bool Material::operator==(const Material &other) const
{
  if(id != other.id)
    return false;

  if(type != other.type)
    return false;

  if(fabs(conductivity - other.conductivity) > std::numeric_limits<float>::epsilon() ||
     fabs(permittivity - other.permittivity) > std::numeric_limits<float>::epsilon())
    return false;

  return true;
}

bool Material::operator!=(const Material &other) const
{
  return !(*this == other);
}

bool Material::is_valid() const
{
  bool flag = std::isfinite(conductivity) &
              std::isfinite(permittivity) &
              (conductivity > 0.) &
              id != -1;

  if(type == IsotropicPolarized)
  {
    flag &= std::isfinite(chargeability) &
            std::isfinite(relaxation_time) &
            std::isfinite(exponent);
  }

  return flag;
}

void Material::reset()
{
  id = -1;
  name = "";
  conductivity = std::numeric_limits<double>::quiet_NaN();
  permittivity = 0;
  chargeability = 0;
  relaxation_time = 0;
  exponent = 0;
}
