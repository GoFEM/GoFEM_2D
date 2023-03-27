#ifndef MATERIAL_H
#define MATERIAL_H

#include <string>
#include <vector>
#include <map>
#include <iostream>

enum MaterialType {UnderfinedMaterial, Isotropic, IsotropicPolarized,
                   TriaxialAnisotropy, TriaxialAnisotropyPolarized};

static std::map<std::string, MaterialType> material_type_name_to_type_table =
{
  {"iso", Isotropic},
  {"iso_polarized", IsotropicPolarized},
  {"vti", TriaxialAnisotropy}
};

static std::map<MaterialType, std::string> material_type_to_name_table =
{
  {Isotropic, "iso"},
  {IsotropicPolarized, "iso_polarized"},
  {TriaxialAnisotropy, "vti"}
};

/*
 * Defines physical material correcponding to the certain ID
 */
class Material
{
public:
  Material(MaterialType t = UnderfinedMaterial);
  Material(double conductivity,
           double permittivity);

  bool operator==(const Material &other) const;
  bool operator!=(const Material &other) const;

  bool is_valid() const;

  void reset();

  MaterialType type;
  std::string name;
  int id;

  double conductivity;
  double permittivity;

  // Cole-Cole properties
  float chargeability, relaxation_time, exponent;

  friend std::istream& operator>> (std::istream &in, Material &r);
  friend std::ostream& operator<< (std::ostream &out, const Material &r);
};

typedef std::vector<Material> MaterialList;

#endif // MATERIAL_H
