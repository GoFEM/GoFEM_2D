#ifndef DIPOLE_SOURCE_H
#define DIPOLE_SOURCE_H

#include "physical_source.h"

#include "common.h"

/*
 * Three types of supported dipoles sources
 */
enum DipoleType {ElectricDipole, MagneticDipole};

/*
 * Defines point dipole source at a specified point.
 * This class also enables us to define elementary
 * point dipole source of an arbitrary extent and current.
 * Note that this source can consist of many elementary
 * dipoles, which allows implementing arbitrarily
 * source configurations.
 */
class DipoleSource: public PhysicalSource
{
public:
  DipoleSource();
  DipoleSource(const std::string name, DipoleType t);

  unsigned n_dipole_elements() const;

  void add_dipole_element(const dvec3d &p, const dvec3d &m, const cvec3d &current);

  void set_dipole_element(const dvec3d &p, const dvec3d &m, const cvec3d &current);

  void set_data(const std::vector<dvec3d> &locations,
                const std::vector<dvec3d> &extents,
                const std::vector<cvec3d> &currents);

  template<class T>
  void position(T &p, const unsigned &dim, unsigned dipole_index = 0) const
  {
    if(dim == 3)
    {
      for(unsigned d = 0; d < dim; ++d)
        p[d] = dipole_locations[dipole_index][d];
    }
    else if(dim == 2)
    {
      p[0] = dipole_locations[dipole_index][1];
      p[1] = dipole_locations[dipole_index][2];
    }
  }

  const std::vector<dvec3d>& get_locations() const;
  const std::vector<dvec3d>& get_extents() const;
  const std::vector<cvec3d>& get_currents() const;

  double position_along_strike(unsigned dipole_index) const;
  const dvec3d& dipole_extent(unsigned dipole_index) const;
  cvec3d current(unsigned dipole_index) const;

  // In case this source consist of a single dipole, these methods returns
  // information about it. Otherwise, an exception is thrown.
  double position_along_strike() const;
  const dvec3d &dipole_extent() const;
  cvec3d current() const;

  DipoleType get_type() const;

  void set_position_along_strike (double p, unsigned dipole_index = 0);
  void set_positions_along_strike (double p);

  void set_dipole_extent(const dvec3d &extent, unsigned dipole_index = 0);

  friend std::istream& operator>> (std::istream &in, DipoleSource &s)
  {
    std::string source_type, dipole_type;
    unsigned n_parts;
    dvec3d j_re, j_im;

    in >> source_type >> s.name >> dipole_type >> n_parts;

    if(trim(source_type).length() == 0)
      return in;

    if(!istrcompare(source_type, string_to_source_type[Dipole]))
    {
      throw std::runtime_error("Source " + s.name + " has type " + source_type
                               + " whereas expected type is " + string_to_source_type[Dipole]);
    }

    if(istrcompare(dipole_type, "E"))
      s.type = ElectricDipole;
    else if(istrcompare(dipole_type, "H"))
      s.type = MagneticDipole;
    else
      throw std::runtime_error("Unknown dipole type. It should be E or H.");

    if(s.type == ElectricDipole || s.type == MagneticDipole)
    {
      s.dipole_locations.resize(n_parts);
      s.dipole_extents.resize(n_parts);
      s.dipole_currents.resize(n_parts);

      for(unsigned n = 0; n < n_parts; ++n)
      {
        for(int d = 0; d < 3; ++d)
          in >> s.dipole_locations[n][d];

        for(int d = 0; d < 3; ++d)
          in >> s.dipole_extents[n][d];

        in >> j_re[0] >> j_im[0] >> j_re[1] >> j_im[1] >> j_re[2] >> j_im[2];
        s.dipole_currents[n][0] = dcomplex(j_re[0], j_im[0]);
        s.dipole_currents[n][1] = dcomplex(j_re[1], j_im[1]);
        s.dipole_currents[n][2] = dcomplex(j_re[2], j_im[2]);
      }
    }
    else
    {
      throw std::runtime_error("Unsupported dipole type.");
    }

    return in;
  }

private:
  // Note: for 2.5D modeling source can in principle be located anywhere along strike direction
  // This position is taken into account upon integration from wavenumber to spatial domain

  // In Cartesian frame the coordinates are: X (north), Y (east), Z (depth)
  // In Spherical frame the coordinates are: phi (rad), theta (rad), r (m)
  std::vector<dvec3d> dipole_locations;  // Location of dipoles comprising the source
  std::vector<dvec3d> dipole_extents;    // Orientation vectors specifying arbitrary spatial extent for each dipole
  std::vector<cvec3d> dipole_currents; // Complex current vector for each elementary dipole of the source
  DipoleType type;
};

#endif // DIPOLE_SOURCE_H
