#include "dipole_source.h"

DipoleSource::DipoleSource():
  PhysicalSource("")
{
}

DipoleSource::DipoleSource(const std::string name, DipoleType t):
  PhysicalSource(name), type(t)
{}

unsigned DipoleSource::n_dipole_elements() const
{
  return dipole_locations.size();
}

void DipoleSource::add_dipole_element(const dvec3d &p, const dvec3d &m, const cvec3d &current)
{
  dipole_extents.push_back(m);
  dipole_locations.push_back(p);
  dipole_currents.push_back(current);
}

void DipoleSource::set_dipole_element(const dvec3d &p, const dvec3d &m, const cvec3d &current)
{
  dipole_extents = {m};
  dipole_locations = {p};
  dipole_currents = {current};
}

void DipoleSource::set_data(const std::vector<dvec3d> &locations,
                            const std::vector<dvec3d> &extents,
                            const std::vector<cvec3d> &currents)
{
  dipole_extents = extents;
  dipole_locations = locations;
  dipole_currents = currents;
}

const std::vector<dvec3d> &DipoleSource::get_locations() const
{
  return dipole_locations;
}

const std::vector<dvec3d> &DipoleSource::get_extents() const
{
  return dipole_extents;
}

const std::vector<cvec3d> &DipoleSource::get_currents() const
{
  return dipole_currents;
}

double DipoleSource::position_along_strike(unsigned dipole_index) const
{
  if(dipole_locations.size() <= dipole_index)
    throw std::runtime_error("dipole_locations: index out of range.");

  return dipole_locations[dipole_index][0];
}

double DipoleSource::position_along_strike() const
{
  if(dipole_locations.size() > 1)
    throw std::runtime_error("This is a multi-dipole source which is treated as a point dipole.");

  return dipole_locations[0][0];
}

const dvec3d& DipoleSource::dipole_extent(unsigned dipole_index) const
{
  if(dipole_extents.size() <= dipole_index)
    throw std::runtime_error("dipole_moment: index out of range.");

  return dipole_extents[dipole_index];
}

cvec3d DipoleSource::current(unsigned dipole_index) const
{
  if(dipole_currents.size() <= dipole_index)
    throw std::runtime_error("current: index out of range.");

  return dipole_currents[dipole_index];
}

const dvec3d& DipoleSource::dipole_extent() const
{
  if(dipole_extents.size() > 1)
    throw std::runtime_error("This is a multi-dipole source which is treated as a point dipole.");

  return dipole_extents[0];
}

cvec3d DipoleSource::current() const
{
  if(dipole_currents.size() > 1)
    throw std::runtime_error("This is a multi-dipole source which is treated as a point dipole.");

  // Currently only real currents are supported
  return dipole_currents[0];
}

DipoleType DipoleSource::get_type() const
{
  return type;
}

void DipoleSource::set_position_along_strike(double p, unsigned dipole_index)
{
  if(dipole_locations.size() <= dipole_index)
    throw std::runtime_error("set_position_along_strike: index out of range.");

  dipole_locations[dipole_index][0] = p;
}

void DipoleSource::set_positions_along_strike(double p)
{
  for(auto &location: dipole_locations)
    location[0] = p;
}

void DipoleSource::set_dipole_extent(const dvec3d &extent, unsigned dipole_index)
{
  if(dipole_extents.size() <= dipole_index)
    throw std::runtime_error("set_dipole_moment: index out of range.");

  dipole_extents[dipole_index] = extent;
}
