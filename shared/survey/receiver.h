#ifndef RECEIVER_H
#define RECEIVER_H

#include <string>
#include <vector>
#include <iostream>

#include "common.h"

/*
 * Defines receiver station where measurements
 * takes place. Note that multiple data can be
 * measurement at a single receiver.
 */
class Receiver
{
public:
  Receiver ():
    electrode_positions(1, {{0, 0, 0}})
  {}

  Receiver (const dvec3d p, const std::string rec_name):
    electrode_positions(1, p), name(rec_name)
  {}

  void set_name(std::string rec_name);
  std::string get_name() const;

  SourceType get_type() const;

  unsigned n_electrodes() const;

  template<class Point>
  Point position(unsigned electrode_index) const
  {
    if(electrode_index >= electrode_positions.size())
      throw std::runtime_error("Receiver electrode index is out of range.");

    Point p;
    for(size_t i = 0; i < electrode_positions[electrode_index].size(); ++i)
      p[i] = electrode_positions[electrode_index][i];

    return p;
  }

  template<class Point>
  void set_position(Point &p, unsigned electrode_index)
  {
    for(size_t i = 0; i < electrode_positions[electrode_index].size(); ++i)
      electrode_positions[electrode_index][i] = p[i];
  }

  friend std::istream& operator>> (std::istream &in, Receiver &r)
  {
    std::string rec_type;
    unsigned n_electrodes;
    in >> rec_type >> r.name;

    if(rec_type.length() < 3)
      return in;

    auto it = source_type_conversion.find(rec_type);
    if(it == source_type_conversion.end())
      throw std::runtime_error("Receiver " + r.name + " has unsupported type " + rec_type);

    r.type = it->second;

    if(r.type == DCDipole || r.type == Dipole)
    {
      in >> n_electrodes;
      r.electrode_positions.clear();
      r.electrode_positions.resize(n_electrodes);
      for(unsigned i = 0; i < n_electrodes; ++i)
      {
        in >> r.electrode_positions[i][0] >> r.electrode_positions[i][1] >> r.electrode_positions[i][2];

        if(r.type == DCDipole)
        {
          double f;
          in >> f;
        }
      }
    }
    else if(r.type == RadialSheet)
    {
      r.electrode_positions.resize(1);
      in >> r.electrode_positions[0][2];
    }

    return in;
  }

private:
  std::vector<dvec3d> electrode_positions;
  std::string name;
  SourceType type;
};

#endif // RECEIVER_H
