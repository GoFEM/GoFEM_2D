#include "receiver.h"

void Receiver::set_name(std::string rec_name)
{
  name = rec_name;
}

std::string Receiver::get_name() const
{
  return name;
}

SourceType Receiver::get_type() const
{
  return type;
}

unsigned Receiver::n_electrodes() const
{
  return electrode_positions.size();
}


