#include "physical_source.h"

PhysicalSource::PhysicalSource (const std::string source_name):
  name(source_name)
{}

std::string PhysicalSource::get_name() const
{
  return name;
}

void PhysicalSource::set_name(const std::string source_name)
{
  name = source_name;
}
