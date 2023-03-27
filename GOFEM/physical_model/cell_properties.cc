#include "cell_properties.h"

#include "common.h"

unsigned CellProperties::invalid_parameter_index = std::numeric_limits<unsigned>::max();

CellProperties::CellProperties():
  material(Isotropic)
{
  global_parameter_index = CellProperties::invalid_parameter_index;
  free_parameter_index = CellProperties::invalid_parameter_index;
}

CellProperties::CellProperties(const Material &material):
  material(material)
{
  global_parameter_index = CellProperties::invalid_parameter_index;
  free_parameter_index = CellProperties::invalid_parameter_index;
}
