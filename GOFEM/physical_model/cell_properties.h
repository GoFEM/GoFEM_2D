#ifndef CELL_PROPERTIES_H
#define CELL_PROPERTIES_H

#include "material.h"

/*
 * This structure stores cell's properties and auxiliary information
 */
class CellProperties
{
public:
  CellProperties();
  CellProperties(const Material& material);

  static unsigned invalid_parameter_index;

  // Global index of the cell among all cells on the parameter mesh
  unsigned global_parameter_index;
  // Index of the cell taking into account free/fixed inversion cells
  unsigned free_parameter_index;

  // Physical properties of the cell
  Material material;
};

#endif
