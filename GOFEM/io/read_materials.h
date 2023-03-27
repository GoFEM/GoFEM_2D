#ifndef READ_MATERIALS_H
#define READ_MATERIALS_H

#include <vector>
#include <fstream>
#include <iterator>

#include "../physical_model/material.h"

std::vector<Material> read_materials (const std::string file)
{
  std::ifstream ifs(file.c_str ());
  std::vector<Material> data;

  if (ifs.is_open ())
  {
    std::string line;
    int pos;

    // skip comments in the beginning
    do
    {
      pos = ifs.tellg ();
      std::getline (ifs, line);
    } while (line[0] == '#');

    ifs.seekg (pos, std::ios::beg);

    std::string material_type_str;
    ifs >> material_type_str;

    auto it = material_type_name_to_type_table.find(material_type_str);
    if(it == material_type_name_to_type_table.end())
      throw std::runtime_error("Unknown or not implemented material type.");

    MaterialType type = it->second;

    Material material(type);
    while(ifs >> material)
      data.push_back(material);
  }
  else
    throw std::ios_base::failure (std::string ("Can not open file " + file).c_str ());

  return data;
}

#endif // READ_MATERIALS_H
