#ifndef READ_MODEL_H
#define READ_MODEL_H

#include "common.h"

void read_1dmodel (const std::string file, dvector& depths, std::vector<unsigned int>& material_ids)
{
  std::ifstream ifs (file.c_str ());

  if (ifs.is_open ())
  {
    while (!ifs.eof ())
    {
      std::string line;
      std::getline(ifs, line);

      if ((line.length () < 3) || (line[0] == '#'))
        continue;

      std::stringstream ss (line);

      double depth;
      unsigned material_id;
      ss >> depth >> material_id;

      depths.push_back (depth);
      material_ids.push_back (material_id);
    }
  }
  else
    throw std::ios_base::failure (std::string ("Can not open file " + file).c_str ());
}

#endif // READ_MODEL_H
