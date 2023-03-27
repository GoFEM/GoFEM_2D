#include "read_xyz_model.h"

#include <boost/algorithm/string.hpp>

#include <fstream>
#include <sstream>

void read_tensor_grid_model(const std::string file, std::vector<std::vector<double>>& cell_sizes,
                            std::vector<double>& origin, std::vector<unsigned> &material_id,
                            std::vector<unsigned> &/*parameter_id*/)
{
  std::ifstream ifs (file.c_str ());

  if (!ifs.is_open ())
    throw std::ios_base::failure (std::string ("Can not open file " + file).c_str ());

  std::string clean, line;

  // read in all data skipping comments and empty lines
  while (!ifs.eof())
  {
    std::getline (ifs, line);

    boost::trim(line);

    if (line.length() == 0 || line[0] == '#')
      continue;

    clean += line + "\n";
  }

  std::stringstream is (clean);

  std::getline (is, line);
  std::stringstream ss(line);

  unsigned dim;
  std::vector<unsigned> np;
  while (ss >> dim)
    np.push_back(dim);

  dim = np.size();

  cell_sizes.resize(dim);
  for(unsigned i = 0; i < dim; ++i)
    cell_sizes[i].reserve (np[i]);

  origin.resize (dim);
  for(unsigned i = 0; i < dim; ++i)
    is >> origin[i];
  is.ignore ();

  double d;
  for(unsigned i = 0; i < dim; ++i)
  {
    std::getline (is, line);
    std::stringstream ss (line);
    while (ss >> d) cell_sizes[i].push_back (d);
  }

  unsigned n_cells = 1;
  for(unsigned i = 0; i < dim; ++i)
      n_cells *= cell_sizes[i].size();

  is >> std::ws;

  // material ids
  unsigned i;
  material_id.clear();
  material_id.reserve (n_cells);
  while (is >> i) material_id.push_back (i);
}
