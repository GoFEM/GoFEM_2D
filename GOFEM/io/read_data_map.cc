#include "read_data_map.h"

#include <fstream>
#include <sstream>

void read_data_map(const std::string &map_file, std::map<std::string, std::vector<std::string> > &data_map)
{
  if(!map_file.empty())
  {
    std::ifstream ifs(map_file);

    if(!ifs.is_open())
      throw std::runtime_error("Cannot open file " + map_file);

    while(!ifs.eof())
    {
      std::string line;
      std::getline(ifs, line);

      if (((line.length() < 2) && line[0] < '0') || (line[0] == '#'))
        continue;

      std::stringstream is (line);

      std::string sname, rname;
      is >> sname >> rname;

      data_map[sname].push_back(rname);
    }
  }
}
