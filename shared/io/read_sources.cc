#include "read_sources.h"

#include "survey/dipole_source.h"

std::vector<PhysicalSourcePtr> read_sources(const std::string sources_file)
{
  std::vector<PhysicalSourcePtr> sources;

  std::ifstream ifs(sources_file);
  std::string clean, line;
  // read in all data skipping comments and empty lines
  while (!ifs.eof())
  {
    std::getline (ifs, line);
    // Trim string
    line.erase(0, line.find_first_not_of(" \t\r\n"));

    // Skip empty lines and comments
    if (line.length() < 1 || line[0] == '!' || line[0] == '%' || line[0] == '#')
      continue;

    clean += line + "\n";
  }

  std::stringstream ss(clean);

  while(ss.good())
  {
    size_t pos = ss.tellg();

    std::string source_type;
    ss >> source_type;

    if( ss.eof() ) break;

    ss.seekg(pos);

    if(istrcompare(source_type, string_to_source_type[Dipole]))
    {
      DipoleSource dipole_src;
      ss >> dipole_src;

      sources.push_back(PhysicalSourcePtr(new DipoleSource(dipole_src)));
    }
    else
    {
      throw std::runtime_error("Unknown source type " + source_type);
    }
  }

  return sources;
}

