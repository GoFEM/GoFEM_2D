#ifndef READ_LIST_H
#define READ_LIST_H

#include <vector>
#include <fstream>
#include <iterator>
#include <iostream>

template<typename T>
std::vector<T> read_list (const std::string file)
{
  std::ifstream ifs(file.c_str ());
  std::vector<T> data;

  if (ifs.is_open ())
  {
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
      T el;
      ss >> el;
      if( ss.eof() ) break;
      data.push_back(el);
    }
  }
  else
    throw std::ios_base::failure (std::string ("Can not open file " + file).c_str ());

  return data;
}

#endif // READ_LIST_H
