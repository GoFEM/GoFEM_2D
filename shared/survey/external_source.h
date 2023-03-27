#ifndef EXTERNAL_SOURCE_H
#define EXTERNAL_SOURCE_H

#include "physical_source.h"

#include "common.h"

/*
 * Defines delta function source at specified point.
 * This class also enables us to define elementary
 * point dipole source of an arbitrary moment.
 * Note that this source can consist of many elementary
 * dipoles, which allows implementing an arbitrarily complex
 * configurations.
 */
class ExternalFileSource: public PhysicalSource
{
public:
  ExternalFileSource():
    PhysicalSource("")
  {}

  const std::string &get_ext_file_name() const
  {
    return ext_file_name;
  }

  friend std::istream& operator>> (std::istream &in, ExternalFileSource &s)
  {
    std::string source_type;

    in >> source_type >> s.name;

    if(!istrcompare(source_type, string_to_source_type[ExternalSourceFile]))
    {
      throw std::runtime_error("Source " + s.name + " has type " + source_type
                               + " whereas expected type is " + string_to_source_type[ExternalSourceFile]);
    }

    in >> s.ext_file_name;

    return in;
  }

private:
  std::string ext_file_name;
};

#endif // EXTERNAL_SOURCE_H
