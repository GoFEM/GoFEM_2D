#ifndef PARAMETER_READER_H
#define PARAMETER_READER_H

#include <deal.II/base/parameter_handler.h>

using namespace dealii;

class ParameterReader : public Subscriptor
{
public:
  ParameterReader (ParameterHandler &);
  void read_parameters(const std::string parameter_file);

private:
  void declare_parameters ();

  ParameterHandler &prm;
};

#endif // PARAMETER_READER_H
