#ifndef COMMAND_LINE_H
#define COMMAND_LINE_H

#include <algorithm>
#include <iostream>

char* get_cmd_option(char ** argv, int argc, const std::string & option)
{
  for(int i = 0; i < argc; ++i)
  {
    if(std::string(argv[i]).find(option) != std::string::npos)
      return argv[i+1];
  }
  return 0;
}

bool cmd_option_exists(std::string arg, const std::string& option, bool verbose = false)
{
  bool ret = arg.find(option) != std::string::npos;
  if(verbose)
      std::cout << "Option " << option << " is " << (ret ? " exists" : "missing") << std::endl;

  return ret;
}

void print_help_message()
{
  std::cout << "-help: prints this message\n"
               "-dim: dimensionality of a problem: 2 or 3\n"
               "-parameter_file: path to parameter file\n"
               "-task: task to do: modeling, inversion, jacobian, generate_mesh\n"
               "-row: for jacobian also specify row index to calculate\n"
               "-time_domain: specify whether to work in tim domain (default is frequency domain)\n\n";
}

#endif // COMMAND_LINE_H
