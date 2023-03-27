#include "hello_message.h"
#include "common/command_line.h"
#include "modeling/EM/forward_modeling_caller.h"
#include "common.h"

#include <stdexcept>

#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/logstream.h>

using namespace dealii;

template<int dim>
void run(int argc, char *argv[])
{
  const std::string task = get_cmd_option(argv, argc, "-task");
  const std::string method = get_cmd_option(argv, argc, "-method");

  if(method_type_conversion.find(method) == method_type_conversion.end())
    throw std::runtime_error ("Unknown method: " + method);

  char* parameter_file = get_cmd_option(argv, argc, "-parameter_file");

  if (task == "modeling" || task == "modelling")
  {
    do_fd_forward_modeling<dim> (parameter_file, method.c_str());
  }
  else
    throw std::runtime_error ("Unknown task: " + task);
}

int main (int argc, char *argv[])
{
  int ierr = 0;

  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv, 1);
  deallog.depth_console (0);

  try
  {
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      print_hello_message ();

      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "This program is run with "
                  << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
                  << " MPI processes." << std::endl;

      unsigned n_mandatory_args = 0;
      for(int i = 0; i < argc; ++i)
      {
        if(cmd_option_exists(argv[i], "-help"))
          print_help_message();

        if(cmd_option_exists(argv[i], "-task", true) ||
           cmd_option_exists(argv[i], "-dim", true) ||
           cmd_option_exists(argv[i], "-parameter_file", true) ||
           cmd_option_exists(argv[i], "-method", true))
          ++n_mandatory_args;
      }

      if(n_mandatory_args != 4)
        throw std::runtime_error("One of the obligatory command line arguments is missing. Use -help to get more info.");
    }

    const int dim = std::atoi(get_cmd_option(argv, argc, "-dim"));

    if(dim == 2)
      run<2>(argc, argv);
    else
      throw std::runtime_error("Wrong dimension. Use 2.");
  }
  catch (std::exception &exc)
  {
    std::cout << "\n\n----------------------------------------------------" << std::endl;
    std::cout << "Exception on processing: \n" << exc.what () << "\nAborting!\n"
              << "----------------------------------------------------" << std::endl;

    ierr = 1;
  }
  catch (...)
  {
    std::cout << "\n\n----------------------------------------------------" << std::endl;
    std::cout << "Unknown exception!\n" << "Aborting!\n"
              << "----------------------------------------------------" << std::endl;
    ierr = 1;
  }

  return ierr;
}
