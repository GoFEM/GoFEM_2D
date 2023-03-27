#include "mpi.h"
#include <iostream>

void check_mpi_error(int ierr)
{
  if (ierr != MPI_SUCCESS)
  {
    char errstr[MPI_MAX_ERROR_STRING];
    int reslen, pid;

    MPI_Comm_rank (MPI_COMM_WORLD, &pid);
    MPI_Error_string (ierr, static_cast<char*> (&errstr[0]), &reslen);

    std::cerr << "Process id " << pid
              << " raised MPI error: " << errstr
              << " . Programm will be finished."
              << std::endl;

    MPI_Finalize ();
  }
}

