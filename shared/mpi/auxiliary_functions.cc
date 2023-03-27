#include "auxiliary_functions.h"

unsigned calculate_local_size (MPI_Comm communicator, unsigned total_size)
{
  int rank, size;
  MPI_Comm_rank(communicator, &rank);
  MPI_Comm_size(communicator, &size);

  unsigned local_size = total_size / static_cast<unsigned>(size)
                      + (static_cast<unsigned>(rank) < total_size % static_cast<unsigned>(size));

  return local_size;
}

