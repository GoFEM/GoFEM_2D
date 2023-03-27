#ifndef AUXILIARY_FUNCTIONS_H
#define AUXILIARY_FUNCTIONS_H

#include "mpi.h"

/*
 * This function splits total_size among all processors in communicator
 * optimally. The sum of the local portions equals total_size
 */
unsigned calculate_local_size (MPI_Comm communicator, unsigned total_size);

#endif // AUXILIARY_FUNCTIONS_H
