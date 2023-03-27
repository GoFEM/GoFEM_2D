#ifndef MAKESUB_H
#define MAKESUB_H

#include "mpi.h"
#include "mpi_error.h"

/*
 * Makes domain substructure for parallel frequency modeling.
 * Optionally, you can exclude rank=0 process from any group
 */
void split_communicator(const MPI_Comm comm, const int n_groups,
                        MPI_Comm &group_comm, int& this_processor_group,
                        bool exclude_root);

#endif // MAKESUB_H
