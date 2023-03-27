#include "makesub.h"

#include "mpi_error.h"

void split_communicator(const MPI_Comm comm, const int n_groups,
                        MPI_Comm& group_comm, int& this_processor_group,
                        bool exclude_root)
{
    MPI_Comm commgroup = 0;
    int pid, size, ierr;

    ierr = MPI_Comm_rank(comm, &pid);
    check_mpi_error(ierr);
    ierr = MPI_Comm_size(comm, &size);
    check_mpi_error(ierr);

    if(exclude_root)
    {
      // parallel groups: commgroup is a group of processes
      this_processor_group = (pid-1) / ((size-1) / n_groups); // integer division;

      if(pid == 0)
        ierr = MPI_Comm_split(comm, MPI_UNDEFINED, 0, &commgroup);
      else
        ierr = MPI_Comm_split(comm, this_processor_group, pid-1, &commgroup);

      check_mpi_error(ierr);

      group_comm = commgroup;
    }
    else
    {
      // parallel groups: commgroup is a group of processes
      this_processor_group = pid / (size / n_groups); // integer division;
      ierr = MPI_Comm_split(comm, this_processor_group, pid, &commgroup);
      check_mpi_error(ierr);

      group_comm = commgroup;
    }
}
