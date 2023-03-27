#include <unistd.h>

#include "counter.h"
#include "mpi_error.h"

#include <iostream>
#include <thread>
#include <chrono>

Counter::Counter (MPI_Comm comm, int maxval):
  count_ndone (0), count_max (maxval),
  count_value (0), communicator (comm)
{}

int Counter::service (int n_groups)
{
  int requester;    // id of process asking for a value
  int flag;         // indicates if any requests have been made
  MPI_Status stat;  // MPI status
  int dummybuf;     // dummy receive buffer
  int ierr;
  int processed_requests = 0;

  do
  {
    ierr = MPI_Iprobe (MPI_ANY_SOURCE,countertag,communicator,&flag,&stat);
    check_mpi_error (ierr);

    if (!flag)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    else
    {
      ++processed_requests;

      requester = stat.MPI_SOURCE;
      ierr = MPI_Recv (&dummybuf,0,MPI_INTEGER,requester,countertag,communicator,&stat);
      check_mpi_error (ierr);

      ++count_value;

      ierr = MPI_Rsend (&count_value,1,MPI_INTEGER,requester,countertag,communicator);
      check_mpi_error(ierr);

      if (count_value > count_max)
        ++count_ndone;
    }
  }
  while(processed_requests < n_groups);

  return count_value;
}

void Counter::finish (int ngroups)
{
  int requester;    // id of process asking for a value
  int flag;         // indicates if any requests have been made
  MPI_Status stat;  // MPI status
  int dummybuf;     // dummy receive buffer
  int ierr;

  do
  {
    ierr = MPI_Iprobe (MPI_ANY_SOURCE,countertag,communicator,&flag,&stat);
    check_mpi_error (ierr);

    if (flag)
    {
      requester = stat.MPI_SOURCE;
      ierr = MPI_Recv(&dummybuf,0,MPI_INTEGER,requester,countertag,communicator,&stat);
      check_mpi_error(ierr);
      ++count_value;
      ierr = MPI_Send(&count_value,1,MPI_INTEGER,requester,countertag,communicator);
      check_mpi_error(ierr);
      ++count_ndone;
    }
    //	else
    //	  sleep(1);
  } while (count_ndone < ngroups - 1);
}

int Counter::increment ()
{
  return ++count_value;
}

int Counter::request ()
{
  int val, ierr;

  ierr = MPI_Send(NULL, 0, MPI_INTEGER, 0, countertag, communicator);
  check_mpi_error(ierr);

  ierr = MPI_Recv(&val, 1, MPI_INTEGER, 0, countertag, communicator, MPI_STATUS_IGNORE);
  check_mpi_error(ierr);

  return val;
}
