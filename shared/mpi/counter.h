#ifndef COUNTER_H
#define COUNTER_H

#include "mpi.h"

static const int countertag = 333;

/*
 * Modeling for counting parallel tasks (e.g. frequencies),
 * keeping track of how many have been done.
 */
class Counter
{
public:
  Counter () {}
  Counter (MPI_Comm comm, int maxval);
  ~Counter () {}

  /*
   * A simple MPI-1 based routine for a counter
   * managed by one process must be called frequently
   */
  int service (int n_groups);

  /*
   * Sends stop signal to all processes after
   * all taks have been done
   */
  void finish (int ngroups);

  /*
   * Increments counter and returns updated value
   */
  int increment ();

  /*
   * Requests value
   */
  int request ();

private:
  int count_ndone;    // number of processes or process groups (except proc0)
                      // which finished all supplied modeling tasks
  int count_max;      // maximum counter value
  int count_value;    // counter value

  MPI_Comm communicator;
};

#endif // COUNTER_H
