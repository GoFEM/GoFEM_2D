#ifndef HELLOMESSAGE_H
#define HELLOMESSAGE_H

#include <iostream>
#include "mpi.h"

void print_hello_message ()
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank == 0)
  {
    std::cout << "################################### GOFEM ###############################\n"
                 "#                                                                       #\n"
                 "#     Adaptive finite-element geo-electromagnetic forward modeling      #\n"
                 "#                  (C) Alexander Grayver 2014-2023                      #\n"
                 "#                                                                       #\n"
                 "#                      email: agrayver@gmail.com                        #\n"
                 "#                                                                       #\n"
                 "#    This software can not be used or distributed without permission    #\n"
                 "#    By running it, you agree with these conditions                     #\n"
                 "#########################################################################\n\n";

    std::cout << "Run with the -help argument to get more information on usage\n\n";
  }
}

#endif // HELLOMESSAGE_H
