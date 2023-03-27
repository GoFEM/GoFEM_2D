#ifndef FORWARDMODELING_H
#define FORWARDMODELING_H

// Frequency-domain wrapper
template<int dim>
void do_fd_forward_modeling (const char* parameter_file, const char* method);

// Time-domain wrapper
template<int dim>
void do_td_forward_modeling (const char* parameter_file);

#endif // FORWARDMODELING_H
