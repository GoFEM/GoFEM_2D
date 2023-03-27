#ifndef MT2DFEM_H
#define MT2DFEM_H

#include "em2dfem.h"

#include <deal.II/numerics/fe_field_function.h>

/*
 * Class that provides interface to implement
 * 2D MT solver
 */
class MT2DFEM: public EM2DFEM
{
public:
  MT2DFEM(MPI_Comm comm,
          const unsigned int order,
          const unsigned int mapping_order,
          const PhysicalModelPtr<2> &model);
  virtual ~MT2DFEM();

  virtual unsigned n_data_at_point () const;
  void tangential_field_at(const Point<2> &p, cvector &F, FieldFormulation field_type) const;
  void tangential_fields_at(const std::vector<Point<2>> &points, std::vector<cvector> &fields,
                            FieldFormulation field_type) const;

protected:
  void create_grid (const unsigned int cycle);
  void copy_local_to_global_system (const Assembly::CopyData::MaxwellSystem& data);

  unsigned get_number_of_constraint_matrices() const;
  void set_boundary_values ();

  // RHS vector assembly
  virtual void assemble_problem_rhs ();

  virtual std::string data_header () const;

  cvector calculate_data_at_receiver (const std::vector<cvector>& E,
                                      const std::vector<cvector>& H) const;

  // first left- top-most point, second right- bottom-most point
  std::vector<Point<2>> model_corner_points;
};

#endif // MT2DFEM_H

