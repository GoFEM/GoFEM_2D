#ifndef PHYSICAL_MODEL_H
#define PHYSICAL_MODEL_H

#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/distributed/shared_tria.h>

#include "common.h"
#include "material.h"
#include "cell_properties.h"

using namespace dealii;

/*
 * Abstract class that implements interface for physical model,
 * it allows to get information about geometry and physical properties
 */
template<int dim>
class PhysicalModel
{
  template<int dim_>
  friend class MeshingScript;

public:
  PhysicalModel () = default;

  PhysicalModel (const std::string materialpath,
                 const std::string modelpath);

  virtual PhysicalModel<dim>* clone() const = 0;

  virtual void create_triangulation (Triangulation<dim> &tria) = 0;

  virtual double conductivity_at (const typename Triangulation<dim>::active_cell_iterator& cell, const Point<dim>& p = Point<dim> ()) const;
  virtual std::complex<double> complex_conductivity_at (const typename Triangulation<dim>::active_cell_iterator& cell, double frequency, const Point<dim>& p = Point<dim> ()) const;
  virtual double permittivity_at (const typename Triangulation<dim>::active_cell_iterator& cell, const Point<dim>& p = Point<dim> ()) const;
  virtual double permeability_at (const typename Triangulation<dim>::active_cell_iterator& cell, const Point<dim>& p = Point<dim> ()) const;
  virtual void polarization_properties_at (const typename Triangulation<dim>::active_cell_iterator& cell, std::vector<double> &properties, const Point<dim>& p = Point<dim> ()) const;

  virtual void conductivity_list (const typename Triangulation<dim>::active_cell_iterator& cell, const std::vector< Point<dim> > &points, std::vector< double > &values) const;
  virtual void complex_conductivity_list (const typename Triangulation<dim>::active_cell_iterator& cell, const std::vector< Point<dim> > &points, std::vector< std::complex<double> > &values, double frequency) const;
  virtual void permittivity_list (const typename Triangulation<dim>::active_cell_iterator& cell, const std::vector< Point<dim> > &points, std::vector< double > &values) const;
  virtual void permeability_list (const typename Triangulation<dim>::active_cell_iterator& cell, const std::vector< Point<dim> > &points, std::vector< double > &values) const;
  virtual void polarization_properties_list (const typename Triangulation<dim>::active_cell_iterator& cell, const std::vector< Point<dim> > &points, std::vector< std::vector<double> > &values) const;

  unsigned n_materials(const Triangulation<dim> &triangulation) const;

  types::material_id get_material_id (std::string name) const;

  /*
   * Return material structure stored for this cell
   */
  const Material& cell_material(const typename Triangulation<dim>::active_cell_iterator &cell) const;

  /*
   * Set cell's material and update cell's material id, which we
   * take from the passed material
   */
  void set_cell_material (const typename Triangulation<dim>::active_cell_iterator &cell,
                          const Material& material);

  void set_cell_conductivity (const typename Triangulation<dim>::active_cell_iterator &cell,
                              double conductivity);

  /*
   * This method sets cell user's pointer to a structure that contains cell's properties
   * by taking this information from the parent cell which in turn is given this information
   * upon construction of the initial grid. This function does not copy information, hence
   * multiple cells can point to the same structure.
   */
  void set_cell_properties_to_children(const Triangulation<dim> &triangulation) const;

  /*
   * In contrast to the previous function this ensures that every cell has unique structure
   * it points to. This is important for inversion where every cell should posses physical
   * property independently since they all may be updated individually.
   */
  void copy_cell_properties_to_children(Triangulation<dim>& triangulation);

  /*
   * Once parameter grid gets refined, we need to deactivate cells which
   * got children upon refinement.
   */
  void deactivate_parent_cells(const Triangulation<dim>& triangulation);

  /*
   * Checks that cell has a valid free parameter index, meaning it is
   * part of the free parameters set in the inversion.
   */
  bool is_free_index_valid (const typename Triangulation<dim>::active_cell_iterator &cell) const;

  /*
   * This method is supposed to be invoked after cells are marked for coarsening
   * and refinement. Currently, this method performs homogenization for cells
   * that are flagged for coarsening. It does so by calculating weighted average
   * of the conductivity using children and assign the value to the parent cell.
   */
  void prepare_coarsening_and_refinement(const Triangulation<dim>& triangulation);

  void transfer_cell_properties(Triangulation<dim>& tria);
  void clear_cell_properties_recursively(const typename Triangulation<dim>::cell_iterator &cell);

  void set_free_parameter_index (const typename Triangulation<dim>::active_cell_iterator &cell, unsigned index);
  void set_global_parameter_index (const typename Triangulation<dim>::active_cell_iterator &cell, unsigned index);
  unsigned free_parameter_index (const typename Triangulation<dim>::active_cell_iterator &cell) const;
  unsigned global_parameter_index (const typename Triangulation<dim>::active_cell_iterator &cell) const;
  unsigned retrieve_global_parameter_index (unsigned free_index) const;
  unsigned retrieve_free_parameter_index (unsigned global_index) const;

  /*
   * Model size in every direction
   */
  virtual Point<dim> model_extension() const = 0;

  virtual double air_thickness() const;

  /*
   * All cells within this box are treated as free parameters.
   */
  void set_free_parameter_box(const dvec3d p1, const dvec3d p2);

  /*
   * This file is optional. It specifies the free/fixed mask
   * for cells .
   */
  void set_free_parameter_mask_file(const std::string filepath);

  /*
   * Return whether cell's properties must be fixed.
   * There can be several reasons for that:
   * 1. Cell is in the air or sea
   * 2. Cell is outside the provided inversion domain
   */
  bool is_cell_fixed(const typename Triangulation<dim>::active_cell_iterator &cell) const;

  bool is_air_cell(const typename Triangulation<dim>::active_cell_iterator &cell) const;

  void save_triangulation(Triangulation<dim> &tria,
                          const std::string &tria_file) const;

  void save_cell_materials(Triangulation<dim> &tria,
                           const std::string &material_file) const;

protected:
  PhysicalModel(const PhysicalModel&) = default;
  PhysicalModel& operator=(PhysicalModel const& that) = default;

  virtual void init_properties_from_materials (Triangulation<dim>& tria);

  /*
   * Checks if cell's center is located within the active inversion domain
   */
  virtual bool is_within_inversion_domain(const typename Triangulation<dim>::active_cell_iterator &cell) const;

  /*
   * This routine assigns parents' properties to all children provided that a child
   * has no properties yet, otherwise it skips a child cell.
   */
  void set_cell_properties_to_empty_children(const typename Triangulation<dim>::cell_iterator &parent_cell,
                                             void* user_pointer) const;

  /*
   * Finds a parent (by going back in the refinement hierachy) that has properties set and
   * sets properties to all empty children by calling @set_cell_properties_to_empty_children
   */
  void set_cell_properties_from_parent(const typename Triangulation<dim>::active_cell_iterator &cell) const;


protected:
  std::map<types::material_id, Material> materials_map;
  std::deque<CellProperties> cell_properties;

  std::string input_model_file;

  std::array<dvec3d, 2> free_parameters_box;
  std::map<std::pair<short, unsigned>, bool> free_parameter_mask;
};

template<int dim>
using PhysicalModelPtr = std::shared_ptr<PhysicalModel<dim>>;

/*
 * Interface for a 1D horizontally layered models
 */
template<int dim>
class BackgroundModel: public PhysicalModel<dim>
{
public:
  BackgroundModel () = default;

  BackgroundModel (dvector depths,
                   std::vector<Material> materials_data);

  BackgroundModel (const std::string modelpath,
                   const std::string materialpath);

  BackgroundModel (const std::string modelpath,
                   const std::string materialpath,
                   const double size);

  BackgroundModel<dim>* clone() const;

  void create_triangulation (Triangulation<dim> &tria);

  double conductivity_at (const typename Triangulation<dim>::active_cell_iterator & cell,
                          const Point<dim>& p) const;
  double permittivity_at (const typename Triangulation<dim>::active_cell_iterator & cell,
                          const Point<dim>& p) const;

  double conductivity_at (const double z_center, const double z) const;
  double permittivity_at (const double z_center, const double z) const;

  types::material_id material_id_at(const double z) const;

  void conductivity_list (const typename Triangulation<dim>::active_cell_iterator& cell,
                          const std::vector<Point<dim> > &points,
                          std::vector<double> &values) const;

  dvector layer_depths () const;
  dvector conductivities() const;
  dvector permittivities() const;
  dvector permeabilities () const;

  unsigned n_layers() const;

  Point<dim> model_extension() const;

  void set_coordinate_system(CoordinateSystem c);

  void set_radius(double r);
  double get_radius() const;

private:
  std::vector<unsigned> layer_index(const typename Triangulation<dim>::active_cell_iterator& cell,
                                    const std::vector<Point<dim>>& points) const;

  unsigned layer_index(const double &z_center, const double &z) const;

private:
  /*
   * Along horizontal dimensions the 1D model is suppossed to be infinite,
   * yet we make it finite for convenience
   * For spherical model this denotes the radius
   */
  double sphere_radius;

  dvector depths;
  std::vector<Material> materials;

  // If Cartesian, the coordinate is z
  // If Spherical, the coordinate is r
  CoordinateSystem cs;
};

/*
 * The model is supposed to be loaded from the XYZ format file.
 * The file contains spacings along three dimensions and material
 * ids for each cell
 * Read GridGenerator::subdivided_hyper_rectangle documentation
 * for further information about this type of grid specification
 */
template<int dim>
class XYZModel: public PhysicalModel<dim>
{
public:
  XYZModel (const std::string xyzpath, const std::string materialpath);

  virtual XYZModel<dim> *clone() const;

  void create_triangulation (Triangulation<dim> &tria);
  Point<dim> model_extension() const;

  double air_thickness() const;

private:
  void read_in(const std::string xyzpath);

private:
  std::vector<dvector> cell_sizes;        // spacings in corresponding dimensions
  std::vector<unsigned> material_id;      // material ids for all cells
  dvector origin;                         // coordinates of the model origin
};

/*
 * The model is supposed to be loaded from the previously saved
 * triangulation file (e.g., produced after grid generation step).
 * Along with triangulation file, also specify material file.
 */
template<int dim>
class TriangulationModel: public PhysicalModel<dim>
{
public:
  TriangulationModel (const std::string triapath,
                      const std::string materialpath);

  virtual TriangulationModel<dim> *clone() const;

  virtual void create_triangulation (Triangulation<dim> &tria);

  Point<dim> model_extension() const;

  double air_thickness() const;

private:
  void retrieve_geometry_data(const Triangulation<dim> &tria);

  void load_cell_properties(Triangulation<dim> &tria);

private:
  // These variables are set after calling create_triangulation
  dvec3d extension;
  double air_layer_thickness;
};

/*
 * The model described a boundary of a dim+1 triangulation
 * with identificator bid. For instance, for a 3D rectangular
 * domain this will be a face.
 */
template<int dim>
class BoundaryModel: public PhysicalModel<dim>
{
public:
  BoundaryModel (const Triangulation<dim+1> &tria, types::boundary_id id);

  virtual BoundaryModel<dim> *clone() const;

  virtual void create_triangulation (Triangulation<dim> &tria);

  virtual Point<dim> model_extension() const;

private:
  const Triangulation<3> *triangulation;
  types::boundary_id bid;
};

#endif // PHYSICAL_MODEL_H
