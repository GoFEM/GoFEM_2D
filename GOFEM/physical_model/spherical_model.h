#ifndef SPHERICAL_MODEL_H
#define SPHERICAL_MODEL_H

#include "physical_model.h"

/*
 * The model is supposed to be loaded from our own XYZ format file.
 * The file contains spacings along three dimensions and material
 * id for each cell
 * Read GridGenerator::subdivided_hyper_rectangle documentation
 * for further information about this type of grid specification
 */
template<int dim>
class SphericalModelTria: public PhysicalModel<dim>
{
public:
  SphericalModelTria (const std::string modelpath,
                      const std::string materialpath);

  SphericalModelTria<dim>* clone() const;

  virtual void create_triangulation (Triangulation<dim> &tria);

  virtual Point<dim> model_extension() const;

  const BackgroundModel<dim> &get_background_model() const;

  double air_thickness() const;

private:
  void read_in(const std::string gdspath);

  void init_properties_from_materials (Triangulation<dim>& tria);

  bool is_within_inversion_domain(const typename Triangulation<dim>::active_cell_iterator &cell) const;

  void create_background_model(const dvector &thickness,
                               const std::vector<unsigned> &material_ids);

  unsigned anomaly_layer_index(const Point<dim> &p, const Point<dim> &cell_center) const;
  std::pair<int, int> anomaly_conductivity_index(const unsigned idx, const Point<dim> &p) const;

  void create_shell(Triangulation<dim> &tria);
  void create_part_shell(Triangulation<dim> &triangulation);
  void create_from_file(Triangulation<dim> &triangulation);

  /*
   * This method construct a Material for a given cell. Along with it, it also returns
   * the flag which is true if cell lies within anomaly and false if cell if within the
   * background model.
   */
  std::pair<Material, bool> create_cell_material(const typename Triangulation<dim>::active_cell_iterator& cell,
                                                 const Point<dim>& p) const;

private:
  double sphere_radius;
  double phi1, theta1, phi2, theta2;
  unsigned nphi, ntheta;
  unsigned shell_type;
  unsigned n_anomalous_layers;
  std::vector<double> radial_layer_thickness;
  std::vector<std::pair<double, double>> anomalous_layer_boundaries;
  std::vector<std::pair<unsigned, unsigned>> anomalous_layer_size;
  std::vector<std::vector<double>> anomalous_layer_conductivities;
  std::vector<std::vector<double>> phi_intervals, theta_intervals;

  BackgroundModel<dim> background_model;

  std::string tria_file;
};

#endif // PHYSICAL_MODEL_H
