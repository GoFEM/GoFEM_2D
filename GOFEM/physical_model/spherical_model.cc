#include "spherical_model.h"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_reordering.h>
#include <deal.II/grid/manifold_lib.h>

#include "mesh_tools/mesh_tools.h"

template<int dim>
SphericalModelTria<dim>::SphericalModelTria(const std::string modelpath,
                                            const std::string materialpath):
  PhysicalModel<dim> (materialpath, modelpath)
{
  read_in(modelpath);
}

template<int dim>
SphericalModelTria<dim> *SphericalModelTria<dim>::clone() const
{
  return new SphericalModelTria<dim>(*this);
}

template<int dim>
void SphericalModelTria<dim>::create_triangulation(Triangulation<dim> &tria)
{
  if(shell_type == 0)
    create_shell(tria);
  else if(shell_type == 1)
    create_part_shell(tria);
  else if(shell_type == 2)
    create_from_file(tria);
  else
    throw std::runtime_error("Wrong shell type.");

//  {
//  DataOutBase::VtkFlags flags;
//  flags.write_higher_order_cells = true;
//  std::ofstream output("mesh_gofem.vtu");
//  DataOut<dim> data_out;
//  data_out.set_flags(flags);
//  data_out.attach_triangulation(tria);
//  MappingQGeneric<dim> mapping(4);
//  data_out.build_patches(mapping, 4, DataOut<dim>::curved_inner_cells);
//  data_out.write_vtu(output);
//  output.close();
//  }

  init_properties_from_materials(tria);

//  {
//  std::ofstream output("mesh_gofem.vtk");
//  GridOut grid_out;
//  grid_out.write_vtk(tria, output);
//  output.close();
//  }

//  std::ofstream output("mesh.vtk");
//  DataOut<dim> data_out;
//  data_out.attach_triangulation(tria);

//  Vector<float> conductivity (tria.n_active_cells ());
//  unsigned index = 0;
//  for (auto cell=tria.begin_active();
//       cell!=tria.end(); ++cell, ++index)
//  {
//    conductivity(index) = this->conductivity_at(cell);
//  }
//  data_out.add_data_vector(conductivity, "conductivity");
//  data_out.build_patches();
//  data_out.write_vtk(output);
//  output.close();
}

template<int dim>
void SphericalModelTria<dim>::create_shell(Triangulation<dim> &tria)
{
  double inner_radius = sphere_radius - radial_layer_thickness[0];
  double outer_radius = sphere_radius;

  Triangulation<dim> layer_tria;

  GridGenerator::hyper_shell (tria, Point<dim>(),
                              inner_radius, outer_radius);

  for(size_t i = 1; i < radial_layer_thickness.size(); ++i)
  {
    outer_radius = inner_radius;
    inner_radius -= radial_layer_thickness[i];
    GridGenerator::hyper_shell (layer_tria, Point<dim>(),
                                inner_radius, outer_radius);

    GridGenerator::merge_triangulations(tria, layer_tria, tria);

    layer_tria.clear();
  }

  tria.reset_all_manifolds();

  // Use a manifold description for all cells
  static const SphericalManifold<dim> spherical_manifold;
  tria.set_all_manifold_ids(0);
  tria.set_manifold (0, spherical_manifold);
}

template<int dim>
void SphericalModelTria<dim>::create_part_shell(Triangulation<dim> &triangulation)
{
  std::unique_ptr<Triangulation<dim>> basetria_ptr;

  auto tria_pft = dynamic_cast<parallel::fullydistributed::Triangulation<dim>*>(&triangulation);
  if(tria_pft)
    basetria_ptr.reset(
          new Triangulation<dim>(Triangulation<dim>::limit_level_difference_at_vertices)
          );
  else
    basetria_ptr.reset(&triangulation);

  const double d2r = numbers::PI / 180.;

  dvector depths;
  depths.push_back(0);
  for(size_t i = 0; i < radial_layer_thickness.size(); ++i)
    depths.push_back(depths.back() + radial_layer_thickness[i]);

  Point<dim> corner_points[2];
  if(dim == 3)
  {
    corner_points[0] = Point<dim>(phi1 * d2r, theta1 * d2r, sphere_radius - depths.back());
    corner_points[1] = Point<dim>(phi2 * d2r, theta2 * d2r, sphere_radius);
  }
  else
  {
    corner_points[0] = Point<dim>(theta1 * d2r, sphere_radius - depths[1]);
    corner_points[1] = Point<dim>(theta2 * d2r, sphere_radius);
  }

  std::vector<std::vector<double>> cell_sizes(3);
  cell_sizes[0].resize(nphi, d2r*(phi2 - phi1) / double(nphi));
  cell_sizes[1].resize(ntheta, d2r*(theta2 - theta1) / double(ntheta));
  cell_sizes[2] = radial_layer_thickness;
  std::reverse(cell_sizes[2].begin(), cell_sizes[2].end());

  GridGenerator::subdivided_hyper_rectangle (*basetria_ptr, cell_sizes,
                                             corner_points[0], corner_points[1]);

  GridTools::transform ([](const Point<dim> & p){ return point_to_cartesian(p); },
                        *basetria_ptr);

  // Reorder cells to conform with the standard deal.II orientation
  std::vector<Point<dim>>      tria_vertices;
  std::vector<CellData<dim>>   tria_cells;
  SubCellData                  tria_subcell_data;
  std::tie(tria_vertices, tria_cells, tria_subcell_data) =
      GridTools::get_coarse_mesh_description(*basetria_ptr);

  GridReordering<dim>::invert_all_cells_of_negative_grid(tria_vertices, tria_cells);
  GridReordering<dim>::reorder_cells(tria_cells, true);

  basetria_ptr->clear();
  basetria_ptr->create_triangulation(tria_vertices, tria_cells, tria_subcell_data);

  if(tria_pft)
  {
    tria_pft->set_partitioner(
          [](dealii::Triangulation<dim> &tria, const unsigned int n_partitions) {
      GridTools::partition_triangulation(n_partitions,
                                         tria,
                                         SparsityTools::Partitioner::metis);
    },
    TriangulationDescription::Settings::default_setting);

    // actually create triangulation
    tria_pft->copy_triangulation(*basetria_ptr);
  }
  else
    basetria_ptr.release();

  triangulation.reset_all_manifolds();

  static const PolarManifold<dim> polar_manifold;
  triangulation.set_all_manifold_ids(0);
  triangulation.set_manifold (0, polar_manifold);
}

template<int dim>
void SphericalModelTria<dim>::create_from_file(Triangulation<dim> &triangulation)
{
//  static const PolarManifold<dim> polar_manifold;
  static const SphericalManifold<dim> polar_manifold;

  if(tria_file.find(".tria") != std::string::npos)
  {
    MyGridTools::load_triangulation(tria_file, triangulation);
  }
  else
  {
    throw std::runtime_error("Invalid triangulation file extention.");
  }

  triangulation.reset_all_manifolds();
  triangulation.set_all_manifold_ids(0);
  triangulation.set_manifold (0, polar_manifold);
}

template<int dim>
void SphericalModelTria<dim>::init_properties_from_materials(Triangulation<dim> &tria)
{
  auto pft_tria = dynamic_cast<parallel::fullydistributed::Triangulation<dim>*>(&tria);

  for (const auto &cell: tria.active_cell_iterators())
  {
    if(pft_tria != nullptr &&
       cell->is_locally_owned() == false)
    {
      continue;
    }

    // Since we use triangulation to store mesh and reset cell properties here,
    // let's clean anything that could have been associated earlier with that cell
    cell->set_user_pointer(nullptr);

    Material material;

    // Depending on the input shell type, we either create a cell material
    // or simply query it from the material table
    if(shell_type < 2)
    {
      const Point<dim> center = cell->center(true);

      const auto p = create_cell_material(cell, center);
      const bool is_anomaly_cell = p.second;
      material = p.first;

      // If cell is within anomaly, it needs a unique material id because its
      // properties are not equal to the background model
      if(is_anomaly_cell)
      {
        const types::material_id max_background_material_id =
            this->materials_map.rbegin()->first;
        material.id = cell->active_cell_index() + max_background_material_id;
        material.name = "Material" + std::to_string(material.id);
      }
    }
    else if(shell_type == 2)
    {
      auto it = this->materials_map.find (cell->material_id());
      if(it == this->materials_map.end ())
        throw std::runtime_error("SphericalModelTria: no material found!");

      material = it->second;
    }
    else
      throw std::runtime_error("Unsupported model type.");

    this->set_cell_material(cell, material);
  }
}

template<int dim>
bool SphericalModelTria<dim>::is_within_inversion_domain(const typename Triangulation<dim>::active_cell_iterator &cell) const
{
  const size_t shift = dim == 2 ? 1 : 0;

  const Point<dim> cell_center = cell->center (true);
  Point<dim> spherical_point = point_to_spherical(cell_center);
  spherical_point[0] *= 180. / numbers::PI;
  spherical_point[1] *= 180. / numbers::PI;

  bool f = true;
  for(int i = 0; i < dim; ++i)
  {
    if(spherical_point[i] <= this->free_parameters_box[0][i + shift] ||
       spherical_point[i] >= this->free_parameters_box[1][i + shift])
    {
      f = false;
      break;
    }
  }

  return f;
}

template<int dim>
double SphericalModelTria<dim>::air_thickness() const
{
  return sphere_radius - earth_radius;
}

template<int dim>
std::pair<Material, bool> SphericalModelTria<dim>::create_cell_material(
    const typename Triangulation<dim>::active_cell_iterator &cell,
    const Point<dim> &p) const
{
  bool is_anomaly_cell = false;

  const auto center = cell->center(true);
  const Point<dim> p_spherical = point_to_spherical(p);
  const Point<dim> c_spherical = point_to_spherical(center);

  const double z_center = sphere_radius - c_spherical[dim - 1];
  const double z = sphere_radius - p_spherical[dim - 1];

  double conductivity = background_model.conductivity_at(z_center, z);
  const unsigned idx = anomaly_layer_index(p_spherical, c_spherical);
  if(idx != std::numeric_limits<unsigned>::max())
  {
    auto indices = anomaly_conductivity_index(idx, p_spherical);
    if(indices.first >= 0 && indices.second >= 0)
    {
      unsigned linear_index = indices.second * anomalous_layer_size[idx].first + indices.first;
      conductivity = anomalous_layer_conductivities[idx][linear_index];
      is_anomaly_cell = true;
    }
  }

  const double permittivity = background_model.permittivity_at(z_center, z);

  Material material(conductivity, permittivity);
  material.id = background_model.material_id_at(z);

  auto it = this->materials_map.find(material.id);
  if(it != this->materials_map.end())
    material.name = it->second.name;

  return {material, is_anomaly_cell};
}

template<int dim>
Point<dim> SphericalModelTria<dim>::model_extension() const
{
  const double r = 2.*sphere_radius;
  return Point<dim>(r, r, r);
}

template<int dim>
const BackgroundModel<dim> &SphericalModelTria<dim>::get_background_model() const
{
  return background_model;
}

template<int dim>
void SphericalModelTria<dim>::read_in(const std::string gdspath)
{
  std::ifstream ifs(gdspath);

  if(!ifs.is_open())
    throw std::runtime_error("Cannot open file " + gdspath);

  std::string clean, line;

  // read in all data skipping comments and empty lines
  while (!ifs.eof())
  {
    std::getline (ifs, line);
    line.erase(0, line.find_first_not_of(" \n\r\t"));
    if (line.length() == 0 || line[0] == '#')
      continue;

    clean += line + "\n";
  }

  std::stringstream is (clean);

  // ################# Mesh parameters #######################
  {
    std::getline(is, line);
    std::stringstream ss(line);
    ss >> sphere_radius;
  }

  {
    std::getline(is, line);
    std::stringstream ss(line);
    ss >> shell_type;

    if(shell_type > 2)
      throw std::runtime_error("Unknown shell type.");

    if(shell_type == 1)
    {
      ss >> phi1 >> theta1
          >> phi2 >> theta2
          >> nphi >> ntheta;
    }
    else if(shell_type == 2)
    {
      ss >> tria_file;
    }
  }

  // ############### Radial discretization ###################
  {
    std::string radial_block_str;
    std::getline(is, radial_block_str);

    std::stringstream ss(radial_block_str);
    double t;
    while(ss >> t)
      radial_layer_thickness.push_back(t);
  }

  // ################# Background model #######################
  std::vector<double> thickness;
  std::vector<unsigned> material_ids;
  unsigned n_background_layers;
  is >> n_background_layers;
  for(unsigned n = 0; n < n_background_layers; ++n)
  {
    double t;
    unsigned id;
    is >> t >> id;

    material_ids.push_back(id);
    thickness.push_back(t);
  }
  create_background_model(thickness, material_ids);

  // ################# Anomalous part #######################
  is >> n_anomalous_layers;
  anomalous_layer_boundaries.resize(n_anomalous_layers);
  anomalous_layer_size.resize(n_anomalous_layers);
  anomalous_layer_conductivities.resize(n_anomalous_layers);
  phi_intervals.resize(n_anomalous_layers);
  theta_intervals.resize(n_anomalous_layers);

  for(unsigned n = 0; n < n_anomalous_layers; ++n)
  {
    double d, t;
    is >> d >> t;

    anomalous_layer_boundaries[n] = std::make_pair(d, d+t);

    unsigned np, nt;
    double aphi1, aphi2, atheta1, atheta2;
    is >> aphi1 >> atheta1
       >> aphi2 >> atheta2
       >> np >> nt;

    anomalous_layer_size[n] = std::make_pair(np, nt);
    anomalous_layer_conductivities[n].resize(nt*np);

    for(unsigned i = 0; i < nt*np; ++i)
    {
      double sigma;
      is >> sigma;
      anomalous_layer_conductivities[n][i] = sigma;
    }

    double theta = atheta1 * M_PI / 180.,
          dtheta = (atheta2 - atheta1) / nt * M_PI / 180.;
    for(unsigned i = 0; i < nt; ++i)
    {
      theta_intervals[n].push_back(theta);
      theta += dtheta;
    }
    theta_intervals[n].push_back(theta);

    double phi = aphi1 * M_PI / 180.,
          dphi = (aphi2 - aphi1) / np * M_PI / 180.;
    for(unsigned i = 0; i < np; ++i)
    {
      phi_intervals[n].push_back(phi);
      phi += dphi;
    }
    phi_intervals[n].push_back(phi);
  }
}

template<int dim>
void SphericalModelTria<dim>::create_background_model(const dvector &thickness,
                                                      const std::vector<unsigned> &material_ids)
{
  dvector depths;

  depths.push_back(0);
  for(unsigned i = 0; i < thickness.size(); ++i)
    depths.push_back(depths.back() + thickness[i]);

  std::vector<Material> materials;
  for(size_t i = 0; i < material_ids.size(); ++i)
  {
    auto it = this->materials_map.find (material_ids[i]);
    if(it == this->materials_map.end ())
      throw std::runtime_error("SphericalModelTria: no material found!");

    materials.push_back(it->second);
  }

  materials.push_back(materials.back());

  background_model = BackgroundModel<dim>(depths, materials);
  background_model.set_coordinate_system(Spherical);
  background_model.set_radius(sphere_radius);
}

template<int dim>
unsigned SphericalModelTria<dim>::anomaly_layer_index(const Point<dim> &p,
                                                      const Point<dim> &cell_center) const
{
  const double z = sphere_radius - p[dim - 1];
  const double cz = sphere_radius - cell_center[dim - 1];
  for(unsigned i = 0; i < anomalous_layer_boundaries.size(); ++i)
  {
    const auto& range = anomalous_layer_boundaries[i];

    if(cz <= z) // approach boundary from above
    {
      if (z > range.first && z <= range.second)
        return i;
    }
    else // approach boundary from below
    {
      if (z >= range.first && z < range.second)
        return i;
    }
  }

  return std::numeric_limits<unsigned>::max();
}

template<int dim>
std::pair<int, int> SphericalModelTria<dim>::anomaly_conductivity_index(const unsigned zidx,
                                                                        const Point<dim> &p) const
{
  auto it = std::upper_bound(phi_intervals[zidx].begin(), phi_intervals[zidx].end(), p[0]);
  unsigned phi_i = std::distance(phi_intervals[zidx].begin(), it) - 1;

  it = std::upper_bound(theta_intervals[zidx].begin(), theta_intervals[zidx].end(), p[1]);
  unsigned theta_i = std::distance(theta_intervals[zidx].begin(), it) - 1;

  if(phi_i >= anomalous_layer_size[zidx].first ||
     theta_i >= anomalous_layer_size[zidx].second)
  {
    if(fabs(p[0] - 2.*numbers::PI) < 1e-6)
      --phi_i;
    else if(fabs(p[1] - numbers::PI) < 1e-6)
      --theta_i;
    else
      return {-1, -1};
  }

  return {phi_i, theta_i};
}

template class SphericalModelTria<3>;
template class SphericalModelTria<2>;
