#include "physical_model.h"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/manifold_lib.h>

#include "io/read_model.h"
#include "io/read_list.h"
#include "io/read_xyz_model.h"
#include "io/read_materials.h"
#include "mesh_tools/mesh_tools.h"

#include <algorithm>
#include <limits>

template<int dim>
PhysicalModel<dim>::PhysicalModel (const std::string materialpath, const std::string modelpath):
  input_model_file(modelpath)
{
  // Read material definition file
  MaterialList materials;
  if(materialpath.length() > 1)
    materials = read_materials (materialpath);

  // For convenience and speed create an associated array
  for(size_t i = 0; i < materials.size (); ++i)
    materials_map.insert(std::make_pair(materials[i].id, materials[i]));

  free_parameters_box[0].fill(0);
  free_parameters_box[1].fill(0);
}

template<int dim>
unsigned PhysicalModel<dim>::n_materials(const Triangulation<dim> &triangulation) const
{
  std::set<types::material_id> ids;
  for (const auto &cell: triangulation.active_cell_iterators())
    ids.insert(cell->material_id());

  return ids.size();
}

template<int dim>
void PhysicalModel<dim>::set_cell_properties_to_children (const Triangulation<dim> &triangulation) const
{
  for (const auto &cell: triangulation.active_cell_iterators())
    if (cell->user_pointer () == nullptr && cell->level () > 0) // Is this an active refined cell without properties?
      set_cell_properties_from_parent(cell);
}

template<int dim>
void PhysicalModel<dim>::set_cell_properties_from_parent(const typename Triangulation<dim>::active_cell_iterator &cell) const
{
  auto cell_parent = cell->parent();
  while(cell_parent->user_pointer () == nullptr && cell_parent->level () > 0)
    cell_parent = cell_parent->parent();

  set_cell_properties_to_empty_children(cell_parent, cell_parent->user_pointer ());
}

template<int dim>
void PhysicalModel<dim>::set_cell_properties_to_empty_children(const typename Triangulation<dim>::cell_iterator &parent_cell,
                                                               void* user_pointer) const
{
  for(unsigned child_index = 0; child_index < parent_cell->n_children(); ++child_index)
  {
    auto child = parent_cell->child(child_index);
    if (child->user_pointer() == nullptr)
    {
      child->set_user_pointer(user_pointer);
      set_cell_properties_to_empty_children(child, user_pointer);
    }
  }
}

template<int dim>
void PhysicalModel<dim>::copy_cell_properties_to_children(Triangulation<dim> &triangulation)
{
  // Collect cells' properties by making a unique copy for every cell
  std::deque<CellProperties> new_cell_properties;

  for (const auto &cell: triangulation.active_cell_iterators())
  {
    if (cell->user_pointer () != nullptr)
    {
      CellProperties* properties = static_cast<CellProperties*> (cell->user_pointer ());
      new_cell_properties.push_back (*properties);
    }
    else
    {
      new_cell_properties.push_back(CellProperties());
    }
  }

  // Save new properties and assign them anew
  cell_properties = new_cell_properties;

  unsigned cell_index = 0;
  for (const auto &cell: triangulation.active_cell_iterators())
  {
    cell->set_user_pointer (&cell_properties[cell_index]);
    ++cell_index;
  }
}

template<int dim>
void PhysicalModel<dim>::deactivate_parent_cells(const Triangulation<dim> &triangulation)
{
  for (const auto &cell: triangulation.cell_iterators())
    if(!cell->is_active() && cell->user_pointer() != nullptr)
    {
      // This cell should not be a parameter in the inversion
      CellProperties* properties = static_cast<CellProperties*> (cell->user_pointer ());
      properties->material.reset();
      properties->free_parameter_index = CellProperties::invalid_parameter_index;

      cell->set_user_pointer(nullptr);
    }
}

template<int dim>
void PhysicalModel<dim>::prepare_coarsening_and_refinement(const Triangulation<dim> &triangulation)
{
  for (const auto &cell: triangulation.active_cell_iterators ())
    if(cell->level() > 0 && cell->coarsen_flag_set())
    {
      auto parent = cell->parent();
      if(parent->user_pointer() != nullptr)
      {
        double average_conductivity = 0.;

        for(unsigned i = 0; i < GeometryInfo<dim>::max_children_per_cell; ++i)
        {
          CellProperties* child_properties = static_cast<CellProperties*> (parent->child(i)->user_pointer ());
          average_conductivity += 1. / GeometryInfo<dim>::max_children_per_cell * child_properties->material.conductivity;
        }

        CellProperties* parent_properties = static_cast<CellProperties*> (parent->user_pointer ());
        parent_properties->material.conductivity = average_conductivity;
      }
      else
        throw std::runtime_error("Parent of a cell that is going to be coarsened has no properties!");
    }
}

template<int dim>
void PhysicalModel<dim>::set_global_parameter_index (const typename Triangulation<dim>::active_cell_iterator &cell, unsigned index)
{
  Assert (cell->user_pointer () != nullptr,
          ExcMessage ("No properties were assigned to this cell!"));

  CellProperties* properties = static_cast<CellProperties*> (cell->user_pointer ());
  properties->global_parameter_index = index;
}

template<int dim>
void PhysicalModel<dim>::set_free_parameter_index (const typename Triangulation<dim>::active_cell_iterator &cell, unsigned index)
{
  Assert (cell->user_pointer () != nullptr,
          ExcMessage ("No properties were assigned to this cell!"));

  CellProperties* properties = static_cast<CellProperties*> (cell->user_pointer ());
  properties->free_parameter_index = index;
}

template<int dim>
void PhysicalModel<dim>::set_cell_conductivity(const typename Triangulation<dim>::active_cell_iterator &cell,
                                               double conductivity)
{
  CellProperties* properties = static_cast<CellProperties*> (cell->user_pointer ());
  if(cell->user_pointer () == nullptr)
    ExcMessage ("No properties were assigned to this cell!");

  properties->material.conductivity = conductivity;
}

template<int dim>
void PhysicalModel<dim>::set_cell_material(const typename Triangulation<dim>::active_cell_iterator &cell,
                                           const Material &material)
{
  if(!material.is_valid())
  {
    std::stringstream ss;
    ss << "Cell " << cell->id() << " (" << cell->center(true)
       << ") assigned an invalid material:\n"
       << material << "\n"
       << "Locally owned = "
       << cell->is_locally_owned();
    throw std::runtime_error(ss.str());
  }

  // if no properties were assigned to this cell, do this
  if(cell->user_pointer () == nullptr)
  {
    cell_properties.emplace_back (CellProperties(material));
    cell->set_user_pointer (&cell_properties[cell_properties.size() - 1]);
  }
  else // just update properties
  {
    CellProperties* current_properties = static_cast<CellProperties*> (cell->user_pointer ());
    current_properties->material = material;
  }

  // Need to update material id assigned to the cell
  cell->set_material_id(material.id);
}

template<int dim>
unsigned PhysicalModel<dim>::free_parameter_index (const typename Triangulation<dim>::active_cell_iterator &cell) const
{
  Assert (cell->user_pointer () != nullptr,
          ExcMessage ("No properties were assigned to this cell!"));

  CellProperties* properties = static_cast<CellProperties*> (cell->user_pointer ());

  return properties->free_parameter_index;
}

template<int dim>
unsigned PhysicalModel<dim>::global_parameter_index (const typename Triangulation<dim>::active_cell_iterator &cell) const
{
  Assert (cell->user_pointer () != nullptr,
          ExcMessage ("No properties were assigned to this cell!"));

  CellProperties* properties = static_cast<CellProperties*> (cell->user_pointer ());

  return properties->global_parameter_index;
}

template<int dim>
unsigned PhysicalModel<dim>::retrieve_global_parameter_index(unsigned free_index) const
{
  for(size_t i = 0; i < cell_properties.size(); ++i)
    if(cell_properties[i].free_parameter_index == free_index)
      return cell_properties[i].global_parameter_index;

  return CellProperties::invalid_parameter_index;
}

template<int dim>
unsigned PhysicalModel<dim>::retrieve_free_parameter_index(unsigned global_index) const
{
  for(size_t i = 0; i < cell_properties.size(); ++i)
    if(cell_properties[i].global_parameter_index == global_index)
      return cell_properties[i].free_parameter_index;

  return CellProperties::invalid_parameter_index;
}

template<int dim>
double PhysicalModel<dim>::air_thickness() const
{
  throw std::runtime_error("air_thickness not implemented.");
}

template<int dim>
const Material &PhysicalModel<dim>::cell_material(const typename Triangulation<dim>::active_cell_iterator &cell) const
{
  Assert (cell->user_pointer () != nullptr,
          ExcMessage ("No properties were assigned to this cell!"));

  CellProperties* properties = static_cast<CellProperties*> (cell->user_pointer ());

  return properties->material;
}

template<int dim>
void PhysicalModel<dim>::init_properties_from_materials (Triangulation<dim> &tria)
{
  tria.clear_user_data ();

  cell_properties.resize (tria.n_active_cells ());
  size_t idx = 0;

  auto pft_tria = dynamic_cast<parallel::fullydistributed::Triangulation<dim>*>(&tria);

  for (const auto &cell: tria.active_cell_iterators())
  {
    if(pft_tria != nullptr &&
       cell->is_locally_owned() == false)
    {
      continue;
    }

    auto it = materials_map.find (cell->material_id ());
    if(it == materials_map.end ())
      throw std::runtime_error("Material with id " +
                               std::to_string(cell->material_id ()) +
                               " not found! Cell is locally owned = " +
                               std::to_string(cell->is_locally_owned()));

    CellProperties cell_properties_struct(it->second);

    cell_properties[idx] = cell_properties_struct;
    cell->set_user_pointer (&cell_properties[idx]);

    ++idx;
  }
}

template<int dim>
bool PhysicalModel<dim>::is_free_index_valid (const typename Triangulation<dim>::active_cell_iterator &cell) const
{
  Assert (cell->user_pointer () != nullptr,
          ExcMessage ("No properties were assigned to this cell!"));

  CellProperties* properties = static_cast<CellProperties*> (cell->user_pointer ());
  return properties->free_parameter_index != CellProperties::invalid_parameter_index;
}

template<int dim>
void PhysicalModel<dim>::set_free_parameter_box(const dvec3d p1, const dvec3d p2)
{
  free_parameters_box[0] = p1;
  free_parameters_box[1] = p2;
}

template<int dim>
void PhysicalModel<dim>::set_free_parameter_mask_file(const std::string filepath)
{
  if(filepath.empty())
    return;

  std::ifstream ifs(filepath);

  if(!ifs.is_open())
    throw std::runtime_error("Cannot open file " + filepath);

  while(!ifs.eof())
  {
    std::string line;
    std::getline(ifs, line);

    if ((line.length() < 2) || (line[0] == '#'))
      continue;

    std::stringstream is (line);

    std::pair<short, unsigned> level_index;
    bool flag;
    is >> level_index.first >> level_index.second >> flag;

    free_parameter_mask.insert({level_index, flag});
  }

  ifs.close();
}

template<int dim>
bool PhysicalModel<dim>::is_cell_fixed(const typename Triangulation<dim>::active_cell_iterator &cell) const
{
  if(free_parameter_mask.size())
  {
    auto it = free_parameter_mask.find({cell->level(), cell->index()});
    if(it == free_parameter_mask.end())
      throw std::runtime_error("Cannot find cell (level = " +
                               std::to_string(cell->level()) + ", index = " +
                               std::to_string(cell->index()) +
                               ") in the parameter mask.");

    const bool is_within_box = is_within_inversion_domain(cell);
    const bool parameter_flag = it->second;

    if(is_within_box && parameter_flag)
      return false;
    else
      return true;
  }
  else
  {
    return !is_within_inversion_domain(cell) ||
           is_air_cell(cell);
  }
}

template<int dim>
types::material_id PhysicalModel<dim>::get_material_id(std::string name) const
{
  std::transform(name.begin(), name.end(), name.begin(), ::tolower);
  // Retrieve material id by name
  for(auto p: materials_map)
  {
    std::string material_name = p.second.name;
    std::transform(material_name.begin(), material_name.end(),
                   material_name.begin(), ::tolower);
    if(material_name == name)
      return p.first;
  }

  return numbers::invalid_material_id;
}

template<int dim>
bool PhysicalModel<dim>::is_air_cell(const typename Triangulation<dim>::active_cell_iterator &cell) const
{
  Assert (cell->user_pointer () != nullptr,
          ExcMessage ("No properties were assigned to this cell!"));

  CellProperties* properties = static_cast<CellProperties*> (cell->user_pointer ());

  std::string material_name = properties->material.name;
  std::transform(material_name.begin(), material_name.end(),
                 material_name.begin(), ::tolower);

  return material_name == "air";
}

template<int dim>
bool PhysicalModel<dim>::is_within_inversion_domain(const typename Triangulation<dim>::active_cell_iterator &cell) const
{
  size_t shift = dim == 2 ? 1 : 0;

  const Point<dim> cell_center = cell->center (true);
  bool f = true;
  for(int i = 0; i < dim; ++i)
  {
    if(cell_center[i] <= free_parameters_box[0][i + shift] ||
       cell_center[i] >= free_parameters_box[1][i + shift])
      f = false;
  }

  return f;
}

template<int dim>
void PhysicalModel<dim>::save_triangulation(Triangulation<dim> &tria,
                                            const std::string &tria_file) const
{
  std::ofstream ofs(tria_file);
  boost::archive::binary_oarchive oa(ofs);

  if(auto shared_tria = dynamic_cast<parallel::shared::Triangulation<dim>*>(&tria))
  {
    shared_tria->save(oa, 1);
  }
  else if(auto tria_pft = dynamic_cast<parallel::fullydistributed::Triangulation<dim>*>(&tria))
  {
    throw std::runtime_error("Cannot save fully distributed mesh.");
  }
  else
    tria.save(oa, 1);
}

template<int dim>
void PhysicalModel<dim>::save_cell_materials(Triangulation<dim> &tria,
                                             const std::string &material_file) const
{
  // Retieve unique materials from active cells
  std::map<types::material_id, Material> active_cell_materials;

  for (const auto &cell: tria.active_cell_iterators())
  {
    auto material = cell_material(cell);
    auto it = active_cell_materials.find(material.id);
    if(it == active_cell_materials.end())
    {
      active_cell_materials.insert({material.id, material});
    }
    else
    {
      if(it->second != material)
        throw std::runtime_error("There are materials with the same id, but different properties.");
    }
  }

  // Check that all materials are of the same type
  std::set<MaterialType> types;
  for(auto p: active_cell_materials)
    types.insert(p.second.type);

  if(types.size() > 1)
    throw std::runtime_error("Mixed material types are not supported yet.");

  // Write them in the standard format
  std::ofstream ofs(material_file);
  ofs << material_type_to_name_table[*types.begin()] << std::endl;
  for(auto p: active_cell_materials)
  {
    const Material& material = p.second;
    ofs << material << std::endl;
  }
}

template<int dim>
double PhysicalModel<dim>::conductivity_at (const typename Triangulation<dim>::active_cell_iterator &cell,
                                            const Point<dim>& /*p*/) const
{
  Assert (cell->user_pointer () != nullptr,
          ExcMessage ("No properties were assigned to cell " + cell->id().to_string()));
  CellProperties* properties = static_cast<CellProperties*> (cell->user_pointer ());
  const double value = properties->material.conductivity;

  if(!std::isfinite(value))
  {
    std::stringstream ss;
    ss << "Cell " << cell->id() << " (" << cell->center(true)
       << ") has invalid conductivity " << value
       << " and it was requested. Locally owned = "
       << cell->is_locally_owned();
    throw std::runtime_error(ss.str());
  }

  return value;
}

template<int dim>
std::complex<double> PhysicalModel<dim>::complex_conductivity_at(const typename Triangulation<dim>::active_cell_iterator &cell,
                                                                 double frequency, const Point<dim> &/*p*/) const
{
  if(cell->user_pointer () == nullptr)
  {
   std::cout << cell->center(true) << "\t" << cell->level() << std::endl;
   throw std::runtime_error("No properties were assigned to this cell!");
  }

  CellProperties* properties = static_cast<CellProperties*> (cell->user_pointer ());

  const Material& material = properties->material;

  if(material.is_valid())
  {
    if(fabs(material.chargeability) == 0.)
      return std::complex<double>(material.conductivity, 0.);
    else
    {
      // Use Cole-Cole formulae to calculate frequency dependent conductivity
      std::complex<double> rho = 1./ material.conductivity *
          (1. - material.chargeability *
           (1. - 1. / (1. + pow(II*2.*numbers::PI*frequency*material.relaxation_time,
                                material.exponent))));

      return 1. / rho;
    }
  }
  else
  {
    std::stringstream ss;
    ss << "Cell " << cell->id() << " (" << cell->center(true)
       << ") has invalid conductivity " << material.conductivity
       << " and it was requested. Locally owned = " << cell->is_locally_owned();
    throw std::runtime_error(ss.str());
  }
}

template<int dim>
double PhysicalModel<dim>::permittivity_at (const typename Triangulation<dim>::active_cell_iterator &cell,
                                            const Point<dim>& /*p*/) const
{
  Assert (cell->user_pointer () != nullptr, ExcMessage ("No properties were assigned to this cell!"));
  CellProperties* properties = static_cast<CellProperties*> (cell->user_pointer ());
  const Material& material = properties->material;

  if(!material.is_valid())
  {
    std::stringstream ss;
    ss << "Cell " << cell->id() << " (" << cell->center(true)
       << ") has invalid permittivity " << material.permittivity
       << " and it was requested. Locally owned = "
       << cell->is_locally_owned();
    throw std::runtime_error(ss.str());
  }

  return material.permittivity * eps0;
}

template<int dim>
double PhysicalModel<dim>::permeability_at (const typename Triangulation<dim>::active_cell_iterator& /*cell*/,
                                            const Point<dim>& /*p*/) const
{
  return mu0;
}

template<int dim>
void PhysicalModel<dim>::polarization_properties_at(const typename Triangulation<dim>::active_cell_iterator &cell,
                                                    std::vector<double> &properties,
                                                    const Point<dim> &/*p*/) const
{
  Assert (cell->user_pointer () != nullptr,
          ExcMessage ("No properties were assigned to this cell!"));

  Assert (properties.size () == 3, ExcDimensionMismatch (properties.size (), 3));

  CellProperties* cell_properties = static_cast<CellProperties*> (cell->user_pointer ());
  const Material& material = cell_properties->material;

  if(std::isfinite(material.chargeability) &&
     std::isfinite(material.relaxation_time) &&
     std::isfinite(material.exponent))
  {
    properties[0] = material.chargeability;
    properties[1] = material.relaxation_time;
    properties[2] = material.exponent;
  }
  else
    throw std::runtime_error("Cell has invalid polarization properties and they were requested.");
}

template<int dim>
void PhysicalModel<dim>::conductivity_list (const typename Triangulation<dim>::active_cell_iterator& cell,
                                            const std::vector<Point<dim> > &points,
                                            std::vector<double> &values) const
{
  Assert (values.size () == points.size (),
          ExcDimensionMismatch (values.size (), points.size ()));

  for (size_t i = 0; i < points.size (); ++i)
    values[i] = conductivity_at (cell, points[i]);
}

template<int dim>
void PhysicalModel<dim>::complex_conductivity_list(const typename Triangulation<dim>::active_cell_iterator &cell,
                                                   const std::vector<Point<dim> > &points,
                                                   std::vector<std::complex<double> > &values,
                                                   double frequency) const
{
  Assert (values.size () == points.size (),
          ExcDimensionMismatch (values.size (), points.size ()));

  for (size_t i = 0; i < points.size (); ++i)
    values[i] = complex_conductivity_at (cell, frequency, points[i]);
}

template<int dim>
void PhysicalModel<dim>::permittivity_list (const typename Triangulation<dim>::active_cell_iterator& cell,
                                            const std::vector<Point<dim> > &points,
                                            std::vector<double> &values) const
{
  Assert (values.size() == points.size (),
          ExcDimensionMismatch (values.size (), points.size ()));

  for (size_t i = 0; i < points.size (); ++i)
    values[i] = permittivity_at (cell, points[i]);
}

template<int dim>
void PhysicalModel<dim>::permeability_list (const typename Triangulation<dim>::active_cell_iterator& cell,
                                            const std::vector<Point<dim> > &points,
                                            std::vector<double> &values) const
{
  Assert (values.size() == points.size (), ExcDimensionMismatch (values.size (), points.size ()));

  for (size_t i = 0; i < points.size (); ++i)
    values[i] = permeability_at (cell, points[i]);
}

template<int dim>
void PhysicalModel<dim>::polarization_properties_list(const typename Triangulation<dim>::active_cell_iterator &cell,
                                                      const std::vector<Point<dim> > &points,
                                                      std::vector<std::vector<double>> &values) const
{
  Assert (values.size() == points.size (),
          ExcDimensionMismatch (values.size (), points.size ()));

  for (size_t i = 0; i < points.size (); ++i)
    this->polarization_properties_at (cell, values[i], points[i]);
}

template<int dim>
void PhysicalModel<dim>::transfer_cell_properties(Triangulation<dim> &tria)
{
  cell_properties.clear();
  for (const auto &cell: tria.active_cell_iterators ())
  {
    if(cell->user_pointer () == nullptr)
      throw std::runtime_error ("No properties were assigned to this cell!");

    CellProperties* properties = static_cast<CellProperties*> (cell->user_pointer ());
    if(!properties->material.is_valid())
      throw std::runtime_error("Invalid cell properties");

    cell_properties.push_back(*properties);
    cell->set_user_pointer (&cell_properties.back());
  }
}

template<int dim>
void PhysicalModel<dim>::clear_cell_properties_recursively(const typename Triangulation<dim>::cell_iterator &cell)
{
  for(unsigned child_index = 0; child_index < cell->n_children(); ++child_index)
  {
    auto child = cell->child(child_index);
    child->set_user_pointer(nullptr);
    clear_cell_properties_recursively(child);
  }
}

template<int dim>
BackgroundModel<dim>::BackgroundModel (const std::string modelpath, const std::string materialpath):
  BackgroundModel<dim> (modelpath, materialpath, earth_radius)
{
}

template<int dim>
BackgroundModel<dim>::BackgroundModel (const std::string modelpath, const std::string materialpath, const double size):
  PhysicalModel<dim> (materialpath, modelpath), sphere_radius (size), cs(Cartesian)
{
  std::vector<types::material_id> material_ids;
  read_1dmodel (modelpath, depths, material_ids);

  for(size_t i = 0; i < material_ids.size(); ++i)
  {
    auto it = this->materials_map.find (material_ids[i]);
    if(it == this->materials_map.end ())
      throw std::runtime_error("BackgroundModel: no material found!");

    materials.push_back(it->second);
  }
}

template<int dim>
BackgroundModel<dim>::BackgroundModel (dvector depths, std::vector<Material> materials_data):
  sphere_radius (earth_radius), depths(depths), materials(materials_data), cs(Cartesian)
{
}

template<int dim>
BackgroundModel<dim> *BackgroundModel<dim>::clone() const
{
  return new BackgroundModel<dim>(*this);
}

template<int dim>
void BackgroundModel<dim>::create_triangulation(Triangulation<dim> &/*tria*/)
{
  throw std::runtime_error("Not implemented.");
}

template<int dim>
double BackgroundModel<dim>::conductivity_at (const typename Triangulation<dim>::active_cell_iterator& cell,
                                              const Point<dim>& p) const
{
  const unsigned idx = layer_index(cell, {p})[0];
  return materials[idx].conductivity;
}

template<int dim>
double BackgroundModel<dim>::conductivity_at(const double z_center, const double z) const
{
  const unsigned idx = layer_index(z_center, z);
  return materials[idx].conductivity;
}

template<int dim>
double BackgroundModel<dim>::permittivity_at (const typename Triangulation<dim>::active_cell_iterator& cell,
                                              const Point<dim>& p) const
{
  const unsigned idx = layer_index(cell, {p})[0];
  return materials[idx].permittivity * eps0;
}

template<int dim>
double BackgroundModel<dim>::permittivity_at(const double z_center, const double z) const
{
  const unsigned idx = layer_index(z_center, z);
  return materials[idx].permittivity * eps0;
}

template<int dim>
void BackgroundModel<dim>::conductivity_list(const typename Triangulation<dim>::active_cell_iterator &cell,
                                             const std::vector<Point<dim> > &points,
                                             std::vector<double> &values) const
{
  const auto indices = layer_index(cell, points);
  for(unsigned i = 0; i < indices.size(); ++i)
    values[i] = materials[indices[i]].conductivity;
}

template<int dim>
dvector BackgroundModel<dim>::conductivities() const
{
  dvector sigma(materials.size());

  for(unsigned i = 0; i < materials.size(); ++i)
    sigma[i] = materials[i].conductivity;

  return sigma;
}

template<int dim>
dvector BackgroundModel<dim>::permittivities () const
{
  dvector eps(materials.size());

  for(unsigned i = 0; i < materials.size(); ++i)
    eps[i] = materials[i].permittivity * eps0;

  return eps;
}

template<int dim>
dvector BackgroundModel<dim>::permeabilities() const
{
  return dvector(materials.size(), mu0);
}

template<int dim>
dvector BackgroundModel<dim>::layer_depths() const
{
  return depths;
}

template<int dim>
unsigned BackgroundModel<dim>::n_layers() const
{
  return depths.size();
}

template<int dim>
double BackgroundModel<dim>::get_radius() const
{
  return sphere_radius;
}

template<int dim>
types::material_id BackgroundModel<dim>::material_id_at(const double z) const
{
  // Special case of a single layer
  if(depths.size() == 1)
  {
    if (z > depths[0])
      return materials[0].id;
  }

  for (size_t i = 1; i < depths.size (); ++i)
  {
    if (z >= depths[i - 1] && z < depths[i])
      return materials[i - 1].id;
  }

  std::stringstream ss;
  ss << "Point is outside model grid. z = " << z;
  throw std::runtime_error(ss.str());

  return numbers::invalid_material_id;
}

template<int dim>
Point<dim> BackgroundModel<dim>::model_extension() const
{
  Point<dim> ext;

  for(int d = 0; d < dim; ++d)
    ext[d] = std::numeric_limits<double>::infinity();

  return ext;
}

template<int dim>
void BackgroundModel<dim>::set_coordinate_system(CoordinateSystem c)
{
  cs = c;
}

template<int dim>
void BackgroundModel<dim>::set_radius(double r)
{
  sphere_radius = r;
}

template<int dim>
std::vector<unsigned> BackgroundModel<dim>::layer_index(const typename Triangulation<dim>::active_cell_iterator& cell,
                                                        const std::vector<Point<dim>> &points) const
{
  std::vector<unsigned> indices(points.size(), std::numeric_limits<unsigned>::max());

  Point<dim> c = cell->center(true);
  if(cs == Spherical)
  {
    const auto ps = point_to_spherical(c);
    c = ps;
    c[dim - 1] = sphere_radius - c[dim - 1];
  }

  for(unsigned n = 0; n < points.size(); ++n)
  {
    double z = points[n](dim - 1);

    if(cs == Spherical)
    {
      auto ps = point_to_spherical(points[n]);
      z = sphere_radius - ps[dim - 1];

      if(fabs(z) < minimum_thickness)
        z = 0;
    }

    indices[n] = layer_index(c[dim - 1], z);
  }

  return indices;
}

template<int dim>
unsigned BackgroundModel<dim>::layer_index(const double &z_center, const double &z) const
{
  // Special case of a single layer
  if(depths.size() == 1)
  {
    if (z > depths[0])
      return 0;
  }

  // Lower half-space
  if(z > depths.back())
    return depths.size () - 1;

  for (size_t i = 1; i < depths.size (); ++i)
  {
    if(z_center <= z) // approach boundary from above
    {
      if (z > depths[i - 1] && z <= depths[i])
        return i - 1;
    }
    else // approach boundary from below
    {
      if (z >= depths[i - 1] && z < depths[i])
        return i - 1;
    }
  }

  std::stringstream ss;
  ss << "Point " << z_center << " is outside model grid";
  throw std::runtime_error(ss.str());

  return std::numeric_limits<unsigned>::max();
}

template<int dim>
XYZModel<dim>::XYZModel (const std::string xyzpath, const std::string materialpath):
  PhysicalModel<dim> (materialpath, xyzpath)
{
  read_in(xyzpath);
}

template<int dim>
XYZModel<dim> *XYZModel<dim>::clone() const
{
  return new XYZModel<dim>(*this);
}

template<int dim>
void XYZModel<dim>::create_triangulation(Triangulation<dim> &tria)
{
  throw std::runtime_error("What dim did you use??");
}

template<int dim>
Point<dim> XYZModel<dim>::model_extension() const
{
  Point<dim> ext;

  for(int d = 0; d < dim; ++d)
    for(size_t i = 0; i < cell_sizes[d].size(); ++i)
      ext[d] += cell_sizes[d][i];

  return ext;
}

template<int dim>
double XYZModel<dim>::air_thickness() const
{
  // Identify thickness of the air layer
  types::material_id air_material_id = numbers::invalid_material_id;

  for(auto p: this->materials_map)
  {
    std::string material_name = p.second.name;
    std::transform(material_name.begin(), material_name.end(),
                   material_name.begin(), ::tolower);

    if(material_name == "air")
      air_material_id = p.first;
  }

  if(air_material_id == numbers::invalid_material_id)
  {
    std::cout << "No material with the name 'air' was found. Assume model has not air layer." << std::endl;
    return 0;
  }
  else
  {
    double air_layer_thickness = 0;
    for(unsigned i = 0; i < this->cell_sizes[dim - 1].size(); ++i)
    {
      if(this->material_id[i] == air_material_id)
        air_layer_thickness += this->cell_sizes[dim - 1][i];
    }

    if(air_layer_thickness < 1e-1)
      throw std::runtime_error("Air thickness is zero, something went wrong.");

    return air_layer_thickness;
  }
}

template<>
void XYZModel<2>::create_triangulation (Triangulation<2> &tria)
{
  Table<2, types::material_id> material_id_table (cell_sizes[0].size (), cell_sizes[1].size ());
  material_id_table.fill (&material_id[0]);

  Point<2> p (origin[0], origin[1]);

  GridGenerator::subdivided_hyper_rectangle (tria, cell_sizes, p, material_id_table);

  init_properties_from_materials(tria);
}

template<>
void XYZModel<3>::create_triangulation (Triangulation<3> &tria)
{
  Table<3, types::material_id> material_id_table (cell_sizes[0].size (),
      cell_sizes[1].size (), cell_sizes[2].size ());
  material_id_table.fill (&material_id[0]);

  Point<3> p (origin[0], origin[1], origin[2]);

  GridGenerator::subdivided_hyper_rectangle (tria, cell_sizes, p, material_id_table);

  init_properties_from_materials(tria);
}

template<int dim>
void XYZModel<dim>::read_in(const std::string xyzpath)
{
  std::vector<unsigned> parameter_id;
  read_tensor_grid_model (xyzpath, cell_sizes, origin,
                                   material_id,
                                   parameter_id);

  unsigned n_read_cells = 1;
  for(size_t i = 0; i < cell_sizes.size(); ++i)
    n_read_cells *= cell_sizes[i].size();

  // If only one model column is given, populate it X^dim-1 times
  if(material_id.size() == cell_sizes[dim-1].size())
  {
    auto column_id = material_id;
    material_id.clear();

    if(dim == 3)
    {
      for(size_t i = 0; i < cell_sizes[0].size(); ++i)
        for(size_t j = 0; j < cell_sizes[1].size(); ++j)
          for(size_t k = 0; k < cell_sizes[2].size(); ++k)
            material_id.push_back(column_id[k]);
    }
    else if(dim == 2)
    {
      for(size_t i = 0; i < cell_sizes[0].size(); ++i)
          for(size_t k = 0; k < cell_sizes[1].size(); ++k)
            material_id.push_back(column_id[k]);
    }
    else
      throw std::runtime_error("Wrong dimension.");
  }

  if (material_id.size() != n_read_cells || dim != cell_sizes.size())
  {
    throw std::runtime_error("Size of material id array " + std::to_string(material_id.size()) +
                             " is not equal to the number of cells " + std::to_string(n_read_cells) +
                             " or the specified model dimension is wrong! Check input file.");
  }
}

template<int dim>
BoundaryModel<dim>::BoundaryModel(const Triangulation<dim+1> &tria, types::boundary_id id)
{
  triangulation = &tria;
  bid = id;
}

template<int dim>
BoundaryModel<dim> *BoundaryModel<dim>::clone() const
{
  return new BoundaryModel<dim>(*this);
}

template<int dim>
Point<dim> BoundaryModel<dim>::model_extension() const
{
  throw std::runtime_error("model_extension not implemented.");
}

template<int dim>
void BoundaryModel<dim>::create_triangulation(Triangulation<dim>& /*triangulation*/)
{
  throw std::runtime_error("create_triangulation not implemented");
}

template<>
void BoundaryModel<2>::create_triangulation(Triangulation<2> &tria)
{
  unsigned skip_dim = 0;

  if(bid == east_boundary_id || bid == west_boundary_id)
    skip_dim = 1;
  else if(bid == top_boundary_id || bid == bottom_boundary_id)
    skip_dim = 2;

  MyGridTools::extract_boundary_mesh(*triangulation, tria, bid, skip_dim);

//  FE_Q<2> fe(1);
//  DoFHandler<2> dof(tria);
//  dof.distribute_dofs(fe);

//  std::ofstream output(std::string("mesh_") + std::to_string(int(this->bid)) + ".vtk");
//  DataOut<2,DoFHandler<2> > data_out;
//  data_out.attach_dof_handler(dof);

//  Vector<float> material_id (tria.n_active_cells ()),
//      detJ (tria.n_active_cells ());
//  unsigned index = 0;
//  FEValues<2> fe_values(fe, Quadrature<2>(Point<2>(0.5, 0.5)), update_JxW_values);
//  for (Triangulation<2>::active_cell_iterator cell=tria.begin_active();
//       cell!=tria.end(); ++cell, ++index)
//  {
//    fe_values.reinit(cell);
//    detJ(index) = fe_values.JxW(0);
//    material_id(index) = cell->material_id();
//  }
//  data_out.add_data_vector(material_id, "material_id");
//  data_out.add_data_vector (detJ, "detJ");
//  data_out.build_patches();
//  data_out.write_vtk(output);
//  output.close();
}

template<int dim>
TriangulationModel<dim>::TriangulationModel(const std::string triapath,
                                            const std::string materialpath):
  PhysicalModel<dim>(materialpath, triapath)
{}

template<int dim>
TriangulationModel<dim> *TriangulationModel<dim>::clone() const
{
  return new TriangulationModel<dim>(*this);
}

template<int dim>
void TriangulationModel<dim>::create_triangulation(Triangulation<dim> &tria)
{
  tria.clear();

  MyGridTools::load_triangulation(this->input_model_file, tria);

  this->init_properties_from_materials(tria);

  retrieve_geometry_data(tria);
}

template<int dim>
Point<dim> TriangulationModel<dim>::model_extension() const
{
  Point<dim> p;

  for(unsigned d = 0; d < dim; ++d)
    p[d] = extension[d];

  return p;
}

template<int dim>
double TriangulationModel<dim>::air_thickness() const
{
  return air_layer_thickness;
}

template<int dim>
void TriangulationModel<dim>::retrieve_geometry_data(const Triangulation<dim> &tria)
{
  dvec3d p0, p1;
  p0.fill(std::numeric_limits<double>::max());
  p1.fill(std::numeric_limits<double>::min());

  double zair_min = std::numeric_limits<double>::max(),
         zair_max =-std::numeric_limits<double>::max();

  for (const auto &cell: tria.active_cell_iterators())
  {
    if(cell->is_locally_owned())
    {
      for(unsigned i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
      {
        for(unsigned d = 0; d < dim; ++d)
        {
          if(cell->vertex(i)[d] < p0[d])
            p0[d] = cell->vertex(i)[d];

          if(cell->vertex(i)[d] > p1[d])
            p1[d] = cell->vertex(i)[d];
        }

        if(this->is_air_cell(cell))
        {
          const double z = cell->vertex(i)[dim-1];

          if(z < zair_min)
            zair_min = z;

          if(z > zair_max)
            zair_max = z;
        }
      }
    }
  }

  if(auto tria_dist = dynamic_cast<const parallel::fullydistributed::Triangulation<dim>*>(&tria))
  {
    zair_min = Utilities::MPI::min(zair_min, tria_dist->get_communicator());
    zair_max = Utilities::MPI::max(zair_max, tria_dist->get_communicator());

    for(int d = 0; d < dim; ++d)
    {
      p0[d] = Utilities::MPI::min(p0[d], tria_dist->get_communicator());
      p1[d] = Utilities::MPI::max(p1[d], tria_dist->get_communicator());
    }
  }

  for(unsigned d = 0; d < dim; ++d)
    extension[d] = p1[d] - p0[d];

  air_layer_thickness = zair_max - zair_min;
}

template class PhysicalModel<3>;
template class BackgroundModel<3>;
template class XYZModel<3>;
template class TriangulationModel<3>;
template class PhysicalModel<2>;
template class BackgroundModel<2>;
template class XYZModel<2>;
template class TriangulationModel<2>;
template class BoundaryModel<2>;
