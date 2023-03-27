#ifndef MESH_TOOLS_H
#define MESH_TOOLS_H

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/fully_distributed_tria.h>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/intergrid_map.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/utilities.h>

#include "physical_model/physical_model.h"
#include "physical_model/spherical_model.h"

namespace MyGridTools
{

inline bool file_exists (const std::string& name)
{
    if (FILE *file = fopen(name.c_str(), "r"))
    {
        fclose(file);
        return true;
    }
    else
    {
        return false;
    }
}

template<int dim>
void load_triangulation(const std::string tria_file,
                        Triangulation<dim> &tria)
{
    if(!file_exists(tria_file))
        throw std::runtime_error("File " + tria_file + " does not exist.");

    std::ifstream ifs(tria_file);
    boost::archive::binary_iarchive ia(ifs);

    if(auto shared_tria = dynamic_cast<parallel::shared::Triangulation<dim>*>(&tria))
    {
        shared_tria->load(ia, 1);
    }
    else if(auto tria_pft = dynamic_cast<parallel::fullydistributed::Triangulation<dim>*>(&tria))
    {
        Triangulation<dim> basetria;
        basetria.load(ia, 1);

        std::map<CellId, types::material_id> cell_mat_id;
        for (const auto &cell: basetria.active_cell_iterators())
            cell_mat_id.insert({cell->id(), cell->material_id()});

        tria_pft->set_partitioner(
                    [](dealii::Triangulation<dim> &tria, const unsigned int n_partitions) {
            GridTools::partition_triangulation(n_partitions,
                                               tria,
                                               SparsityTools::Partitioner::metis);
        },
        TriangulationDescription::Settings::default_setting);

        // actually create triangulation
        tria_pft->copy_triangulation(basetria);

        for (auto &cell: tria_pft->active_cell_iterators())
        {
            auto it = cell_mat_id.find(cell->id());
            if(it != cell_mat_id.end())
                cell->set_material_id(it->second);
        }

        for (auto &cell: tria_pft->active_cell_iterators())
        {
            if(cell->is_locally_owned() &&
                    cell->material_id() == numbers::invalid_material_id &&
                    cell->level () > 0)
            {
                auto cell_parent = cell->parent();
                while(cell_parent->material_id () == numbers::invalid_material_id &&
                      cell_parent->level () > 0)
                    cell_parent = cell_parent->parent();

                cell_parent->recursively_set_material_id(cell_parent->material_id ());
            }
        }
    }
    else
        tria.load(ia, 1);
}

/*
 * Refines cells which neighbor specified points where the
 * accuracy is especially important (e.g. receiver positions)
 */
template<int dim>
void refine_grid_around_points_cache (parallel::shared::Triangulation<dim> &triangulation,
                                      const Mapping<dim> &mapping,
                                      const std::vector<Point<dim>>& points)
{
    GridTools::Cache<dim> cache(triangulation, mapping);
    std::vector<CellId> local_cells;
    {
        IteratorFilters::LocallyOwnedCell locally_owned_cell_predicate;
        std::vector<BoundingBox<dim>>     local_bbox =
                GridTools::compute_mesh_predicate_bounding_box(
                    cache.get_triangulation(),
                    std::function<bool(
                        const typename Triangulation<dim>::active_cell_iterator &)>(
                        locally_owned_cell_predicate));

        // Obtaining the global mesh description through an all to all communication
        std::vector<std::vector<BoundingBox<dim>>> global_bboxes;
        global_bboxes = Utilities::MPI::all_gather(triangulation.get_communicator(), local_bbox);

        // Using the distributed version of compute point location
        auto output_tuple =
                GridTools::distributed_compute_point_locations(cache, points, global_bboxes);

        const auto &local_cell_iterators   = std::get<0>(output_tuple);

        for(const auto cell: local_cell_iterators)
        {
            local_cells.push_back(cell->id());
            // Refine neigbors as well
            for (auto face_no = 0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no)
                if (!cell->at_boundary (face_no))
                {
                    const auto neighbor = cell->neighbor (face_no);

                    // Refine neighbors of neighbors
                    for (auto face_no_neighbor=0; face_no_neighbor < GeometryInfo<dim>::faces_per_cell; ++face_no_neighbor)
                        if (!neighbor->at_boundary (face_no_neighbor))
                        {
                            const auto neighbor_of_neighbor = neighbor->neighbor (face_no_neighbor);

                            if (neighbor_of_neighbor->is_active ())
                                local_cells.push_back(neighbor_of_neighbor->id());
                        }

                    if (neighbor->is_active ())
                        local_cells.push_back(neighbor->id());
                }
        }
    }

    std::vector<char> local_buffer = Utilities::pack(local_cells);
    std::vector<std::vector<char>> all_buffers =
            Utilities::MPI::all_gather(triangulation.get_communicator(), local_buffer);

    for (auto & buffer: all_buffers)
    {
        auto ids = Utilities::unpack<std::vector<CellId>>(buffer);
        for (auto & id: ids)
        {
            auto cell = id.to_cell(triangulation);
            cell->set_refine_flag();
        }
    }

    triangulation.prepare_coarsening_and_refinement ();
    triangulation.execute_coarsening_and_refinement ();
}

/*
 * Refines cells which neighbor specified points where the
 * accuracy is especially important (e.g. receiver positions)
 */
template<int dim>
void refine_grid_around_points (parallel::shared::Triangulation<dim> &triangulation,
                                const Mapping<dim> &mapping,
                                const std::vector<Point<dim>>& points)
{
    std::vector<CellId> local_cells;

    // Loop over all points
    for (size_t i = 0; i < points.size (); ++i)
    {
        // Check if current point is on subdomain owned by me
        auto cell_point = GridTools::find_active_cell_around_point (mapping, triangulation, points[i]);

        if (cell_point.first == triangulation.end() ||
                cell_point.first->is_locally_owned() == false)
            continue;

        local_cells.push_back(cell_point.first->id());

        // Refine neigbors as well
        for (auto face_no = 0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no)
            if (!cell_point.first->at_boundary (face_no))
            {
                const auto neighbor = cell_point.first->neighbor (face_no);

                // Refine neighbors of neighbors
                for (auto face_no_neighbor=0; face_no_neighbor < GeometryInfo<dim>::faces_per_cell; ++face_no_neighbor)
                    if (!neighbor->at_boundary (face_no_neighbor))
                    {
                        const auto neighbor_of_neighbor = neighbor->neighbor (face_no_neighbor);

                        if (neighbor_of_neighbor->is_active ())
                            local_cells.push_back(neighbor_of_neighbor->id());
                    }

                if (neighbor->is_active ())
                    local_cells.push_back(neighbor->id());
            }
    }

    std::vector<char> local_buffer = Utilities::pack(local_cells);
    std::vector<std::vector<char>> all_buffers =
            Utilities::MPI::all_gather(triangulation.get_communicator(), local_buffer);

    for (auto & buffer: all_buffers)
    {
        auto ids = Utilities::unpack<std::vector<CellId>>(buffer);
        for (auto & id: ids)
        {
            auto cell = id.to_cell(triangulation);
            cell->set_refine_flag();
        }
    }

    triangulation.prepare_coarsening_and_refinement ();
    triangulation.execute_coarsening_and_refinement ();
}

/*
 * For inversion it is important to ensure that the cells
 * which contain receivers do not have coarser neighbours.
 * Neighbours include all cells that share face, edge and/or
 * vertex. Hence, there are at least 26 of them in 3D.
 * This method checks this condition and refines coarser
 * neighbours. It returns true if more coarser neighbors
 * exist and false if no coarser neighbors were detected
 * after refinement.
 * TODO: this routine is rather slow, think about optimization.
 */
template<int dim>
bool refine_coarser_neighbours_around_receivers_parallel (Triangulation<dim> &triangulation,
                                                          const Mapping<dim>& /*mapping*/,
                                                          const std::vector<Point<dim>> &points,
                                                          MPI_Comm communicator)
{
    unsigned char coarser_neighbors_exist_local = false,
            coarser_neighbors_exist_global = false;

    std::set<unsigned> visited_vertices;

    for (auto cell: triangulation.active_cell_iterators ())
        if (!cell->is_artificial ())
        {
            // Loop over all points
            for (size_t i = 0; i < points.size (); ++i)
            {
                // !!! for non-rectangular meshes this call has to be replaced
                if(!cell->point_inside(points[i]))
                    continue;

                typedef std::set<typename Triangulation<dim>::active_cell_iterator> CellSet;
                CellSet adjacent_cells;

                for(unsigned vertex = 0; vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex)
                {
                    unsigned vertex_index = cell->vertex_index(vertex);
                    if(visited_vertices.find(vertex_index) != visited_vertices.end())
                        continue;
                    else
                        visited_vertices.insert(vertex_index);

                    std::vector<typename Triangulation<dim>::active_cell_iterator> adjacent_cells_vec
                            = GridTools::find_cells_adjacent_to_vertex (triangulation, vertex_index);

                    std::copy (adjacent_cells_vec.begin(), adjacent_cells_vec.end(),
                               std::inserter (adjacent_cells, adjacent_cells.begin()));
                }

                int level = cell->level();
                for (typename CellSet::const_iterator it = adjacent_cells.begin(); it != adjacent_cells.end(); ++it)
                    if ((*it)->level() < level)
                    {
                        (*it)->set_refine_flag ();
                        coarser_neighbors_exist_local = true;
                    }
            }
        }

    triangulation.prepare_coarsening_and_refinement ();
    triangulation.execute_coarsening_and_refinement ();

    // All processes make another loop if there were coarser neighbors on at least one process
    MPI_Allreduce ((void*)&coarser_neighbors_exist_local,
                   (void*)&coarser_neighbors_exist_global,
                   1, MPI_UNSIGNED_CHAR, MPI_LOR, communicator);

    return coarser_neighbors_exist_global;
}

/*
 * Same as above, but faster and works only for serial cases
 */
template<int dim>
bool refine_coarser_neighbours_around_receivers_serial (parallel::shared::Triangulation<dim> &triangulation,
                                                        const Mapping<dim> &mapping,
                                                        const std::vector<Point<dim>>& points)
{
    std::vector<CellId> local_cells;

    bool coarser_neighbors_exist = false;

    std::set<unsigned> visited_vertices;

    // Loop over all points
    for (size_t i = 0; i < points.size (); ++i)
    {
        auto cell_point = GridTools::find_active_cell_around_point (mapping, triangulation, points[i]);

        if (cell_point.first == triangulation.end() ||
                cell_point.first->is_locally_owned() == false)
            continue;


        typedef std::set<typename Triangulation<dim>::active_cell_iterator> CellSet;
        CellSet adjacent_cells;

        for(unsigned vertex = 0; vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex)
        {
            unsigned vertex_index = cell_point.first->vertex_index(vertex);
            if(visited_vertices.find(vertex_index) == visited_vertices.end())
                visited_vertices.insert(vertex_index);
            else
                continue;

            auto adjacent_cells_vec = GridTools::find_cells_adjacent_to_vertex (triangulation,
                                                                                vertex_index);

            std::copy (adjacent_cells_vec.begin(),
                       adjacent_cells_vec.end(),
                       std::inserter (adjacent_cells, adjacent_cells.begin()));
        }

        int level = cell_point.first->level();
        for (auto cell: adjacent_cells)
        {
            if (cell->level() < level)
                local_cells.push_back(cell->id ());
        }
    }

    std::vector<char> local_buffer = Utilities::pack(local_cells);
    std::vector<std::vector<char>> all_buffers =
            Utilities::MPI::all_gather(triangulation.get_communicator(), local_buffer);

    for (auto & buffer: all_buffers)
    {
        auto ids = Utilities::unpack<std::vector<CellId>>(buffer);

        if(ids.size() > 0)
            coarser_neighbors_exist = true;

        for (auto & id: ids)
        {
            auto cell = id.to_cell(triangulation);
            cell->set_refine_flag();
        }
    }

    triangulation.prepare_coarsening_and_refinement ();
    triangulation.execute_coarsening_and_refinement ();

    return coarser_neighbors_exist;
}

/*
 * Same as above, but uses cache
 */
template<int dim>
bool refine_coarser_neighbours_around_receivers_serial_cache (parallel::shared::Triangulation<dim> &triangulation,
                                                              const Mapping<dim> &mapping,
                                                              const std::vector<Point<dim>>& points)
{
    bool coarser_neighbors_exist = false;

    std::set<unsigned> visited_vertices;

    GridTools::Cache<dim> cache(triangulation, mapping);
    std::vector<CellId> local_cells;
    {
        IteratorFilters::LocallyOwnedCell locally_owned_cell_predicate;
        std::vector<BoundingBox<dim>>     local_bbox =
                GridTools::compute_mesh_predicate_bounding_box(
                    cache.get_triangulation(),
                    std::function<bool(
                        const typename Triangulation<dim>::active_cell_iterator &)>(
                        locally_owned_cell_predicate));

        // Obtaining the global mesh description through an all to all communication
        std::vector<std::vector<BoundingBox<dim>>> global_bboxes;
        global_bboxes = Utilities::MPI::all_gather(triangulation.get_communicator(), local_bbox);

        // Using the distributed version of compute point location
        auto output_tuple =
                GridTools::distributed_compute_point_locations(cache, points, global_bboxes);

        const auto &local_cell_iterators   = std::get<0>(output_tuple);

        for(const auto cell: local_cell_iterators)
        {
            typedef std::set<typename Triangulation<dim>::active_cell_iterator> CellSet;
            CellSet adjacent_cells;

            for(unsigned vertex = 0; vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex)
            {
                unsigned vertex_index = cell->vertex_index(vertex);
                if(visited_vertices.find(vertex_index) == visited_vertices.end())
                    visited_vertices.insert(vertex_index);
                else
                    continue;

                const auto& adjacent_cells_vec = cache.get_vertex_to_cell_map()[vertex_index];
                std::copy (adjacent_cells_vec.begin(),
                           adjacent_cells_vec.end(),
                           std::inserter (adjacent_cells, adjacent_cells.begin()));
            }

            int level = cell->level();
            for (const auto adjacent_cell: adjacent_cells)
            {
                if (adjacent_cell->level() < level)
                    local_cells.push_back(adjacent_cell->id ());
            }
        }
    }

    std::vector<char> local_buffer = Utilities::pack(local_cells);
    std::vector<std::vector<char>> all_buffers =
            Utilities::MPI::all_gather(triangulation.get_communicator(), local_buffer);

    for (auto & buffer: all_buffers)
    {
        auto ids = Utilities::unpack<std::vector<CellId>>(buffer);

        if(ids.size() > 0)
            coarser_neighbors_exist = true;

        for (auto & id: ids)
        {
            auto cell = id.to_cell(triangulation);
            cell->set_refine_flag();
        }
    }

    triangulation.prepare_coarsening_and_refinement ();
    triangulation.execute_coarsening_and_refinement ();

    return coarser_neighbors_exist;
}

template <int dim>
std::map<typename Triangulation<dim-1>::cell_iterator,
typename Triangulation<dim>::face_iterator>
extract_boundary_mesh (const Triangulation<dim>       &volume_mesh,
                       Triangulation<dim-1>           &surface_mesh,
                       const types::boundary_id bid,
                       const unsigned skip_dim)
{
    if(dim != 3)
        throw std::runtime_error("Dimension mismatch in extract_boundary_mesh");

    // This function works using the following assumption:
    //    Triangulation::create_triangulation(...) will create cells that preserve
    //    the order of cells passed in using the CellData argument; also,
    //    that it will not reorder the vertices.

    int surface_to_volume_face_child_map[4] = {0, 1, 2, 3};
    if(skip_dim == 1)
        std::swap (surface_to_volume_face_child_map[1],
                surface_to_volume_face_child_map[2]);

    std::map<typename Triangulation<dim-1>::cell_iterator,
            typename Triangulation<dim>::face_iterator>
            surface_to_volume_mapping;

    std::map<typename Triangulation<dim>::face_iterator,void*>
            user_pointers_mapping;

    // First create surface mesh and mapping
    // from only level(0) cells of volume_mesh
    std::vector<typename Triangulation<dim>::face_iterator> mapping;  // temporary map for level==0

    std::vector<bool> touched (volume_mesh.get_triangulation().n_vertices(), false);
    std::vector<CellData<dim-1>> cells;
    SubCellData                  subcell_data;
    std::vector<Point<dim-1>>    vertices;

    std::map<unsigned int,unsigned int> map_vert_index; //volume vertex indices to surf ones

    for (typename Triangulation<dim>::cell_iterator
         cell = volume_mesh.begin(0);
         cell != volume_mesh.end(0);
         ++cell)
        for (unsigned int i=0; i < GeometryInfo<dim>::faces_per_cell; ++i)
        {
            const typename Triangulation<dim>::face_iterator
                    face = cell->face(i);

            if ( face->at_boundary()
                 &&
                 face->boundary_id() == bid )
            {
                CellData< dim-1 > c_data;

                for (unsigned int j=0;
                     j<GeometryInfo<dim-1>::vertices_per_cell; ++j)
                {
                    const unsigned int v_index = face->vertex_index(j);

                    if ( !touched[v_index] )
                    {
                        Point<dim-1> p;
                        unsigned dnew = 0;
                        for(unsigned d = 0; d < dim; ++d)
                        {
                            if(d != skip_dim)
                                p[dnew++] = face->vertex(j)[d];
                        }

                        vertices.push_back(p);
                        map_vert_index[v_index] = vertices.size() - 1;
                        touched[v_index] = true;
                    }

                    c_data.vertices[j] = map_vert_index[v_index];
                    c_data.material_id = cell->material_id();
                }

                // if we start from a 3d mesh, then we have copied the
                // vertex information in the same order in which they
                // appear in the face; however, this means that we
                // impart a coordinate system that is right-handed when
                // looked at *from the outside* of the cell if the
                // current face has index 0, 2, 4 within a 3d cell, but
                // right-handed when looked at *from the inside* for the
                // other faces. we fix this by flipping opposite
                // vertices if we are on a face 1, 3, 5
                if(i == 2 || i == 3)
                    std::swap (c_data.vertices[1], c_data.vertices[2]);

                // in 3d, we also need to make sure we copy the manifold
                // indicators from the edges of the volume mesh to the
                // edges of the surface mesh
                //
                // one might think that we we can also prescribe
                // boundary indicators for edges, but this is only
                // possible for edges that aren't just on the boundary
                // of the domain (all of the edges we consider are!) but
                // that would actually end up at the boundary of the
                // surface mesh. there is no easy way to check this, so
                // we simply don't do it and instead set it to an
                // invalid value that makes sure
                // Triangulation::create_triangulation doesn't copy it
                if (dim == 3)
                    for (unsigned int e=0; e<4; ++e)
                    {
                        // see if we already saw this edge from a
                        // neighboring face, either in this or the reverse
                        // orientation. if so, skip it.
                        {
                            bool edge_found = false;
                            for (unsigned int i=0; i<subcell_data.boundary_lines.size(); ++i)
                                if (((subcell_data.boundary_lines[i].vertices[0]
                                      == map_vert_index[face->line(e)->vertex_index(0)])
                                     &&
                                     (subcell_data.boundary_lines[i].vertices[1]
                                      == map_vert_index[face->line(e)->vertex_index(1)]))
                                        ||
                                        ((subcell_data.boundary_lines[i].vertices[0]
                                          == map_vert_index[face->line(e)->vertex_index(1)])
                                         &&
                                         (subcell_data.boundary_lines[i].vertices[1]
                                          == map_vert_index[face->line(e)->vertex_index(0)])))
                                {
                                    edge_found = true;
                                    break;
                                }
                            if (edge_found == true)
                                continue;   // try next edge of current face
                        }

                        CellData<1> edge;
                        edge.vertices[0] = map_vert_index[face->line(e)->vertex_index(0)];
                        edge.vertices[1] = map_vert_index[face->line(e)->vertex_index(1)];
                        edge.boundary_id = numbers::internal_face_boundary_id;
                        edge.manifold_id = face->line(e)->manifold_id();

                        subcell_data.boundary_lines.push_back (edge);
                    }


                cells.push_back(c_data);
                mapping.push_back(face);
            }
        }

    // create level 0 surface triangulation
    Assert (cells.size() > 0, ExcMessage ("No boundary faces selected"));
    surface_mesh.create_triangulation (vertices, cells, subcell_data);

    //  std::ofstream output(std::string("mesh_coarse.vtk"));
    //  DataOut<2,DoFHandler<2> > data_out;
    //  data_out.attach_triangulation(surface_mesh);
    //  data_out.build_patches();
    //  data_out.write_vtk(output);
    //  output.close();

    // Make the actual mapping
    for (typename Triangulation<dim-1>::active_cell_iterator
         cell = surface_mesh.begin(0);
         cell!=surface_mesh.end(0); ++cell)
    {
        surface_to_volume_mapping[cell] = mapping.at(cell->index());
    }

    // Store user pointers the actual mapping
    for (typename Triangulation<dim>::active_cell_iterator
         cell = volume_mesh.begin();
         cell!= volume_mesh.end(); ++cell)
    {
        for (unsigned int i=0; i < GeometryInfo<dim>::faces_per_cell; ++i)
        {
            const typename Triangulation<dim>::face_iterator
                    face = cell->face(i);

            if ( face->at_boundary() &&
                 face->boundary_id() == bid )
            {
                user_pointers_mapping[face] = cell->user_pointer();
            }
        }
    }

    do
    {
        bool changed = false;

        for (typename Triangulation<dim-1>::active_cell_iterator
             cell = surface_mesh.begin_active(); cell!=surface_mesh.end(); ++cell)
            if (surface_to_volume_mapping[cell]->has_children() == true )
            {
                cell->set_refine_flag ();
                changed = true;
            }

        if (changed)
        {
            surface_mesh.execute_coarsening_and_refinement();

            for (typename Triangulation<dim-1>::cell_iterator
                 surface_cell = surface_mesh.begin(); surface_cell!=surface_mesh.end(); ++surface_cell)
                for (unsigned int c=0; c<surface_cell->n_children(); c++)
                    if (surface_to_volume_mapping.find(surface_cell->child(c)) == surface_to_volume_mapping.end())
                    {
                        if(surface_cell->n_children() == surface_to_volume_mapping[surface_cell]->n_children())
                        {
                            surface_to_volume_mapping[surface_cell->child(c)]
                                    = surface_to_volume_mapping[surface_cell]->child(surface_to_volume_face_child_map[c]);
                        }
                        else
                        {
                            surface_to_volume_mapping[surface_cell->child(c)] =
                                    surface_to_volume_mapping[surface_cell];
                        }
                    }
        }
        else
            break;
    }
    while (true);

    //  for (typename Triangulation<dim-1>::cell_iterator
    //       cell = surface_mesh.begin_active(0);
    //       cell!=surface_mesh.end(0); ++cell)
    //  {
    //    if(cell->has_children())
    //      std::cout << cell->center(true) << "\t"
    //                << cell->child(0)->center(true) << "\t"
    //                << cell->child(1)->center(true) << "\t"
    //                << cell->child(2)->center(true) << "\t"
    //                << cell->child(3)->center(true) << "\n";
    //  }

    //  for (typename Triangulation<dim>::cell_iterator
    //       cell = volume_mesh.begin(0);
    //       cell!= volume_mesh.end(0); ++cell)
    //  {
    //    for (unsigned int i=0; i < GeometryInfo<dim>::faces_per_cell; ++i)
    //    {
    //      const typename Triangulation<dim>::face_iterator
    //          face = cell->face(i);

    //      if ( face->at_boundary() &&
    //           face->boundary_id() == bid )
    //      {
    //        if(face->has_children())
    //          std::cout << face->center(true) << "\t"
    //                    << face->child(0)->center(true) << "\t"
    //                    << face->child(1)->center(true) << "\t"
    //                    << face->child(2)->center(true) << "\t"
    //                    << face->child(3)->center(true) << "\n";
    //      }
    //    }
    //  }

    for (typename Triangulation<dim-1>::active_cell_iterator
         cell = surface_mesh.begin_active();
         cell!=surface_mesh.end(); ++cell)
    {
        auto it = surface_to_volume_mapping.find(cell);
        if(it == surface_to_volume_mapping.end())
            throw std::runtime_error("Could not find volume mesh cell");

        auto face = it->second;

        auto itf = user_pointers_mapping.find(face);
        if(itf == user_pointers_mapping.end())
            throw std::runtime_error("Could not find surface mesh cell pointer");

        if(face->center(true).distance(itf->first->center(true)) > 1e-10)
            std::cout << "Face -> surface cell mapping" << std::endl;

        cell->set_user_pointer(itf->second);

        //    std::cout << cell->center(true) << "\t" << cell->level() << "\t" << cell->active() << "\t" << cell->user_pointer() << std::endl;
    }

    return surface_to_volume_mapping;
}

template<int dim>
void copy_model_with_triangulation(const parallel::shared::Triangulation<dim> &triangulation,
                                   const PhysicalModelPtr<dim> &model,
                                   parallel::TriangulationBase<dim> &other_triangulation,
                                   PhysicalModelPtr<dim> &other_model)
{
    //  std::cout << "Copy tria.\n";
    //  std::cout << Utilities::MPI::this_mpi_process(other_triangulation.get_communicator())
    //            << std::endl;

    if(auto shared_tria = dynamic_cast<parallel::shared::Triangulation<dim>*>(&other_triangulation))
    {
        shared_tria->clear();
        shared_tria->copy_triangulation(triangulation);
    }
    else if(auto distributed_tria =
            dynamic_cast<parallel::fullydistributed::Triangulation<dim>*>(&other_triangulation))
    {
        distributed_tria->clear();
        distributed_tria->copy_triangulation(triangulation);
    }

    //  std::cout << "Original:\n";
    //  std::cout << Utilities::MPI::this_mpi_process(triangulation.get_communicator()) << "\t"
    //            << triangulation.n_active_cells() << "\t" << triangulation.n_locally_owned_active_cells() << "\t"
    //            << triangulation.get_communicator() << std::endl;

    //  MPI_Barrier(MPI_COMM_WORLD);

    //  std::cout << "Copied:\n";
    //  std::cout << Utilities::MPI::this_mpi_process(other_triangulation.get_communicator()) << "\t"
    //            << other_triangulation.n_active_cells() << "\t" << other_triangulation.n_locally_owned_active_cells() << "\t"
    //            << other_triangulation.get_communicator() << std::endl;

    other_model.reset(model->clone());
    other_model->transfer_cell_properties(other_triangulation);
}

/*
 * Constructs projection operator between coarse and fine meshes.
 * This function requires the fine mesh to be a refined version of
 * the coarse mesh. The later also has to be owned locally in its entirety.
 */
template<int dim, typename MATRIX>
void create_intergrid_transfer_operator (MATRIX &intergrid_operator,
                                         const DoFHandler<dim> &dof_coarse,
                                         const DoFHandler<dim> &dof_fine)
{
    // Create intergrid mapping, read doc of InterGridMap
    InterGridMap<DoFHandler<dim> > grid_1_to_2_map;

    grid_1_to_2_map.make_mapping (dof_fine, dof_coarse);
    std::vector<types::global_dof_index> fine_dof_index(dof_fine.get_fe().dofs_per_cell),
            coarse_dof_index(dof_coarse.get_fe().dofs_per_cell);

    // Save DoFs indices of the local cells on the coarse grid
    std::map<CellId, types::global_dof_index> local_dof_indices, global_dof_indices;
    for (auto cell_coarse = dof_coarse.begin_active ();
         cell_coarse != dof_coarse.end (); ++cell_coarse)
    {
        if(cell_coarse->is_locally_owned())
        {
            cell_coarse->get_dof_indices(coarse_dof_index);
            local_dof_indices[cell_coarse->id()] = coarse_dof_index[0];
        }
    }

    // Collect all DoFs on all processors
    MPI_Comm communicator = MPI_COMM_SELF;
    if(const typename parallel::TriangulationBase<dim> *parallel_tria =
            dynamic_cast<const parallel::TriangulationBase<dim>*>(&dof_coarse.get_triangulation()))
        communicator = parallel_tria->get_communicator();

    auto collected_dof_indices = Utilities::MPI::all_gather(communicator, local_dof_indices);
    for(auto &local_indices: collected_dof_indices)
        for(auto &p: local_indices)
            global_dof_indices.insert(p);

    // Fill the operator
    for (auto cell = dof_fine.begin_active (); cell != dof_fine.end (); ++cell)
    {
        if(cell->is_locally_owned())
        {
            typename DoFHandler<dim>::active_cell_iterator cell_coarse = grid_1_to_2_map[cell];
            if(cell_coarse->level() > cell->level())
                throw std::runtime_error("Fine and coarse meshes are not hierarchically related.");

            cell->get_dof_indices(fine_dof_index);

            auto it = global_dof_indices.find(cell_coarse->id());
            if(it == global_dof_indices.end())
                throw std::runtime_error("Cannot find global dof index of the coarse mesh.");

            //      const double inv_weight = 1. / pow(2., dim * (cell->level() - cell_coarse->level()));

            intergrid_operator.set(fine_dof_index[0], it->second, 1.);
        }
    }

    //  grid_1_to_2_map.make_mapping (dof_coarse, dof_fine);
    //  std::vector<std::map<types::global_dof_index, float> > transfer_representation;
    //  DoFTools::compute_intergrid_transfer_representation (dof_coarse, 0, dof_fine, 0,
    //                                                       grid_1_to_2_map, transfer_representation);
    //  for (size_t i = 0; i < transfer_representation.size(); ++i)
    //  {
    //    for (auto it = transfer_representation[i].begin(); it != transfer_representation[i].end(); ++it)
    //      intergrid_operator.set(i, it->first, it->second);
    //  }
}

}

#endif // MESH_TOOLS_H
