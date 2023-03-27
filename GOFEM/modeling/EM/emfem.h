#ifndef EMFEM_H
#define EMFEM_H

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_block_vector.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/numerics/data_out.h>

#ifndef SHARED_TRIANGULATION
#include <deal.II/distributed/fully_distributed_tria.h>
#else
#include <deal.II/distributed/shared_tria.h>
#endif

#include "common.h"
#include "physical_model/physical_model.h"
#include "survey/receiver.h"
#include "survey/dipole_source.h"
#include "functions/current_function.h"
#include "functions/solution_function.h"
#include "core/assembly.h"
#include "modelled_data.h"

using namespace dealii;

typedef std::map<std::string, std::vector<std::string>> SrcRecMap;

void create_petsc_sparse_matrix(PETScWrappers::MPI::SparseMatrix &matrix,
                                const MPI_Comm &communicator,
                                const types::global_dof_index  m,
                                const types::global_dof_index  n,
                                const unsigned int  local_rows,
                                const unsigned int  local_columns,
                                const unsigned int  n_nonzero_per_row,
                                const bool          is_symmetric = false,
                                const unsigned int  n_offdiag_nonzero_per_row = 0);

/*
 * Abstract class that mainly serves as parameters' storage
 * for FEM solvers
 */
template<int dim>
class EMFEM
{
    friend class ForwardProblemEM3D;
    friend class ForwardProblemEM2D;
    friend class ForwardCalculatorMT3D;

public:
    EMFEM(MPI_Comm comm,
          const unsigned int order,
          const unsigned int mapping_order,
          const PhysicalModelPtr<dim> &model,
          const BackgroundModel<dim> &bg_model,
          const FieldFormulation formulation,
          bool face_orientation);
    EMFEM(MPI_Comm comm,
          const unsigned int order,
          const unsigned int mapping_order,
          const PhysicalModelPtr<dim> &model,
          const FieldFormulation formulation,
          bool face_orientation);
    virtual ~EMFEM();

    // Clear object for next run
    virtual void clear ();

    // Runs modelling step that computes solutiond for the two
    // source polarizations and calculates magnetic fields afterwards
    virtual void run ();

    // If f = true the programm ensures there are no coarser
    // neighbors after grid refinement
    void set_no_coarser_neighbors_around_receivers (bool f);

    void set_estimate_error_on_last_cycle (bool f);

    // Sets frequency in Hz that we want to calculate fields for
    void set_frequency (double f);

    // Sets different user defined parameters
    void set_boundary_conditions_id (const std::string& id);
    void set_adaptivity_type_id (const std::string& id);
    void set_refinement_strategy_id (const std::string& id);
    void set_theta (const double t);
    void set_initial_refinements (const unsigned n);
    void set_refinement_steps (const unsigned n);
    void set_target_error_reduction (const double n);
    void set_dofs_budget (const unsigned n);
    void set_output_type (const std::string type);
    void set_output_format (const std::string format);
    void set_output_mesh_order (const unsigned order);
    void set_parallel_output(bool f);
    void set_physical_model (const PhysicalModel<dim>& model);
    void set_background_model(const BackgroundModel<dim> &model);
    virtual void set_receivers (const std::vector<Receiver>& recs);
    virtual void set_physical_sources (const std::vector<PhysicalSourcePtr> &sources);
    void set_data_map(const SrcRecMap &data_mapping = SrcRecMap());
    void set_output_file (const std::string fname);
    void set_refinement_around_receivers (const unsigned cycles);
    void set_preconditioner_type (const std::string preconditioner);
    void set_maximum_solver_iterations (const unsigned max_it);
    void set_solver_residual (const double residual);
    void set_adjoint_solver_residual (double residual);
    void set_inner_solver_residual (const double residual);
    void set_inner_solver_maximum_iterations (const unsigned max_it);
    void set_perform_topography_adjustment(bool f);
    void set_verbosity(bool flag);
    void set_reuse_data_structures(bool f);
    void set_local_model(const PhysicalModelPtr<dim> &model);

    // Get source-receiver map
    const SrcRecMap& get_data_map() const;
    // Return calculated data according to the source-receiver map
    virtual void get_survey_data(ModelledData<dcomplex> &modelled_data) const;

    PhysicalSourcePtr get_physical_source(const std::string &srcname);

protected:
    // Mesh-related routines
    virtual void create_grid (const unsigned int cycle);
    void setup_initial_triangulation(PhysicalModelPtr<dim> &model,
                                     Triangulation<dim> &tria);
    virtual void set_boundary_indicators(Triangulation<dim> &tria);

    virtual double get_wavenumber() const;

    // System matrix setup and assembly
    virtual void assemble_system_matrix () = 0;
    virtual void copy_local_to_global_system (const Assembly::CopyData::MaxwellSystem& data) = 0;
    virtual void setup_system (const unsigned n_rhs) = 0;
    virtual void set_boundary_values () = 0;
    virtual unsigned get_number_of_constraint_matrices() const = 0;
    virtual void setup_preconditioner ();
    void zero_matrices_and_vectors();

    virtual void assemble_problem_rhs () = 0;
    virtual void assemble_dipole_rhs_vector (const DipoleSource &phys_source,
                                             const AffineConstraints<double> &constraints,
                                             PETScWrappers::MPI::BlockVector& rhs_vector);

    virtual void estimate_error () = 0 ;

    virtual void solve (std::vector<PETScWrappers::MPI::BlockVector> &solution_vectors,
                        std::vector<PETScWrappers::MPI::BlockVector> &rhs_vectors,
                        const std::vector<unsigned> &constraints_indices,
                        bool adjoint = false, bool verbose = true, unsigned start_index = 0) = 0;

    // This function is called right after solve(...). Inherited classes can
    // reimplement it to carry out some post-solve operations.
    virtual void post_solve();

    // Output of results at different stages
    virtual void output_results (const unsigned int cycle) const;

    void output_volume_data(const unsigned int cycle) const;

    // Allows derived classes to add specific vectors to output
    virtual void output_specific_information
    (std::vector<std::shared_ptr<DataPostprocessor<dim>>> &data,
     DataOut<dim> &data_out) const;

    virtual void output_surface_data(unsigned cycle) const;

    // Outputs data taking into account source-receiver data map
    virtual void output_point_data (const unsigned int cycle) const;
    virtual void output_point_data (const std::vector<cvector> &data,
                                    const unsigned int cycle) const;
    // Returns requested data at receiver locations
    virtual std::vector<cvector> data_at_receivers () const;
    virtual std::string data_header () const = 0;
    virtual unsigned n_data_at_point () const = 0;

    virtual void field_at_point (const typename DoFHandler<dim>::active_cell_iterator& cell,
                                 const Point<dim>& p, cvector& E, cvector& H) const = 0;

    virtual cvector calculate_data_at_receiver (const std::vector<cvector> &E,
                                                const std::vector<cvector>& H) const = 0;

    // Collect field values at receiver locations on root process. Note that receivers can
    // consist of several electrodes, which reside in different cells, hence the resulting
    // array can have more elements than number of receivers.
    typedef std::map<std::string, std::vector<cvector>> StringVectorMap;
    void collect_fields_on_root(StringVectorMap& E, StringVectorMap& H) const;
    void collect_fields_on_root_cache(StringVectorMap& E, StringVectorMap& H) const;

    // This method takes a set of locations and finds locally owned cells for all locations
    // Each element of the input vector contains locations for a single receiver
    // Each element of the output vector contains iterators to locally owned cells.
    // If a receiver has no locally owned cells on this MPI rank, the returned iterator will
    // have an invalid state.
    std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator> >
    find_all_receiver_positions_cells(const std::vector<std::vector<Point<dim>>> &receiver_locations) const;

    void construct_survey_data(const std::vector<cvector>& data, ModelledData<dcomplex> &modelled_data) const;

    BackgroundModel<dim> get_boundary_model(const PhysicalModelPtr<dim> &model,
                                            const Triangulation<dim> &tria,
                                            const types::boundary_id bid) const;

    unsigned get_source_index(const std::string &name) const;
    unsigned get_receiver_index(const std::string &name) const;

    void get_unique_electrodes(const std::vector<Point<dim>> &positions,
                               std::vector<Point<dim> > &unique_positions,
                               double min_distance);

    void print_memory_consumption() const;

protected:
    // Order of the quadrature used for modelling
    unsigned quadrature_order;

    MPI_Comm mpi_communicator;
    const unsigned int this_mpi_process;

    ApproachType approach_type;
    FieldFormulation formulation;

#ifdef SHARED_TRIANGULATION
    parallel::shared::Triangulation<dim> triangulation;
#else
    parallel::fullydistributed::Triangulation<dim> triangulation;
    Triangulation<dim> local_copy_triangulation;
#endif
    DoFHandler<dim> dof_handler;
    FESystem<dim> fe;
    MappingQGeneric<dim> mapping;

    // We need to have multiple constraint matrcies in case
    // inhomogeneous constraints are required for Dirichlet BCs.
    // Note: last element of this matrix always stores homogeneous matrix.
    std::vector<AffineConstraints<double>> constraints;
    // Every solution vector has an individual (but possibly shared) constraint
    // matrix from "constraints" vector. This array stores indices to matrices.
    std::vector<unsigned> solution_constraints_indices;

    // Number of physical sources (dipoles, plane waves, etc.)
    unsigned n_physical_sources;
    std::vector<CurrentFunctionPtr<dim>> current_functions;
    std::vector<PhysicalSourcePtr> physical_sources;

    // Background solutions for all sources used for
    // inhomogeneous BCs or secondary source calculations
    std::vector<std::shared_ptr<SolutionFunction<3>>> background_solutions;

    // A list of receivers
    std::vector<Receiver> receivers;
    // A list of unique point positions at which solution is sampled
    std::vector<Point<dim>> unique_point_receiver_positions;

    // Definition of background and main models
    PhysicalModelPtr<dim> phys_model;
    // Background model (used for secondary field approach)
    BackgroundModel<dim> background_model;

#ifndef SHARED_TRIANGULATION
    // Fully locally owned copy of the model (needed for inhomogeneous BCs)
    PhysicalModelPtr<dim> local_copy_model;
#endif

    // Linear algebra objects
    PETScWrappers::MPI::BlockSparseMatrix         system_matrix;
    std::vector<PETScWrappers::MPI::BlockVector>  system_rhs;
    mutable std::vector<PETScWrappers::MPI::BlockVector> ghosted_rhs_vectors;
    std::vector<PETScWrappers::MPI::BlockVector>  solutions;
    PETScWrappers::MPI::BlockVector               completely_distributed_solution;

    // Information on locally owned and relevant index partitionings
    std::vector<IndexSet> locally_owned_partitioning,
    locally_relevant_partitioning;

    BoundaryConditions boundary_conditions;

    // Mesh related parameters
    AdaptivityStrategy adaptivity_type;
    RefinementStrategy refinement_strategy;
    unsigned initial_refinements;
    unsigned refinement_steps;
    unsigned refine_around_receivers;
    double theta;
    double target_error_reduction;
    // If true then error is estimated at the last refinement cycle
    // otherwise error estimation is skipped and thus some time is saved
    bool estimate_error_on_last_cycle;
    // If true, the mesh gets transformed in accordance with
    // input topography/bathymetry files
    bool fit_mesh_to_topography;
    // If true, the programm ensures there are no coarser neighbors
    // around receivers' cells after refinement (should be true for inversion)
    bool ensure_no_coarser_neighbors;
    // Maximum number of DoFs allowed
    unsigned n_maximum_dofs;

    // Estimated global solution errors for all sources and cycles
    std::vector<std::vector<double> > error_estimates;
    // Estimated cell-based errors for all sources
    std::vector<Vector<double>>  estimated_cell_errors;

    // Solver parameters
    PreconditionerType preconditioner_type;
    unsigned max_iterations, max_inner_iterations;
    double solver_residual, adjoint_residual; // Normalized residual
    double inner_solver_residual;
    // If AutoSelection type of solver is chosen, this threshold specifies
    // number of DoFs at which programm switches between direct and iterative
    // solvers
    unsigned direct_solver_dofs_threshold;

    // Frequency being modelled in Hz
    double frequency;

    // Output-related variables
    ConditionalOStream pcout;
    std::string output_type;
    std::string output_format;
    std::string output_data_file;
    unsigned mesh_order;
    bool parallel_output;

    // If true, triangulation and all matrices and system vectors are reused upon
    // calling run() method.
    bool reuse_data_structures;

    // For each source (key) we store an array of receiver names
    // for which data need to be calculated. This source-receiver
    // map can be read from a file and in case we work with many
    // source-receiver layouts, this can result in substantial
    // computational savings.
    std::map<std::string, std::vector<std::string>> data_map;
};

#endif // EMFEM_H

