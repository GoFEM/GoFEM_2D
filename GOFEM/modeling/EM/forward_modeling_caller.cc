#include "forward_modeling_caller.h"

#include "mpi/counter.h"
#include "mpi/makesub.h"
#include "io/parameter_reader.h"
#include "io/read_list.h"
#include "io/read_data_map.h"
#include "survey/receiver.h"
#include "physical_model/physical_model.h"
#include "2D/mt2dfem.h"
#include "2D/csem25dfem.h"
#include "functions/exact_solution.h"

#include <fstream>
#include <boost/algorithm/string.hpp>

template<int dim>
void calc_whole_space(const std::vector<DipoleSource> &sources,
                      const std::vector<Receiver> &receivers,
                      const dvector &frequencies, double sigma,
                      std::ostream &os)
{
  for(size_t k = 0; k < frequencies.size(); ++k)
    for(size_t i = 0; i < sources.size(); ++i)
      for(size_t j = 0; j < receivers.size(); ++j)
      {
        ExactSolutionCSEMSpace exs(sources[i], sigma, eps0, frequencies[k], PhaseLead);
        Point<3> p = receivers[j].position<Point<3>>(0);
        Vector<double> E(6), H(6);
        exs.vector_value(p, E);
        exs.set_field(HField);
        exs.vector_value(p, H);

        os << sources[i].get_name() << "\t" << receivers[j].get_name() << "\t"
           << dcomplex(E[0], E[3]) << "\t" << dcomplex(E[1], E[4]) << "\t" << dcomplex(E[2], E[5]) << "\t"
           << dcomplex(H[0], H[3]) << "\t" << dcomplex(H[1], H[4]) << "\t" << dcomplex(H[2], H[5]) << "\n";
      }
}

template<int dim>
void calc_1D_MT(const std::vector<Receiver> &receivers,
                const dvector &frequencies,
                const BackgroundModel<dim> &bgmodel)
{
  std::vector<PlaneWavePolarization> polarizations = {NS, EW};
  Tensor<1, dim, dcomplex> E, H;

  for(size_t k = 0; k < frequencies.size(); ++k)
  {
    ExactSolutionMT1D<dim> exs(bgmodel, frequencies[k], polarizations[0], PhaseLead, EField);
    std::ofstream ofs("MT_fields_" + std::to_string(frequencies[k]) + ".txt");

    for(size_t j = 0; j < receivers.size(); ++j)
    {
      Point<dim> point = receivers[j].position<Point<dim>>(0);
      ofs << point << "\t";

      for(size_t p = 0; p < 2; ++p)
      {
        exs.set_polarization(polarizations[p]);
        exs.electric_field(point, E);
        exs.magnetic_field(point, H);

        ofs << E << "\t" << H << "\t";
      }

      ofs << "\n";
    }
  }
}

template<int dim>
EMFEM<dim>* create_em_modeling(ParameterHandler &prm,
                               unsigned fe_order,
                               unsigned mapping_order,
                               MPI_Comm communicator,
                               const PhysicalModelPtr<dim> &model,
                               BackgroundModel<dim> &bg_model,
                               const char* method_str,
                               bool standard_orientation)
{
  throw std::runtime_error("Wrong dimension. Use 2 or 3.");
}

template<>
EMFEM<2>* create_em_modeling(ParameterHandler &prm,
                             unsigned fe_order,
                             unsigned mapping_order,
                             MPI_Comm communicator,
                             const PhysicalModelPtr<2> &model,
                             BackgroundModel<2> &bg_model,
                             const char* method_str,
                             bool standard_orientation)
{
  unsigned rank = Utilities::MPI::this_mpi_process (MPI_COMM_WORLD);

  EMFEM<2> *em_modeling;

  if(boost::iequals(method_str, "MT"))
  {
    em_modeling = new MT2DFEM (communicator, fe_order, mapping_order, model);
#ifndef SHARED_TRIANGULATION
    PhysicalModelPtr<2> local_model(model->clone());
    em_modeling->set_local_model(local_model);
#endif
  }
  else if(boost::iequals(method_str, "CSEM"))
  {
    prm.enter_subsection ("Survey parameters");

    const std::string srcs_file = prm.get ("Sources file");
    std::vector<DipoleSource> sources;
    if(srcs_file.length() > 1)
      sources = read_list<DipoleSource> (srcs_file);
    else
      throw std::runtime_error("You have to specify source file for CSEM modeling.");

    prm.leave_subsection ();

    prm.enter_subsection ("2.5D parameters");
    const std::string filter_file = prm.get ("Strike filter");
    const unsigned nk = prm.get_integer("Number of wavenumbers");
    const std::string wave_range = prm.get ("Minimum and maximun wavenumbers");
    std::vector<double> range = Utilities::string_to_double(Utilities::split_string_list(wave_range));
    prm.leave_subsection ();

//    prm.enter_subsection ("Survey parameters");
//    const std::string freqs_file = prm.get ("Frequencies file");
//    const dvector frequencies = read_list<double> (freqs_file);
//    const std::string recvs_file = prm.get ("Stations file");
//    const std::vector<Receiver> receivers = read_list<Receiver> (recvs_file);
//    prm.leave_subsection ();
//    calc_whole_space<3>(sources, receivers, frequencies, 0.1, std::cout);

    CSEM25DFEM* csem2d = new CSEM25DFEM (communicator, fe_order, mapping_order, model, bg_model);
    csem2d->set_dipole_sources(sources);
    csem2d->set_digital_filter_file(filter_file);
    csem2d->set_wavenumber_range(range[0], range[1], nk);
    //csem2d->set_verbosity(false);
    em_modeling = csem2d;
  }
  else
    throw std::runtime_error(std::string("Unsupported method ") + method_str);

  if(rank == 0)
    std::cout << "Start 2D " << method_str << " modeling.\n";

  return em_modeling;
}

template<int dim>
void do_fd_forward_modeling (const char *parameter_file, const char *method)
{
  /*
   * Read input data
   */
  ParameterHandler  prm;
  ParameterReader   param(prm);
  param.read_parameters (parameter_file);

  prm.enter_subsection ("Modeling parameters");
  const unsigned int fe_order = prm.get_integer ("Order");
  const unsigned int mapping_order = prm.get_integer ("Mapping order");
  const std::string bc_id = prm.get ("BC");
  const std::string adaptivity_strategy = prm.get ("Adaptive strategy");
  const std::string refinement_strategy = prm.get ("Refinement strategy");
  const double theta = prm.get_double ("Theta");
  const unsigned int refinement_steps = prm.get_integer ("Number of refinements");
  const unsigned int initial_steps = prm.get_integer ("Number of initial refinements");
  const unsigned int maximum_dofs = prm.get_integer ("DoFs budget");
  const double error_reduction = prm.get_double ("Error reduction");
  const unsigned int n_freq_parallel = prm.get_integer ("Number of parallel frequencies");
  const unsigned int refine_around_receivers = prm.get_integer ("Refine cells around receivers");
  const bool standard_orientation = prm.get_bool("Standard orientation");
  prm.leave_subsection ();


  prm.enter_subsection ("Solver parameters");
  const std::string preconditioner_type = prm.get ("Preconditioner");
  const unsigned int max_iterations = prm.get_integer ("Iterations");
  const double residual = prm.get_double ("Residual");
  const double preconditioner_residual = prm.get_double ("Preconditioner residual");
  const unsigned int max_inner_iterations = prm.get_integer ("Preconditioner iterations");
  prm.leave_subsection ();


  prm.enter_subsection ("Model parameters");
  const std::string model_file = prm.get ("Model definition file");
  const std::string bg_model_file = prm.get ("Background model definition file");
  const std::string materials_file = prm.get ("Materials definition file");
  std::string bg_materials_file = prm.get ("Background materials definition file");

  if(bg_materials_file == "")
    bg_materials_file = materials_file;

  prm.leave_subsection ();

  BackgroundModel<dim> bg_model;
  if(bg_model_file.length() > 1)
    bg_model = BackgroundModel<dim>(bg_model_file, bg_materials_file);

  PhysicalModelPtr<dim> model;
  if (model_file.find(".xyz") != std::string::npos)
    model.reset(new XYZModel<dim> (model_file, materials_file));
  else if (model_file.find(".tria") != std::string::npos)
    model.reset(new TriangulationModel<dim> (model_file, materials_file));
  else
    throw std::runtime_error("The mesh file " + model_file + " has unknown extension.");

  prm.enter_subsection ("Survey parameters");
  const std::string freqs_file = prm.get ("Frequencies file");
  const dvector frequencies = read_list<double> (freqs_file);

  const std::string recvs_file = prm.get ("Stations file");
  const std::vector<Receiver> receivers = read_list<Receiver> (recvs_file);

  const std::string map_file = prm.get ("Sources-receiver map");
  SrcRecMap data_map;
  if(map_file.length() > 1)
    read_data_map(map_file, data_map);
  prm.leave_subsection ();

  prm.enter_subsection ("Output parameters");
  const std::string output_type = prm.get ("Type");
  const std::string output_file = prm.get ("Data file");
  unsigned output_mesh_order = prm.get_integer("Mesh order");
  bool parallel_output = prm.get_bool("Parallel output");
  prm.leave_subsection ();

  /*
   * Create subcommunicators and task manager for parallelization over frequencies
   */
  unsigned int nproc = Utilities::MPI::n_mpi_processes (MPI_COMM_WORLD);
  if (nproc % n_freq_parallel)
  {
    throw std::runtime_error("Error in user input: number of processes is not a"
                             "multiple of number of parallel frequencies.");
  }

  MPI_Comm commfreq;
  int this_frequency_group;
  split_communicator (MPI_COMM_WORLD, n_freq_parallel, commfreq, this_frequency_group, false);
  Counter counter (MPI_COMM_WORLD, frequencies.size ());

  /*
   * Run loop over all frequencies, calculate numerical solutions and
   * extract responses at receiver locations
   */
  int pid, subpid;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_rank(commfreq, &subpid);

  // Print parameters
  if(pid == 0)
    prm.print_parameters (std::cout, ParameterHandler::ShortText);

  /*
   * Create object that handles numerical solution and set input parameters
   */
  EMFEM<dim>* em_modeling = create_em_modeling<dim>(prm, fe_order,
                                                    mapping_order, commfreq,
                                                    model, bg_model, method,
                                                    standard_orientation);

  // Mesh related parameters
  em_modeling->set_boundary_conditions_id (bc_id);
  em_modeling->set_adaptivity_type_id (adaptivity_strategy);
  em_modeling->set_refinement_strategy_id(refinement_strategy);
  em_modeling->set_theta (theta);
  em_modeling->set_initial_refinements (initial_steps);
  em_modeling->set_dofs_budget(maximum_dofs);
  em_modeling->set_refinement_steps (refinement_steps);
  em_modeling->set_target_error_reduction (error_reduction);
  em_modeling->set_refinement_around_receivers (refine_around_receivers);
  em_modeling->set_no_coarser_neighbors_around_receivers (false);

  // Solver parameters
  em_modeling->set_preconditioner_type (preconditioner_type);
  em_modeling->set_maximum_solver_iterations (max_iterations);
  em_modeling->set_solver_residual (residual);
  em_modeling->set_inner_solver_residual (preconditioner_residual);
  em_modeling->set_inner_solver_maximum_iterations(max_inner_iterations);

  // Parameters for output
  em_modeling->set_output_type (output_type);
  em_modeling->set_output_file (output_file);
  em_modeling->set_output_mesh_order (output_mesh_order);
  em_modeling->set_parallel_output(parallel_output);
  em_modeling->set_receivers (receivers);
  em_modeling->set_data_map(data_map);

  while(true)
  {
    unsigned int frequency_index;

    if(this_frequency_group == 0)
    {
      // Distribute frequencies if I am the master
      if(pid == 0)
      {
        // Service all groups except this one
        counter.service (n_freq_parallel - 1);

        // Get one frequency for this group
        frequency_index = counter.increment ();
      }

      // Distribute frequency index within the group
      int ierr = MPI_Bcast(&frequency_index, 1, MPI_INTEGER, 0, commfreq);
      check_mpi_error(ierr);

      if (frequency_index > frequencies.size ())
      {
        if (pid == 0)
          counter.finish(n_freq_parallel);
        break;
      }
    }
    else // Request frequency otherwise
    {
      // Get a frequency from process 0
      if (subpid == 0)
        frequency_index = counter.request ();

      // Distribute frequency index within the group
      int ierr = MPI_Bcast(&frequency_index, 1, MPI_INTEGER, 0, commfreq);
      check_mpi_error(ierr);

      if (frequency_index > frequencies.size ())
        break;
    }

    if(subpid == 0)
        std::cout << "Solve problem for frequency " << frequencies[frequency_index - 1] <<  " Hz" << std::endl;
    em_modeling->set_frequency (frequencies[frequency_index - 1]);
    em_modeling->run ();
    em_modeling->clear ();
  }

  delete em_modeling;
}

template void do_fd_forward_modeling<2>(const char *parameter_file, const char *method);
