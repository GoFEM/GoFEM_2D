#include "parameter_reader.h"

#include <fstream>

#include <deal.II/numerics/data_out.h>

ParameterReader::ParameterReader (ParameterHandler &paramhandler)
  : prm (paramhandler)
{}

void ParameterReader::declare_parameters ()
{
  prm.enter_subsection ("Modeling parameters");
  {
    prm.declare_entry ("Adaptive strategy", "global",
                       Patterns::Anything(),
                       "Adaptivity type (global, residual-based, goal-oriented)");

    prm.declare_entry ("Theta", "0.12",
                       Patterns::Double(0.0, 1.0),
                       "Fraction of refinement (1.0 -- global refinement)");

    prm.declare_entry ("Refinement strategy", "Number",
                       Patterns::Anything(),
                       "Refinement strategy (refine fixed Fraction or Number).");

    prm.declare_entry ("Number of refinements", "0",
                       Patterns::Integer(0, 100),
                       "Number of mesh refinement steps "
                       "applied to the grid");

    prm.declare_entry ("Error reduction", "1000",
                       Patterns::Double(1., 1e13),
                       "Relative error reduction to reach during adaptive refinement");

    prm.declare_entry ("DoFs budget", "10000000",
                       Patterns::Integer(1000, 1000000000),
                       "Maximum number of degrees of freedom allowed");

    prm.declare_entry ("Number of initial refinements", "0",
                       Patterns::Integer(0, 10),
                       "Number of global mesh refinement steps "
                       "applied to initial coarse grid");

    prm.declare_entry ("Order", "1",
                       Patterns::Integer(1, 10),
                       "Polynomial order of FE basis functions");

    prm.declare_entry ("Mapping order", "1",
                       Patterns::Integer(1, 4),
                       "Polynomial order of mapping transformation");

    prm.declare_entry ("BC", "Dirichlet",
                       Patterns::Anything(),
                       "Type of boundary conditions applied (Dirichlet, Neumann, Dirichlet2D)");

    prm.declare_entry ("Number of parallel frequencies", "1",
                       Patterns::Integer(1),
                       "Number of frequencies that will be solved in parallel");

    prm.declare_entry ("Refine cells around receivers", "0",
                       Patterns::Integer(0),
                       "Refine cells around receiver stations (> 0 - nr. cycles, 0 - no). "
                       "For CSEM also around sources. This helps increase accuracy.");

    prm.declare_entry ("Field formulation", "E",
                       Patterns::Anything(),
                       "Specifies which formulation to use (E, EStabilized or H, HStabilized).");

    prm.declare_entry ("Field approach", "Total",
                       Patterns::Anything(),
                       "Specifies which formulation to use (Scattered, Total).");

    prm.declare_entry ("Source type", "SH",
                       Patterns::Anything(),
                       "Specifies how to parameterize MT source on a sphere (SH, Sheet).");

    prm.declare_entry ("Standard orientation", "true",
                       Patterns::Bool(),
                       "Specify if all faces in the mesh have standard orientation.");
  }
  prm.leave_subsection ();

  prm.enter_subsection ("2.5D parameters");
  {
    prm.declare_entry ("Minimum and maximun wavenumbers", "1e-5,1e-1",
                       Patterns::List(Patterns::Double(1e-10, 100.), 2, 2),
                       "Minimum and maximum spatial wavenumber values");

    prm.declare_entry ("Number of wavenumbers", "30",
                       Patterns::Integer(8, 100),
                       "Number of wave numbers used to integrate along strike direction.");

    prm.declare_entry ("Strike filter", "",
                       Patterns::Anything(),
                       "Path to the file storing sin-cos digital filter coefficients for k2x transformation.");

    prm.declare_entry ("Time filter", "",
                       Patterns::Anything(),
                       "Path to the file storing sin-cos digital filter coefficients for f2t transformation.");
  }
  prm.leave_subsection ();

  prm.enter_subsection ("Solver parameters");
  {
    prm.declare_entry ("Preconditioner", "Direct",
                       Patterns::Anything(),
                       "Type of preconditioner for the system (Direct, CGAMS, AMS, AutoSelection)");

    prm.declare_entry ("Iterations", "100",
                       Patterns::Integer(1, 1000),
                       "Maximum number of the outer solver iterations");

    prm.declare_entry ("Residual", "1e-9",
                       Patterns::Double(1e-14, 1.),
                       "Normalized residual for the outer solver in forward modeling");

    prm.declare_entry ("Adjoint residual", "1e-6",
                       Patterns::Double(1e-14, 1.),
                       "Normalized residual for the outer solver in adjoint/inverse modeling");

    prm.declare_entry ("Preconditioner iterations", "30",
                       Patterns::Integer(1, 100),
                       "Maximum number of the inner solver iterations (relevant for AMS preconditioner only)");

    prm.declare_entry ("Preconditioner residual", "1e-2",
                       Patterns::Double(1e-14, 1.),
                       "Normalized residual to achieve during inner solve step (relevant for AMS preconditioner only)");
  }
  prm.leave_subsection ();

  prm.enter_subsection ("Model parameters");
  {
    prm.declare_entry ("Model definition file", "",
                       Patterns::Anything(),
                       "Mesh file contains geometry and topology definition");

    prm.declare_entry ("Inversion model definition file", "",
                       Patterns::Anything(),
                       "Mesh file contains geometry and topology definition "
                       "for parameter grid. If empty, then use Model definition file");

    prm.declare_entry ("Materials definition file", "",
                       Patterns::Anything(),
                       "Contains actual values of the physical properties "
                       "for each cell from the mesh file");

    prm.declare_entry ("Inversion materials definition file", "",
                       Patterns::Anything(),
                       "Contains actual values of the physical properties "
                       "for each cell from the inversion mesh file");

    prm.declare_entry ("Background model definition file", "",
                       Patterns::Anything(),
                       "File contains conductivities, permittivities"
                       "and layers' top depths of the layered background "
                       "model used to compute incident field (for MT 3D only)");

    prm.declare_entry ("Background materials definition file", "",
                       Patterns::Anything(),
                       "Contains actual values of the physical properties "
                       "corresponding to ids from the background file");

    prm.declare_entry ("Active domain box", "0,0,0,0,0,0",
                       Patterns::List(Patterns::Double()),
                       "Coordinates (x1,y1,z1,x2,y2,z2 in meters) of the box within which "
                       "cells will be treated as active during the inversion "
                       "(note, cells in the air are ignored even if they lie within the box).");

    prm.declare_entry ("Active domain mask", "",
                       Patterns::Anything(),
                       "This file lists cell flags which identify which cells are free to "
                       "invert for. Note that this file is optional and comes in addition to "
                       "the domain box.");

    prm.declare_entry ("Cell weights", "",
                       Patterns::Anything(),
                       "This file specifies cell weights which specify how much cell contributes to"
                       "the penalty term.");
  }
  prm.leave_subsection ();

  prm.enter_subsection ("Mesh generation");
  {
    prm.declare_entry ("Output triangulation file", "",
                       Patterns::Anything(),
                       "This file specifies resulting refined and deformed"
                       "triangulation with.");

    prm.declare_entry ("Output material file", "",
                       Patterns::Anything(),
                       "This file specifies file containing all material properties.");

    prm.declare_entry ("Topography definition file", "",
                       Patterns::Anything(),
                       "File that specifies topography which the "
                       "mesh will be adapted to (empty -- no topography).");

    prm.declare_entry ("Coastline definition file", "",
                       Patterns::Anything(),
                       "WKT format file containing multi-polygon specifying"
                       "coastline (empty -- no coastline).");

    prm.declare_entry ("Refinement script", "",
                       Patterns::Anything(),
                       "File that specifies refinement commands with parameters.");

    prm.declare_entry ("Project receivers on topography", "false",
                       Patterns::Bool(),
                       "If true, projects receivers on topography making them slighly below "
                       "surface on land or slighly above seafloor offshore. Otherwise, "
                       "take vertical positions from the receiver file.");
  }
  prm.leave_subsection ();

  prm.enter_subsection ("Survey parameters");
  {
    prm.declare_entry ("Frequencies file", "frequencies",
                       Patterns::Anything(),
                       "Frequencies file lists at which frequencies "
                       "we want to compute solution (in Hz)");

    prm.declare_entry ("Times file", "",
                       Patterns::Anything(),
                       "Time file lists at what times (in seconds)"
                       "we want to compute solution. Relevant for time-domain modeling only.");

    prm.declare_entry ("Stations file", "receivers",
                       Patterns::Anything(),
                       "Stations file lists positions of receiver "
                       "stations where we extract solution");

    prm.declare_entry ("Sources file", "",
                       Patterns::Anything(),
                       "Source file lists positions of sources "
                       "and their types");

    prm.declare_entry ("Sources-receiver map", "",
                       Patterns::Anything(),
                       "Lists pairs of source-receivers for which "
                       "time-domain data need to be calculated.");
  }
  prm.leave_subsection ();

  prm.enter_subsection ("Output parameters");
  {
    prm.declare_entry ("Type", "",
                       Patterns::Anything(),
                       "Type of output (point,surface,volume)");

    prm.declare_entry ("Data file", "output_data",
                       Patterns::Anything(),
                       "Name of the output file for calculated "
                       "transfer functions (without extension).");

    prm.declare_entry ("Mesh order", "1",
                       Patterns::Integer(1, 4),
                       "Order of the VTK mesh (use > 1 for HO elements or manifolds)");

    prm.declare_entry ("Parallel output", "true",
                       Patterns::Bool(),
                       "If true, each process outputs its chunk, otherwise output "
                       "is collected on the master");

    DataOutInterface<1>::declare_parameters (prm);
  }
  prm.leave_subsection ();

  prm.enter_subsection ("Inversion parameters");
  {
    prm.declare_entry ("Inversion input data", "invdata.dat",
                       Patterns::Anything(),
                       "Name of the file containing observed data");

    prm.declare_entry ("Number of iterations", "20",
                       Patterns::Integer(0, 1000),
                       "Number of outer Gauss-Newton iterations.");

    prm.declare_entry ("Target RMS", "1.0",
                       Patterns::Double(0., 1000.),
                       "Target normalized root-mean square value.");

    prm.declare_entry ("Scaling factor", "1.",
                       Patterns::List(Patterns::Double(0., 1.0e4)),
                       "Scaling factors (0. means only implicit LSQR regularization) "
                       "applied at each iteration. At i-th iteration, we take min(i, N)-tn "
                       "element of this vector, where N is the size of the vector.");

    prm.declare_entry ("Regularization operator", "Identity",
                       Patterns::Anything(),
                       "Type of regularization: Identity, Laplacian or Roughness.");

    prm.declare_entry ("Regularization update", "1",
                       Patterns::Double(1, 100),
                       "Recalculate regularization at each n-th iteration");

    prm.declare_entry ("Use starting model as reference", "false",
                       Patterns::Bool(),
                       "When true, use starting model as the reference model "
                       "in the regularization term ||R*(m - m_ref)||_2");

    prm.declare_entry ("Face weighting", "1,1,1",
                       Patterns::List(Patterns::Double()),
                       "Relative scalar weighting on the smoothing in x/y/z directions.");

    prm.declare_entry ("Non-conforming interface weighting", "1.",
                       Patterns::Double(0.01, 100.),
                       "Weighting across non-confirming refinement interfaces.");

    prm.declare_entry ("Cell weighting", "0,0,0,1.",
                       Patterns::List(Patterns::Double()),
                       "List of four numbers rx,ry,rz,w which specify weighting w "
                       "for cells located less than ellipsoid(rx,ry,rz) meters away "
                       "from a receiver. Leave it empty if you don't want this function.");

    prm.declare_entry ("Model transformation", "LOG",
                       Patterns::Anything(),
                       "Type of model parameter transformation (LOG - natural logarithm, "
                       "BOUNDED - bounded log-transform, BANDPASS - band-pass transform, "
                       "LINEAR - linear tranformation).");

    prm.declare_entry ("Minimum conductivity", "1e-5",
                       Patterns::Double(1e-6, 1.0e3),
                       "In case one uses BOUNDED transform this parameter "
                       "specifies minimum conductivity value.");

    prm.declare_entry ("Maximum conductivity", "1e2",
                       Patterns::Double(1e-6, 1.0e3),
                       "In case one uses BOUNDED or BANDPASS transform "
                       "this parameter specifies maximum conductivity value.");

    prm.declare_entry ("Number of inner iterations", "20",
                       Patterns::Integer(1, 500),
                       "Number of inner iterations for solving linearized system.");

    prm.declare_entry ("Inner solver type", "Krylov",
                       Patterns::Anything(),
                       "Solver linearized system (Krylov -- CG, LSQR, Decomposition -- EVD, SVD).");

    prm.declare_entry ("Refine initial grid nr. times", "0",
                       Patterns::Integer(0, 10),
                       "Defines how many times the parameter grid will be refined "
                       "using resolution matrix estimates");

    prm.declare_entry ("Refine grid every nr. iterations", "100",
                       Patterns::Integer(1, 150),
                       "Defines how often the parameter grid should be refined/coarsened.");

    prm.declare_entry ("Refinement strategy", "Number",
                       Patterns::Anything(),
                       "Refinement strategy (refine fixed Fraction or Number).");

    prm.declare_entry ("Refinement coefficient", "0.1",
                       Patterns::Double(0.0, 1.0),
                       "Fraction or number of refinement");

    prm.declare_entry ("Coarsening coefficient", "0.0",
                       Patterns::Double(0.0, 1.0),
                       "Fraction number of coarsening");

    prm.declare_entry ("Reset model after refinement", "0",
                       Patterns::Integer(0, 1),
                       "If equals one, interpolates starting model to the mesh after refinement "
                       "(currently works only if starting model is a homogeneous halfspace)");

    prm.declare_entry ("Steplength iterations", "1",
                       Patterns::Integer(1, 5),
                       "Number of steplength selection iterations (1 -- no line-search,"
                       " i.e. use full steplength 1.)");

    prm.declare_entry ("Step lengths file", "",
                       Patterns::Anything(),
                       "File that stores scalar values (one per line) "
                       "to be used as step lengths in inversion");

    prm.declare_entry ("Output files prefix", "invout",
                       Patterns::Anything(),
                       "Prefix that will be added to all output files");

    prm.declare_entry ("Output iteration increment", "0",
                       Patterns::Integer(),
                       "Increment added to the iteration number in the output file name");

    DataOutInterface<1>::declare_parameters (prm);
  }
  prm.leave_subsection ();
}

void ParameterReader::read_parameters (const std::string parameter_file)
{
  declare_parameters ();
  std::ifstream ifs(parameter_file);
  prm.parse_input (ifs, parameter_file);
}
