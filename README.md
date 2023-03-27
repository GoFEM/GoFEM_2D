### Create a conda environment

You need to create a pyGoFEM environment without the deal.II library (we will compile it manually as decribed below):

>> git clone https://github.com/GoFEM/pyGoFEM.git
>> conda env create -f pyGoFEM_nodealii.yml
>> conda activate pygofem

### Compilation

First, create a directory where libraries will be compiled and go there:
>> mkdir lib; cd lib

#### Build PETSc

Download and unpack the library:
>> wget https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-3.18.5.tar.gz
>> tar -zxvf petsc-3.18.5.tar.gz
>> cd petsc-3.18.5

Configure the library and build it following the instructions from the command line:
>> ./configure --with-petsc-arch=gcc-release-shared --with-x=0 --with-scalar-type=real --with-precision=double  --with-shared-libraries=1 --with-debugging=0 --download-mumps --download-metis --download-fblaslapack=1 --download-scalapack --download-blacs --download-blas-lapack COPTFLAGS="-O3" CXXOPTFLAGS="-O3" FOPTFLAGS="-O3" FC=mpif90 CC=mpicc CXX=mpicxx

#### Build deal.II

Download and unpack the library:
>> wget https://github.com/dealii/dealii/releases/download/v9.3.2/dealii-9.3.2.tar.gz
>> tar -zxvf dealii-9.3.2.tar.gz
>> cd dealii-9.3.2
>> mkdir build; cd build

Configure the library (replace the "/path/to/lib" with your relevant paths):
>> cmake -DCMAKE_BUILD_TYPE=DebugRelease -DCMAKE_INSTALL_PREFIX=/path/to/lib/deal.II -DDEAL_II_STATIC_EXECUTABLE=OFF -DDEAL_II_WITH_PETSC=ON -DPETSC_DIR=/path/to/lib/petsc-3.18.5 -DPETSC_ARCH=gcc-release-shared -DDEAL_II_WITH_MPI=ON -DDEAL_II_WITH_UMFPACK=OFF -DDEAL_II_COMPONENT_EXAMPLES=OFF -DDEAL_II_WITH_METIS=ON -DMETIS_DIR=/path/to/lib/petsc-3.18.5/gcc-release-shared -DCMAKE_C_COMPILER="mpicc" -DCMAKE_CXX_COMPILER="mpicxx" -DDEAL_II_FORCE_BUNDLED_BOOST=OFF -DDEAL_II_COMPONENT_PYTHON_BINDINGS=ON ../
>> make -j4

#### Build GoFEM

Clone the github repository to and go to the appropriate directory:
>> cd GOFEM/applications/em_modeling
>> mkdir build; cd build

Configure and build the code (replace the "/path/to/lib" with your relevant paths):
>> cmake -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=/path/to/lib/dealii-9.3.2/build -DSHARED_PATH=/path/to/GOFEM_2D/shared -DCMAKE_CXX_FLAGS="-DUSE_MUMPS -DUSE_EMFEM -DSHARED_TRIANGULATION" -DCMAKE_C_COMPILER="mpicc" -DCMAKE_CXX_COMPILER="mpicxx" ../
>> make
