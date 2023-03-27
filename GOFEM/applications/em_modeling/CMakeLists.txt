##
#  CMake script for GOFEM2D
#  Alexander Grayver 2023
##

CMAKE_MINIMUM_REQUIRED(VERSION 3.18)

# Set the name of the project and target:
SET(TARGET "gofem")
SET(COMMON_INCLUDE "../../")

# Declare all source files the target consists of:
SET(TARGET_SRC
  ${COMMON_INCLUDE}/modeling/EM/core/assembly.cc
  ${COMMON_INCLUDE}/modeling/EM/emfem.cc
  ${COMMON_INCLUDE}/modeling/EM/2D/em2dfem.cc
  ${COMMON_INCLUDE}/modeling/EM/2D/mt2dfem.cc
  ${COMMON_INCLUDE}/modeling/EM/2D/csem2dfem.cc
  ${COMMON_INCLUDE}/modeling/EM/2D/csem25dfem.cc
  ${COMMON_INCLUDE}/modeling/EM/forward_modeling_caller.cc
  ${COMMON_INCLUDE}/common/sincos_transform.cc
  ${COMMON_INCLUDE}/common/sparse_direct.cc
  ${SHARED_PATH}/common.cc
  ${SHARED_PATH}/analytical/mt1d.cc
  ${SHARED_PATH}/analytical/mt1d_normalized.cc
  ${SHARED_PATH}/survey/physical_source.cc
  ${SHARED_PATH}/survey/dipole_source.cc
  ${SHARED_PATH}/survey/receiver.cc
  ${COMMON_INCLUDE}/functions/secondary_source.cc
  ${COMMON_INCLUDE}/functions/exact_solution.cc
  ${COMMON_INCLUDE}/functions/current_function.cc
  ${COMMON_INCLUDE}/physical_model/material.cc
  ${COMMON_INCLUDE}/physical_model/cell_properties.cc
  ${COMMON_INCLUDE}/physical_model/physical_model.cc
  ${COMMON_INCLUDE}/physical_model/spherical_model.cc
  ${COMMON_INCLUDE}/physical_model/survey.cc
  ${COMMON_INCLUDE}/physical_model/receiver_data.cc
  ${SHARED_PATH}/mpi/counter.cc
  ${SHARED_PATH}/mpi/mpi_error.cc
  ${SHARED_PATH}/mpi/makesub.cc
  ${SHARED_PATH}/mpi/auxiliary_functions.cc
  ${COMMON_INCLUDE}/io/parameter_reader.cc
  ${COMMON_INCLUDE}/io/read_xyz_model.cc
  ${COMMON_INCLUDE}/io/read_data_map.cc
  ${SHARED_PATH}/io/read_sources.cc
  ${TARGET}.cc
)

FIND_PACKAGE(deal.II 9.3 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

INCLUDE_DIRECTORIES(${DEAL_II_INCLUDE_DIRS} ${COMMON_INCLUDE} ${SHARED_PATH})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DEAL_II_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS_RELEASE ${DEAL_II_CXX_FLAGS_RELEASE})
SET(CMAKE_CXX_FLAGS_DEBUG ${DEAL_II_CXX_FLAGS_DEBUG})

PROJECT(${TARGET})
ADD_EXECUTABLE(${TARGET} ${TARGET_SRC})

TARGET_LINK_LIBRARIES(${TARGET} ${DEAL_II_LIBRARIES})

SET_TARGET_PROPERTIES(${TARGET} PROPERTIES
      COMPILE_DEFINITIONS
      "${DEAL_II_USER_DEFINITIONS}"
      COMPILE_DEFINITIONS_DEBUG
      "${DEAL_II_USER_DEFINITIONS_DEBUG}"
      COMPILE_DEFINITIONS_RELEASE
      "${DEAL_II_USER_DEFINITIONS_RELEASE}"
      )

SET_TARGET_PROPERTIES(${TARGET} PROPERTIES
      LINK_FLAGS
      "${DEAL_II_LINKER_FLAGS}"
      LINK_FLAGS_DEBUG
      "${DEAL_II_LINKER_FLAGS_DEBUG}"
      LINK_FLAGS_RELEASE
      "${DEAL_II_LINKER_FLAGS_RELEASE}"
      )

