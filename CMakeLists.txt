# cmake version to be used
cmake_minimum_required( VERSION 3.0 )
set(CMAKE_FIND_DEBUG_MODE 1)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# project name
project(watsonfit)

# target
add_library(watsonfit SHARED src/bonndit/utilc/watsonfit.cpp)

# external libs
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenMP REQUIRED)

target_compile_options(watsonfit PRIVATE -DCYTHON_PEP489_MULTI_PHASE_INIT=0)

target_include_directories(watsonfit
  PRIVATE
    ${CERES_INCLUDE_DIRS}
)
target_link_libraries(watsonfit
  PRIVATE
    ${CERES_LIBRARIES}
)
target_link_libraries(watsonfit PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(watsonfit PRIVATE fftw3)
target_link_libraries(watsonfit PRIVATE cerf)

set_target_properties(watsonfit PROPERTIES PUBLIC_HEADER "src/bonndit/utilc/watsonfit.h")
INSTALL(TARGETS watsonfit
        LIBRARY DESTINATION /usr/lib
        PUBLIC_HEADER DESTINATION /usr/include/
)