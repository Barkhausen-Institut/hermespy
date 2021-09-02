cmake_minimum_required(VERSION 3.15...3.19)
project(ldpc VERSION "0.0.1")

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.3 NO_MODULE)

pybind11_add_module(ldpc MODULE src/ldpc_binding.cpp)
target_include_directories(ldpc PUBLIC include/)

# Since Eigen3 is an optional import, we revert back to the 3rdparty folder if it is not available on system
if(Eigen3)
    target_link_libraries(ldpc PRIVATE Eigen3::Eigen)
else()
    target_include_directories(ldpc PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/)
endif()

install(TARGETS ldpc DESTINATION .)