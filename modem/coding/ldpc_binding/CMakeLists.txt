cmake_minimum_required(VERSION 3.15...3.19)
project(ldpc VERSION "0.0.1")

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

pybind11_add_module(ldpc MODULE src/ldpc_binding.cpp)
target_include_directories(ldpc PUBLIC include/)
target_link_libraries(ldpc PRIVATE Eigen3::Eigen)

install(TARGETS ldpc DESTINATION .)