cmake_minimum_required(VERSION 3.15...3.19)
project(ldpc VERSION "0.0.1")

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(ldpc_binding MODULE src/ldpc_binding.cpp)
target_include_directories(ldpc_binding PUBLIC include/)
target_compile_definitions(ldpc_binding PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_include_directories(ldpc_binding PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/)