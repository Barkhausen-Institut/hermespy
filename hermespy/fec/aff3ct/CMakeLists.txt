set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(aff3ct_binding VERSION "1.3.0")

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

python_add_library(ldpc MODULE ldpc.cpp WITH_SOABI)
target_compile_definitions(ldpc PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(ldpc PRIVATE aff3ct-static-lib pybind11::headers)

python_add_library(polar MODULE polar.cpp WITH_SOABI)
target_compile_definitions(polar PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(polar PRIVATE aff3ct-static-lib pybind11::headers)

python_add_library(rsc MODULE rsc.cpp WITH_SOABI)
target_compile_definitions(rsc PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(rsc PRIVATE aff3ct-static-lib pybind11::headers)

python_add_library(turbo MODULE turbo.cpp WITH_SOABI)
target_compile_definitions(turbo PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(turbo PRIVATE aff3ct-static-lib pybind11::headers)

python_add_library(rs MODULE rs.cpp WITH_SOABI)
target_compile_definitions(rs PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(rs PRIVATE aff3ct-static-lib pybind11::headers)

python_add_library(bch MODULE bch.cpp WITH_SOABI)
target_compile_definitions(bch PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(bch PRIVATE aff3ct-static-lib pybind11::headers)