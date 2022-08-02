set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(aff3ct_binding VERSION "0.3.0")
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(ldpc MODULE ldpc.cpp)
target_compile_definitions(ldpc PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(ldpc PUBLIC aff3ct-static-lib)

pybind11_add_module(polar MODULE polar.cpp)
target_compile_definitions(polar PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(polar PUBLIC aff3ct-static-lib)

pybind11_add_module(rsc MODULE rsc.cpp)
target_compile_definitions(rsc PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(rsc PUBLIC aff3ct-static-lib)

pybind11_add_module(turbo MODULE turbo.cpp)
target_compile_definitions(turbo PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(turbo PUBLIC aff3ct-static-lib)

pybind11_add_module(rs MODULE rs.cpp)
target_compile_definitions(rs PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(rs PUBLIC aff3ct-static-lib)

pybind11_add_module(bch MODULE bch.cpp)
target_compile_definitions(bch PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(bch PUBLIC aff3ct-static-lib)