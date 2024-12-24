set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PYBIND11_NEWPYTHON ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(ldpc ldpc.cpp)
target_link_libraries(ldpc PRIVATE aff3ct-shared-lib pybind11::headers)

pybind11_add_module(polar polar.cpp)
target_link_libraries(polar PRIVATE  aff3ct-shared-lib pybind11::headers)

pybind11_add_module(rsc rsc.cpp)
target_link_libraries(rsc PRIVATE aff3ct-shared-lib pybind11::headers)

pybind11_add_module(turbo turbo.cpp)
target_link_libraries(turbo PRIVATE aff3ct-shared-lib pybind11::headers)

pybind11_add_module(rs rs.cpp)
target_link_libraries(rs PRIVATE aff3ct-shared-lib pybind11::headers)

pybind11_add_module(bch bch.cpp)
target_link_libraries(bch PRIVATE aff3ct-shared-lib pybind11::headers)

install(TARGETS ldpc polar rsc turbo rs bch LIBRARY DESTINATION .)