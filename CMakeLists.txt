cmake_minimum_required(VERSION 3.15...3.19)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(hermespy)

# Add and configure aff3ct
set(AFF3CT_COMPILE_EXE OFF CACHE INTERNAL "Disable executable target" FORCE)
set(AFF3CT_COMPILE_STATIC_LIB ON CACHE INTERNAL "Enable static library target" FORCE)
set(AFF3CT_COMPILE_SHARED_LIB OFF CACHE INTERNAL "Disable shared library target" FORCE)
set(AFF3CT_OVERRIDE_VERSION ON CACHE INTERNAL "Enable build without .git" FORCE)
set(AFF3CT_COMPILE_OBJ OFF CACHE INTERNAL "Disable object building" FORCE)
set(AFF3ECT_INCLUDE_SPU_LIB OFF CACHE INTERNAL "Disable SPU" FORCE)
set(CLI_COMPILE_STATIC_LIB OFF CACHE INTERNAL "Disable CLI" FORCE)
set(CLI_COMPILE_SHARED_LIB OFF CACHE INTERNAL "Disable CLI" FORCE)

add_subdirectory(submodules/affect)
add_subdirectory(hermespy/fec/aff3ct)

install(TARGETS ldpc polar rs rsc turbo bch DESTINATION hermespy/fec/aff3ct)
