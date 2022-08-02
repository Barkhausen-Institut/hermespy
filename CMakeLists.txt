cmake_minimum_required(VERSION 3.15...3.19)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(hermespy)

if(SKBUILD)
  execute_process(
    COMMAND
      "${PYTHON_EXECUTABLE}" -c
      "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()

# Add and configure aff3ct
set(AFF3CT_COMPILE_EXE OFF CACHE INTERNAL "Disable executable target" FORCE)
set(AFF3CT_COMPILE_STATIC_LIB ON CACHE INTERNAL "Enable static library target" FORCE)
set(AFF3CT_COMPILE_SHARED_LIB OFF CACHE INTERNAL "Disable shared library target" FORCE)
add_subdirectory(submodules/affect)

# Add aff3ct bindings to hermes api
add_subdirectory(hermespy/fec/aff3ct)
install(TARGETS ldpc polar rs rsc turbo bch DESTINATION hermespy/fec/aff3ct)
