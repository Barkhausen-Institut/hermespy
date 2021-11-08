cmake_minimum_required(VERSION 3.15...3.19)
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

add_subdirectory(hermespy/coding/ldpc_binding)
install(TARGETS ldpc_binding DESTINATION hermespy/coding/ldpc_binding/bin)