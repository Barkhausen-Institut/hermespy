#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <vector>
#include <iostream>

namespace py = pybind11;

std::vector<py::array_t<int>> encode(
    const std::vector<py::array_t<int>> &data_bits,
    const py::array_t<int> &G, //check if not 2d
    const py::array_t<int> &Z,
    const int encoded_bits_n,
    const int bits_in_frame)
{
    return data_bits;
}

std::vector<py::array_t<int>> decode(
    const std::vector<py::array_t<double>> &encoded_bits,
    const int encoded_bits_n,
    const int code_blocks,
    const int number_parity_bits,
    const int num_total_bits,
    const py::array_t<int> &Z,
    const int no_iterations,
    const Eigen::MatrixXd &H,
    const int num_info_bits)
{
    std::vector<py::array_t<int>> result = {py::array_t<int>(encoded_bits[0].size())};
    return result;
}
PYBIND11_MODULE(ldpc_binding, m)
{
    m.doc() = "python binding of ldpc encoding function";
    m.def("encode", &encode, "Encoding.");
    m.def("decode", &decode, "Decoding.");
}