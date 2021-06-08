#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <vector>
#include <iostream>

namespace py = pybind11;

std::vector<py::array_t<int>> encode(const std::vector<py::array_t<int>> &data_bits)
{
    return data_bits;
}

std::vector<py::array_t<int>> decode(const std::vector<py::array_t<double>> &encoded_bits)
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