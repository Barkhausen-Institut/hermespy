#include "ldpc_binding.hpp"

PYBIND11_MODULE(ldpc_binding, m)
{
    m.doc() = "python binding of ldpc encoding function";
    m.def("encode", &encode, "Encoding.");
    m.def("decode", &decode, "Decoding.");
}