#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <vector>
#include <iostream>
#include <Eigen/LU>

namespace py = pybind11;
using Eigen::Dynamic;
typedef Eigen::Matrix<int, 1, Eigen::Dynamic> RowVector;

std::vector<RowVector> encode(
    const std::vector<RowVector> &data_bits,
    const Eigen::Matrix<int, Dynamic, Dynamic> &G, //check if not 2d
    const int Z,
    const int num_info_bits,
    const int encoded_bits_n,
    const int data_bits_k,
    const int code_blocks,
    const int bits_in_frame)
{
    int no_bits = 0;
    std::vector<RowVector> encoded_words;
    for (auto block : data_bits)
    {
        if (block.size() % data_bits_k != 0)
            exit(0);

        for (int code_block_idx = 0; code_block_idx < code_blocks; code_block_idx++)
        {
            RowVector code_word = block(0, Eigen::seq(0, num_info_bits - 1)) * G;
            code_word = code_word.unaryExpr([](auto x)
                                            { return x % 2; });
            code_word = code_word(0, Eigen::seq(2 * Z, Eigen::last));
            block = block(0, Eigen::seq(num_info_bits, block.cols() - 1));
            no_bits += encoded_bits_n;
        }

        if ((bits_in_frame - no_bits) > 0)
        {
            RowVector fillup_data = (Eigen::MatrixXi::Random(1, bits_in_frame - no_bits).array() + 1) / 2;

            std::cout << "adding " << fillup_data << std::endl;
            encoded_words.push_back(fillup_data);
        }
    }
    return encoded_words;
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