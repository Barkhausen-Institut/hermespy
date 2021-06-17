#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <vector>
#include <iostream>
#include <Eigen/LU>
#include <cmath>
#include <limits>

namespace py = pybind11;
using Eigen::Dynamic;
template <class T>
using RowVector = Eigen::Matrix<T, 1, Eigen::Dynamic>;

RowVector<int> soft_to_hard_bits(const RowVector<double> &vec, const int num_info_bits);
int get_no_negative_elements(const RowVector<double> &vec);
template <class T>
int sign(const T &x);

template <class T>
std::vector<int> nonzero(const RowVector<T> &vec);

std::vector<RowVector<int>> encode(
    const std::vector<RowVector<int>> &data_bits,
    const Eigen::Matrix<int, Dynamic, Dynamic> &G, //check if not 2d
    const int Z,
    const int num_info_bits,
    const int encoded_bits_n,
    const int data_bits_k,
    const int code_blocks,
    const int bits_in_frame)
{
    int no_bits = 0;
    std::vector<RowVector<int>> encoded_words;
    for (auto block : data_bits)
    {
        if (block.size() % data_bits_k != 0)
            exit(0);

        for (int code_block_idx = 0; code_block_idx < code_blocks; code_block_idx++)
        {
            RowVector<int> code_word = block(0, Eigen::seq(0, num_info_bits - 1)) * G;
            code_word = code_word.unaryExpr([](auto x)
                                            { return x % 2; });
            code_word = code_word(0, Eigen::seq(2 * Z, Eigen::last));
            block = block(0, Eigen::seq(num_info_bits, block.cols() - 1));
            no_bits += encoded_bits_n;
        }

        if ((bits_in_frame - no_bits) > 0)
        {
            RowVector<int> fillup_data = (Eigen::MatrixXi::Random(1, bits_in_frame - no_bits).array() + 1) / 2;

            std::cout << "adding " << fillup_data << std::endl;
            encoded_words.push_back(fillup_data);
        }
    }
    return encoded_words;
}

std::vector<RowVector<int>> decode(
    const std::vector<RowVector<double>> &encoded_bits,
    const int encoded_bits_n,
    const int code_blocks,
    const int number_parity_bits,
    const int num_total_bits,
    const int &Z,
    const int no_iterations,
    const Eigen::MatrixXi &H,
    const int num_info_bits)
{
    double eps = std::numeric_limits<double>::epsilon();
    std::vector<RowVector<int>> decoded_blocks;

    for (auto block : encoded_bits)
    {
        RowVector<int> dec_block(1, code_blocks * num_info_bits);
        for (int i = 0; i < code_blocks; i++)
        {
            RowVector<double> curr_code_block = -block(0, Eigen::seq(0, encoded_bits_n));
            if (curr_code_block.cols() < encoded_bits_n)
                continue;

            Eigen::MatrixXd Rcv(number_parity_bits, num_total_bits + 2 * Z);
            Rcv = Eigen::MatrixXd::Zero(number_parity_bits, num_total_bits + 2 * Z);
            RowVector<double> punc_bits = Eigen::MatrixXd::Zero(1, 2 * Z);
            RowVector<double> Qcv(1, punc_bits.cols() + curr_code_block.cols());
            Qcv << punc_bits, curr_code_block;

            for (int spa_ind = 0; spa_ind < no_iterations; spa_ind++)
            {
                for (int check_ind = 0; check_ind < number_parity_bits; check_ind++)
                {
                    RowVector<int> H_row = H(check_ind, Eigen::all);
                    std::vector<int> nb_var_nodes = nonzero<int>(H_row);

                    RowVector<double> temp_llr = Qcv(0, nb_var_nodes) - Rcv(check_ind, nb_var_nodes);
                    double S_mag = (-((temp_llr.array().abs() / 2).tanh() + eps).log()).sum();
                    int S_sign = 0;

                    if (get_no_negative_elements(temp_llr) % 2 == 0)
                        S_sign = 1;
                    else
                        S_sign = -1;

                    for (size_t var_ind = 0; var_ind < nb_var_nodes.size(); var_ind++)
                    {
                        int var_pos = nb_var_nodes[var_ind];
                        double Q_temp = Qcv(0, var_pos) - Rcv(check_ind, var_pos);
                        double Q_temp_mag = -std::log(std::tanh(std::abs(Q_temp) / 2) + eps);
                        int Q_temp_sign = sign<double>(Q_temp + eps);
                        Rcv(check_ind, var_pos) = S_sign * Q_temp_sign * (-std::log(std::tanh(std::abs(S_mag - Q_temp_mag) / 2) + eps));

                        Qcv(0, var_pos) = Q_temp + Rcv(check_ind, var_pos);
                    }
                }
            }
            RowVector<int> code_block = soft_to_hard_bits(Qcv, num_info_bits);

            dec_block(0, Eigen::seq(i * num_info_bits, (i + 1) * num_info_bits)) = code_block;

            block = block(0, Eigen::lastN(encoded_bits_n));
        }
        decoded_blocks.push_back(dec_block);
    }
    return decoded_blocks;
}

template <class T>
int sign(const T &x)
{
    if (x >= 0)
        return 1;
    else
        return -1;
}
template <class T>
std::vector<int> nonzero(const RowVector<T> &vec)
{
    std::vector<int> indices;
    int idx = 0;
    for (auto el : vec)
    {
        if (el != 0)
            indices.push_back(idx);
        idx++;
    }

    return indices;
}
RowVector<int> soft_to_hard_bits(const RowVector<double> &vec, const int num_info_bits)
{
    RowVector<int> hard_bits(1, num_info_bits);

    int idx = 0;
    for (auto llr : vec(0, Eigen::seq(0, num_info_bits - 1)))
    {
        if (llr < 0)
            hard_bits(0, idx) = 1;
        else
            hard_bits(0, idx) = 0;

        idx++;
    }

    return hard_bits;
}
int get_no_negative_elements(const RowVector<double> &vec)
{
    int idx = 0;
    for (auto el : vec)
    {
        if (el < 0)
            idx++;
    }
    return idx;
}