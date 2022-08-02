#include <memory>
#include <vector>
#include <cmath>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Module/Encoder/LDPC/From_H/Encoder_LDPC_from_H.hpp"
#include "Module/Decoder/LDPC/BP/Flooding/Decoder_LDPC_BP_flooding.hpp"
#include "Tools/Code/LDPC/Matrix_handler/LDPC_matrix_handler.hpp"
#include "Tools/Code/LDPC/Update_rule/SPA/Update_rule_SPA.hpp"

using namespace aff3ct::module;
using namespace aff3ct::tools;
namespace py = pybind11;


class LDPC
{
public:

    LDPC(const int numIterations, const std::string& hSourcePath, const std::string& gSavePath, const bool syndromeChecking, const int minNumIterations) :
        numIterations(numIterations)
    {
        // Read the H matrix
        this->infoBitPos = std::vector<uint32_t>(dataBlockSize);
        this->H = std::make_unique<Sparse_matrix>(LDPC_matrix_handler::read(hSourcePath, &infoBitPos));
        
        // Infer parameters
        this->updateRule = std::make_unique<Update_rule_SPA<float>>((unsigned int)this->H->get_cols_max_degree());
        this->dataBlockSize = this->H->get_n_cols();
        this->codeBlockSize = this->H->get_n_rows();

        this->encoder = std::make_unique<Encoder_LDPC_from_H<int>>(this->dataBlockSize, this->codeBlockSize, *this->H, "IDENTITY", gSavePath);   
        if (this->infoBitPos.size() < 1) this->infoBitPos = this->encoder->get_info_bits_pos();

        this->decoder = std::make_unique<Decoder_LDPC_BP_flooding<int, float>>(this->dataBlockSize, this->codeBlockSize, numIterations, *this->H, infoBitPos, *this->updateRule, syndromeChecking, minNumIterations);
    }

    py::array_t<int> encode(py::array_t<int>& data)
    {
        py::array_t<int> code(this->codeBlockSize);
        this->encoder->encode(static_cast<int*>(data.mutable_data()), static_cast<int*>(code.mutable_data()));

        return code;
    }

    py::array_t<int> decode(py::array_t<int>& code)
    {
        py::array_t<int> data(this->dataBlockSize);
        this->decoder->decode_hiho(static_cast<int*>(code.mutable_data()), static_cast<int*>(data.mutable_data()));
    
        return data;
    }

    int getDataBlockSize() const
    {
        return this->dataBlockSize;
    }

    int getCodeBlockSize() const
    {
        return this->codeBlockSize;
    }

    int getNumIterations() const
    {
        return this->numIterations;
    }

    void setNumIterations(const int num)
    {
        if (num < 1) throw std::invalid_argument("Number of decoding iterations must be greater than zero");
        if (num == this->numIterations) return;
        
        this->numIterations = num;
        this->decoder = std::make_unique<Decoder_LDPC_BP_flooding<int, float>>(this->dataBlockSize, this->codeBlockSize, numIterations, *this->H.get(), this->infoBitPos, *this->updateRule.get());
    }

protected:

    int dataBlockSize;
    int codeBlockSize;
    int numIterations;

    std::vector<uint32_t> infoBitPos;
    std::unique_ptr<Update_rule_SPA<float>> updateRule;
    std::unique_ptr<Encoder_LDPC_from_H<int>> encoder;
    std::unique_ptr<Decoder_LDPC_BP_flooding<int, float>> decoder;
    std::unique_ptr<Sparse_matrix> H;
};

PYBIND11_MODULE(ldpc, m)
{
    m.doc() = R"pbdoc(
        ==============================
        Low Differential Parity Checks
        ==============================

        A Python wrapper around the AFF3CT\ :footcite:`2019:cassagne` project,
        transferring the LDPC\ :footcite:`1963:gallager` implementations to the Hermes forward error correction pipeline structure.
    )pbdoc";

    py::class_<LDPC>(m, "LDPCCoding")
       .def(py::init<const int, const std::string&, const std::string&, const bool, const int>(), R"pbdoc(
            Args:

                num_iterations (int):
                    Number of iterations during decoding.

                h_source_path (str):
                    Location of the H matrix savefile.
                
                g_save_path (str):
                    Location of the generated G matrix savefile.

                enable_syndrome_checking (bool):
                    Enable premature decoding stop if syndroms indicate a likely success.

                min_num_iterations (int):
                    Minimum number of decoding iterations when syndrome checking is enabled.
        )pbdoc")
        
        .def("encode", &LDPC::encode, R"pbdoc(
            Encode a block of data bits to code bits.

            Args:

                data (numpy.ndarray):
                    The data bit block to be encoded.

            Returns:
                The code bit block after encoding.
        )pbdoc")

        .def("decode", &LDPC::decode, R"pbdoc(
            Decode a block of code bits to data bits.

            Args:

                code (numpy.ndarray):
                    The code bit block to be decoded.

            Returns:
                The data bit block after decoding.
        )pbdoc")

        .def_property("bit_block_size", &LDPC::getDataBlockSize, nullptr, R"pbdoc(
            Number of bits within a data block to be encoded.
        )pbdoc")

        .def_property("code_block_size", &LDPC::getCodeBlockSize, nullptr, R"pbdoc(
            Number of bits within a code block to be decoded.
        )pbdoc")

        .def_property("num_iterations", &LDPC::getNumIterations, &LDPC::setNumIterations, R"pbdoc(
            Number of iterations during decoding.
        )pbdoc");
}