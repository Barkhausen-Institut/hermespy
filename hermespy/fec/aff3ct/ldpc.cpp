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
        numIterations(numIterations),
        hSourcePath(hSourcePath),
        gSavePath(gSavePath),
        minNumIterations(minNumIterations),
        syndromeChecking(syndromeChecking)
    {
        // Read the H matrix
        this->H = std::make_unique<Sparse_matrix>(LDPC_matrix_handler::read(hSourcePath));
        this->codeBlockSize = this->H->get_n_rows();  // N
        this->dataBlockSize = this->codeBlockSize - this->H->get_n_cols();  // K

        // Create the encoder and decoder
        this->updateRule = std::make_unique<Update_rule_SPA<float>>((unsigned int)this->H->get_cols_max_degree());

        this->encoder = std::make_unique<Encoder_LDPC_from_H<int>>(this->dataBlockSize, this->codeBlockSize, *this->H, "IDENTITY", gSavePath, false);   
        this->infoBitPos = this->encoder->get_info_bits_pos();

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

    float getRate() const
    {
        return (float)this->dataBlockSize / (float)this->codeBlockSize;
    }

    int getNumIterations() const
    {
        return this->numIterations;
    }

    std::string getHSourcePath() const
    {
        return this->hSourcePath;
    }

    std::string getGSavePath() const
    {
        return this->gSavePath;
    }

    int getMinNumIterations() const
    {
        return this->minNumIterations;
    }

    bool getSyndromeChecking() const
    {
        return this->syndromeChecking;
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
    int minNumIterations;
    bool syndromeChecking;
    std::string hSourcePath;
    std::string gSavePath;

    std::vector<uint32_t> infoBitPos;
    std::unique_ptr<Update_rule_SPA<float>> updateRule;
    std::unique_ptr<Encoder_LDPC_from_H<int>> encoder;
    std::unique_ptr<Decoder_LDPC_BP_flooding<int, float>> decoder;
    std::unique_ptr<Sparse_matrix> H;
};

PYBIND11_MODULE(ldpc, m)
{
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

        .def_property_readonly("rate", &LDPC::getRate, R"pbdoc(
            Coding rate of the LDPC code.
        )pbdoc")

        .def_property("num_iterations", &LDPC::getNumIterations, &LDPC::setNumIterations, R"pbdoc(
            Number of iterations during decoding.
        )pbdoc")

        .def_property_readonly_static("enabled", [](py::object){return true;}, R"pbdoc(
            C++ bindings are always enabled.
        )pbdoc")

        .def("__getstate__", [](const LDPC& ldpc) {
            return py::make_tuple(ldpc.getNumIterations(), ldpc.getHSourcePath(), ldpc.getGSavePath(), ldpc.getSyndromeChecking(), ldpc.getMinNumIterations());
        })

        .def("__setstate__", [](LDPC& ldpc, py::tuple t) {
            new (&ldpc) LDPC{t[0].cast<int>(), t[1].cast<std::string>(), t[2].cast<std::string>(), t[3].cast<bool>(), t[4].cast<int>()};
        });
}