#include <memory>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Module/Encoder/RSC/Encoder_RSC_generic_sys.hpp"
#include "Module/Decoder/RSC/BCJR/Seq_generic/Decoder_RSC_BCJR_seq_generic_std.hpp"

using namespace aff3ct::module;
using namespace aff3ct::tools;
namespace py = pybind11;

class RSC
{
public:

    RSC(const int nDataBits, const int nCodeBits, const bool buffered_encoding, const int polyA, const int polyB) :
        dataBlockSize(nDataBits),
        codeBlockSize(nCodeBits)
    {
        const std::vector<int> poly = { polyA, polyB };
        this->encoder = std::make_unique<Encoder_RSC_generic_sys<int8_t>>(nDataBits, nCodeBits, buffered_encoding, poly);
        this->decoder = std::make_unique<Decoder_RSC_BCJR_seq_generic_std<int8_t, int8_t>>(nDataBits, this->encoder->get_trellis(), buffered_encoding);
    }

    py::array_t<int8_t> encode(py::array_t<int8_t>& data)
    {
        py::array_t<int8_t> code(this->codeBlockSize);
        this->encoder->encode(static_cast<int8_t*>(data.mutable_data()), static_cast<int8_t*>(code.mutable_data()));

        return code;
    }

    py::array_t<int8_t> decode(py::array_t<int8_t>& code)
    {
        py::array_t<int8_t> data(this->dataBlockSize);
        this->decoder->decode_hiho(static_cast<int8_t*>(code.mutable_data()), static_cast<int8_t*>(data.mutable_data()));
    
        return data;
    }

protected:

    const int dataBlockSize;
    const int codeBlockSize;

    std::unique_ptr<Encoder_RSC_generic_sys<int8_t>> encoder;
    std::unique_ptr<Decoder_RSC_BCJR_seq_generic_std<int8_t, int8_t>> decoder;
};

PYBIND11_MODULE(rsc, m)
{
    m.doc() = R"pbdoc(
        ==========
        RSC Coding
        ==========

        A Python wrapper around the AFF3CT\ :footcite:`2019:cassagne` project,
        transferring the Recursive Systematic Convolutional BCJR\ :footcite:`1974:bahl` implementations to the Hermes forward error correction pipeline structure.
    )pbdoc";

    py::class_<RSC>(m, "RSCCoding")

        .def(py::init<const int, const int, const bool, const int, const int>(), R"pbdoc(
            Args:

                bit_block_size (int):
                    Number of data bits per block to be encoded.

                code_block_size (int):
                   Number of code bits per encoded block.

                buffered_coding (bool):
                    Enable bufferd coding.

                poly_a (int):
                    Trellis graph polynomial dimension alpha.

                poly_b (int):
                    Trellis graph polynomial dimension beta.
        )pbdoc")
        
        .def("encode", &RSC::encode, R"pbdoc(
            Encode a block of data bits to code bits.

            Args:

                data (numpy.ndarray):
                    The data bit block to be encoded.

            Returns:
                The code bit block after encoding.
        )pbdoc")

        .def("decode", &RSC::decode, R"pbdoc(
            Decode a block of code bits to data bits.

            Args:

                code (numpy.ndarray):
                    The code bit block to be decoded.

            Returns:
                The data bit block after decoding.
        )pbdoc");
}