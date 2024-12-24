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
        codeBlockSize(nCodeBits),
        bufferedEncoding(buffered_encoding),
        polyA(polyA),
        polyB(polyB)
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

    int getDataBlockSize() const 
    {
        return this->dataBlockSize;
    }

    int getCodeBlockSize() const 
    {
        return this->codeBlockSize;
    }

    bool getBufferedEncoding() const 
    {
        return this->bufferedEncoding;
    }

    int getPolyA() const 
    {
        return this->polyA;
    }

    int getPolyB() const 
    {
        return this->polyB;
    }

    float getRate() const 
    {
        return (float)this->dataBlockSize / (float)this->codeBlockSize;
    }

protected:

    const int dataBlockSize;
    const int codeBlockSize;
    const bool bufferedEncoding;
    const int polyA;
    const int polyB;

    std::unique_ptr<Encoder_RSC_generic_sys<int8_t>> encoder;
    std::unique_ptr<Decoder_RSC_BCJR_seq_generic_std<int8_t, int8_t>> decoder;
};

PYBIND11_MODULE(rsc, m)
{
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
        )pbdoc")

        .def_property("bit_block_size", &RSC::getDataBlockSize, nullptr, R"pbdoc(
            Number of bits within a data block to be encoded.
        )pbdoc")

        .def_property("code_block_size", &RSC::getCodeBlockSize, nullptr, R"pbdoc(
            Number of bits within a code block to be decoded.
        )pbdoc")

        .def_property_readonly_static("enabled", [](py::object){return true;}, R"pbdoc(
            C++ bindings are always enabled.
        )pbdoc")

        .def_property_readonly("rate", &RSC::getRate, R"pbdoc(
            The coding rate.
        )pbdoc")

        .def(py::pickle(
            [](const RSC& rsc) {
                return py::make_tuple(rsc.getDataBlockSize(), rsc.getCodeBlockSize(), rsc.getBufferedEncoding(), rsc.getPolyA(), rsc.getPolyB());
            },
            [](py::tuple t) {
                return RSC(t[0].cast<int>(), t[1].cast<int>(), t[2].cast<bool>(), t[3].cast<int>(), t[4].cast<int>());
            }
        ));
}