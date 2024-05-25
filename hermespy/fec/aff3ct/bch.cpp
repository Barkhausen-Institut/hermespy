#include <memory>
#include <vector>
#include <cmath>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Module/Encoder/BCH/Encoder_BCH.hpp"
#include "Module/Decoder/BCH/Standard/Decoder_BCH_std.hpp"
#include "Tools/Code/BCH/BCH_polynomial_generator.hpp"

using namespace aff3ct::module;
using namespace aff3ct::tools;
namespace py = pybind11;


class BCH
{
public:

    BCH(const int dataBlockSize, const int codeBlockSize, const int correctionPower) :
        dataBlockSize(dataBlockSize),
        codeBlockSize(codeBlockSize),
        correctionPower(correctionPower)
    {
        // Build the coders
        this->rebuild();
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

    void setDataBlockSize(const int size)
    {
        if (size < 1) throw std::invalid_argument("Data block size must be greater than zero");
        if (size == this->dataBlockSize) return;

        this->dataBlockSize = size;
        this->rebuild();
    }

    int getCodeBlockSize() const
    {
        return this->codeBlockSize;
    }

    void setCodeBlockSize(const int size)
    {
        if (size < 1) throw std::invalid_argument("Code block size must be greater than zero");
        if (size == this->codeBlockSize) return;

        this->codeBlockSize = size;
        this->rebuild();
    }

    int getCorrectionPower() const
    {
        return this->correctionPower;
    }

    void setCorrectionPower(const int power)
    {
        this->correctionPower = correctionPower;
        this->rebuild();
    }

protected:

    int dataBlockSize;
    int codeBlockSize;
    int correctionPower;

    void rebuild()
    {
        // Generate polynomial generator and coders
        std::unique_ptr<BCH_polynomial_generator<int>> polyGen = std::make_unique<BCH_polynomial_generator<int>>(this->codeBlockSize, this->correctionPower);
        std::unique_ptr<Encoder_BCH<int>> encoder = std::make_unique<Encoder_BCH<int>>(this->dataBlockSize, this->codeBlockSize, *polyGen.get());
        std::unique_ptr<Decoder_BCH_std<int, float>> decoder = std::make_unique<Decoder_BCH_std<int, float>>(this->dataBlockSize, this->codeBlockSize, *polyGen.get());
       
        // Update active polynomial genrator and coders if the new instances were created successfully
        this->polyGen = std::move(polyGen);
        this->encoder = std::move(encoder);
        this->decoder = std::move(decoder);
    }

    std::unique_ptr<BCH_polynomial_generator<int>> polyGen;
    std::unique_ptr<Encoder_BCH<int>> encoder;
    std::unique_ptr<Decoder_BCH_std<int, float>> decoder;
};

PYBIND11_MODULE(bch, m)
{
    py::class_<BCH>(m, "BCHCoding")

        .def(py::init<const int, const int, const int>(), R"pbdoc(
            Args:

                data_block_size (int):
                    Number of data bits per block to be encoded.

                code_block_size (int):
                    Number of code bits per encoded block.

                power (int):
                    Number of corretable bit errors.
        )pbdoc")

        .def("encode", &BCH::encode, R"pbdoc(
            Encode a block of data bits to code bits.

            Args:

                data (numpy.ndarray):
                    The data bit block to be encoded.

            Returns:
                The code bit block after encoding.
        )pbdoc")

        .def("decode", &BCH::decode, R"pbdoc(
            Decode a block of code bits to data bits.

            Args:

                code (numpy.ndarray):
                    The code bit block to be decoded.

            Returns:
                The data bit block after decoding.
        )pbdoc")

        .def_property("bit_block_size", &BCH::getDataBlockSize, &BCH::setDataBlockSize, R"pbdoc(
            Number of bits within a data block to be encoded.
        )pbdoc")

        .def_property("code_block_size", &BCH::getCodeBlockSize, &BCH::setCodeBlockSize, R"pbdoc(
            Number of bits within a code block to be decoded.
        )pbdoc")

        .def_property("correction_power", &BCH::getCorrectionPower, &BCH::setCorrectionPower, R"pbdoc(
            Number of corretable bit errors.
        )pbdoc")

        .def_property_readonly_static("enabled", [](py::object){return true;}, R"pbdoc(
            C++ bindings are always enabled.
        )pbdoc")

        .def(py::pickle(
            [](const BCH& bch) {
                return py::make_tuple(bch.getDataBlockSize(), bch.getCodeBlockSize(), bch.getCorrectionPower());
            },
            [](py::tuple t) {
                return BCH(t[0].cast<int>(), t[1].cast<int>(), t[2].cast<int>());
            }
        ));
}