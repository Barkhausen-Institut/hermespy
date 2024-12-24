#include <memory>
#include <vector>
#include <cmath>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Module/Encoder/RS/Encoder_RS.hpp"
#include "Module/Decoder/RS/Standard/Decoder_RS_std.hpp"
#include "Tools/Code/RS/RS_polynomial_generator.hpp"

using namespace aff3ct::module;
using namespace aff3ct::tools;
namespace py = pybind11;


class ReedSolomon
{
public:

    ReedSolomon(const int dataBlockSize, const int correctionPower) :
        dataBlockSize(dataBlockSize),
        correctionPower(correctionPower)
    {
        // Build the coders
        this->rebuild();
    }

    py::array_t<int> encode(py::array_t<int>& data)
    {
        py::array_t<int> code(this->getCodeBlockSize());
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
        return this->dataBlockSize + 2 * this->correctionPower;
    }

    int getCorrectionPower() const
    {
        return this->correctionPower;
    }

    void setCorrectionPower(const int power)
    {
        if (power < 1) throw std::invalid_argument("Correction power must be greater than zero");
        if (power == this->correctionPower) return;

        this->correctionPower = power;
        this->rebuild();
    }

    float getRate() const
    {
        return (float)this->dataBlockSize / (float)this->getCodeBlockSize();
    }

protected:

    int dataBlockSize;
    int correctionPower;

    void rebuild()
    {
        // Infer parameters
        const int codeBlockSize = this->getCodeBlockSize();

        // Generate polynomial generator and coders
        std::unique_ptr<RS_polynomial_generator> polyGen = std::make_unique<RS_polynomial_generator>(codeBlockSize, this->correctionPower);
        std::unique_ptr<Encoder_RS<int>> encoder = std::make_unique<Encoder_RS<int>>(this->dataBlockSize, codeBlockSize, *polyGen.get());
        std::unique_ptr<Decoder_RS_std<int, float>> decoder = std::make_unique<Decoder_RS_std<int, float>>(this->dataBlockSize, codeBlockSize, *polyGen.get());
       
        // Update active polynomial genrator and coders if the new instances were created successfully
        this->polyGen = std::move(polyGen);
        this->encoder = std::move(encoder);
        this->decoder = std::move(decoder);
    }

    std::unique_ptr<RS_polynomial_generator> polyGen;
    std::unique_ptr<Encoder_RS<int>> encoder;
    std::unique_ptr<Decoder_RS_std<int, float>> decoder;
};

PYBIND11_MODULE(rs, m)
{
    py::class_<ReedSolomon>(m, "ReedSolomonCoding")

        .def(py::init<const int, const int>(), R"pbdoc(
            Args:

                data_block_size (int):
                    Number of data bits per block to be encoded.

                correction power (int):
                    Number of symbol errors the coding may correct.
        )pbdoc")

        .def("encode", &ReedSolomon::encode, R"pbdoc(
            Encode a block of data bits to code bits.

            Args:

                data (numpy.ndarray):
                    The data bit block to be encoded.

            Returns:
                The code bit block after encoding.
        )pbdoc")

        .def("decode", &ReedSolomon::decode, R"pbdoc(
            Decode a block of code bits to data bits.

            Args:

                code (numpy.ndarray):
                    The code bit block to be decoded.

            Returns:
                The data bit block after decoding.
        )pbdoc")

        .def_property("bit_block_size", &ReedSolomon::getDataBlockSize, &ReedSolomon::setDataBlockSize, R"pbdoc(
            Number of bits within a data block to be encoded.
        )pbdoc")

        .def_property("code_block_size", &ReedSolomon::getCodeBlockSize, nullptr, R"pbdoc(
            Number of bits within a code block to be decoded.
        )pbdoc")

        .def_property("correction_power", &ReedSolomon::getCorrectionPower, &ReedSolomon::setCorrectionPower, R"pbdoc(
            Number of symbol errors the coding may correct.
        )pbdoc")

        .def_property_readonly_static("enabled", [](py::object){return true;}, R"pbdoc(
            C++ bindings are always enabled.
        )pbdoc")

        .def_property_readonly("rate", &ReedSolomon::getRate, R"pbdoc(
            The coding rate.
        )pbdoc")

        .def(py::pickle(
            [](const ReedSolomon& rs) {
                return py::make_tuple(rs.getDataBlockSize(), rs.getCorrectionPower());
            },
            [](py::tuple t) {
                return ReedSolomon(t[0].cast<int>(), t[1].cast<int>());
            }
        ));
}