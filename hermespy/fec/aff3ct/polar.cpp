#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Tools/Noise/Event_probability.hpp"
// #include "Tools/Code/Polar/Frozenbits_generator/Frozenbits_generator_5G.hpp"
#include "Tools/Code/Polar/Frozenbits_generator/Frozenbits_generator_BEC.hpp"
#include "Module/Encoder/Polar/Encoder_polar.hpp"
// #include "Module/Decoder/Polar/ASCL/Decoder_polar_ASCL_fast_CA_sys.hpp"
#include "Module/Decoder/Polar/SC/Decoder_polar_SC_naive.hpp"
#include "Module/Decoder/Polar/SCL/Decoder_polar_SCL_naive.hpp"

using namespace aff3ct::module;
using namespace aff3ct::tools;
namespace py = pybind11;


template<class D>
class PolarCoding
{
public:

    PolarCoding(const int dataBlockSize, const int codeBlockSize, const float ber) :
        dataBlockSize(dataBlockSize),
        codeBlockSize(codeBlockSize),
        ber(ber),
        frozenBits(codeBlockSize)
    {
        // Generate frozen bits after the 5G standard
        Frozenbits_generator_BEC frozenBitsGenerator(dataBlockSize, codeBlockSize, "", false);
        this->noise = std::make_unique<Event_probability<float>>(ber);
        frozenBitsGenerator.set_noise(*this->noise.get());
        frozenBitsGenerator.generate(frozenBits);

        // Initialize encoder
        this->encoder = std::make_unique<Encoder_polar<int>>(dataBlockSize, codeBlockSize, frozenBits);
    }

    py::array_t<int> encode(py::array_t<int>& data)
    {
        py::array_t<int> code(this->codeBlockSize);
        int* dataPtr = static_cast<int*>(data.request().ptr);
        int* codePtr = static_cast<int*>(code.request().ptr);
        this->encoder->encode(dataPtr, codePtr);

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

    int getBer() const
    {
        return this->ber;
    }

    float getRate() const
    {
        return (float)this->dataBlockSize / (float) this->codeBlockSize;
    }

protected:

    const int dataBlockSize;
    const int codeBlockSize;
    const int ber;
    std::vector<bool> frozenBits;

    std::unique_ptr<Event_probability<float>> noise;
    std::unique_ptr<Encoder_polar<int>> encoder;
    std::unique_ptr<D> decoder;
};

class PolarSCLCoding : public PolarCoding<Decoder_polar_SCL_naive<int, int>>
{
public:

    PolarSCLCoding(const int dataBlockSize, const int codeBlockSize, const float ber, const int numPaths) :
        PolarCoding(dataBlockSize, codeBlockSize, ber),
        numPaths(numPaths)
    {
        this->decoder = std::make_unique<Decoder_polar_SCL_naive<int, int>>(dataBlockSize, codeBlockSize, numPaths, this->frozenBits);
    }

    int getNumPaths() const
    {
        return this->numPaths;
    }

protected:

    const int numPaths;
};


class PolarSCCoding : public PolarCoding<Decoder_polar_SC_naive<int, int>>
{
public:

    PolarSCCoding(const int dataBlockSize, const int codeBlockSize, const float ber) :
        PolarCoding(dataBlockSize, codeBlockSize, ber)
    {
        this->decoder = std::make_unique<Decoder_polar_SC_naive<int, int>>(dataBlockSize, codeBlockSize, this->frozenBits);
    }
};


PYBIND11_MODULE(polar, m)
{
    py::class_<PolarSCCoding>(m, "PolarSCCoding")

        .def(py::init<const int, const int, const float>(), R"pbdoc(
            Args:

                data_block_size (int):
                    Number of data bits per block to be encoded.

                code_block_size (int):
                    Number of code bits per encoded block.

                ber (float):
                    Assumed bit error rate.

                num_paths (int):
                    Number of decoding paths.
        )pbdoc")

        .def("encode", &PolarSCCoding::encode, R"pbdoc(
            Encode a block of data bits to code bits.

            Args:

                data (numpy.ndarray):
                    The data bit block to be encoded.

            Returns:
                The code bit block after encoding.
        )pbdoc")

        .def("decode", &PolarSCCoding::decode, R"pbdoc(
            Decode a block of code bits to data bits.

            Args:

                code (numpy.ndarray):
                    The code bit block to be decoded.

            Returns:
                The data bit block after decoding.
        )pbdoc")

        .def_property("bit_block_size", &PolarSCCoding::getDataBlockSize, nullptr, R"pbdoc(
            Number of bits within a data block to be encoded.
        )pbdoc")

        .def_property("code_block_size", &PolarSCCoding::getCodeBlockSize, nullptr, R"pbdoc(
            Number of bits within a code block to be decoded.
        )pbdoc")  
        
        .def_property_readonly_static("enabled", [](py::object){return true;}, R"pbdoc(
            C++ bindings are always enabled.
        )pbdoc")

        .def_property_readonly("rate", &PolarSCCoding::getRate, R"pbdoc(
            Coding rate of the polar code.
        )pbdoc")

        .def(py::pickle(
            [](const PolarSCCoding& polar) {
                return py::make_tuple(polar.getDataBlockSize(), polar.getCodeBlockSize(), polar.getBer());
            },
            [](py::tuple t) {
                return PolarSCCoding(t[0].cast<int>(), t[1].cast<int>(), t[2].cast<int>());
            }
        ));

    py::class_<PolarSCLCoding>(m, "PolarSCLCoding")

        .def(py::init<const int, const int, const float, const int>(), R"pbdoc(
            Successive Cancellation List Polar Codes.
            Refer to :footcite:p:`2015:tal` for further information.
            
            Args:

                data_block_size (int):
                    Number of data bits per block to be encoded.

                code_block_size (int):
                    Number of code bits per encoded block.

                ber (float):
                    Assumed bit error rate.

                num_paths (int):
                    Number of considered decoding paths.
        )pbdoc")

        .def("encode", &PolarSCLCoding::encode, R"pbdoc(
            Encode a block of data bits to code bits.

            Args:

                data (numpy.ndarray):
                    The data bit block to be encoded.

            Returns:
                The code bit block after encoding.
        )pbdoc")

        .def("decode", &PolarSCLCoding::decode, R"pbdoc(
            Decode a block of code bits to data bits.

            Args:

                code (numpy.ndarray):
                    The code bit block to be decoded.

            Returns:
                The data bit block after decoding.
        )pbdoc")

        .def_property("bit_block_size", &PolarSCLCoding::getDataBlockSize, nullptr, R"pbdoc(
            Number of bits within a data block to be encoded.
        )pbdoc")

        .def_property("code_block_size", &PolarSCLCoding::getCodeBlockSize, nullptr, R"pbdoc(
            Number of bits within a code block to be decoded.
        )pbdoc")  

        .def_property_readonly_static("enabled", [](py::object){return true;}, R"pbdoc(
            C++ bindings are always enabled.
        )pbdoc")

        .def_property_readonly("rate", &PolarSCLCoding::getRate, R"pbdoc(
            Coding rate of the polar code.
        )pbdoc")

        .def(py::pickle(
            [](const PolarSCLCoding& polar) {
                return py::make_tuple(polar.getDataBlockSize(), polar.getCodeBlockSize(), polar.getBer(), polar.getNumPaths());
            },
            [](py::tuple t) {
                return PolarSCLCoding(t[0].cast<int>(), t[1].cast<int>(), t[2].cast<int>(), t[3].cast<int>());
            }
        ));
}