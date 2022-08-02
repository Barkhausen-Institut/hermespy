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
        frozenBits(codeBlockSize)
    {
        // Generate frozen bits after the 5G standard
        Frozenbits_generator_BEC frozenBitsGenerator(dataBlockSize, codeBlockSize, "", false);
        this->noise = std::make_unique<Event_probability<float>>(ber);
        frozenBitsGenerator.set_noise(*this->noise.get());
        frozenBitsGenerator.generate(frozenBits);

        // Initialize encoder
        this->encoder = std::make_unique<Encoder_polar<int32_t>>(dataBlockSize, codeBlockSize, frozenBits);
    }

    py::array_t<int32_t> encode(py::array_t<int32_t>& data)
    {
        py::array_t<int32_t> code(this->codeBlockSize);
        this->encoder->encode(static_cast<int32_t*>(data.mutable_data()), static_cast<int32_t*>(code.mutable_data()));

        return code;
    }

    py::array_t<int32_t> decode(py::array_t<int32_t>& code)
    {
        py::array_t<int32_t> data(this->dataBlockSize);
        this->decoder->decode_hiho(static_cast<int32_t*>(code.mutable_data()), static_cast<int32_t*>(data.mutable_data()));
    
        return data;
    }

protected:

    const int dataBlockSize;
    const int codeBlockSize;
    std::vector<bool> frozenBits;

    std::unique_ptr<Event_probability<float>> noise;
    std::unique_ptr<Encoder_polar<int32_t>> encoder;
    std::unique_ptr<D> decoder;
};

class PolarSCLCoding : public PolarCoding<Decoder_polar_SCL_naive<int32_t, int32_t>>
{
public:

    PolarSCLCoding(const int dataBlockSize, const int codeBlockSize, const float ber, const int numPaths) :
        PolarCoding(dataBlockSize, codeBlockSize, ber)
    {
        this->decoder = std::make_unique<Decoder_polar_SCL_naive<int32_t, int32_t>>(dataBlockSize, codeBlockSize, numPaths, this->frozenBits);
    }
};


class PolarSCCoding : public PolarCoding<Decoder_polar_SC_naive<int32_t, int32_t>>
{
public:

    PolarSCCoding(const int dataBlockSize, const int codeBlockSize, const float ber) :
        PolarCoding(dataBlockSize, codeBlockSize, ber)
    {
        this->decoder = std::make_unique<Decoder_polar_SC_naive<int32_t, int32_t>>(dataBlockSize, codeBlockSize, this->frozenBits);
    }
};


PYBIND11_MODULE(polar, m)
{
    m.doc() = R"pbdoc(
        ============
        Polar Coding
        ============

        A Python wrapper around the AFF3CT\ :footcite:`2019:cassagne` project,
        transferring the Polar\ :footcite:`2009:arikan` implementations to the Hermes forward error correction pipeline structure.
    )pbdoc";

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
        )pbdoc");

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
        )pbdoc");
}