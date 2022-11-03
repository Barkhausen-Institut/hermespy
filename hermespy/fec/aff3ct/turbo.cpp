#include <memory>
#include <vector>
#include <cmath>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Module/Encoder/Encoder.hpp"
#include "Module/Decoder/Decoder.hpp"
#include "Tools/Interleaver/LTE/Interleaver_core_LTE.hpp"
#include "Module/Encoder/RSC/Encoder_RSC_generic_sys.hpp"
#include "Module/Decoder/RSC/BCJR/Seq_generic/Decoder_RSC_BCJR_seq_generic_std.hpp"
#include "Module/Encoder/Turbo/Encoder_turbo.hpp"
#include "Module/Decoder/Turbo/Decoder_turbo_std.hpp"
#include "Module/Decoder/RSC/BCJR/Inter/Decoder_RSC_BCJR_inter_std.hpp"

using namespace aff3ct::module;
using namespace aff3ct::tools;
namespace py = pybind11;


class Turbo
{
public:

    Turbo(const int dataBlockSize, const int polyA, const int polyB, const int nIterations) :
        dataBlockSize(dataBlockSize),
        polyA(polyA),
        polyB(polyB),
        numIterations(nIterations)
    {
        // Buuild the turbo coding
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

    void setNumIterations(const int n)
    {
        if (n < 1) throw std::invalid_argument("Number of iterations must be greater than zero");
        if (n == this->numIterations) return;

        this->numIterations = n;
        this->rebuild();
    }

    int getNumIterations() const
    {
        return this->numIterations;
    }

    int getPolyA() const 
    {
        return this->polyA;
    }

    int getPolyB() const 
    {
        return this->polyB;
    }

protected:

    int dataBlockSize;
    int codeBlockSize;
    int polyA;
    int polyB;
    int numIterations;

    void rebuild()
    {
        // Infer parameters
        const std::vector<int> poly = { this->polyA, this->polyB };
        const int n_ff = (int) std::floor(std::log2(this->polyA));
        const int rscCodeBlockSize = 2 * this->dataBlockSize  + 2 * n_ff;
        const int codeBlockSize = 3 * this->dataBlockSize + 4 * n_ff;
        const bool buffered = true;

        // RSC sub-coders
        Encoder_RSC_generic_sys<int> rscNaturalEncoder(this->dataBlockSize, rscCodeBlockSize, buffered, poly);
        Decoder_RSC_BCJR_seq_generic_std<int, float> rscNaturalDecoder(this->dataBlockSize, rscNaturalEncoder.get_trellis(), buffered);
        Encoder_RSC_generic_sys<int> rscInterleavedEncoder(this->dataBlockSize, rscCodeBlockSize, buffered, poly);
        Decoder_RSC_BCJR_seq_generic_std<int, float> rscInterleavedDecoder(this->dataBlockSize, rscInterleavedEncoder.get_trellis(), buffered);

        // Final interleaver
        std::unique_ptr<Interleaver_core_LTE<uint32_t>> interleaverCore = std::make_unique<Interleaver_core_LTE<uint32_t>>(this->dataBlockSize);
        Interleaver<int, uint32_t> interleaver_hiho = Interleaver<int, uint32_t>(*interleaverCore.get());
        Interleaver<float, uint32_t> interleaver_siho = Interleaver<float, uint32_t>(*interleaverCore.get());

        // Instantiate turbo en- and decoder
        std::unique_ptr<Encoder_turbo<int>> encoder = std::make_unique<Encoder_turbo<int>>(this->dataBlockSize, codeBlockSize, rscNaturalEncoder, rscInterleavedEncoder, interleaver_hiho);
        std::unique_ptr<Decoder_turbo_std<int, float>> decoder = std::make_unique<Decoder_turbo_std<int, float>>(this->dataBlockSize, codeBlockSize, this->numIterations, rscNaturalDecoder, rscInterleavedDecoder, interleaver_siho, buffered);

        this->codeBlockSize = codeBlockSize;
        this->interleaverCore = std::move(interleaverCore);
        this->encoder = std::move(encoder);
        this->decoder = std::move(decoder);
    }

    std::unique_ptr<Interleaver_core_LTE<uint32_t>> interleaverCore;
    std::unique_ptr<Encoder_turbo<int>> encoder;
    std::unique_ptr<Decoder_turbo_std<int, float>> decoder;
};

PYBIND11_MODULE(turbo, m)
{
    m.doc() = R"pbdoc(
        ============
        Turbo Coding
        ============

        A Python wrapper around the AFF3CT\ :footcite:`2019:cassagne` project,
        transferring the Turbo\ :footcite:`1993:berrou` implementations to the Hermes forward error correction pipeline structure.
    )pbdoc";

    py::class_<Turbo>(m, "TurboCoding")

        .def(py::init<const int, const int, const int, const int>(), R"pbdoc(
            Args:

                bit_block_size (int):
                    Number of data bits per block to be encoded.

                poly_a (int):
                    Trellis graph polynomial dimension alpha.

                poly_b (int):
                    Trellis graph polynomial dimension beta.

                num_iterations (int):
                    Number of iterations during decoding.
        )pbdoc")

        .def("encode", &Turbo::encode, R"pbdoc(
            Encode a block of data bits to code bits.

            Args:

                data (numpy.ndarray):
                    The data bit block to be encoded.

            Returns:
                The code bit block after encoding.
        )pbdoc")

        .def("decode", &Turbo::decode, R"pbdoc(
            Decode a block of code bits to data bits.

            Args:

                code (numpy.ndarray):
                    The code bit block to be decoded.

            Returns:
                The data bit block after decoding.
        )pbdoc")

        .def_property("bit_block_size", &Turbo::getDataBlockSize, &Turbo::setDataBlockSize, R"pbdoc(
            Number of bits within a data block to be encoded.
        )pbdoc")

        .def_property("code_block_size", &Turbo::getCodeBlockSize, nullptr, R"pbdoc(
            Number of bits within a code block to be decoded.
        )pbdoc")

        .def_property("num_iterations", &Turbo::getNumIterations, &Turbo::setNumIterations, R"pbdoc(
            Number of iterations during decoding.
        )pbdoc")

        .def_property_readonly_static("enabled", [](py::object){return true;}, R"pbdoc(
            C++ bindings are always enabled.
        )pbdoc")

        .def(py::pickle(
            [](const Turbo& turbo) {
                return py::make_tuple(turbo.getDataBlockSize(), turbo.getPolyA(), turbo.getPolyB(), turbo.getNumIterations());
            },
            [](py::tuple t) {
                return Turbo(t[0].cast<int>(), t[1].cast<int>(), t[2].cast<int>(), t[3].cast<int>());
            }
        ));
}