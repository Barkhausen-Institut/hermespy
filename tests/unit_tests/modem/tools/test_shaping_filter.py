import unittest

import numpy as np
import numpy.testing as npt
from modem.tools.shaping_filter import ShapingFilter
from matplotlib import pyplot as plt
from scipy import signal


class TestShapingFilter(unittest.TestCase):

    def setUp(self) -> None:
        self.rnd = np.random.RandomState(42)

    def test_raised_cosine(self) -> None:
        """
        Test if a raise cosine pulse satisfies the desired properties, i.e., given a discrete sequence x[k], with x(t) a
        zero-padded upsampled signal, then if y(t) = x(t) * h(t), with h(t) a raised-cosine filter, then
        y(kTs) = A s[k].
        A sequence of 100 random symbols is generated, filtered by a raised cosine filter, and sampled. The sampled
        signal must be (nearly) equal to the original sequence, except for a multiplying factor.
        This is tested for several different oversampling rates and FIR lengths.
        """

        # parameters
        number_of_symbols = 128
        samples_per_symbol = 128
        length_in_symbols = 32
        roll_off = 0.3

        # generate a random sequence of complex symbols and upsample it
        input_signal = self.rnd.randn(
            number_of_symbols) + 1j * self.rnd.randn(number_of_symbols)
        oversampled_signal = np.vstack(
            (input_signal, np.zeros([samples_per_symbol - 1, number_of_symbols])))
        oversampled_signal = oversampled_signal.flatten('F')

        # filter with a root-raised cosine filter
        tx_filter = ShapingFilter("RAISED_COSINE", samples_per_symbol, length_in_symbols=length_in_symbols,
                                  roll_off=roll_off)
        filtered_signal = tx_filter.filter(oversampled_signal)

        # downsample filtered output
        sampling_indices = np.arange(
            number_of_symbols) * samples_per_symbol + tx_filter.delay_in_samples
        output_signal = filtered_signal[sampling_indices]

        # calculate normalization power
        factor = np.abs(input_signal[0] / output_signal[0])

        npt.assert_allclose(
            input_signal,
            output_signal *
            factor,
            rtol=1e-5,
            atol=0)

    def test_root_raised_cosine(self) -> None:
        """
        Test if a root-raise cosine pulse satisfies the desired properties, i.e., given a discrete sequence x[k], with
        x(t) a zero-padded upsampled signal, then if y(t) = x(t) * h(t) * h(t), with h(t) a root-raised-cosine filter,
        then y(kTs) = A s[k].
        A sequence of 100 random symbols is generated, filtered twice by a root-raised-cosine filter, and sampled. The
        sampled signal must be (nearly) equal to the original sequence.
        However, this is true only for very long impulse responses (because of truncation), and only one combination of
        filter length and oversampling rate is tested.
        """

        # parameters
        number_of_symbols = 100
        samples_per_symbol = 128
        length_in_symbols = 128
        roll_off = self.rnd.random_sample() / 4

        # generate a random sequence of complex symbols and upsample it
        input_signal = self.rnd.randn(
            number_of_symbols) + 1j * self.rnd.randn(number_of_symbols)
        oversampled_signal = np.vstack(
            (input_signal, np.zeros([samples_per_symbol - 1, number_of_symbols])))
        oversampled_signal = oversampled_signal.flatten('F')

        # filter twice with root-raised cosine filters
        self.filter = ShapingFilter(
            "ROOT_RAISED_COSINE",
            samples_per_symbol,
            length_in_symbols,
            1.0,
            roll_off)
        filtered_signal = self.filter.filter(
            self.filter.filter(oversampled_signal))

        # downsample filtered output
        delay_in_samples = 2 * self.filter.delay_in_samples
        sampling_indices = np.arange(
            number_of_symbols) * samples_per_symbol + delay_in_samples
        output_signal = filtered_signal[sampling_indices]

        # calculate normalization power
        factor = np.abs(input_signal[0] / output_signal[0])

        npt.assert_allclose(
            input_signal,
            output_signal *
            factor,
            rtol=1e-2,
            atol=0)

    def test_none(self) -> None:
        """
        Test if a shaping pulse without filtering behaves as expected.
        """

        number_of_symbols = 100
        samples_per_symbol = 128

        self.filter = ShapingFilter("NONE", samples_per_symbol)

        input_signal = self.rnd.randn(
            number_of_symbols) + 1j * self.rnd.randn(number_of_symbols)
        oversampled_signal = np.vstack(
            (input_signal, np.zeros([samples_per_symbol - 1, number_of_symbols])))
        oversampled_signal = oversampled_signal.flatten('F')

        filtered_signal = self.filter.filter(
            self.filter.filter(oversampled_signal))

        npt.assert_array_equal(filtered_signal, oversampled_signal)

    def test_rectangular(self) -> None:
        """
        Test if filtering with a rectangular pulse behaves as expected.
        A sequence of 100 random symbols is generated, and upsampled with zero-padding. The filtered signal must consist
        of a series of rectangular pulses of the desired length and complex amplitude given by the random symbols.
        """

        # parameters
        number_of_symbols = 100
        samples_per_symbol = 8
        bandwidth_factor = 4

        # generate a random sequence of complex symbols and upsample it
        input_signal = self.rnd.randn(
            number_of_symbols) + 1j * self.rnd.randn(number_of_symbols)
        oversampled_signal = np.vstack(
            (input_signal, np.zeros([samples_per_symbol - 1, number_of_symbols])))
        oversampled_signal = oversampled_signal.flatten('F')

        # filter with rectangular pulse
        self.filter = ShapingFilter(
            "RECTANGULAR",
            samples_per_symbol,
            bandwidth_factor=bandwidth_factor)
        filtered_signal = self.filter.filter(oversampled_signal)

        # generate desired output with rectangular pulses
        desired_output = np.vstack((np.tile(input_signal, (int(samples_per_symbol / bandwidth_factor), 1)),
                                    np.zeros((int(samples_per_symbol * (1 - 1 / bandwidth_factor)), number_of_symbols))))
        desired_output = desired_output.flatten(
            'F') / np.sqrt(samples_per_symbol / bandwidth_factor)

        npt.assert_array_equal(
            filtered_signal[:desired_output.size], desired_output)

    def test_fmcw(self) -> None:
        """
        Test if filtering with an FMCW pulse behaves as expected.
        An FMCW pulse or chirp filter will ideally simply delay different frequency components with different delay
        values. In this test we generate two complex carrier waves (CW) with different frequencies but with same time
        reference. Both are independently filtered by the same FMCW filter and their delays are estimated. The
        difference between the measured delays is then compared with the expected value according to the parameters.
        Given a chirp with sweep bandwidth B and duration Tc, and considering two CWs of frequencies f1 and f2, then the
        difference between the delays of the filtered CWs should be equal to dt = (f2-f1)/S, where S = B/Tc is the chirp
        slope.
        """

        # parameters
        chirp_duration = 2e-6
        symbol_interval = .1e-6
        chirp_bandwidth = 200e6
        samples_per_symbol = 128
        sampling_interval = symbol_interval / samples_per_symbol
        length_in_symbols = chirp_duration / symbol_interval

        # generate two complex CWs at frequencies -BW/4 and BW/4
        freq = chirp_bandwidth / 4
        t = np.arange(samples_per_symbol) * sampling_interval
        input1 = np.exp(-2j * np.pi * freq * t)
        input2 = np.exp(2j * np.pi * freq * t)

        # filter both CWs with a chirp
        self.filter = ShapingFilter("FMCW", samples_per_symbol, length_in_symbols=length_in_symbols,
                                    bandwidth_factor=chirp_bandwidth * symbol_interval)
        output1 = self.filter.filter(input1)
        output2 = self.filter.filter(input2)

        # estimate the difference between the delays of both CWs after
        # filtering
        delay1 = np.argmax(np.abs(np.correlate(output1, input1, mode='full')))
        delay2 = np.argmax(np.abs(np.correlate(output2, input2, mode='full')))
        delay_in_samples = delay2 - delay1

        self.assertEqual(
            delay_in_samples, int(
                length_in_symbols * samples_per_symbol / 2))


def plot_fmcw() -> None:
    """
    Generate plots of the FMCW impulse response for visual inspection.
    The following plots are generated:
    1 - Short-time Fourier Transform (STFT) of a baseband upchirp
    2 - impulse response of a passband upconverted upchirp
    3 - STFT of a passband upconverted upchirp
    4 - Short-time Fourier Transform (STFT) of a baseband downchirp
    """
    #####################################
    # create a complex baseband up-chirp
    chirp_duration = 1e-6
    symbol_interval = .1e-6
    chirp_bandwidth = 200e6
    chirp_oversampling_factor = 8
    sampling_rate = chirp_bandwidth * chirp_oversampling_factor
    samples_per_symbol = int(symbol_interval * sampling_rate)

    fmcw = ShapingFilter("FMCW", samples_per_symbol,
                         length_in_symbols=(chirp_duration / symbol_interval),
                         bandwidth_factor=chirp_bandwidth * symbol_interval)
    input_signal = np.array([1])
    impulse_response = fmcw.filter(input_signal)

    # upconvert chirp to generate a passband real-valued chirp signal
    time = np.arange(impulse_response.size) / sampling_rate
    impulse_response_upconverted = np.real(
        impulse_response *
        np.exp(
            1j *
            2 *
            np.pi *
            chirp_bandwidth *
            time))

    ##################
    # generate plots

    # STFT of baseband chirp
    plt.figure()
    f, t, Zxx = signal.stft(
        impulse_response, sampling_rate, return_onesided=False)
    f = np.fft.fftshift(f)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    plt.pcolormesh(t, f, np.abs(Zxx))
    plt.title("chirp STFT")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")

    # time-domain impulse response of upconverted chirp
    plt.figure()
    plt.title("chirp in time domain")
    plt.plot(time, impulse_response_upconverted)

    # STFT of upconverted passband chirp
    plt.figure()
    f, t, Zxx = signal.stft(impulse_response_upconverted,
                            sampling_rate, return_onesided=False)
    f = np.fft.fftshift(f)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    plt.pcolormesh(t, f, np.abs(Zxx))
    plt.title("upconverted chirp STFT")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")

    #####################################
    # create a complex baseband down-chirp
    chirp_duration = 10e-6
    symbol_interval = .1e-6
    chirp_bandwidth = -200e6
    chirp_oversampling_factor = 4

    sampling_rate = np.abs(chirp_bandwidth) * chirp_oversampling_factor
    samples_per_symbol = int(symbol_interval * sampling_rate)

    fmcw = ShapingFilter("FMCW", samples_per_symbol,
                         length_in_symbols=(chirp_duration / symbol_interval),
                         bandwidth_factor=chirp_bandwidth * symbol_interval)
    input_signal = np.array([1])
    impulse_response = fmcw.filter(input_signal)

    ##################
    # generate plots

    # STFT of baseband chirp
    plt.figure()
    f, t, Zxx = signal.stft(
        impulse_response, sampling_rate, return_onesided=False)
    f = np.fft.fftshift(f)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    plt.pcolormesh(t, f, np.abs(Zxx))
    plt.title("down-chirp STFT")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")


def plot_raised_cosine() -> None:
    """
    Generate plots of the raised-cosine impulse response for visual inspection.
    The following plots are generated:
    1 - impulse response of a raised cosine filter with different roll-of factors
    2 - impulse response of a raised cosine filter with different bandwidths
    """
    # plot impulse response of a raised cosine filter for different roll-off
    # factors
    samples_per_symbol = 128
    length_in_symbols = 32

    input_signal = np.array([1])

    plt.figure()

    roll_off = 0.
    rc_filter = ShapingFilter("RAISED_COSINE", samples_per_symbol,
                              length_in_symbols=length_in_symbols, roll_off=roll_off)
    impulse_response = rc_filter.filter(input_signal)
    time = np.arange(impulse_response.size) / samples_per_symbol
    plt.plot(time, impulse_response, label="0")

    roll_off = 0.25
    rc_filter = ShapingFilter("RAISED_COSINE", samples_per_symbol,
                              length_in_symbols=length_in_symbols, roll_off=roll_off)
    impulse_response = rc_filter.filter(input_signal)
    plt.plot(time, impulse_response, label="0.25")

    roll_off = 0.75
    rc_filter = ShapingFilter("RAISED_COSINE", samples_per_symbol,
                              length_in_symbols=length_in_symbols, roll_off=roll_off)
    impulse_response = rc_filter.filter(input_signal)
    plt.plot(time, impulse_response, label="0.75")

    plt.title("Raised-Cosine Filter")
    plt.legend(title='roll-off factor')

    plt.figure()

    roll_off = 0.
    rc_filter = ShapingFilter("RAISED_COSINE", samples_per_symbol,
                              length_in_symbols=length_in_symbols, roll_off=roll_off)
    impulse_response = rc_filter.filter(input_signal)
    time = np.arange(impulse_response.size) / samples_per_symbol
    plt.plot(time, impulse_response, label="0")

    roll_off = 0.
    rc_filter = ShapingFilter("RAISED_COSINE", samples_per_symbol,
                              length_in_symbols=length_in_symbols, roll_off=roll_off, bandwidth_factor=1.5)
    impulse_response = rc_filter.filter(input_signal)
    plt.plot(time, impulse_response, label="1.5")

    plt.title("Raised-Cosine Filters with different bandwidths")
    plt.legend(title='bandwidth expansion')


if __name__ == '__main__':

    plot_raised_cosine()
    plot_fmcw()
    plt.show()

    unittest.main()
