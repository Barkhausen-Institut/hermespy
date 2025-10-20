# -*- coding: utf-8 -*-
"""Waveform Generation for Phase-Shift-Keying Quadrature Amplitude Modulation Testing"""

from unittest import TestCase
from unittest.mock import Mock, patch
from typing_extensions import override

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_almost_equal
from scipy.constants import pi

from hermespy.modem import FilteredSingleCarrierWaveform, StatedSymbols, RaisedCosineWaveform, RootRaisedCosineWaveform, RectangularWaveform, FMCWWaveform, SingleCarrierCorrelationSynchronization, SingleCarrierMinimumMeanSquareChannelEqualization
from hermespy.modem.waveform_single_carrier import SingleCarrierLeastSquaresChannelEstimation, RolledOffSingleCarrierWaveform
from .test_waveform import TestCommunicationWaveform

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockSingleCarrierWaveform(FilteredSingleCarrierWaveform):
    """Implementation of the abstract single carrier waveform base class for testing"""

    @override
    def _transmit_filter(self, oversampling_factor: int) -> np.ndarray:
        filter = np.zeros(oversampling_factor, dtype=np.complex128)
        filter[0] = 1.0
        return filter

    @override
    def _receive_filter(self, oversampling_factor: int) -> np.ndarray:
        filter = np.zeros(oversampling_factor, dtype=np.complex128)
        filter[0] = 1.0
        return filter

    @override
    def _filter_delay(self, oversampling_factor: int) -> int:
        return oversampling_factor - 1


class MockRolledOfSingleCarrierWaveform(RolledOffSingleCarrierWaveform):
    """Implementation of the abstract single carrier waveform base class for testing"""

    @override
    def _base_filter(self, oversampling_factor: int) -> np.ndarray:
        filter = np.zeros(self.filter_length, dtype=np.complex128)
        filter[int(.5 * self.filter_length)] = 1.0
        return filter


class TestFilteredSingleCarrierWaveform(TestCommunicationWaveform):
    """Test the Phase-Shift-Keying / Quadrature Amplitude Modulation Waveform Generator"""

    waveform: FilteredSingleCarrierWaveform

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        self.symbol_rate = 125e3
        self.pilot_rate = 4
        self.oversampling_factor = 16
        self.modulation_order = 16
        self.guard_interval = 1e-3
        self.num_data_symbols = 1000

        self.waveform = MockSingleCarrierWaveform(
            pilot_rate=self.pilot_rate,
            modulation_order=self.modulation_order,
            guard_interval=self.guard_interval,
            num_preamble_symbols=1,
            num_data_symbols=self.num_data_symbols,
        )

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as object attributes"""

        self.assertEqual(self.guard_interval, self.waveform.guard_interval)
        self.assertEqual(self.num_data_symbols, self.waveform.num_data_symbols)

    def test_num_preamble_symbols_validation(self) -> None:
        """Number of preamble symbols property should raise ValueError on arguments smaller than zero"""

        with self.assertRaises(ValueError):
            self.waveform.num_preamble_symbols = -1

    def test_num_preamble_symbols_setget(self) -> None:
        """Number of preamble symbols property getter should return setter argument"""

        num_preamble_symbols = 1
        self.waveform.num_preamble_symbols = num_preamble_symbols

        self.assertEqual(num_preamble_symbols, self.waveform.num_preamble_symbols)

    def test_num_postamble_symbols_validation(self) -> None:
        """Number of postamble symbols property should raise ValueError on arguments smaller than zero"""

        with self.assertRaises(ValueError):
            self.waveform.num_postamble_symbols = -1

    def test_num_postamble_symbols_setget(self) -> None:
        """Number of postamble symbols property getter should return setter argument"""

        num_postamble_symbols = 1
        self.waveform.num_postamble_symbols = num_postamble_symbols

        self.assertEqual(num_postamble_symbols, self.waveform.num_postamble_symbols)

    def test_modulation_order_setget(self) -> None:
        """Modulation order property getter should return setter argument"""

        modulation_order = 16
        self.waveform.modulation_order = modulation_order

        self.assertEqual(modulation_order, self.waveform.modulation_order)

    def test_pilot_signal_property(self) -> None:
        """Pilot signal property should return correct pilot signal"""

        self.waveform.num_preamble_symbols = 1
        pilot_signal = self.waveform.pilot_signal(self.symbol_rate, self.oversampling_factor)
        self.assertEqual(self.waveform.num_preamble_symbols * self.oversampling_factor, pilot_signal.num_samples)

        self.waveform.num_preamble_symbols = 0
        pilot_signal = self.waveform.pilot_signal(self.symbol_rate, self.oversampling_factor)
        self.assertEqual(0, pilot_signal.num_samples)

    def test_guard_interval_setget(self) -> None:
        """Guard interval property getter should return setter argument"""

        guard_interval = 1.23
        self.waveform.guard_interval = 1.23

        self.assertEqual(guard_interval, self.waveform.guard_interval)

    def test_guard_interval_validation(self) -> None:
        """Guard interval property should raise ValueError on arguments smaller than zero"""

        with self.assertRaises(ValueError):
            self.waveform.guard_interval = -1.0

        try:
            self.waveform.guard_interval = 0.0

        except ValueError:
            self.fail()

    def test_pilot_rate_validation(self) -> None:
        """Pilot rate property should raise ValueError on arguments smaller than zero"""

        with self.assertRaises(ValueError):
            self.waveform.pilot_rate = -1

        try:
            self.waveform.pilot_rate = 0

        except ValueError:
            self.fail()

    def test_pilot_rate_setget(self) -> None:
        """Pilot rate property getter should return setter argument"""

        pilot_rate = 4
        self.waveform.pilot_rate = pilot_rate

        self.assertEqual(pilot_rate, self.waveform.pilot_rate)

    def test_num_data_symbols_setget(self) -> None:
        """Number of pilot symbols property getter should return setter argument"""

        num_data_symbols = 123
        self.waveform.num_data_symbols = 123
        self.assertEqual(num_data_symbols, self.waveform.num_data_symbols)

    def test_num_data_symbols_validation(self) -> None:
        """Number of pilot symbols property setter should raise ValueError on arguments smaller than zero"""

        with self.assertRaises(ValueError):
            self.waveform.num_data_symbols = -1

        try:
            self.waveform.num_data_symbols = 0
            self.waveform.num_data_symbols = 10

        except ValueError:
            self.fail()

    def test_plot_filter_correlation(self) -> None:
        """Plotting the filter correlation should generate a figure"""

        with patch("matplotlib.pyplot.subplots") as subplots_mock:
            figure = Mock()
            axes = Mock()
            subplots_mock.return_value = (figure, axes)

            generated_figure = self.waveform.plot_filter_correlation()

            axes.plot.assert_called_once()
            self.assertIs(figure, generated_figure)

    def test_plot_filter(self) -> None:
        """Plotting the filter should generate a figure"""

        with patch("matplotlib.pyplot.subplots") as subplots_mock:
            figure = Mock()
            axes = Mock()
            subplots_mock.return_value = (figure, axes)

            generated_figure = self.waveform.plot_filter()

            axes.plot.assert_called()
            self.assertIs(figure, generated_figure)

    @override
    def test_power(self) -> None:
        # Skip this test since we're not working with a real waveform
        self.skipTest("Not a real waveform")

    @override
    def test_energy(self) -> None:
        # Skip this test since we're not working with a real waveform
        self.skipTest("Not a real waveform")

    @override
    def test_modulate_demodulate(self) -> None:
        # Skip this test since we're not working with a real waveform
        self.skipTest("Not a real waveform")


class TestSingleCarrierCorrelationSynchronization(TestCase):
    """Test the correlation-based synchronization routine"""

    def setUp(self) -> None:
        self.rng = default_rng(42)

        self.bandwidth = 1e6
        self.oversampling_factor = 4

        self.synchronization = SingleCarrierCorrelationSynchronization()
        self.waveform = MockSingleCarrierWaveform(num_preamble_symbols=10, num_data_symbols=50)
        self.waveform.synchronization = self.synchronization
        self.waveform.guard_interval = 0.0

    def test_delay_synchronization(self) -> None:
        """Test synchronization with arbitrary sample offset"""

        bits = self.rng.integers(0, 2, self.waveform.bits_per_frame(self.waveform.num_data_symbols))
        symbols = self.waveform.map(bits)

        signal = self.waveform.modulate(self.waveform.place(symbols), self.bandwidth, self.oversampling_factor)

        for offset in [0, 1, 10, 15, 20]:
            samples = np.append(np.zeros((1, offset), dtype=complex), signal)

            pilot_indices = self.synchronization.synchronize(samples, self.bandwidth, self.oversampling_factor)
            self.assertSequenceEqual([offset], pilot_indices)

    def test_phase_shift_synchronization(self) -> None:
        """Test synchronization with arbitrary sample offset and phase shift"""

        bits = self.rng.integers(0, 2, self.waveform.bits_per_frame(self.waveform.num_data_symbols))
        symbols = self.waveform.map(bits)

        samples = self.waveform.modulate(self.waveform.place(symbols), self.bandwidth, self.oversampling_factor) * np.exp(0.24567j * pi)
        padded_samples = np.append(np.zeros((1, 15), dtype=complex), samples)

        pilot_indices = self.synchronization.synchronize(padded_samples, self.bandwidth, self.oversampling_factor)
        self.assertSequenceEqual([15], pilot_indices)


class TestLeastSquaresChannelEstimation(TestCase):
    """Test channel estimation of single carrier waveforms"""

    def setUp(self) -> None:
        self.rng = default_rng(42)

        self.bandwidth = 1e6
        self.oversampling_factor = 2
        self.waveform = MockSingleCarrierWaveform(num_preamble_symbols=3, num_postamble_symbols=3, num_data_symbols=100, pilot_rate=10)

        self.symbols = self.waveform.place(self.waveform.map(self.rng.uniform(0, 2, self.waveform.bits_per_frame(self.waveform.num_data_symbols))))

        self.estimation = SingleCarrierLeastSquaresChannelEstimation()
        self.waveform.channel_estimation = self.estimation

    def test_least_squares_validation(self) -> None:
        """Least squares channel estimation should raise FloatingError if not assigned to a waveform"""

        self.estimation.waveform = None

        with self.assertRaises(RuntimeError):
            self.estimation.estimate_channel(self.symbols, self.bandwidth, self.oversampling_factor)

    def test_least_squares(self) -> None:
        """Least squares channel estimation"""

        stated_symbols = self.estimation.estimate_channel(self.symbols, self.bandwidth, self.oversampling_factor)
        self.assertEqual(self.waveform._num_frame_symbols, stated_symbols.num_blocks)


class TestChannelEqualization(TestCase):
    """Test channel equalization of single carrier waveforms"""

    def setUp(self) -> None:
        self.rng = default_rng(42)

        self.waveform = MockSingleCarrierWaveform(num_preamble_symbols=3, num_postamble_symbols=3, num_data_symbols=100, pilot_rate=10)

        self.raw_symbols = self.waveform.map(self.rng.uniform(0, 2, self.waveform.bits_per_frame(self.waveform.num_data_symbols)))
        self.raw_state = np.ones((1, 1, self.raw_symbols.num_blocks, self.raw_symbols.num_symbols))
        self.symbols = StatedSymbols(self.raw_symbols.raw, self.raw_state)

    def test_mmse_validation(self) -> None:
        """MMSE channel equalization should raise RuntimeError in the MISO case"""

        equalization = SingleCarrierMinimumMeanSquareChannelEqualization(self.waveform)
        propagated_symbols = StatedSymbols(self.raw_symbols.raw, np.repeat(self.raw_state, 2, axis=1))
        with self.assertRaises(RuntimeError):
            equalization.equalize_channel(propagated_symbols)

    def test_mmse_siso(self) -> None:
        """Test MMSE equalization in the SISO case"""

        equalization = SingleCarrierMinimumMeanSquareChannelEqualization(self.waveform)
        equalized_symbols = equalization.equalize_channel(self.symbols)

        assert_array_almost_equal(self.symbols.raw, equalized_symbols.raw)

    def test_mmse_simo(self) -> None:
        """Test MMSE equalization in the SIMO case"""

        self.raw_state = np.ones((2, 1, self.raw_symbols.num_blocks, self.raw_symbols.num_symbols))
        propagated_symbols = StatedSymbols(np.repeat(self.raw_symbols.raw, 2, axis=0), self.raw_state)

        equalization = SingleCarrierMinimumMeanSquareChannelEqualization(self.waveform)
        equalized_symbols = equalization.equalize_channel(propagated_symbols)

        assert_array_almost_equal(self.symbols.raw, equalized_symbols.raw)


class TestRolledOfFilteredSingleCarrierWaveform(TestCommunicationWaveform):
    """Test the rolled-off filtered single carrier waveform generator"""

    waveform: RolledOffSingleCarrierWaveform

    def setUp(self) -> None:
        self.rng = default_rng(42)
        self.num_preamble_symbols = 0
        self.num_data_symbols = 1000
        self.waveform = MockRolledOfSingleCarrierWaveform(num_preamble_symbols=self.num_preamble_symbols, num_data_symbols=self.num_data_symbols)

    def test_filter_length_validation(self) -> None:
        """Filter length property should raise ValueError on arguments smaller than zero"""

        with self.assertRaises(ValueError):
            self.waveform.filter_length = -1

    def test_filter_length_setget(self) -> None:
        """Filter length property getter should return setter argument"""

        filter_length = 1
        self.waveform.filter_length = filter_length

        self.assertEqual(filter_length, self.waveform.filter_length)

    def test_relative_bandwidth_validation(self) -> None:
        """Relative bandwidth property should raise ValueError on arguments smaller than zero"""

        with self.assertRaises(ValueError):
            self.waveform.relative_bandwidth = -1

        with self.assertRaises(ValueError):
            self.waveform.relative_bandwidth = 0.0

    def test_relative_bandwidth_setget(self) -> None:
        """Relative bandwidth property getter should return setter argument"""

        relative_bandwidth = 1
        self.waveform.relative_bandwidth = relative_bandwidth

        self.assertEqual(relative_bandwidth, self.waveform.relative_bandwidth)

    def test_roll_off_validation(self) -> None:
        """Roll-off property should raise ValueError on arguments smaller than zero or bigger than one"""

        with self.assertRaises(ValueError):
            self.waveform.roll_off = -0.1

        with self.assertRaises(ValueError):
            self.waveform.roll_off = 1.1

    @override
    def test_power(self) -> None:
        self.skipTest("Not a real waveform")

    @override
    def test_energy(self) -> None:
        self.skipTest("Not a real waveform")

    @override
    def test_modulate_demodulate(self) -> None:
        self.skipTest("Not a real waveform")


class TestRootRaisedCosineWaveform(TestCommunicationWaveform):
    """Test the Root-Raised-Cosine pulse shape for single carrier modulation"""

    waveform: RootRaisedCosineWaveform

    def setUp(self) -> None:
        self.rng = default_rng(42)

        self.relative_bandwidth = 1.0
        self.roll_off = 0.25
        self.num_preamble_symbols = 10
        self.num_data_symbols = 1000

        self.waveform = RootRaisedCosineWaveform(
            relative_bandwidth=self.relative_bandwidth,
            roll_off=self.roll_off,
            num_preamble_symbols=self.num_preamble_symbols,
            num_data_symbols=self.num_data_symbols,
        )

    def test_rolled_off_modulate_demodulate(self) -> None:
        """Test the successful modulation and demodulation of data symbols with roll-off"""

        self.waveform.roll_off = 1.0
        self.test_modulate_demodulate()

    def test_no_roll_off_modulate_demodulate(self) -> None:
        """Test the successful modulation and demodulation of data symbols without roll-off"""

        self.waveform.filter_length = 64
        self.waveform.roll_off = 0.0
        self.test_modulate_demodulate()


class TestRaisedCosineWaveform(TestCommunicationWaveform):
    """Test the Root-Raised-Cosine pulse shape for single carrier modulation"""

    waveform: RaisedCosineWaveform

    def setUp(self) -> None:
        self.rng = default_rng(42)

        self.relative_bandwidth = 1.0
        self.roll_off = 0.0
        self.num_preamble_symbols = 10
        self.num_data_symbols = 1000
        self.filter_length = 128

        self.waveform = RaisedCosineWaveform(
            relative_bandwidth=self.relative_bandwidth,
            roll_off=self.roll_off,
            num_preamble_symbols=self.num_preamble_symbols,
            num_data_symbols=self.num_data_symbols,
            filter_length=self.filter_length,
        )

    @override
    def test_power(self) -> None:
        self.waveform.num_preamble_symbols = 0
        self.waveform.num_postamble_symbols = 0
        self.waveform.guard_interval = 0.0
        self.waveform.filter_length = 16
        super().test_power()


class TestRectangularWaveform(TestCommunicationWaveform):
    """Test the Root-Raised-Cosine pulse shape for single carrier modulation"""

    waveform: RectangularWaveform

    def setUp(self) -> None:
        self.rng = default_rng(42)

        self.oversampling_factor = 4
        self.num_preamble_symbols = 10
        self.num_data_symbols = 1000

        self.waveform = RectangularWaveform(
            num_preamble_symbols=self.num_preamble_symbols,
            num_data_symbols=self.num_data_symbols,
        )


class TestFMCWWaveform(TestCommunicationWaveform):
    """Test the FMCW pulse shape for single carrier modulation"""

    waveform: FMCWWaveform

    def setUp(self) -> None:
        self.rng = default_rng(42)

        self.oversampling_factor = 4
        self.bandwidth = .25e6
        self.num_preamble_symbols = 10
        self.num_data_symbols = 1000
        self.num_samples_per_chirp = 256
        self.moduluation_order = 4

        self.waveform = FMCWWaveform(
            num_samples_per_chirp=self.num_samples_per_chirp,
            num_preamble_symbols=self.num_preamble_symbols,
            num_data_symbols=self.num_data_symbols,
            modulation_order=self.moduluation_order,
        )

    @override
    def test_power(self) -> None:
        self.waveform.num_preamble_symbols = 0
        self.waveform.num_postamble_symbols = 0
        self.waveform.guard_interval = 0.0
        super().test_power()

    @override
    def test_energy(self) -> None:
        self.waveform.num_preamble_symbols = 0
        self.waveform.num_postamble_symbols = 0
        self.waveform.guard_interval = 0.0
        super().test_energy()

    def test_num_samples_per_chirp_validation(self) -> None:
        """Number of samples per chirp property should raise ValueError on arguments smaller than zero"""

        with self.assertRaises(ValueError):
            self.waveform.num_samples_per_chirp = -1

        with self.assertRaises(ValueError):
            self.waveform.num_samples_per_chirp = 0

    def test_num_samples_per_chirp_setget(self) -> None:
        """Number of samples per chirp property getter should return setter argument"""

        expected_num_samples = 2048
        self.waveform.num_samples_per_chirp = expected_num_samples

        self.assertEqual(expected_num_samples, self.waveform.num_samples_per_chirp)
