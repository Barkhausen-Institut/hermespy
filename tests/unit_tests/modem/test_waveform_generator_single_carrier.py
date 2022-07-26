# -*- coding: utf-8 -*-
"""Waveform Generation for Phase-Shift-Keying Quadrature Amplitude Modulation Testing."""

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.constants import pi

from hermespy.core.channel_state_information import ChannelStateInformation
from hermespy.modem import FilteredSingleCarrierWaveform, Symbols, RaisedCosineWaveform, RootRaisedCosineWaveform, RectangularWaveform, FMCWWaveform, SingleCarrierCorrelationSynchronization

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockSingleCarrierWaveform(FilteredSingleCarrierWaveform):
    """Implementation of the abstract single carrier waveform base class for testing"""

    def _transmit_filter(self) -> np.ndarray:

        return np.ones(1, dtype=complex)

    def _receive_filter(self) -> np.ndarray:

        return np.ones(1, dtype=complex)

    @property
    def _filter_delay(self) -> int:

        return 0

    @property
    def bandwidth(self) -> float:

        return self.symbol_rate


class FilteredSingleCarrierWaveform(TestCase):
    """Test the Phase-Shift-Keying / Quadrature Amplitude Modulation Waveform Generator"""

    def setUp(self) -> None:

        self.rng = np.random.default_rng(42)

        self.modem = Mock()
        self.modem.carrier_frequency = 100e6

        self.symbol_rate = 125e3
        self.pilot_rate = 4
        self.oversampling_factor = 16
        self.modulation_order = 16
        self.guard_interval = 1e-3
        self.num_data_symbols = 1000

        self.waveform = MockSingleCarrierWaveform(
            modem=self.modem,
            symbol_rate=self.symbol_rate,
            pilot_rate=self.pilot_rate,
            oversampling_factor=self.oversampling_factor,
            modulation_order=self.modulation_order,
            guard_interval=self.guard_interval,
            num_preamble_symbols = 1,
            num_data_symbols=self.num_data_symbols
        )

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as object attributes"""

        self.assertIs(self.modem, self.waveform.modem)
        self.assertEqual(self.symbol_rate, self.waveform.symbol_rate)
        self.assertEqual(self.oversampling_factor, self.waveform.oversampling_factor)
        self.assertEqual(self.guard_interval, self.waveform.guard_interval)
        self.assertEqual(self.num_data_symbols, self.waveform.num_data_symbols)

    def test_symbol_rate_setget(self) -> None:
        """Symbol rate property getter should return setter argument"""

        symbol_rate = 10
        self.waveform.symbol_rate = symbol_rate

        self.assertEqual(symbol_rate, self.waveform.symbol_rate)

    def test_symbol_rate_validation(self) -> None:
        """Symbol rate property should raise ValueError on arguments smaller than zero"""

        with self.assertRaises(ValueError):
            self.waveform.symbol_rate = -1.0

        with self.assertRaises(ValueError):
            self.waveform.symbol_rate = 0.

    def test_map_unmap(self) -> None:
        """Mapping and subsequently un-mapping a bit stream should yield identical bits"""

        expected_bits = self.rng.integers(0, 2, self.waveform.bits_per_frame)

        symbols = self.waveform.map(expected_bits)
        bits = self.waveform.unmap(symbols)

        assert_array_equal(expected_bits, bits)

    def test_modulate_demodulate(self) -> None:
        """Modulating and subsequently de-modulating a symbol stream should yield identical symbols"""

        expected_symbols = Symbols(np.exp(2j * self.rng.uniform(0, pi, self.waveform.symbols_per_frame)))

        baseband_signal = self.waveform.modulate(expected_symbols)
        channel_state = ChannelStateInformation.Ideal(num_samples=baseband_signal.num_samples)
        symbols, _, _ = self.waveform.demodulate(baseband_signal.samples[0, :], channel_state)

        assert_array_almost_equal(expected_symbols.raw, symbols.raw, decimal=1)
        
    def test_guard_interval_setget(self) -> None:
        """Guard interval property getter should return setter argument."""
        
        guard_interval = 1.23
        self.waveform.guard_interval = 1.23
        
        self.assertEqual(guard_interval, self.waveform.guard_interval)
        
    def test_guard_interval_validation(self) -> None:
        """Guard interval property should raise ValueError on arguments smaller than zero"""

        with self.assertRaises(ValueError):
            self.waveform.guard_interval = -1.0

        try:
            self.waveform.guard_interval = 0.

        except ValueError:
            self.fail()

    def test_pilot_rate_setget(self) -> None:
        """Pilot rate property getter should return setter argument"""

        pilot_rate = 4
        self.waveform.pilot_rate = pilot_rate

        self.assertEqual(pilot_rate, self.waveform.pilot_rate)

    def test_pilot_rate_validation(self) -> None:
        """Pilot rate property should raise ValueError on arguments smaller than zero."""

        with self.assertRaises(ValueError):
            self.waveform.pilot_rate = -1

        try:
            self.waveform.pilot_rate = 0

        except ValueError:
            self.fail()

    def test_samples_in_frame(self) -> None:
        """Samples in frame property should compute the correct sample count."""

        symbols = Symbols(np.exp(2j * self.rng.uniform(0, pi, self.waveform.symbols_per_frame)))
        signal = self.waveform.modulate(symbols)

        self.assertEqual(signal.num_samples, self.waveform.samples_in_frame)
            
    def test_num_data_symbols_setget(self) -> None:
        """Number of pilot symbols property getter should return setter argument."""

        num_data_symbols = 1.23
        self.waveform.num_data_symbols = 1.23

        self.assertEqual(num_data_symbols, self.waveform.num_data_symbols)

    def test_num_data_symbols_validation(self) -> None:
        """Number of pilot symbols property setter should raise ValueError on arguments smaller than zero."""

        with self.assertRaises(ValueError):
            self.waveform.num_data_symbols = -1

        try:
            self.waveform.num_data_symbols = 0
            self.waveform.num_data_symbols = 10

        except ValueError:
            self.fail()

    def test_bits_per_frame(self) -> None:
        """Bits per frame property should compute correct amount of data bits per frame."""

        signal = (self.rng.normal(0, 1.0, self.waveform.samples_in_frame) +
                  1j * self.rng.normal(0, 1.0, self.waveform.samples_in_frame))
        channel_state = ChannelStateInformation.Ideal(self.waveform.samples_in_frame)

        data_symbols, _, _ = self.waveform.demodulate(signal, channel_state)
        bits = self.waveform.unmap(data_symbols)

        self.assertEqual(len(bits), self.waveform.bits_per_frame)

    def test_symbols_per_frame(self) -> None:
        """Symbols per frame property should compute correct amount of symbols per frame."""

        signal = (self.rng.normal(0, 1.0, self.waveform.samples_in_frame) +
                  1j * self.rng.normal(0, 1.0, self.waveform.samples_in_frame))
        channel_state = ChannelStateInformation.Ideal(self.waveform.samples_in_frame)

        symbols, _, _ = self.waveform.demodulate(signal, channel_state)

        self.assertEqual(len(symbols.raw.flatten()), self.waveform.symbols_per_frame)

    def test_bit_energy(self) -> None:
        """Bit energy property should compute correct bit energy."""

        self.waveform.pilot_rate = 0.
        self.waveform.guard_interval = 0.

        data_bits = self.rng.integers(0, 2, self.waveform.bits_per_frame)
        data_symbols = self.waveform.map(data_bits)
        signal = self.waveform.modulate(data_symbols)

        energy = np.linalg.norm(signal.samples) ** 2 / self.waveform.bits_per_frame
        self.assertAlmostEqual(energy, self.waveform.bit_energy, places=2)

    def test_symbol_energy(self) -> None:
        """Symbol energy property should compute correct symbol energy."""

        self.waveform.pilot_rate = 0.
        self.waveform.guard_interval = 0.

        data_bits = self.rng.integers(0, 2, self.waveform.bits_per_frame)
        data_symbols = self.waveform.map(data_bits)
        signal = self.waveform.modulate(data_symbols)

        energy = np.linalg.norm(signal.samples) ** 2 / self.waveform.symbols_per_frame
        self.assertAlmostEqual(energy, self.waveform.symbol_energy, places=1)

    def test_power(self) -> None:
        """Power property should compute correct bit power."""

        self.waveform.pilot_rate = 0.
        self.waveform.guard_interval = 0.

        data_bits = self.rng.integers(0, 2, self.waveform.bits_per_frame)
        data_symbols = self.waveform.map(data_bits)
        signal = self.waveform.modulate(data_symbols)

        energy = np.linalg.norm(signal.samples) ** 2 / self.waveform.samples_in_frame
        self.assertAlmostEqual(energy, self.waveform.power, places=2)

    def test_sampling_rate(self) -> None:
        """Sampling rate property should compute correct sampling rate."""

        self.assertEqual(self.oversampling_factor * self.symbol_rate, self.waveform.sampling_rate)


class TestPskQamCorrelationSynchronization(TestCase):
    """Test the correlation-based synchronization routine."""

    def setUp(self) -> None:

        self.rng = default_rng(42)
        self.synchronization = SingleCarrierCorrelationSynchronization()
        self.waveform = MockSingleCarrierWaveform(symbol_rate=1e6, num_preamble_symbols=10, num_data_symbols=50)
        self.waveform.synchronization = self.synchronization
        self.waveform.guard_interval = 0.

    def test_delay_synchronization(self) -> None:
        """Test synchronization with arbitrary sample offset."""

        bits = self.rng.integers(0, 2, self.waveform.bits_per_frame)
        symbols = self.waveform.map(bits)

        signal = self.waveform.modulate(symbols)

        for offset in [0, 1, 10, 15, 20]:

            samples = np.append(np.zeros((1, offset), dtype=complex), signal.samples)

            frames = self.synchronization.synchronize(samples, ChannelStateInformation.Ideal(signal.num_samples))

            self.assertEqual(len(frames), 1)
            assert_array_equal(frames[0][0], signal.samples)

    def test_phase_shift_synchronization(self) -> None:
        """Test synchronization with arbitrary sample offset and phase shift."""

        bits = self.rng.integers(0, 2, self.waveform.bits_per_frame)
        symbols = self.waveform.map(bits)

        samples = self.waveform.modulate(symbols).samples * np.exp(0.24567j * pi)
        padded_samples = np.append(np.zeros((1, 15), dtype=complex), samples)

        frames = self.synchronization.synchronize(padded_samples, ChannelStateInformation.Ideal(len(padded_samples)))

        self.assertEqual(len(frames), 1)
        assert_array_almost_equal(frames[0][0], samples)


class TestRootRaisedCosineWaveform(TestCase):
    """Test the Root-Raised-Cosine pulse shape for single carrier modulation"""

    def setUp(self) -> None:

        self.rng = default_rng(42)

        self.oversampling_factor = 8
        self.relative_bandwidth = 1.
        self.roll_off = .9
        self.num_preamble_symbols = 10
        self.num_data_symbols = 40
        self.symbol_rate = 1e6

        self.waveform = RootRaisedCosineWaveform(
            oversampling_factor=self.oversampling_factor,
            relative_bandwidth=self.relative_bandwidth,
            roll_off=self.roll_off,
            num_preamble_symbols=self.num_preamble_symbols,
            num_data_symbols=self.num_data_symbols,
            symbol_rate=self.symbol_rate
        )

    def test_modulate_demodulate(self) -> None:
        """Test the successful modulation and demodulation of data symbols"""

        data_symbols = Symbols(np.exp(2j * pi * self.rng.uniform(0, 1, self.num_data_symbols)))

        modulation = self.waveform.modulate(data_symbols)
        demodulation, _, _ = self.waveform.demodulate(modulation.samples[0, :], ChannelStateInformation.Ideal(modulation.num_samples))

        assert_array_almost_equal(data_symbols.raw, demodulation.raw, decimal=1)

class TestRaisedCosineWaveform(TestCase):
    """Test the Root-Raised-Cosine pulse shape for single carrier modulation"""

    def setUp(self) -> None:

        self.rng = default_rng(42)

        self.oversampling_factor = 8
        self.relative_bandwidth = 1.
        self.roll_off = .9
        self.num_preamble_symbols = 0
        self.num_data_symbols = 1
        self.symbol_rate = 1e6

        self.waveform = RaisedCosineWaveform(
            oversampling_factor=self.oversampling_factor,
            relative_bandwidth=self.relative_bandwidth,
            roll_off=self.roll_off,
            num_preamble_symbols=self.num_preamble_symbols,
            num_data_symbols=self.num_data_symbols,
            symbol_rate=self.symbol_rate
        )

    def test_modulate_demodulate(self) -> None:
        """Test the successful modulation and demodulation of data symbols"""

        data_symbols = Symbols(np.exp(2j * pi * self.rng.uniform(0, 1, self.num_data_symbols)))

        modulation = self.waveform.modulate(data_symbols)
        demodulation, _, _ = self.waveform.demodulate(modulation.samples[0, :], ChannelStateInformation.Ideal(modulation.num_samples))

        assert_array_almost_equal(data_symbols.raw, demodulation.raw, decimal=1)


class TestRectangularWaveform(TestCase):
    """Test the Root-Raised-Cosine pulse shape for single carrier modulation"""

    def setUp(self) -> None:

        self.rng = default_rng(42)

        self.oversampling_factor = 4
        self.relative_bandwidth = 1.
        self.num_preamble_symbols = 10
        self.num_data_symbols = 40
        self.symbol_rate = 1e6

        self.waveform = RectangularWaveform(
            oversampling_factor=self.oversampling_factor,
            relative_bandwidth=self.relative_bandwidth,
            num_preamble_symbols=self.num_preamble_symbols,
            num_data_symbols=self.num_data_symbols,
            symbol_rate=self.symbol_rate
        )

    def test_modulate_demodulate(self) -> None:
        """Test the successful modulation and demodulation of data symbols"""

        data_symbols = Symbols(np.exp(2j * pi * self.rng.uniform(0, 1, self.num_data_symbols)))

        modulation = self.waveform.modulate(data_symbols)
        demodulation, _, _ = self.waveform.demodulate(modulation.samples[0, :], ChannelStateInformation.Ideal(modulation.num_samples))

        assert_array_almost_equal(data_symbols.raw, demodulation.raw, decimal=1)


class TestFMCWWaveform(TestCase):
    """Test the FMCW pulse shape for single carrier modulation"""

    def setUp(self) -> None:

        self.rng = default_rng(42)

        self.oversampling_factor = 4
        self.bandwidth = 2.5e6
        self.num_preamble_symbols = 10
        self.num_data_symbols = 40
        self.symbol_rate = 1e6

        self.waveform = FMCWWaveform(
            oversampling_factor=self.oversampling_factor,
            bandwidth=self.bandwidth,
            num_preamble_symbols=self.num_preamble_symbols,
            num_data_symbols=self.num_data_symbols,
            symbol_rate=self.symbol_rate
        )

    def test_modulate_demodulate(self) -> None:
        """Test the successful modulation and demodulation of data symbols"""

        data_symbols = Symbols(np.exp(2j * pi * self.rng.uniform(0, 1, self.num_data_symbols)))

        modulation = self.waveform.modulate(data_symbols)
        demodulation, _, _ = self.waveform.demodulate(modulation.samples[0, :], ChannelStateInformation.Ideal(modulation.num_samples))

        assert_array_almost_equal(data_symbols.raw, demodulation.raw, decimal=1)
