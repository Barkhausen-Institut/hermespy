# -*- coding: utf-8 -*-
"""Waveform Generation for Phase-Shift-Keying Quadrature Amplitude Modulation Testing."""

import unittest
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.constants import pi

from hermespy.channel import ChannelStateInformation
from hermespy.modem.waveform_generator_psk_qam import WaveformGeneratorPskQam
from hermespy.modem.tools import ShapingFilter

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestWaveformGeneratorPskQam(unittest.TestCase):
    """Test the Phase-Shift-Keying / Quadrature Amplitude Modulation Waveform Generator."""

    def setUp(self) -> None:

        self.rng = np.random.default_rng(42)

        self.modem = Mock()
        self.modem.carrier_frequency = 100e6

        self.filter_type = 'ROOT_RAISED_COSINE'
        self.symbol_rate = 125e3
        self.oversampling_factor = 8
        self.modulation_order = 16
        self.guard_interval = 1e-3
        self.num_data_symbols = 1000

        shaping_filter = ShapingFilter(filter_type=self.filter_type,
                                       samples_per_symbol=self.oversampling_factor)

        self.tx_filter = ShapingFilter(filter_type=self.filter_type,
                                       samples_per_symbol=self.oversampling_factor,
                                       is_matched=False,
                                       length_in_symbols=shaping_filter.length_in_symbols,
                                       roll_off=shaping_filter.roll_off,
                                       bandwidth_factor=1.)

        self.rx_filter = ShapingFilter(filter_type=self.filter_type,
                                       samples_per_symbol=self.oversampling_factor,
                                       is_matched=True,
                                       length_in_symbols=shaping_filter.length_in_symbols,
                                       roll_off=shaping_filter.roll_off,
                                       bandwidth_factor=1.)

        self.generator = WaveformGeneratorPskQam(modem=self.modem, symbol_rate=self.symbol_rate,
                                                 oversampling_factor=self.oversampling_factor,
                                                 modulation_order=self.modulation_order,
                                                 guard_interval=self.guard_interval,
                                                 tx_filter=self.tx_filter,
                                                 rx_filter=self.rx_filter,
                                                 num_data_symbols=self.num_data_symbols)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as object attributes."""

        self.assertIs(self.modem, self.generator.modem)
        self.assertEqual(self.symbol_rate, self.generator.symbol_rate)
        self.assertEqual(self.oversampling_factor, self.generator.oversampling_factor)
        self.assertEqual(self.guard_interval, self.generator.guard_interval)
        self.assertIs(self.rx_filter, self.generator.rx_filter)
        self.assertIs(self.tx_filter, self.generator.tx_filter)
        self.assertEqual(self.num_data_symbols, self.generator.num_data_symbols)

    def test_symbol_rate_setget(self) -> None:
        """Symbol rate property getter should return setter argument."""

        symbol_rate = 10
        self.generator.symbol_rate = symbol_rate

        self.assertEqual(symbol_rate, self.generator.symbol_rate)

    def test_symbol_rate_validation(self) -> None:
        """Symbol rate property should raise ValueError on arguments smaller than zero."""

        with self.assertRaises(ValueError):
            self.generator.symbol_rate = -1.0

        try:
            self.generator.symbol_rate = 0.

        except ValueError:
            self.fail()

    def test_map_unmap(self) -> None:
        """Mapping and subsequently un-mapping a bit stream should yield identical bits."""

        expected_bits = self.rng.integers(0, 2, self.generator.bits_per_frame)

        symbols = self.generator.map(expected_bits)
        bits = self.generator.unmap(symbols)

        assert_array_equal(expected_bits, bits)

    def test_modulate_demodulate_no_filter(self) -> None:
        """Modulating and subsequently de-modulating a symbol stream should yield identical symbols."""

        self.generator.rx_filter = ShapingFilter(ShapingFilter.FilterType.NONE, self.oversampling_factor)
        self.generator.tx_filter = ShapingFilter(ShapingFilter.FilterType.NONE, self.oversampling_factor)

        expected_symbols = (np.exp(2j * self.rng.uniform(0, pi, self.generator.symbols_per_frame)) *
                            np.arange(1, 1 + self.generator.symbols_per_frame))

        baseband_signal = self.generator.modulate(expected_symbols)
        channel_state = ChannelStateInformation.Ideal(num_samples=baseband_signal.num_samples)
        symbols, _, _ = self.generator.demodulate(baseband_signal.samples[0, :], channel_state)

        assert_array_almost_equal(expected_symbols, symbols)

    def test_modulate_demodulate(self) -> None:
        """Modulating and subsequently de-modulating a symbol stream should yield identical symbols."""

        expected_symbols = (np.exp(2j * self.rng.uniform(0, pi, self.generator.symbols_per_frame)) *
                            np.arange(1, 1 + self.generator.symbols_per_frame))

        baseband_signal = self.generator.modulate(expected_symbols)
        channel_state = ChannelStateInformation.Ideal(num_samples=baseband_signal.num_samples)
        symbols, _, _ = self.generator.demodulate(baseband_signal.samples[0, :], channel_state)

        assert_array_almost_equal(expected_symbols, symbols)

    def test_equalization_setget(self) -> None:
        """Equalization property getter should return setter argument."""

        equalization = self.generator.Equalization.MMSE
        self.generator.equalization = equalization

        self.assertEqual(equalization, self.generator.equalization)
        
    def test_guard_interval_setget(self) -> None:
        """Guard interval property getter should return setter argument."""
        
        guard_interval = 1.23
        self.generator.guard_interval = 1.23
        
        self.assertEqual(guard_interval, self.generator.guard_interval)
        
    def test_guard_interval_validation(self) -> None:
        """Guard interval property should raise ValueError on arguments smaller than zero."""

        with self.assertRaises(ValueError):
            self.generator.guard_interval = -1.0

        try:
            self.generator.guard_interval = 0.

        except ValueError:
            self.fail()

    def test_pilot_rate_setget(self) -> None:
        """Pilot rate property getter should return setter argument."""

        pilot_rate = 1.23
        self.generator.pilot_rate = 1.23

        self.assertEqual(pilot_rate, self.generator.pilot_rate)

    def test_symbol_samples_in_frame(self) -> None:
        """Symbol samples in frame property should compute the correct sample count."""

        self.generator.tx_filter = ShapingFilter(ShapingFilter.FilterType.NONE, self.oversampling_factor)
        symbols = np.exp(2j * self.rng.uniform(0, pi, self.generator.symbols_per_frame))
        signal = self.generator.modulate(symbols)

        self.assertEqual(signal.num_samples, self.generator.symbol_samples_in_frame)

    def test_samples_in_frame(self) -> None:
        """Samples in frame property should compute the correct sample count."""

        symbols = np.exp(2j * self.rng.uniform(0, pi, self.generator.symbols_per_frame))
        signal = self.generator.modulate(symbols)

        self.assertEqual(signal.num_samples, self.generator.samples_in_frame)

    def test_pilot_rate_validation(self) -> None:
        """Pilot rate property should raise ValueError on arguments smaller than zero."""

        with self.assertRaises(ValueError):
            self.generator.pilot_rate = -1.0

        try:
            self.generator.pilot_rate = 0.

        except ValueError:
            self.fail()
            
    def test_num_data_symbols_setget(self) -> None:
        """Number of pilot symbols property getter should return setter argument."""

        num_data_symbols = 1.23
        self.generator.num_data_symbols = 1.23

        self.assertEqual(num_data_symbols, self.generator.num_data_symbols)

    def test_num_data_symbols_validation(self) -> None:
        """Number of pilot symbols property setter should raise ValueError on arguments smaller than zero."""

        with self.assertRaises(ValueError):
            self.generator.num_data_symbols = -1

        try:
            self.generator.num_data_symbols = 0
            self.generator.num_data_symbols = 10

        except ValueError:
            self.fail()

    def test_bits_per_frame(self) -> None:
        """Bits per frame property should compute correct amount of data bits per frame."""

        signal = (self.rng.normal(0, 1.0, self.generator.samples_in_frame) +
                  1j * self.rng.normal(0, 1.0, self.generator.samples_in_frame))
        channel_state = ChannelStateInformation.Ideal(self.generator.samples_in_frame)

        data_symbols, _, _ = self.generator.demodulate(signal, channel_state)
        bits = self.generator.unmap(data_symbols)

        self.assertEqual(len(bits), self.generator.bits_per_frame)

    def test_symbols_per_frame(self) -> None:
        """Symbols per frame property should compute correct amount of symbols per frame."""

        signal = (self.rng.normal(0, 1.0, self.generator.samples_in_frame) +
                  1j * self.rng.normal(0, 1.0, self.generator.samples_in_frame))
        channel_state = ChannelStateInformation.Ideal(self.generator.samples_in_frame)

        symbols, _, _ = self.generator.demodulate(signal, channel_state)

        self.assertEqual(len(symbols), self.generator.symbols_per_frame)

    def test_bit_energy(self) -> None:
        """Bit energy property should compute correct bit energy."""

        self.generator.pilot_rate = 0.
        self.generator.guard_interval = 0.

        data_bits = self.rng.integers(0, 2, self.generator.bits_per_frame)
        data_symbols = self.generator.map(data_bits)
        signal = self.generator.modulate(data_symbols)

        energy = np.linalg.norm(signal.samples) ** 2 / self.generator.bits_per_frame
        self.assertAlmostEqual(energy, self.generator.bit_energy, places=2)

    def test_symbol_energy(self) -> None:
        """Symbol energy property should compute correct symbol energy."""

        self.generator.pilot_rate = 0.
        self.generator.guard_interval = 0.

        data_bits = self.rng.integers(0, 2, self.generator.bits_per_frame)
        data_symbols = self.generator.map(data_bits)
        signal = self.generator.modulate(data_symbols)

        energy = np.linalg.norm(signal.samples) ** 2 / self.generator.symbols_per_frame
        self.assertAlmostEqual(energy, self.generator.symbol_energy, places=1)

    def test_power(self) -> None:
        """Power property should compute correct bit power."""

        self.generator.pilot_rate = 0.
        self.generator.guard_interval = 0.

        data_bits = self.rng.integers(0, 2, self.generator.bits_per_frame)
        data_symbols = self.generator.map(data_bits)
        signal = self.generator.modulate(data_symbols)

        energy = np.linalg.norm(signal.samples) ** 2 / self.generator.samples_in_frame
        self.assertAlmostEqual(energy, self.generator.power, places=2)

    def test_sampling_rate(self) -> None:
        """Sampling rate property should compute correct sampling rate."""

        self.assertEqual(8 * 125e3, self.generator.sampling_rate)

    def test_to_yaml(self) -> None:
        """Serialization to YAML."""
        pass

    def test_from_yaml(self) -> None:
        """Serialization from YAML."""
        pass
