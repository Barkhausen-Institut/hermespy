# -*- coding: utf-8 -*-
"""Waveform Generation for Phase-Shift-Keying Quadrature Amplitude Modulation Testing"""

from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.constants import pi

from hermespy.core import Signal
from hermespy.modem import FilteredSingleCarrierWaveform, StatedSymbols, Symbols, RaisedCosineWaveform, RootRaisedCosineWaveform, RectangularWaveform, FMCWWaveform, \
    SingleCarrierCorrelationSynchronization, SingleCarrierIdealChannelEstimation, SingleCarrierZeroForcingChannelEqualization, SingleCarrierMinimumMeanSquareChannelEqualization
from hermespy.modem.waveform_single_carrier import SingleCarrierLeastSquaresChannelEstimation, RolledOffSingleCarrierWaveform
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockSingleCarrierWaveform(FilteredSingleCarrierWaveform):
    """Implementation of the abstract single carrier waveform base class for testing"""

    def _transmit_filter(self) -> np.ndarray:

        filter = np.zeros(self.oversampling_factor, dtype=np.complex_)
        filter[0] = 1.
        return filter

    def _receive_filter(self) -> np.ndarray:

        filter = np.zeros(self.oversampling_factor, dtype=np.complex_)
        filter[0] = 1.
        return filter

    @property
    def _filter_delay(self) -> int:

        return 0

    @property
    def bandwidth(self) -> float:

        return self.symbol_rate
    
    
class MockRolledOfSingleCarrierWaveform(RolledOffSingleCarrierWaveform):
    """Implementation of the abstract single carrier waveform base class for testing"""
    
    def _base_filter(self) -> np.ndarray:
        
        filter = np.zeros(self.oversampling_factor, dtype=np.complex_)
        filter[0] = 1.
        return filter
    

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
        pilot_signal = self.waveform.pilot_signal
        self.assertEqual(self.waveform.num_preamble_symbols * self.waveform.oversampling_factor, pilot_signal.num_samples)

        self.waveform.num_preamble_symbols = 0       
        pilot_signal = self.waveform.pilot_signal
        self.assertEqual(0, pilot_signal.num_samples)

    def test_map_unmap(self) -> None:
        """Mapping and subsequently un-mapping a bit stream should yield identical bits"""

        expected_bits = self.rng.integers(0, 2, self.waveform.bits_per_frame(self.waveform.num_data_symbols))

        symbols = self.waveform.map(expected_bits)
        bits = self.waveform.unmap(symbols)

        assert_array_equal(expected_bits, bits)

    def test_modulate_demodulate(self) -> None:
        """Modulating and subsequently de-modulating a symbol stream should yield identical symbols"""

        expected_symbols = Symbols(np.exp(2j * self.rng.uniform(0, pi, (1, self.waveform._num_frame_symbols, 1))))

        baseband_signal = self.waveform.modulate(expected_symbols)
        symbols = self.waveform.demodulate(baseband_signal)

        assert_array_almost_equal(expected_symbols.raw, symbols.raw, decimal=1)
        
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
            self.waveform.guard_interval = 0.

        except ValueError:
            self.fail()
            
    def test_num_guard_samples(self) -> None:
        """Number of guard samples property should compute correct number of guard samples"""
        
        self.waveform.guard_interval = 1e-3
        self.assertEqual(self.waveform.guard_interval * self.waveform.sampling_rate, self.waveform.num_guard_samples)

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

    def test_samples_per_frame(self) -> None:
        """Samples per frame property should compute the correct sample count"""

        symbols = Symbols(np.exp(2j * self.rng.uniform(0, pi, self.waveform._num_frame_symbols)))
        signal = self.waveform.modulate(symbols)

        self.assertEqual(signal.shape[0], self.waveform.samples_per_frame)
            
    def test_num_data_symbols_setget(self) -> None:
        """Number of pilot symbols property getter should return setter argument"""

        num_data_symbols = 1.23
        self.waveform.num_data_symbols = 1.23

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

    def test_bit_energy(self) -> None:
        """Bit energy property should compute correct bit energy"""

        self.waveform.pilot_rate = 0.
        self.waveform.guard_interval = 0.

        bits_per_frame = self.waveform.bits_per_frame(self.waveform.num_data_symbols)
        data_bits = self.rng.integers(0, 2, bits_per_frame)
        data_symbols = self.waveform.map(data_bits)
        signal = self.waveform.modulate(data_symbols)

        energy = np.linalg.norm(signal) ** 2 / bits_per_frame
        self.assertAlmostEqual(energy, self.waveform.bit_energy, places=2)

    def test_symbol_energy(self) -> None:
        """Symbol energy property should compute correct symbol energy"""

        self.waveform.pilot_rate = 0.
        self.waveform.guard_interval = 0.

        data_bits = self.rng.integers(0, 2, self.waveform.bits_per_frame(self.waveform.num_data_symbols))
        data_symbols = self.waveform.map(data_bits)
        signal = self.waveform.modulate(data_symbols)

        energy = np.linalg.norm(signal) ** 2 / self.waveform._num_frame_symbols
        self.assertAlmostEqual(energy, self.waveform.symbol_energy, places=1)

    def test_power(self) -> None:
        """Power property should compute correct bit power"""

        self.waveform.pilot_rate = 0.
        self.waveform.guard_interval = 0.

        data_bits = self.rng.integers(0, 2, self.waveform.num_data_symbols)
        data_symbols = self.waveform.map(data_bits)
        signal_samples = self.waveform.modulate(data_symbols)

        self.assertAlmostEqual(Signal(signal_samples, self.waveform.sampling_rate).power[0], self.waveform.power, places=2)

    def test_sampling_rate(self) -> None:
        """Sampling rate property should compute correct sampling rate"""

        self.assertEqual(self.oversampling_factor * self.symbol_rate, self.waveform.sampling_rate)

    def test_plot_filter_correlation(self) -> None:
        """Plotting the filter correlation should generate a figure"""

        with patch('matplotlib.pyplot.subplots') as subplots_mock:
            
            figure = Mock()
            axes = Mock()
            subplots_mock.return_value = (figure, axes)
            
            generated_figure = self.waveform.plot_filter_correlation()
            
            axes.plot.assert_called_once()
            self.assertIs(figure, generated_figure)
            
    def test_plot_filter(self) -> None:
        """Plotting the filter should generate a figure"""
        
        with patch('matplotlib.pyplot.subplots') as subplots_mock:
            
            figure = Mock()
            axes = Mock()
            subplots_mock.return_value = (figure, axes)
            
            generated_figure = self.waveform.plot_filter()
            
            axes.plot.assert_called()
            self.assertIs(figure, generated_figure)


class TestSingleCarrierCorrelationSynchronization(TestCase):
    """Test the correlation-based synchronization routine"""

    def setUp(self) -> None:

        self.rng = default_rng(42)
        self.synchronization = SingleCarrierCorrelationSynchronization()
        self.waveform = MockSingleCarrierWaveform(symbol_rate=1e6, num_preamble_symbols=10, num_data_symbols=50)
        self.waveform.synchronization = self.synchronization
        self.waveform.guard_interval = 0.

    def test_delay_synchronization(self) -> None:
        """Test synchronization with arbitrary sample offset"""

        bits = self.rng.integers(0, 2, self.waveform.bits_per_frame(self.waveform.num_data_symbols))
        symbols = self.waveform.map(bits)

        signal = self.waveform.modulate(self.waveform.place(symbols))

        for offset in [0, 1, 10, 15, 20]:

            samples = np.append(np.zeros((1, offset), dtype=complex), signal)

            pilot_indices = self.synchronization.synchronize(samples)
            self.assertCountEqual([offset], pilot_indices)

    def test_phase_shift_synchronization(self) -> None:
        """Test synchronization with arbitrary sample offset and phase shift"""

        bits = self.rng.integers(0, 2, self.waveform.bits_per_frame(self.waveform.num_data_symbols))
        symbols = self.waveform.map(bits)

        samples = self.waveform.modulate(self.waveform.place(symbols)) * np.exp(0.24567j * pi)
        padded_samples = np.append(np.zeros((1, 15), dtype=complex), samples)

        pilot_indices = self.synchronization.synchronize(padded_samples,)
        self.assertCountEqual([15], pilot_indices)


class TestSingleCarrierChannelEstimation(TestCase):
    """Test channel estimation of single carrier waveforms"""
    
    def setUp(self) -> None:
        
        self.rng = default_rng(42)
        
        self.waveform = MockSingleCarrierWaveform(symbol_rate=1e6,
                                                  num_preamble_symbols=3,
                                                  num_postamble_symbols=3,
                                                  num_data_symbols=100,
                                                  pilot_rate=10)
        
        mapped_data_symbols = self.waveform.map(self.rng.uniform(0, 2, self.waveform.bits_per_frame(self.waveform.num_data_symbols)))
        self.symbols = self.waveform.place(mapped_data_symbols)
        
    def test_least_squares_validation(self) -> None:
        """Least squares channel estimation should raise FloatingError if not assigned to a waveform"""
        
        estimation = SingleCarrierLeastSquaresChannelEstimation()
        with self.assertRaises(RuntimeError):
            estimation.estimate_channel(self.symbols)
        
    def test_least_squares(self) -> None:
        """Least squares channel estimation"""
        
        estimation = SingleCarrierLeastSquaresChannelEstimation()
        self.waveform.channel_estimation = estimation
        
        stated_symbols, _ = estimation.estimate_channel(self.symbols)
        self.assertEqual(self.waveform._num_frame_symbols, stated_symbols.num_blocks)
        
    def test_ideal(self) -> None:
        """Ideal channel estimation"""

        estimation = SingleCarrierIdealChannelEstimation()
        self.waveform.channel_estimation = estimation
        
        
        with patch('hermespy.modem.waveform.IdealChannelEstimation._csi') as csi_mock:
            
            expected_csi = self.rng.standard_normal((1, 1, self.waveform._filter_delay + self.waveform.samples_per_frame, 1))
            state_mock = Mock()
            state_mock.state = expected_csi
            csi_mock.return_value = state_mock
            
            symbols, csi = estimation.estimate_channel(self.symbols)
            #self.assertEqual(self.waveform.num_data_symbols, csi.num_samples)


class TestChannelEqualization(TestCase):
    """Test channel equalization of single carrier waveforms"""
    
    def setUp(self) -> None:
        
        self.rng = default_rng(42)
        
        self.waveform = MockSingleCarrierWaveform(symbol_rate=1e6,
                                                  num_preamble_symbols=3,
                                                  num_postamble_symbols=3,
                                                  num_data_symbols=100,
                                                  pilot_rate=10)
        
        self.raw_symbols = self.waveform.map(self.rng.uniform(0, 2, self.waveform.bits_per_frame(self.waveform.num_data_symbols)))
        self.raw_state = np.ones((1, 1, self.raw_symbols.num_blocks, self.raw_symbols.num_symbols))
        self.symbols = StatedSymbols(self.raw_symbols.raw, self.raw_state)
    
        self.modem_patch = patch('hermespy.modem.waveform.WaveformGenerator.modem')
        self.modem_mock = self.modem_patch.start()  
        self.modem_mock.receiving_device.snr = np.inf
        
    def tearDown(self) -> None:
        
        self.modem_patch.stop()
        
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


class TestRolledOfFilteredSingleCarrierWaveform(TestCase):
    """Test the rolled-off filtered single carrier waveform generator"""
    
    def setUp(self) -> None:
        
        self.symbol_rate = 1e6
        self.num_preamble_symbols = 0
        self.num_data_symbols = 10
        self.waveform = MockRolledOfSingleCarrierWaveform(symbol_rate=self.symbol_rate, num_preamble_symbols=self.num_preamble_symbols, num_data_symbols=self.num_data_symbols)
        
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
            self.waveform.relative_bandwidth = 0.
            
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
        
    def test_bandwdith(self) -> None:
        """Bandwidth property should compute correct bandwidth"""
        
        self.waveform.symbol_rate = 1e6
        self.waveform.relative_bandwidth = 1.
        self.waveform.roll_off = 0.
        
        self.assertEqual(self.waveform.symbol_rate, self.waveform.bandwidth)


class TestRootRaisedCosineWaveform(TestCase):
    """Test the Root-Raised-Cosine pulse shape for single carrier modulation"""

    def setUp(self) -> None:

        self.rng = default_rng(42)

        self.oversampling_factor = 8
        self.relative_bandwidth = 1.
        self.roll_off = .25
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

    def test_rolled_off_modulate_demodulate(self) -> None:
        """Test the successful modulation and demodulation of data symbols with roll-off"""

        expected_symbols = Symbols(np.exp(2j * pi * self.rng.uniform(0, 1, (1, self.waveform._num_frame_symbols, 1))))
        self.waveform.filter_length = 15 * self.oversampling_factor
        
        waveform = self.waveform.modulate(expected_symbols)
        symbols = self.waveform.demodulate(waveform)

        assert_array_almost_equal(expected_symbols.raw, symbols.raw, decimal=1)
            
    def test_no_rolled_off_modulate_demodulate(self) -> None:
        """Test the successful modulation and demodulation of data symbols without roll-off"""

        self.waveform.roll_off = 0.
        self.waveform.filter_length = 15
        expected_symbols = Symbols(np.exp(2j * pi * self.rng.uniform(0, 1, (1, self.waveform._num_frame_symbols, 1))))

        waveform = self.waveform.modulate(expected_symbols)
        symbols = self.waveform.demodulate(waveform)

       # assert_array_almost_equal(expected_symbols.raw, symbols.raw, decimal=1)
              
    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.waveform)


class TestRaisedCosineWaveform(TestCase):
    """Test the Root-Raised-Cosine pulse shape for single carrier modulation"""

    def setUp(self) -> None:

        self.rng = default_rng(42)

        self.oversampling_factor = 8
        self.relative_bandwidth = 1.
        self.roll_off = .5
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

    def test_rolled_off_modulate_demodulate(self) -> None:
        """Test the successful modulation and demodulation of data symbols with roll-off"""

        expected_symbols = Symbols(np.exp(2j * pi * self.rng.uniform(0, 1, self.waveform._num_frame_symbols)))
        self.waveform.filter_length = 7 * self.oversampling_factor

        waveform = self.waveform.modulate(expected_symbols)
        symbols = self.waveform.demodulate(waveform)

        assert_array_almost_equal(expected_symbols.raw, symbols.raw, decimal=1)
            
    def test_no_roll_off_modulate_demodulate(self) -> None:
        """Test the successful modulation and demodulation of data symbols without roll-off"""

        expected_symbols = Symbols(np.exp(2j * pi * self.rng.uniform(0, 1, self.waveform._num_frame_symbols)))
        self.waveform.filter_length = 7 * self.oversampling_factor
        self.waveform.roll_off = 0.

        waveform = self.waveform.modulate(expected_symbols)
        symbols = self.waveform.demodulate(waveform)

        assert_array_almost_equal(expected_symbols.raw, symbols.raw, decimal=1)
            
    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.waveform)


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
        
    def test_relative_bandwidth_validation(self) -> None:
        """Relative bandwidth property should raise ValueError on arguments smaller than zero"""
        
        with self.assertRaises(ValueError):
            self.waveform.relative_bandwidth = -1
            
        with self.assertRaises(ValueError):
            self.waveform.relative_bandwidth = 0.
            
    def test_relative_bandwidth_setget(self) -> None:
        """Relative bandwidth property getter should return setter argument"""
        
        relative_bandwidth = 1
        self.waveform.relative_bandwidth = relative_bandwidth
        
        self.assertEqual(relative_bandwidth, self.waveform.relative_bandwidth)

    def test_bandwidth(self) -> None:
        """Bandwidth property should compute correct bandwidth"""
        
        self.waveform.symbol_rate = 1e6
        self.waveform.relative_bandwidth = 1.
        
        self.assertEqual(self.waveform.symbol_rate, self.waveform.bandwidth)

    def test_modulate_demodulate(self) -> None:
        """Test the successful modulation and demodulation of data symbols"""

        expected_symbols = Symbols(np.exp(2j * pi * self.rng.uniform(0, 1, (1, self.waveform._num_frame_symbols, 1))))

        waveform = self.waveform.modulate(expected_symbols)
        symbols = self.waveform.demodulate(waveform)

        assert_array_almost_equal(expected_symbols.raw, symbols.raw, decimal=1)
            
    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.waveform)


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
        
    def test_chirp_duration_validation(self) -> None:
        """Chirp duration property should raise ValueError on arguments smaller than zero"""
        
        with self.assertRaises(ValueError):
            self.waveform.chirp_duration = -1
            
    def test_chirp_duration_setget(self) -> None:
        """Chirp duration property getter should return setter argument"""
        
        chirp_duration = .5 * self.symbol_rate
        self.waveform.chirp_duration = chirp_duration
        
        self.assertEqual(chirp_duration, self.waveform.chirp_duration)
        
    def test_bandwidth_validation(self) -> None:
        """Bandwidth property should raise ValueError on arguments smaller than zero"""
        
        with self.assertRaises(ValueError):
            self.waveform.bandwidth = -1
            
        with self.assertRaises(ValueError):
            self.waveform.bandwidth = 0.
            
    def test_bandwidth_setget(self) -> None:
        """Bandwidth property getter should return setter argument"""
        
        bandwidth = 1.2345
        self.waveform.bandwidth = bandwidth
        
        self.assertEqual(bandwidth, self.waveform.bandwidth)

    def test_modulate_demodulate(self) -> None:
        """Test the successful modulation and demodulation of data symbols"""

        expected_symbols = Symbols(np.exp(2j * pi * self.rng.uniform(0, 1, (1, self.waveform._num_frame_symbols, 1))))

        waveform = self.waveform.modulate(expected_symbols)
        symbols = self.waveform.demodulate(waveform)
        assert_array_almost_equal(expected_symbols.raw, symbols.raw, decimal=1)
            
        self.waveform.chirp_duration = .5 / self.symbol_rate

        waveform = self.waveform.modulate(expected_symbols)
        symbols = self.waveform.demodulate(waveform)
        assert_array_almost_equal(expected_symbols.raw, symbols.raw, decimal=1)
        
    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.waveform)
