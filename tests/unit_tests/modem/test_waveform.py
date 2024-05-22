# -*- coding: utf-8 -*-
"""Test prototype for waveform generation modeling"""

from typing import Optional
import unittest
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
import numpy.random as rnd
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.constants import pi

from hermespy.core import ChannelStateFormat, ChannelStateInformation, Signal
from hermespy.modem.symbols import StatedSymbols, Symbols
from hermespy.modem.waveform import ChannelEqualization, ChannelEstimation
from hermespy.modem import ChannelEstimation, ConfigurablePilotWaveform, CustomPilotSymbolSequence, CommunicationWaveform, UniformPilotSymbolSequence, StatedSymbols, Synchronization, Symbols, ChannelEqualization, ZeroForcingChannelEqualization
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockCommunicationWaveform(CommunicationWaveform):
    """Mock communication waveform for modem testing"""

    symbol_rate = 1e9

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, modulation_order=2, **kwargs)

    @property
    def symbols_per_frame(self) -> int:
        return 100

    @property
    def num_data_symbols(self) -> int:
        return self.symbols_per_frame

    @property
    def bit_energy(self) -> float:
        return 1.0

    @property
    def symbol_energy(self) -> float:
        return 1.0

    @property
    def power(self) -> float:
        return 1.0

    @property
    def samples_per_frame(self) -> int:
        return self.symbols_per_frame * self.oversampling_factor

    @property
    def symbol_duration(self) -> float:
        return 1 / self.symbol_rate

    def map(self, data_bits: np.ndarray) -> Symbols:
        return Symbols(data_bits[np.newaxis, :, np.newaxis])

    def unmap(self, symbols: Symbols) -> np.ndarray:
        return symbols.raw.real.flatten()

    def place(self, symbols: Symbols) -> Symbols:
        return symbols

    def pick(self, placed_symbols: StatedSymbols) -> StatedSymbols:
        return placed_symbols

    def modulate(self, data_symbols: Symbols) -> np.ndarray:
        return data_symbols.raw.flatten().repeat(self.oversampling_factor)

    def demodulate(self, signal: np.ndarray) -> Symbols:
        symbols = Symbols(signal[np.newaxis, : self.oversampling_factor * self.symbols_per_frame : self.oversampling_factor, np.newaxis])
        return symbols

    @property
    def bandwidth(self) -> float:
        return self.sampling_rate

    @property
    def sampling_rate(self) -> float:
        return self.symbol_rate * self.oversampling_factor


class MockPilotCommunicationWaveform(MockCommunicationWaveform, ConfigurablePilotWaveform):
    """Mock communication waveform for modem testing"""

    def pilot_signal(self) -> Signal:
        return Signal.Create(np.zeros(self.samples_per_frame), self.sampling_rate)


class TestSynchronization(unittest.TestCase):
    """Test waveform generator synchronization base class"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.synchronization = Synchronization()
        self.waveform = MockCommunicationWaveform()
        self.waveform.synchronization = self.synchronization

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as object attributes"""

        self.assertIs(self.waveform, self.synchronization.waveform)

    def test_waveform_setget(self) -> None:
        """Waveform generator property getter should return setter argument"""

        expected_waveform = Mock()
        self.synchronization.waveform = expected_waveform

        self.assertIs(expected_waveform, self.synchronization.waveform)

        self.synchronization.waveform = None
        self.assertIs(None, self.synchronization.waveform)

    def test_synchronize_validation(self) -> None:
        """Synchronization shoul raise RuntimeError if no waveform is assigned"""

        self.synchronization.waveform = None

        with self.assertRaises(RuntimeError):
            _ = self.synchronization.synchronize(np.zeros(10))

    def test_synchronize(self) -> None:
        """Default synchronization should properly split signals into frame-sections"""

        num_streams = 3
        num_frames = 1
        num_offset_samples = 2
        num_samples = num_frames * self.waveform.samples_per_frame + num_offset_samples

        signal = np.exp(2j * self.rng.uniform(0, pi, (num_streams, 1))) @ np.exp(2j * self.rng.uniform(0, pi, (1, num_samples)))
        frames = self.synchronization.synchronize(signal)
        self.assertEqual(num_frames, len(frames))

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        with patch("hermespy.modem.waveform.Synchronization.property_blacklist", new_callable=PropertyMock) as blacklist:
            blacklist.return_value = {"waveform"}
            test_yaml_roundtrip_serialization(self, self.synchronization)


class TestChannelEstimation(unittest.TestCase):
    def setUp(self) -> None:
        self.waveform = MockCommunicationWaveform()
        self.estimation = ChannelEstimation(self.waveform)

    def test_init(self) -> None:
        """Initialization should properly set class properties"""

        self.assertIs(self.waveform, self.estimation.waveform)

    def test_waveform(self) -> None:
        """Waveform generator property should properly rebind estimations"""

        new_waveform = MockCommunicationWaveform()

        self.estimation.waveform = new_waveform
        self.assertIsNot(self.waveform.channel_estimation, self.estimation)
        self.assertIs(new_waveform, self.estimation.waveform)

        self.estimation.waveform = None
        self.assertIsNot(new_waveform, self.estimation.waveform)
        self.assertIsNone(self.estimation.waveform)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        with patch("hermespy.modem.waveform.ChannelEstimation.property_blacklist", new_callable=PropertyMock) as blacklist:
            blacklist.return_value = {"waveform"}
            test_yaml_roundtrip_serialization(self, self.estimation)


class TestChannelEqualization(unittest.TestCase):
    """Test channel equalization base class"""

    def setUp(self) -> None:
        self.waveform = MockCommunicationWaveform()
        self.equalization = ChannelEqualization(self.waveform)

    def test_waveform_validation(self) -> None:
        """Waveform setter should raise RuntimeError if already assigned to a waveform"""

        with self.assertRaises(RuntimeError):
            self.equalization.waveform = MockCommunicationWaveform()

    def test_equalize_channel(self) -> None:
        """Equalization should be a stub returing the symbols"""

        symbols = Mock()
        equalized_symbols = self.equalization.equalize_channel(symbols)

        self.assertIs(symbols, equalized_symbols)


class TestZeroForcingEqualization(unittest.TestCase):
    """Test zero-forcing channel equalization"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        self.waveform = MockCommunicationWaveform()
        self.equalization = ZeroForcingChannelEqualization(self.waveform)

        num_bits = self.waveform.bits_per_frame(self.waveform.num_data_symbols)
        self.raw_symbols = self.waveform.map(self.rng.uniform(0, 2, num_bits))
        self.raw_state = np.ones((1, 1, self.raw_symbols.num_blocks, self.raw_symbols.num_symbols))
        self.symbols = StatedSymbols(self.raw_symbols.raw, self.raw_state)

    def test_siso_equalization(self) -> None:
        """Test ZF equalization in the SISO case"""

        equalized_symbols = self.equalization.equalize_channel(self.symbols)
        assert_array_almost_equal(self.symbols.raw, equalized_symbols.raw)

    def test_simo_equalization(self) -> None:
        """Test ZF equalization in the SIMO case"""

        self.raw_state = np.ones((2, 1, self.raw_symbols.num_blocks, self.raw_symbols.num_symbols))
        propagated_symbols = StatedSymbols(np.repeat(self.raw_symbols.raw, 2, axis=0), self.raw_state)

        equalized_symbols = self.equalization.equalize_channel(propagated_symbols)
        assert_array_almost_equal(self.symbols.raw, equalized_symbols.raw)


class TestCommunicationWaveform(unittest.TestCase):
    """Test the communication waveform generator unit"""

    def setUp(self) -> None:
        self.rnd = rnd.default_rng(42)
        self.modem = Mock()

        self.waveform = MockCommunicationWaveform(modem=self.modem)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as attributes"""

        self.assertIs(self.modem, self.waveform.modem)

    def test_oversampling_factor_validation(self) -> None:
        """Oversampling factor property setter should raise a ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.waveform.oversampling_factor = 0

        with self.assertRaises(ValueError):
            self.waveform.oversampling_factor = -1

    def test_oversampling_factor(self) -> None:
        """Oversampling factor property getter should return setter argument"""

        oversampling_factor = 4
        self.waveform.oversampling_factor = oversampling_factor

        self.assertEqual(oversampling_factor, self.waveform.oversampling_factor)

    def test_modulation_order_setget(self) -> None:
        """Modulation order property getter should return setter argument"""

        order = 256
        self.waveform.modulation_order = order

        self.assertEqual(order, self.waveform.modulation_order)

    def test_modulation_order_validation(self) -> None:
        """Modulation order property setter should raise ValueErrors on arguments which aren't powers of two"""

        with self.assertRaises(ValueError):
            self.waveform.modulation_order = 0

        with self.assertRaises(ValueError):
            self.waveform.modulation_order = -1

        with self.assertRaises(ValueError):
            self.waveform.modulation_order = 20

    def test_synchronize(self) -> None:
        """Default synchronization routine should properly split signals into frame-sections"""

        num_streams = 3
        num_samples_test = [50, 100, 150, 200]
        expected_num_frames_candidates = [0, 1, 1, 1]

        for num_samples, expected_num_frames in zip(num_samples_test, expected_num_frames_candidates):
            signal = np.exp(2j * self.rnd.uniform(0, pi, (num_streams, 1))) @ np.exp(2j * self.rnd.uniform(0, pi, (1, num_samples)))

            synchronized_frames = self.waveform.synchronization.synchronize(signal)

            # Number of frames is the number of frames that fit into the samples
            num_frames = len(synchronized_frames)
            self.assertEqual(expected_num_frames, num_frames)

    def test_synchronize_validation(self) -> None:
        """Synchronization should raise a ValueError if the signal shape does match the stream response shape"""

        with self.assertRaises(ValueError):
            _ = self.waveform.synchronization.synchronize(np.zeros(10), ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE, np.zeros((10, 2))))

        with self.assertRaises(ValueError):
            _ = self.waveform.synchronization.synchronize(np.zeros((10, 2)), ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE, np.zeros((10, 2))))

    def test_data_rate(self) -> None:
        """Data rate method should compute the correct data rate"""

        num_data_symbols = 10
        a = self.waveform.bits_per_frame(num_data_symbols)
        b = self.waveform.frame_duration

        self.assertAlmostEqual(a / b, self.waveform.data_rate(num_data_symbols))

    def test_modem_set_none(self) -> None:
        """Modem property setter should abort if nothings to be done"""

        self.waveform = MockCommunicationWaveform()
        self.waveform.modem = None

        self.assertIsNone(self.waveform.modem)

    def test_modem_setget(self) -> None:
        """Modem property getter should return setter argument"""

        modem = Mock()

        self.waveform.modem = modem

        self.assertIs(self.waveform, modem.waveform)
        self.assertIs(self.waveform.modem, modem)

        self.waveform.modem = None

        self.assertIsNone(modem.waveform)
        self.assertIsNone(self.waveform.modem)

    def test_symbol_precoding_support(self) -> None:
        """Symbol precoding should be supported"""

        self.assertTrue(self.waveform.symbol_precoding_support)


class TestUniformPilotSymbolSequence(unittest.TestCase):
    """Test the uniform pilot symbol sequence"""

    def test_sequence(self) -> None:
        """The generated sequence should be an array containing only the configured symbol"""

        expected_symbol = 1.234 - 1234j
        uniform_sequence = UniformPilotSymbolSequence(expected_symbol)

        assert_array_equal(np.array([expected_symbol], dtype=complex), uniform_sequence.sequence)


class TestConfigurablePilotWaveform(unittest.TestCase):
    """Test the configurable pilot waveform"""

    def setUp(self) -> None:
        self.pilot_symbols = np.array([1.234 - 1234j, 2.345 + 2345j, 3.456 - 3456j])
        self.pilot_sequence = CustomPilotSymbolSequence(self.pilot_symbols)

        self.waveform = MockPilotCommunicationWaveform()
        self.waveform.pilot_symbol_sequence = self.pilot_sequence

    def test_pilot_symbols_validation(self) -> None:
        """Pilot symbol generator should raise RuntimeError if repetition required but flag disabled"""

        self.waveform.repeat_pilot_symbol_sequence = False

        with self.assertRaises(RuntimeError):
            self.waveform.pilot_symbols(9)

    def test_pilot_symbols(self) -> None:
        """Pilot symbols generator routine should compute correct pilot symbol sequences"""

        symbols = self.waveform.pilot_symbols(3)
        assert_array_equal(self.pilot_symbols, symbols)
