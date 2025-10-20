# -*- coding: utf-8 -*-

from __future__ import annotations
import unittest
from unittest.mock import Mock
from itertools import product
from typing_extensions import override

import numpy as np
import numpy.random as rnd
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.constants import pi

from hermespy.core import Signal
from hermespy.modem.symbols import StatedSymbols, Symbols
from hermespy.modem.waveform import ChannelEqualization, ChannelEstimation
from hermespy.modem import ConfigurablePilotWaveform, CustomPilotSymbolSequence, CommunicationWaveform, UniformPilotSymbolSequence, Synchronization, ZeroForcingChannelEqualization
from unit_tests.core.test_factory import test_roundtrip_serialization  # type: ignore

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockCommunicationWaveform(CommunicationWaveform):
    """Mock communication waveform for modem testing"""

    symbol_rate = 1e9

    def __init__(self, *args, **kwargs) -> None:
        kwargs['modulation_order'] = 2
        super().__init__(*args, **kwargs)

    @property
    def symbols_per_frame(self) -> int:
        return 100

    @property
    def num_data_symbols(self) -> int:
        return self.symbols_per_frame

    def symbol_energy(self, bandwidth: float, oversampling_factor: int) -> float:
        return oversampling_factor

    @property
    def power(self) -> float:
        return 1.0

    @override
    def samples_per_frame(self, bandwidth: float, oversampling_factor: int) -> int:
        return self.symbols_per_frame * oversampling_factor

    @override
    def frame_duration(self, bandwidth: float) -> float:
        return self.symbols_per_frame / bandwidth

    @property
    def symbol_duration(self) -> float:
        return 1 / self.symbol_rate

    @override
    def map(self, data_bits: np.ndarray) -> Symbols:
        return Symbols(data_bits[np.newaxis, :, np.newaxis])

    @override
    def unmap(self, symbols: Symbols) -> np.ndarray:
        return symbols.raw.real.flatten()

    @override
    def place(self, symbols: Symbols) -> Symbols:
        return symbols

    @override
    def pick(self, placed_symbols: StatedSymbols) -> StatedSymbols:
        return placed_symbols

    @override
    def modulate(self, data_symbols: Symbols, bandwidth: float, oversampling_factor: int) -> np.ndarray:
        return data_symbols.raw.flatten().repeat(oversampling_factor)

    @override
    def demodulate(self, signal: np.ndarray, bandwidth: float, oversampling_factor: int) -> Symbols:
        symbols = Symbols(signal[np.newaxis, : oversampling_factor * self.symbols_per_frame : oversampling_factor, np.newaxis])
        return symbols

    @override
    def serialize(self, process: object) -> None:
        return

    @classmethod
    @override
    def Deserialize(cls, process: object) -> MockCommunicationWaveform:
        return MockCommunicationWaveform()


class MockPilotCommunicationWaveform(MockCommunicationWaveform, ConfigurablePilotWaveform):
    """Mock communication waveform for modem testing"""

    @override
    def pilot_signal(self, bandwidth: float, oversampling_factor: int) -> Signal:
        return Signal.Create(np.zeros(self.samples_per_frame(bandwidth, oversampling_factor)), bandwidth * oversampling_factor)


class TestSynchronization(unittest.TestCase):
    """Test waveform generator synchronization base class"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        self.bandwidth = 1e6
        self.oversampling_factor = 4

        self.synchronization: Synchronization = Synchronization()
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
            _ = self.synchronization.synchronize(np.zeros(10), self.bandwidth, self.oversampling_factor)

    def test_synchronize(self) -> None:
        """Default synchronization should properly split signals into frame-sections"""

        num_streams = 3
        num_frames = 1
        num_offset_samples = 2
        num_samples = num_frames * self.waveform.samples_per_frame(self.bandwidth, self.oversampling_factor) + num_offset_samples

        signal = np.exp(2j * self.rng.uniform(0, pi, (num_streams, 1))) @ np.exp(2j * self.rng.uniform(0, pi, (1, num_samples)))
        frames = self.synchronization.synchronize(signal, self.bandwidth, self.oversampling_factor)
        self.assertEqual(num_frames, len(frames))

    def test_serialization(self) -> None:
        """Test synchronization serialization"""

        test_roundtrip_serialization(self, self.synchronization, {'waveform'})


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
        """Test channel estimation serialization"""

        test_roundtrip_serialization(self, self.estimation, {'waveform'})


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

    def test_serialization(self) -> None:
        """Test channel equalization serialization"""

        test_roundtrip_serialization(self, self.equalization, {'waveform'})


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

    def test_serialization(self) -> None:
        """Test zero-forcing channel equalization serialization"""

        test_roundtrip_serialization(self, self.equalization, {'waveform'})


class TestCommunicationWaveform(unittest.TestCase):
    """Base class for communication waveform tests"""

    waveform: CommunicationWaveform
    rng: np.random.Generator

    def test_map_unmap(self) -> None:
        """Mapping and subsequently un-mapping a bit stream should yield identical bits"""

        # Dont run the test if no waveform is assigned
        if not hasattr(self, 'waveform'):
            self.skipTest("No waveform assigned")

        expected_bits = self.rng.integers(0, 2, self.waveform.bits_per_frame(self.waveform.num_data_symbols))

        symbols = self.waveform.map(expected_bits)
        bits = self.waveform.unmap(symbols)

        assert_array_equal(expected_bits, bits)

    def test_modulate_demodulate(self) -> None:
        """Modulating and subsequently de-modulating a symbol stream should yield identical symbols"""

        # Dont run the test if no waveform is assigned
        if not hasattr(self, 'waveform'):
            self.skipTest("No waveform assigned")

        expected_symbols = self.waveform.map(self.rng.integers(0, 2, self.waveform.bits_per_frame(self.waveform.num_data_symbols)))

        test_bandwidths = [0.5e6, 2e6, 5e6]
        test_oversampling_factors = [2, 4, 16]

        for bandwidth, oversampling_factor in product(test_bandwidths, test_oversampling_factors):
            with self.subTest(bandwidth=bandwidth, oversampling_factor=oversampling_factor):

                baseband_signal = self.waveform.modulate(self.waveform.place(expected_symbols), bandwidth, oversampling_factor)
                symbols = self.waveform.pick(
                    self.waveform.estimate_channel(
                        self.waveform.demodulate(baseband_signal, bandwidth, oversampling_factor),
                        bandwidth,
                        oversampling_factor,
                    )
                )

                assert_array_almost_equal(expected_symbols.raw, symbols.raw, decimal=1)

    def test_samples_per_frame(self) -> None:
        """Samples per frame method should compute the correct sample count"""

        # Dont run the test if no waveform is assigned
        if not hasattr(self, 'waveform'):
            self.skipTest("No waveform assigned")

        expected_symbols = self.waveform.map(self.rng.integers(0, 2, self.waveform.bits_per_frame(self.waveform.num_data_symbols)))

        test_bandwidths = [0.5e6, 2e6, 5e6]
        test_oversampling_factors = [1, 4, 16]

        for bandwidth, oversampling_factor in product(test_bandwidths, test_oversampling_factors):
            with self.subTest(bandwidth=bandwidth, oversampling_factor=oversampling_factor):

                baseband_signal = self.waveform.modulate(self.waveform.place(expected_symbols), bandwidth, oversampling_factor)
                expected_samples_per_frame = self.waveform.samples_per_frame(bandwidth, oversampling_factor)

                self.assertEqual(expected_samples_per_frame, baseband_signal.size)

    def test_energy(self) -> None:
        """Test the expected energy of the modulated waveform"""

        # Dont run the test if no waveform is assigned
        if not hasattr(self, 'waveform'):
            self.skipTest("No waveform assigned")

        test_bandwidths = [0.5e6, 2e6, 5e6]
        test_oversampling_factors = [2, 4, 7]

        symbols = self.waveform.map(self.rng.integers(0, 2, self.waveform.bits_per_frame(self.waveform.num_data_symbols)))
        for bandwidth, oversampling_factor in product(test_bandwidths, test_oversampling_factors):
            with self.subTest(bandwidth=bandwidth, oversampling_factor=oversampling_factor):
                baseband_waveform = self.waveform.modulate(self.waveform.place(symbols), bandwidth, oversampling_factor)

                expected_symbol_energy = self.waveform.symbol_energy(bandwidth, oversampling_factor)
                estimated_energy: float = float(np.linalg.norm(baseband_waveform) ** 2)
                estimated_symbol_energy = estimated_energy / self.waveform.num_data_symbols

                self.assertAlmostEqual(expected_symbol_energy, estimated_symbol_energy, places=1, msg="Symbol energy mismatch")

    def test_power(self) -> None:
        """Test the expected power of the modulated waveform"""

        # Dont run the test if no waveform is assigned
        if not hasattr(self, 'waveform'):
            self.skipTest("No waveform assigned")

        test_bandwidths = [0.5e6, 2e6, 5e6]
        test_oversampling_factors = [2, 4, 7]

        symbols = self.waveform.map(self.rng.integers(0, 2, self.waveform.bits_per_frame(self.waveform.num_data_symbols)))
        for bandwidth, oversampling_factor in product(test_bandwidths, test_oversampling_factors):
            with self.subTest(bandwidth=bandwidth, oversampling_factor=oversampling_factor):
                baseband_waveform = self.waveform.modulate(self.waveform.place(symbols), bandwidth, oversampling_factor)

                expected_power = self.waveform.power
                estimated_power: float = float(np.linalg.norm(baseband_waveform) ** 2) / baseband_waveform.size

                self.assertAlmostEqual(expected_power, estimated_power, places=1, msg="Power mismatch")

    def test_serialization(self) -> None:
        """Test communication waveform serialization"""

        # Dont run the test if no waveform is assigned
        if not hasattr(self, 'waveform'):
            self.skipTest("No waveform assigned")

        test_roundtrip_serialization(self, self.waveform, {'modem'})


class TestCommunicationWaveformBase(unittest.TestCase):
    """Test the communication waveform generator unit"""

    waveform: CommunicationWaveform
    rng: np.random.Generator

    def setUp(self) -> None:
        self.rng = rnd.default_rng(42)
        self.waveform = MockCommunicationWaveform()

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
        expected_num_frames_candidates = [0, 1, 1, 2]

        for num_samples, expected_num_frames in zip(num_samples_test, expected_num_frames_candidates):
            signal = np.exp(2j * self.rng.uniform(0, pi, (num_streams, 1))) @ np.exp(2j * self.rng.uniform(0, pi, (1, num_samples)))

            synchronized_frames = self.waveform.synchronization.synchronize(signal, 1.0, 1)

            # Number of frames is the number of frames that fit into the samples
            num_frames = len(synchronized_frames)
            self.assertEqual(expected_num_frames, num_frames)

    def test_data_rate(self) -> None:
        """Data rate method should compute the correct data rate"""

        num_data_symbols = 10
        a = self.waveform.bits_per_frame(num_data_symbols)
        b = self.waveform.frame_duration(1.0)

        self.assertAlmostEqual(a / b, self.waveform.data_rate(num_data_symbols, 1.0))

    def test_symbol_precoding_support(self) -> None:
        """Symbol precoding should be supported"""

        self.assertTrue(self.waveform.symbol_precoding_support)


class TestUniformPilotSymbolSequence(unittest.TestCase):
    """Test the uniform pilot symbol sequence"""

    def setUp(self) -> None:
        self.pilot_symbol = 1.234 - 1234j
        self.uniform_sequence = UniformPilotSymbolSequence(self.pilot_symbol)

    def test_sequence(self) -> None:
        """The generated sequence should be an array containing only the configured symbol"""

        assert_array_equal(np.array([self.pilot_symbol], dtype=complex), self.uniform_sequence.sequence)

    def test_serialization(self) -> None:
        """Test pilot symbol sequence serialization"""

        test_roundtrip_serialization(self, self.uniform_sequence)


class TestCustomPilotSymbolSequence(unittest.TestCase):
    """Test the custom pilot symbol sequence"""

    def setUp(self) -> None:
        self.pilot_symbols = np.array([1.234 - 1234j, 2.345 + 2345j, 3.456 - 3456j])
        self.pilot_sequence = CustomPilotSymbolSequence(self.pilot_symbols)

    def test_sequence(self) -> None:
        """The generated sequence should be an array containing the configured symbols"""

        assert_array_equal(self.pilot_symbols, self.pilot_sequence.sequence)

    def test_serialization(self) -> None:
        """Test pilot symbol sequence serialization"""

        test_roundtrip_serialization(self, self.pilot_sequence)


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
