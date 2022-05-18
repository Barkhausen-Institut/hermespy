# -*- coding: utf-8 -*-
"""Test HermesPy Orthogonal Frequency Division Multiplexing Waveform Generation."""

import unittest
from typing import Tuple
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from numpy.random import default_rng
from scipy.constants import pi

from hermespy.channel import ChannelStateInformation
from hermespy.modem.modem import Symbols
from hermespy.modem import WaveformGeneratorOfdm, FrameSymbolSection, FrameGuardSection, FrameResource
from hermespy.modem.waveform_generator_ofdm import ChannelEstimation, FrameElement, ElementType, FrameSection

__author__ = "André Noll Barreto"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["André Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestFrameResource(unittest.TestCase):
    """Test a single OFDM frame resource."""

    def setUp(self) -> None:

        self.repetitions = 2
        self.cp_ratio = 0.01
        self.elements = [FrameElement(ElementType.DATA, 2),
                         FrameElement(ElementType.REFERENCE, 1),
                         FrameElement(ElementType.NULL, 3)]

        self.resource = FrameResource(self.repetitions, self.cp_ratio, self.elements)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes."""

        self.assertEqual(self.repetitions, self.resource.repetitions)
        self.assertEqual(self.cp_ratio, self.resource.cp_ratio)
        self.assertCountEqual(self.elements, self.resource.elements)

    def test_repetitions_setget(self) -> None:
        """Repetitions property getter should return setter argument."""

        repetitions = 10
        self.resource.repetitions = repetitions

        self.assertEqual(repetitions, self.resource.repetitions)

    def test_repetitions_validation(self) -> None:
        """Repetitions property setter should raise ValueError on arguments smaller than one."""

        with self.assertRaises(ValueError):
            self.resource.repetitions = 0

        with self.assertRaises(ValueError):
            self.resource.repetitions = -1

    def test_cp_ratio_setget(self) -> None:
        """Cyclic prefix ratio property getter should return setter argument."""

        cp_ratio = .5
        self.resource.cp_ratio = .5

        self.assertEqual(cp_ratio, self.resource.cp_ratio)

    def test_cp_ratio_validation(self) -> None:
        """Cyclic prefix ratio property setter should raise ValueError on arguments
        smaller than zero or bigger than one."""

        with self.assertRaises(ValueError):
            self.resource.cp_ratio = -1.0

        with self.assertRaises(ValueError):
            self.resource.cp_ratio = 1.5

        try:
            self.resource.cp_ratio = 0.0
            self.resource.cp_ratio = 1.0

        except ValueError:
            self.fail()

    def test_num_subcarriers(self) -> None:
        """Number of subcarriers property should return the correct subcarrier count."""

        self.assertEqual(12, self.resource.num_subcarriers)

    def test_num_symbols(self) -> None:
        """Number of symbols property should return the correct data symbol count."""

        self.assertEqual(4, self.resource.num_symbols)

    def test_num_references(self) -> None:
        """Number of references property should return the correct reference symbol count."""

        self.assertEqual(2, self.resource.num_references)

    def test_resource_mask(self) -> None:
        """Resource mask property should return a mask selecting the proper elements."""

        expected_mask = np.zeros((3, 12), bool)
        expected_mask[1, [0, 1, 6, 7]] = True           # Data symbol mask
        expected_mask[0, [2, 8]] = True                 # Reference symbol mask
        expected_mask[2, [3, 4, 5, 9, 10, 11]] = True   # NULL symbol mask

        assert_array_equal(expected_mask, self.resource.mask)


class FrameSectionMock(FrameSection):

    def __init__(self, *args) -> None:
        FrameSection.__init__(self, *args)

    @property
    def num_samples(self) -> int:
        return 1

    @property
    def duration(self) -> float:
        return 1.

    def modulate(self, symbols: np.ndarray) -> np.ndarray:
        pass

    def demodulate(self,
                   baseband_signal: np.ndarray,
                   channel_state: ChannelStateInformation) -> Tuple[np.ndarray, ChannelStateInformation]:
        pass


class TestFrameSection(unittest.TestCase):
    """Test OFDM frame section."""

    def setUp(self) -> None:

        self.frame = Mock()
        self.num_repetitions = 2

        self.section = FrameSectionMock(self.num_repetitions, self.frame)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as attributes."""

        self.assertIs(self.frame, self.section.frame)
        self.assertEqual(self.num_repetitions, self.section.num_repetitions)

    def test_num_repetitions_setget(self) -> None:
        """Number of repetitions property getter should return setter argument."""

        num_repetitions = 3
        self.section.num_repetitions = num_repetitions

        self.assertEqual(num_repetitions, self.section.num_repetitions)

    def test_num_repetitions_validation(self) -> None:
        """Number of repetitions property setter should raise ValueError on arguments smaller than zero."""

        with self.assertRaises(ValueError):
            self.section.num_repetitions = 0

        with self.assertRaises(ValueError):
            self.section.num_repetitions = -1


class TestFrameSymbolSection(unittest.TestCase):
    """Test OFDM frame symbol section."""

    def setUp(self) -> None:

        self.rnd = np.random.default_rng(42)

        self.repetitions_a = 2
        self.cp_ratio_a = 0.1
        self.elements_a = [FrameElement(ElementType.DATA, 2),
                           FrameElement(ElementType.REFERENCE, 1),
                           FrameElement(ElementType.NULL, 3)]
        self.resource_a = FrameResource(self.repetitions_a, self.cp_ratio_a, self.elements_a)

        self.repetitions_b = 3
        self.cp_ratio_b = 0.0
        self.elements_b = [FrameElement(ElementType.REFERENCE, 2),
                           FrameElement(ElementType.DATA, 1),
                           FrameElement(ElementType.NULL, 3)]
        self.resource_b = FrameResource(self.repetitions_b, self.cp_ratio_b, self.elements_b)

        self.frame = Mock()
        self.frame.num_subcarriers = 20
        self.frame.oversampling_factor = 4
        self.frame.resources = [self.resource_a, self.resource_b]
        self.num_repetitions = 2
        self.pattern = [0, 1, 0]

        self.section = FrameSymbolSection(self.num_repetitions, self.pattern, self.frame)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as object attributes."""

        self.assertEqual(self.num_repetitions, self.section.num_repetitions)
        self.assertCountEqual(self.pattern, self.section.pattern)
        self.assertIs(self.frame, self.section.frame)

    def test_num_symbols(self) -> None:
        """Number of symbols property should return the correct data symbol count."""

        self.assertEqual(22, self.section.num_symbols)

    def test_num_references(self) -> None:
        """Number of references property should return the correct reference symbol count."""

        self.assertEqual(20, self.section.num_references)

    def test_num_words(self) -> None:
        """Number of words property should return the correct word count."""

        self.assertIs(6, self.section.num_words)

    def test_num_subcarriers(self) -> None:
        """Number of subcarriers property should return the correct number of occupied subcarriers"""

        self.assertEqual(18, self.section.num_subcarriers)

    def test_modulate_demodulate(self) -> None:
        """Modulating and subsequently de-modulating an OFDM symbol section should yield correct symbols."""

        expected_symbols = np.exp(2j * self.rnd.uniform(0, pi, self.section.num_symbols))

        modulated_signal = self.section.modulate(expected_symbols)
        channel_state = ChannelStateInformation.Ideal(num_samples=modulated_signal.shape[0])
        ofdm_grid, channel_state_grid = self.section.demodulate(modulated_signal, channel_state)

        symbols = ofdm_grid.T[self.section.resource_mask[ElementType.DATA.value].T]
        assert_array_almost_equal(expected_symbols, symbols)

    def test_resource_mask(self) -> None:
        _ = self.section.resource_mask

    def test_num_samples(self) -> None:
        """Number of samples property should compute the correct sample count."""

        expected_num_samples = self.section.num_samples
        modulated_signal = self.section.modulate(np.ones(self.section.num_symbols, dtype=complex))

        self.assertEqual(modulated_signal.shape[0], expected_num_samples)


class TestFrameGuardSection(unittest.TestCase):
    """Test OFDM frame guard section."""

    def setUp(self) -> None:

        self.duration = 1.23e-3
        self.num_repetitions = 3
        self.frame = Mock()
        self.frame.sampling_rate = 1e5
        self.frame.num_subcarriers = 10

        self.section = FrameGuardSection(self.duration, self.num_repetitions, self.frame)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as object attributes."""

        self.assertEqual(self.num_repetitions, self.section.num_repetitions)
        self.assertEqual(self.duration, self.section.duration)
        self.assertIs(self.frame, self.section.frame)

    def test_duration_setget(self) -> None:
        """Duration property getter should return setter argument."""

        duration = 4.56
        self.section.duration = duration

        self.assertEqual(duration, self.section.duration)

    def test_duration_validation(self) -> None:
        """Duration property setter should raise ValueError on arguments smaller than zero."""

        with self.assertRaises(ValueError):
            self.section.duration = -1.0

        try:
            self.section.duration = 0.

        except ValueError:
            self.fail()

    def test_num_samples(self) -> None:
        """Number of samples property should compute the correct amount of samples."""

        expected_num_samples = int(self.num_repetitions * self.duration * self.frame.sampling_rate)
        self.assertEqual(expected_num_samples, self.section.num_samples)

    def test_modulate(self) -> None:
        """Modulation should return a zero-vector."""

        expected_signal = np.zeros(self.section.num_samples, dtype=complex)
        signal = self.section.modulate(np.empty(0, dtype=complex))

        assert_array_equal(expected_signal, signal)

    def test_demodulate(self) -> None:
        """Demodulation should return an empty tuple."""

        _ = self.section.demodulate(np.empty(0, dtype=complex), ChannelStateInformation.Ideal(0))


class TestWaveformGeneratorOFDM(unittest.TestCase):
    """Test Orthogonal Frequency Division Multiplexing Waveform Generator."""

    def setUp(self) -> None:

        self.subcarrier_spacing = 1e3
        self.num_subcarriers = 100
        self.oversampling_factor = 2

        self.rng = default_rng(42)

        self.modem = Mock()
        self.modem.random_generator = self.rng
        self.modem.carrier_frequency = 100e6

        self.repetitions_a = 2
        self.cp_ratio_a = 0.1
        self.elements_a = [FrameElement(ElementType.DATA, 2),
                           FrameElement(ElementType.REFERENCE, 1),
                           FrameElement(ElementType.NULL, 3)]
        self.resource_a = FrameResource(self.repetitions_a, self.cp_ratio_a, self.elements_a)

        self.repetitions_b = 3
        self.cp_ratio_b = 0.0
        self.elements_b = [FrameElement(ElementType.REFERENCE, 2),
                           FrameElement(ElementType.DATA, 1),
                           FrameElement(ElementType.NULL, 3)]
        self.resource_b = FrameResource(self.repetitions_b, self.cp_ratio_b, self.elements_b)

        self.section_a = FrameSymbolSection(2, [1, 0, 1])
        self.section_b = FrameGuardSection(1e-3)
        self.section_c = FrameSymbolSection(2, [0, 1, 0])

        self.resources = [self.resource_a, self.resource_b]
        self.sections = [self.section_a, self.section_b, self.section_c]

        self.generator = WaveformGeneratorOfdm(subcarrier_spacing=self.subcarrier_spacing, modem=self.modem,
                                               resources=self.resources, structure=self.sections,
                                               num_subcarriers=self.num_subcarriers,
                                               oversampling_factor=self.oversampling_factor)

    def test_init(self) -> None:
        """Object initialization arguments should be properly stored as class attributes."""

        self.assertIs(self.modem, self.generator.modem)
        self.assertEqual(self.subcarrier_spacing, self.generator.subcarrier_spacing)

    def test_add_resource(self) -> None:
        """Added resources should be properly appended to the resource list."""

        resource = Mock()
        self.generator.add_resource(resource)

        self.assertIn(resource, self.generator.resources)

    def test_add_section(self) -> None:
        """Added sections should be properly appended to the section list."""

        section = Mock()
        self.generator.add_section(section)

        self.assertIn(section, self.generator.structure)
        self.assertIs(self.generator, section.frame)

    def test_subcarrier_spacing_setget(self) -> None:
        """Subcarrier spacing property getter should return setter argument."""

        spacing = 123
        self.generator.subcarrier_spacing = spacing

        self.assertEqual(spacing, self.generator.subcarrier_spacing)

    def test_subcarrier_spacing_assert(self) -> None:
        """Subcarrier spacing property setter should raise ValueError on arguments zero or smaller."""

        with self.assertRaises(ValueError):
            self.generator.subcarrier_spacing = -1.

        with self.assertRaises(ValueError):
            self.generator.subcarrier_spacing = 0.

    def test_reference_based_channel_estimation(self) -> None:
        """Reference-based channel estimation should properly estimate channel at reference points."""

        self.generator.channel_estimation_algorithm = ChannelEstimation.REFERENCE

        expected_bits = self.rng.integers(0, 2, self.generator.bits_per_frame)
        expected_symbols = self.generator.map(expected_bits)
        signal = self.generator.modulate(expected_symbols)
        expected_csi = ChannelStateInformation.Ideal(signal.num_samples)

        symbols, csi, _ = self.generator.demodulate(signal.samples[0, :], expected_csi)

        assert_array_almost_equal(np.ones(csi.state.shape, dtype=complex), csi.state)        

    def test_modulate_demodulate(self) -> None:
        """Modulating and subsequently de-modulating a data frame should yield identical symbols."""

        expected_symbols = Symbols(np.exp(2j * self.rng.uniform(0, pi, self.generator.symbols_per_frame)) *
                                   np.arange(1, 1 + self.generator.symbols_per_frame))

        baseband_signal = self.generator.modulate(expected_symbols)
        channel_state = ChannelStateInformation.Ideal(num_samples=baseband_signal.num_samples)
        symbols, _, _ = self.generator.demodulate(baseband_signal.samples[0, :], channel_state)

        assert_array_almost_equal(expected_symbols.raw, symbols.raw)
