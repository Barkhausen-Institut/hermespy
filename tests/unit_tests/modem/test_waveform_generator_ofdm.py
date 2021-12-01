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
from hermespy.modem import WaveformGeneratorOfdm, FrameSymbolSection, FrameResource
from hermespy.modem.waveform_generator_ofdm import FrameElement, ElementType, FrameSection

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


class TestWaveformGeneratorOFDM(unittest.TestCase):
    """Test Orthogonal Frequency Division Multiplexing Waveform Generator."""

    def setUp(self) -> None:

        self.subcarrier_spacing = 1e3
        self.num_subcarriers = 100

        self.random_generator = default_rng(42)

        self.modem = Mock()
        self.modem.random_generator = self.random_generator
        self.modem.scenario.sampling_rate = 4 * self.num_subcarriers * self.subcarrier_spacing

        self.waveform_generator = WaveformGeneratorOfdm(subcarrier_spacing=self.subcarrier_spacing, modem=self.modem)

    def test_init(self) -> None:
        """Object initialization arguments should be properly stored as class attributes."""

        self.assertIs(self.modem, self.waveform_generator.modem)
        self.assertEqual(self.subcarrier_spacing, self.waveform_generator.subcarrier_spacing)

    def test_add_resource(self) -> None:
        """Added resources should be properly appended to the resource list."""

        resource = Mock()
        self.waveform_generator.add_resource(resource)

        self.assertIn(resource, self.waveform_generator.resources)

    def test_add_section(self) -> None:
        """Added sections should be properly appended to the section list."""

        section = Mock()
        self.waveform_generator.add_section(section)

        self.assertIn(section, self.waveform_generator.structure)
        self.assertIs(self.waveform_generator, section.frame)

    def test_subcarrier_spacing_setget(self) -> None:
        """Subcarrier spacing property getter should return setter argument."""

        spacing = 123
        self.waveform_generator.subcarrier_spacing = spacing

        self.assertEqual(spacing, self.waveform_generator.subcarrier_spacing)

    def test_subcarrier_spacing_assert(self) -> None:
        """Subcarrier spacing property setter should raise ValueError on arguments zero or smaller."""

        with self.assertRaises(ValueError):
            self.waveform_generator.subcarrier_spacing = -1.

        with self.assertRaises(ValueError):
            self.waveform_generator.subcarrier_spacing = 0.

    def test_reference_based_channel_estimation(self) -> None:
        """Reference-based channel estimation should properly estimate channel at reference points."""

        pass
