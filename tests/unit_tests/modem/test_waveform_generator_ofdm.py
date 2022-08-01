# -*- coding: utf-8 -*-
"""Test HermesPy Orthogonal Frequency Division Multiplexing Waveform Generation."""

from itertools import product
from multiprocessing.sharedctypes import Value
from signal import signal
from typing import Tuple
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from numpy.random import default_rng
from scipy.constants import pi
from scipy.fft import fft, fftshift

from hermespy.channel import ChannelStateInformation
from hermespy.modem.modem import Symbols
from hermespy.modem import OFDMWaveform, FrameSymbolSection, FrameGuardSection, FrameResource, symbols
from hermespy.modem.waveform_generator_ofdm import FrameElement, ElementType, FrameSection, OFDMCorrelationSynchronization, OFDMIdealChannelEstimation, PilotSection, SchmidlCoxPilotSection, SchmidlCoxSynchronization, ChannelEstimation

__author__ = "André Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["André Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestFrameResource(TestCase):
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


class TestFrameSection(TestCase):
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


class TestFrameSymbolSection(TestCase):
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
        self.frame.dc_suppression = False
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

    def test_modulate_demodulate_no_dc_suppression(self) -> None:
        """Modulating and subsequently de-modulating an OFDM symbol section should yield correct symbols."""

        num_blocks = self.section.num_words
        num_symbols = self.section.num_subcarriers
        expected_symbols = np.exp(2j * self.rnd.uniform(0, pi, (num_blocks, num_symbols)))

        modulated_signal = self.section.modulate(expected_symbols)
        demodulated_symbols = self.section.demodulate(modulated_signal)

        assert_array_almost_equal(expected_symbols, demodulated_symbols)
        
    def test_modulate_demodulate_dc_suppression(self) -> None:
        """Modulating and subsequently de-modulating an OFDM symbol section should yield correct symbols."""

        self.frame.dc_suppression = True
        
        num_blocks = self.section.num_words
        num_symbols = self.section.num_subcarriers
        expected_symbols = np.exp(2j * self.rnd.uniform(0, pi, (num_blocks, num_symbols)))

        modulated_signal = self.section.modulate(expected_symbols)
        demodulated_symbols = self.section.demodulate(modulated_signal)

        assert_array_almost_equal(expected_symbols, demodulated_symbols)

    def test_resource_mask(self) -> None:
        _ = self.section.resource_mask

    def test_num_samples(self) -> None:
        """Number of samples property should compute the correct sample count."""

        expected_num_samples = self.section.num_samples
        
        num_blocks = self.section.num_words
        num_symbols = self.section.num_subcarriers
        expected_symbols = np.exp(2j * self.rnd.uniform(0, pi, (num_blocks, num_symbols)))
        modulated_signal = self.section.modulate(expected_symbols)

        self.assertEqual(modulated_signal.shape[0], expected_num_samples)


class TestFrameGuardSection(TestCase):
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

        _ = self.section.demodulate(np.empty(0, dtype=complex))


class TestOFDMWaveform(TestCase):
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

        self.ofdm = OFDMWaveform(subcarrier_spacing=self.subcarrier_spacing, modem=self.modem,
                                               resources=self.resources, structure=self.sections,
                                               num_subcarriers=self.num_subcarriers,
                                               oversampling_factor=self.oversampling_factor)

    def test_init(self) -> None:
        """Object initialization arguments should be properly stored as class attributes."""

        self.assertIs(self.modem, self.ofdm.modem)
        self.assertEqual(self.subcarrier_spacing, self.ofdm.subcarrier_spacing)

    def test_add_resource(self) -> None:
        """Added resources should be properly appended to the resource list."""

        resource = Mock()
        self.ofdm.add_resource(resource)

        self.assertIn(resource, self.ofdm.resources)

    def test_add_section(self) -> None:
        """Added sections should be properly appended to the section list."""

        section = Mock()
        self.ofdm.add_section(section)

        self.assertIn(section, self.ofdm.structure)
        self.assertIs(self.ofdm, section.frame)
        
    def test_pilot_setget(self) -> None:
        """Pilot property getter should return setter argument"""
        
        pilot = Mock()
        self.ofdm.pilot_section = pilot
        
        self.assertIs(pilot, self.ofdm.pilot_section)
        
    def test_pilot_registration(self) -> None:
        """Setting a pilot should register the respective frame as a reference"""

        pilot = Mock()
        self.ofdm.pilot_section = pilot
        
        self.assertIs(self.ofdm, pilot.frame)
        
    def test_pilot_signal(self) -> None:
        """The pilot signal property should generate the correct pilot samples"""
        
        self.ofdm.pilot_section = None
        empty_pilot_signal = self.ofdm.pilot_signal
        
        self.assertEqual(0, empty_pilot_signal.num_samples)
        self.assertEqual(self.ofdm.sampling_rate, empty_pilot_signal.sampling_rate)

        expected_samples = np.arange(100, dtype=complex)
        pilot_mock = Mock()
        pilot_mock.modulate.return_value = expected_samples
        self.ofdm.pilot_section = pilot_mock
        pilot_signal = self.ofdm.pilot_signal
        
        assert_array_equal(expected_samples[None, :], pilot_signal.samples)
        self.assertEqual(self.ofdm.sampling_rate, pilot_signal.sampling_rate)

    def test_subcarrier_spacing_setget(self) -> None:
        """Subcarrier spacing property getter should return setter argument."""

        spacing = 123
        self.ofdm.subcarrier_spacing = spacing

        self.assertEqual(spacing, self.ofdm.subcarrier_spacing)

    def test_subcarrier_spacing_assert(self) -> None:
        """Subcarrier spacing property setter should raise ValueError on arguments zero or smaller."""

        with self.assertRaises(ValueError):
            self.ofdm.subcarrier_spacing = -1.

        with self.assertRaises(ValueError):
            self.ofdm.subcarrier_spacing = 0.   

    def test_modulate_demodulate(self) -> None:
        """Modulating and subsequently de-modulating a data frame should yield identical symbols."""

        bits = self.rng.integers(0, 2, self.ofdm.bits_per_frame)
        expected_symbols = self.ofdm.map(bits)

        baseband_signal = self.ofdm.modulate(expected_symbols)
        symbols = self.ofdm.demodulate(baseband_signal.samples[0, :])

        assert_array_almost_equal(expected_symbols.raw, symbols.raw)
        
    def test_map_unmap(self) -> None:
        """Mappding and numapping should result in identical bit sequences"""
        
        expected_bits = self.rng.integers(0, 2, self.ofdm.bits_per_frame)
        symbols = self.ofdm.map(expected_bits)
        bits = self.ofdm.unmap(symbols)
                
        assert_array_equal(expected_bits, bits)
        
    def test_transmit_receive(self) -> None:
        """Mapping and modulation should resultin a correct information transmission"""

        expected_bits = self.rng.integers(0, 2, self.ofdm.bits_per_frame)
        
        symbols = self.ofdm.map(expected_bits)
        signal = self.ofdm.modulate(symbols)
        demodulated_symbols = self.ofdm.demodulate(signal.samples[0, :])
        bits = self.ofdm.unmap(demodulated_symbols)
        
        assert_array_equal(expected_bits, bits)


class TestPilotSection(TestCase):
    """Test the general base class for OFDM pilot sections."""
    
    def setUp(self) -> None:
        
        self.rng = default_rng(42)
        self.subsymbols = Symbols(np.array([1., -1., 1.j, -1.j], dtype=complex))
        self.frame = OFDMWaveform(oversampling_factor=4)
        
        self.pilot_section = PilotSection(pilot_elements=self.subsymbols, frame=self.frame)
        
    def test_init(self) -> None:
        """Initialization arguments should be properly stored as class attributes"""
        
        self.assertIs(self.subsymbols, self.pilot_section.pilot_elements)
        self.assertIs(self.frame, self.pilot_section.frame)
        
    def test_num_samples(self) -> None:
        """The number of samples property should compute the correct sample count"""
        
        self.assertEqual(4 * self.frame.num_subcarriers, self.pilot_section.num_samples)
        
    def test_pilot_subsymbols_setget(self) -> None:
        """Pilot subsymbol getter should return setter argument"""
        
        self.pilot_section.pilot_elements = None
        self.assertIs(None, self.pilot_section.pilot_elements)
        
        expected_subsymbols = Symbols(np.array([-1., 1.]))
        self.pilot_section.pilot_elements = expected_subsymbols
        self.assertIs(expected_subsymbols, self.pilot_section.pilot_elements)
        
    def test_pilot_subsymbols_validation(self) -> None:
        """Pilot subsymbol setter should raise ValueErrors on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.pilot_section.pilot_elements = Symbols(np.array([[1], [1]]))
            
        with self.assertRaises(ValueError):
            self.pilot_section.pilot_elements = Symbols()
            
    def test_pseudorandom_pilot_sequence(self) -> None:
        """Unspecified subsymbols should result in the generation of constant valid pilot sequence"""
        
        self.pilot_section.pilot_elements = None
        
        first_pilot_sequence = self.pilot_section._pilot_sequence(self.frame.num_subcarriers)
        second_pilot_sequence = self.pilot_section._pilot_sequence()
        
        self.assertEqual(1, first_pilot_sequence.num_streams)
        self.assertEqual(self.frame.num_subcarriers, first_pilot_sequence.num_symbols)
        assert_array_equal(first_pilot_sequence.raw, second_pilot_sequence.raw)
        
    def test_configured_pilot_sequence(self) -> None:
        """Specified subsymbols should result in the generation of a valid pilot sequence"""
        
        self.pilot_section.pilot_elements = Symbols(np.array([[[1., -1., 1.j, -1.j]]], dtype=complex))
        pilot_sequence = self.pilot_section._pilot_sequence()
        
        self.assertEqual(1, pilot_sequence.num_streams)
        self.assertEqual(self.frame.num_subcarriers, pilot_sequence.num_symbols)
        assert_array_equal(self.pilot_section.pilot_elements.raw, pilot_sequence.raw[:, :, :4])
        
    def test_modulate(self) -> None:
        """Modulation should return a valid pilot section"""
        
        expected_pilot_symbols = np.exp(2j * pi * self.rng.uniform(0, 1, (1, 1, self.frame.num_subcarriers)))
        self.pilot_section.pilot_elements = Symbols(expected_pilot_symbols)
        
        pilot = self.pilot_section.modulate()
        cached_pilot = self.pilot_section.modulate()
        assert_array_equal(pilot, cached_pilot)
        
    def test_demodulate(self) -> None:
        """Demodulation should always return an empty tuple"""
        
        symbols, csi = self.pilot_section.demodulate()
        
        self.assertEqual(0, len(symbols))
        
    def test_pilot(self) -> None:
        """Pilot samples should be the inverse Fourier transform of subsymbols"""
        
        expected_pilot_symbols = np.exp(2j * pi * self.rng.uniform(0, 1, (1, 1, self.frame.num_subcarriers)))
        self.pilot_section.pilot_elements = Symbols(expected_pilot_symbols)
    
        pilot = self.pilot_section._pilot()

        padded_num_subcarriers = self.frame.num_subcarriers * self.frame.oversampling_factor
        subgrid_start_idx = int(.5 * (padded_num_subcarriers - self.frame.num_subcarriers))
        dc_index = int(.5 * padded_num_subcarriers)
        pilot_symbols = fftshift(fft(pilot, norm='ortho'))
        pilot_symbols[dc_index:] = np.roll(pilot_symbols[dc_index:], -1)
        pilot_symbols = pilot_symbols[subgrid_start_idx:subgrid_start_idx+self.frame.num_subcarriers]

        assert_array_almost_equal(expected_pilot_symbols[0, 0, :], pilot_symbols)


class TestSchmidlCoxPilotSection(TestCase):
    """Test the Schmidl Cox Algorithm Pilot section implementation."""
    
    def setUp(self) -> None:
        
        self.frame = OFDMWaveform(oversampling_factor=4)
        self.pilot = SchmidlCoxPilotSection(frame=self.frame)
        
    def test_num_samples(self) -> None:
        """Number of samples property should return two frame symbols worth of samples"""
        
        self.assertTrue(2 * self.frame.oversampling_factor * self.frame.num_subcarriers, self.pilot.num_samples)
        
    def test_pilot(self) -> None:
        """A valid pilot section should be generated"""
        
        pilot = self.pilot.modulate()
        self.assertEqual(self.pilot.num_samples, len(pilot), "Invalid number of samples generated")
        self.assertEqual(2 * self.frame.num_subcarriers * self.frame.oversampling_factor, len(pilot))

        synchronization_symbol = pilot[:int(.5 * len(pilot))]
        half_symbol_length = int(.5 * len(synchronization_symbol))
        first_half_symbol = synchronization_symbol[:half_symbol_length]
        second_half_symbol = synchronization_symbol[half_symbol_length:]
        
        assert_array_almost_equal(first_half_symbol, second_half_symbol, err_msg="Synchronization symbol not symmmetric")


class TestCorrelationSynchronization(TestCase):
    """Test OFDM Synchronization via pilot section correlation"""
    
    def setUp(self) -> None:

        self.rng = default_rng(42)

        test_resource = FrameResource(repetitions=1, elements=[FrameElement(ElementType.DATA, repetitions=1200)])
        test_payload = FrameSymbolSection(num_repetitions=3, pattern=[0])
        self.frame = OFDMWaveform(oversampling_factor=4, resources=[test_resource], structure=[test_payload])
        self.frame.pilot_section = PilotSection()

        self.synchronization = OFDMCorrelationSynchronization()
        self.frame.synchronization = self.synchronization

        self.num_streams = 3
        self.delays_in_samples = [0, 9, 80]
        self.num_frames = [1, 2, 3]
        
    def test_synchronize(self) -> None:
        """Test the proper estimation of delays during correlation synchronization"""

        for d, n in product(self.delays_in_samples, self.num_frames):

            symbols = np.exp(2j * pi * self.rng.uniform(0, 1, (n, self.frame.symbols_per_frame)))
            frames = [np.exp(2j * pi * self.rng.uniform(0, 1, (self.num_streams, 1))) @ self.frame.modulate(Symbols(symbols[f, :])).samples for f in range(n)]
            
            signal = np.empty((self.num_streams, 0), dtype=complex)
            for frame in frames:

                signal = np.concatenate((signal, np.zeros((self.num_streams, d), dtype=complex), frame), axis=1)
            
            frame_delays = self.synchronization.synchronize(signal)

            self.assertCountEqual(d + (d + self.frame.samples_in_frame) * np.arange(n), frame_delays)


class TestSchmidlCoxSynchronization(TestCase):
    """Test the Schmidl Cox Algorithm implementation."""

    def setUp(self) -> None:

        self.rng = default_rng(90)

        test_resource = FrameResource(repetitions=1, elements=[FrameElement(ElementType.DATA, repetitions=1200)])
        test_payload = FrameSymbolSection(num_repetitions=3, pattern=[0])
        self.frame = OFDMWaveform(oversampling_factor=1, resources=[test_resource], structure=[test_payload])
        self.frame.pilot_section = SchmidlCoxPilotSection()
        self.frame.dc_suppression = False

        self.synchronization = SchmidlCoxSynchronization(self.frame)

        self.num_streams = 3
        self.delays_in_samples = [0, 9, 80]
        self.num_frames = [1, 2, 3]

    def test_synchronize(self) -> None:
        """Test the proper estimation of delays during Schmidl-Cox synchronization"""

        for d, n in product(self.delays_in_samples, self.num_frames):

            symbols = np.exp(2j * pi * self.rng.uniform(0, 1, (n, self.frame.symbols_per_frame)))
            frames = [np.exp(2j * pi * self.rng.uniform(0, 1, (self.num_streams, 1))) @ self.frame.modulate(Symbols(symbols[f, :])).samples for f in range(n)]
            
            signal = np.empty((self.num_streams, 0), dtype=complex)
            for frame in frames:

                signal = np.concatenate((signal, np.zeros((self.num_streams, d), dtype=complex), frame), axis=1)
            
            channel_state = ChannelStateInformation.Ideal(len(signal), self.num_streams)

            synchronization = self.synchronization.synchronize(signal, channel_state)

            self.assertEqual(n, len(synchronization))
            for frame, (synchronized_frame, _) in zip(frames, synchronization):
                assert_array_equal(frame, synchronized_frame)


class TestIdealChannelEstimation(TestCase):
    """Test ideal channel estimation for OFDM waveforms."""
    
    def setUp(self) -> None:
        
        self.waveform = Mock()
        self.channel_estimation = OFDMIdealChannelEstimation()
