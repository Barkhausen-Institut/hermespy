# -*- coding: utf-8 -*-

from itertools import product
from typing import Tuple
from unittest import TestCase
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from numpy.random import default_rng
from scipy.constants import pi

from hermespy.core import ChannelStateInformation, Signal
from hermespy.modem import OrthogonalWaveform, SymbolSection, GuardSection, GridResource, StatedSymbols, Symbols, CustomPilotSymbolSequence, GridElement, ElementType, PrefixType, GridSection, OFDMCorrelationSynchronization, PilotSection, ReferencePosition, OrthogonalLeastSquaresChannelEstimation, OrthogonalChannelEqualization, OrthogonalZeroForcingChannelEqualization
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization
from unit_tests.utils import SimulationTestContext

__author__ = "André Noll Barreto"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["André Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestGridElement(TestCase):
    """Test a single grid element"""

    def setUp(self) -> None:
        self.element = GridElement(ElementType.DATA, repetitions=1)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.element)


class TestGridResource(TestCase):
    """Test a single grid resource."""

    def setUp(self) -> None:
        self.repetitions = 2
        self.prefix_type = PrefixType.CYCLIC
        self.prefix_ratio = 0.01
        self.elements = [GridElement(ElementType.DATA, 2), GridElement(ElementType.REFERENCE, 1), GridElement(ElementType.NULL, 3)]

        self.resource = GridResource(self.repetitions, self.prefix_type, self.prefix_ratio, self.elements)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes."""

        self.assertEqual(self.repetitions, self.resource.repetitions)
        self.assertEqual(self.prefix_type, self.resource.prefix_type)
        self.assertEqual(self.prefix_ratio, self.resource.prefix_ratio)
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

    def test_prefix_ratio_setget(self) -> None:
        """Cyclic prefix ratio property getter should return setter argument."""

        prefix_ratio = 0.5
        self.resource.prefix_ratio = 0.5

        self.assertEqual(prefix_ratio, self.resource.prefix_ratio)

    def test_prefix_ratio_validation(self) -> None:
        """Cyclic prefix ratio property setter should raise ValueError on arguments
        smaller than zero or bigger than one."""

        with self.assertRaises(ValueError):
            self.resource.prefix_ratio = -1.0

        with self.assertRaises(ValueError):
            self.resource.prefix_ratio = 1.5

        try:
            self.resource.prefix_ratio = 0.0
            self.resource.prefix_ratio = 1.0

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
        expected_mask[1, [0, 1, 6, 7]] = True  # Data symbol mask
        expected_mask[0, [2, 8]] = True  # Reference symbol mask
        expected_mask[2, [3, 4, 5, 9, 10, 11]] = True  # NULL symbol mask

        assert_array_equal(expected_mask, self.resource.mask)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.resource)


class GridSectionMock(GridSection):
    """Grid section implementation for testing purposes."""

    @property
    def num_samples(self) -> int:
        return 1

    @property
    def duration(self) -> float:
        return 1.0

    def modulate(self, symbols: np.ndarray) -> np.ndarray:
        pass

    def demodulate(self, baseband_signal: np.ndarray, channel_state: ChannelStateInformation) -> Tuple[np.ndarray, ChannelStateInformation]:
        pass
    
    def pick_samples(self, signal: np.ndarray) -> np.ndarray:
        return signal
    
    def place_samples(self, signal: np.ndarray) -> np.ndarray:
        return signal

class TestGridSection(TestCase):
    """Test the grid section base class section."""

    def setUp(self) -> None:
        self.wave = Mock()
        self.num_repetitions = 2

        self.section = GridSectionMock(self.num_repetitions, wave=self.wave)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as attributes."""

        self.assertIs(self.wave, self.section.wave)
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


class TestSymbolSection(TestCase):
    """Test grid symbol section."""

    def setUp(self) -> None:
        self.rnd = np.random.default_rng(42)

        self.repetitions_a = 2
        self.prefix_type_a = PrefixType.CYCLIC
        self.prefix_ratio_a = 0.1
        self.elements_a = [GridElement(ElementType.DATA, 2), GridElement(ElementType.REFERENCE, 1), GridElement(ElementType.NULL, 3)]
        self.resource_a = GridResource(self.repetitions_a, self.prefix_type_a, self.prefix_ratio_a, self.elements_a)

        self.repetitions_b = 3
        self.prefix_type_b = PrefixType.CYCLIC
        self.prefix_ratio_b = 0.2
        self.elements_b = [GridElement(ElementType.REFERENCE, 2), GridElement(ElementType.DATA, 1), GridElement(ElementType.NULL, 3)]
        self.resource_b = GridResource(self.repetitions_b, self.prefix_type_b, self.prefix_ratio_b, self.elements_b)

        self.num_repetitions = 2
        self.pattern = [0, 1, 0]
        self.section = SymbolSection(self.num_repetitions, self.pattern, 2)
        self.wave = OrthogonalWaveformMock(20, grid_resources=[self.resource_a, self.resource_b], grid_structure=[self.section])

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as object attributes."""

        self.assertEqual(self.num_repetitions, self.section.num_repetitions)
        self.assertCountEqual(self.pattern, self.section.pattern)
        self.assertIs(self.wave, self.section.wave)

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

    def test_place_samples_validation(self) -> None:
        """Place samples should raise a RuntimeError on invalid prefix types"""
        
        samples = np.empty((self.section.num_words, self.section.num_subcarriers), dtype=np.complex128)
        with patch.object(self.resource_a, 'prefix_type', new_callable=PropertyMock) as type_mock:
            type_mock.return_value = Mock()
            with self.assertRaises(RuntimeError):
                _ = self.section.place_samples(samples)        

    def test_resource_mask(self) -> None:
        _ = self.section.resource_mask

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        with patch("hermespy.modem.waveforms.orthogonal.waveform.SymbolSection.wave", new_callable=PropertyMock) as wave:
            wave.return_value = self.wave
            test_yaml_roundtrip_serialization(self, self.section)


class TestGuardSection(TestCase):
    """Test the grid guard section."""

    def setUp(self) -> None:
        self.duration = 1.23e-3
        self.num_repetitions = 3
        self.wave = Mock()
        self.wave.sampling_rate = 1e5
        self.wave.num_subcarriers = 10

        self.section = GuardSection(self.duration, self.num_repetitions, self.wave)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as object attributes."""

        self.assertEqual(self.num_repetitions, self.section.num_repetitions)
        self.assertEqual(self.duration, self.section.duration)
        self.assertIs(self.wave, self.section.wave)

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
            self.section.duration = 0.0

        except ValueError:
            self.fail()

    def test_num_samples(self) -> None:
        """Number of samples property should compute the correct amount of samples."""

        expected_num_samples = int(self.num_repetitions * self.duration * self.wave.sampling_rate)
        self.assertEqual(expected_num_samples, self.section.num_samples)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        with patch("hermespy.modem.waveforms.orthogonal.waveform.GuardSection.wave", new_callable=PropertyMock) as wave:
            wave.return_value = self.wave
            test_yaml_roundtrip_serialization(self, self.section)


class OrthogonalWaveformMock(OrthogonalWaveform):
    """Mock class for testing the abstract base class."""
    
    def _forward_transformation(self, symbol_grid: np.ndarray) -> np.ndarray:
        return symbol_grid.repeat(self.oversampling_factor, axis=-1)

    def _backward_transformation(self, signal_grid: np.ndarray) -> np.ndarray:
        return signal_grid[::, ::self.oversampling_factor] 
    
    @property
    def sampling_rate(self) -> float:
        return self.oversampling_factor
    
    @property
    def bandwidth(self) -> float:
        return 1.
    
    def _correct_sample_offset(self, symbol_subgrid: np.ndarray, sample_offset: int) -> np.ndarray:
        return symbol_subgrid


class TestOrthogonalWaveform(TestCase):
    """Test the general base class for orthogonal multicarrier waveforms"""

    def setUp(self) -> None:
        self.num_subcarriers = 100
        self.oversampling_factor = 2

        self.rng = default_rng(42)

        self.modem = Mock()
        self.modem.random_generator = self.rng
        self.modem.carrier_frequency = 100e6

        self.repetitions_a = 2
        self.prefix_type_a = PrefixType.CYCLIC
        self.prefix_ratio_a = 0.1
        self.elements_a = [GridElement(ElementType.DATA, 2), GridElement(ElementType.REFERENCE, 1), GridElement(ElementType.NULL, 3)]
        self.resource_a = GridResource(self.repetitions_a, self.prefix_type_a, self.prefix_ratio_a, self.elements_a)

        self.repetitions_b = 3
        self.prefix_type_b = PrefixType.ZEROPAD
        self.prefix_ratio_b = 0.2
        self.elements_b = [GridElement(ElementType.REFERENCE, 2), GridElement(ElementType.DATA, 1), GridElement(ElementType.NULL, 3)]
        self.resource_b = GridResource(self.repetitions_b, self.prefix_type_b, self.prefix_ratio_b, self.elements_b)

        self.section_a = SymbolSection(2, [1, 0, 1])
        self.section_b = GuardSection(1e-3)
        self.section_c = SymbolSection(2, [0, 1, 0])

        self.grid_resources = [self.resource_a, self.resource_b]
        self.grid_sections = [self.section_a, self.section_b, self.section_c]

        self.waveform = OrthogonalWaveformMock(
            modem=self.modem,
            grid_resources=self.grid_resources,
            grid_structure=self.grid_sections,
            num_subcarriers=self.num_subcarriers,
            oversampling_factor=self.oversampling_factor,
        )

    def test_init(self) -> None:
        """Object initialization arguments should be properly stored as class attributes."""

        self.assertIs(self.modem, self.waveform.modem)
        self.assertEqual(self.num_subcarriers, self.waveform.num_subcarriers)

    def test_plot_grid(self) -> None:
        """Test the grid visualization"""
        
        with SimulationTestContext():
            visualization = self.waveform.plot_grid()
            
    def test_pilot_setget(self) -> None:
        """Pilot property getter should return setter argument"""

        pilot = Mock()
        self.waveform.pilot_section = pilot

        self.assertIs(pilot, self.waveform.pilot_section)

    def test_pilot_registration(self) -> None:
        """Setting a pilot should register the respective frame as a reference"""

        pilot = Mock()
        self.waveform.pilot_section = pilot

        self.assertIs(self.waveform, pilot.wave)

    def test_pilot_signal(self) -> None:
        """The pilot signal property should generate the correct pilot samples"""

        self.waveform.pilot_section = None
        empty_pilot_signal = self.waveform.pilot_signal

        self.assertEqual(0, empty_pilot_signal.num_samples)
        self.assertEqual(self.waveform.sampling_rate, empty_pilot_signal.sampling_rate)

        expected_samples = np.arange(100, dtype=np.complex128)
        pilot_mock = Mock()
        pilot_mock.generate.return_value = expected_samples
        self.waveform.pilot_section = pilot_mock
        pilot_signal = self.waveform.pilot_signal

        assert_array_equal(expected_samples[None, :], pilot_signal.getitem())
        self.assertEqual(self.waveform.sampling_rate, pilot_signal.sampling_rate)

    def test_symbols_per_frame(self) -> None:
        """Symbols per frame property should return the correct number of symbols per frame."""

        self.assertEqual(90, self.waveform.symbols_per_frame)

    def test_num_data_symbols(self) -> None:
        """Number of data symbols property should report the correct number of data symbols."""

        self.assertEqual(42, self.waveform.num_data_symbols)

    def test_words_per_frame(self) -> None:
        """The number of words per frame should be the sum of the number of words of all sections"""

        self.assertEqual(self.section_a.num_words + self.section_b.num_words + self.section_c.num_words, self.waveform.words_per_frame)

    def test_references_per_frame(self) -> None:
        """The number of references per frame should be the sum of the number of references of all sections"""

        self.assertEqual(self.section_a.num_references + self.section_b.num_references + self.section_c.num_references, self.waveform.references_per_frame)

    def test_samples_per_frame(self) -> None:
        """Samples per frame property should return the correct sample count."""

        expected_bits = self.rng.integers(0, 2, self.waveform.bits_per_frame(self.waveform.num_data_symbols))
        mapped_symbols = self.waveform.map(expected_bits)
        placed_symbols = self.waveform.place(mapped_symbols)
        signal = self.waveform.modulate(placed_symbols)

        self.assertEqual(signal.size, self.waveform.samples_per_frame)

    def test_symbol_duration(self) -> None:
        """Symbol duration property should report the correct symbol duration."""

        self.assertEqual(1 / self.waveform.bandwidth, self.waveform.symbol_duration)

    def test_map_validation(self) -> None:
        """Mapping should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.waveform.map(np.random.standard_normal(2))

    def test_map_unmap(self) -> None:
        """Mappding and numapping should result in identical bit sequences"""

        expected_bits = self.rng.integers(0, 2, self.waveform.bits_per_frame(self.waveform.num_data_symbols))
        symbols = self.waveform.map(expected_bits)
        bits = self.waveform.unmap(symbols)

        assert_array_equal(expected_bits, bits)

    def test_place_validation(self) -> None:
        """Placing should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.waveform.place(Symbols(np.random.standard_normal(2)))

    def test_modulate_demodulate(self) -> None:
        """Modulating and subsequently de-modulating a data frame should yield identical symbols."""

        bits = self.rng.integers(0, 2, self.waveform.bits_per_frame(self.waveform.num_data_symbols))
        expected_symbols = self.waveform.place(self.waveform.map(bits))

        baseband_signal = self.waveform.modulate(expected_symbols)
        symbols = self.waveform.demodulate(baseband_signal)

        assert_array_almost_equal(expected_symbols.raw, symbols.raw)

    def test_modulate_demodulate_pilot(self) -> None:
        """Modulating and subsequently de-modulating a data frame with pilot section should yield identical symbols."""

        self.waveform.pilot_section = PilotSection()
        bits = self.rng.integers(0, 2, self.waveform.bits_per_frame(self.waveform.num_data_symbols))
        expected_symbols = self.waveform.place(self.waveform.map(bits))

        baseband_signal = self.waveform.modulate(expected_symbols)
        symbols = self.waveform.demodulate(baseband_signal)

        assert_array_almost_equal(expected_symbols.raw, symbols.raw)

    def test_modulate_demodulate_reference_only(self) -> None:
        """Modulating and subsequently demodulating a frame of only reference symbols should yield a valid channel estimate"""

        resources = [GridResource(1, PrefixType.NONE, 0, [GridElement(ElementType.REFERENCE, self.num_subcarriers)])]
        structure = [SymbolSection(1, [0])]

        ofdm = OrthogonalWaveformMock(num_subcarriers=self.num_subcarriers, grid_resources=resources, grid_structure=structure)
        ofdm.channel_estimation = OrthogonalLeastSquaresChannelEstimation()

        symbols = ofdm.map(np.empty(0, dtype=np.int_))
        tx_signal = ofdm.modulate(ofdm.place(symbols))
        received_symbols = ofdm.demodulate(tx_signal)

        expected_state = np.ones((1, 1, 1, self.num_subcarriers), dtype=np.complex128)
        stated_symbols = ofdm.channel_estimation.estimate_channel(received_symbols)

        assert_array_almost_equal(expected_state, stated_symbols.states)

    def test_transmit_receive(self) -> None:
        """Mapping and modulation should resultin a correct information transmission"""

        expected_bits = self.rng.integers(0, 2, self.waveform.bits_per_frame(self.waveform.num_data_symbols))

        mapped_symbols = self.waveform.map(expected_bits)
        placed_symbols = self.waveform.place(mapped_symbols)
        signal = self.waveform.modulate(placed_symbols)

        demodulated_symbols = self.waveform.demodulate(signal)
        stated_symbols = StatedSymbols(demodulated_symbols.raw, np.ones((1, 1, demodulated_symbols.num_blocks, demodulated_symbols.num_symbols)))
        picked_symbols = self.waveform.pick(stated_symbols)
        bits = self.waveform.unmap(picked_symbols)

        assert_array_equal(expected_bits, bits)

    def test_frame_duration(self) -> None:
        """The frame duration should be the sum of the durations of all sections"""

        self.assertEqual(self.waveform.samples_per_frame / self.waveform.sampling_rate, self.waveform.frame_duration)

    def test_resource_mask(self) -> None:
        """The resource mask should be the sum of the masks of all resources"""

        resource_mask = self.waveform.resource_mask
        self.assertSequenceEqual((3, self.waveform.words_per_frame, self.waveform.num_subcarriers), resource_mask.shape)

    def test_bit_energy(self) -> None:
        """The bit energy should be the sum of the bit energies of all sections"""

        self.assertEqual(0.25, self.waveform.bit_energy)

    def test_symbol_energy(self) -> None:
        """The symbol energy should be the sum of the symbol energies of all sections"""

        self.assertEqual(1, self.waveform.symbol_energy)

    def test_power(self) -> None:
        """The signal power should be computed correctly"""

        self.elements_a = [GridElement(ElementType.DATA, self.num_subcarriers)]
        self.resource_a = GridResource(1, PrefixType.NONE, 0, self.elements_a)
        self.section_a = SymbolSection(100, [0])

        self.waveform = OrthogonalWaveformMock(modem=self.modem, grid_resources=[self.resource_a], grid_structure=[self.section_a], num_subcarriers=self.num_subcarriers, oversampling_factor=self.oversampling_factor)

        symbols = self.waveform.map(self.rng.integers(0, 2, self.waveform.bits_per_frame(self.waveform.num_data_symbols)))
        placed_symbols = self.waveform.place(symbols)
        frame_samples = self.waveform.modulate(placed_symbols)
        frame_signal = Signal.Create(frame_samples, self.waveform.sampling_rate)

        self.assertAlmostEqual(self.waveform.power, frame_signal.power[0], places=2)

    def test_num_subcarriers_validation(self) -> None:
        """Number of subcarries property setter should raise ValueError on arguments smaller than one."""

        with self.assertRaises(ValueError):
            self.waveform.num_subcarriers = 0

        with self.assertRaises(ValueError):
            self.waveform.num_subcarriers = -1


class TestPilotSection(TestCase):
    """Test the general base class for OFDM pilot sections."""

    def setUp(self) -> None:
        self.rng = default_rng(42)
        self.subsymbols = Symbols(np.array([1.0, -1.0, 1.0j, -1.0j], dtype=np.complex128))
        self.wave = OrthogonalWaveformMock(128, [], [], oversampling_factor=4)

        self.pilot_section = PilotSection(pilot_elements=self.subsymbols, wave=self.wave)

    def test_init(self) -> None:
        """Initialization arguments should be properly stored as class attributes"""

        self.assertIs(self.subsymbols, self.pilot_section.pilot_elements)
        self.assertIs(self.wave, self.pilot_section.wave)
        self.assertEqual(0, self.pilot_section.num_symbols)
        
    def test_num_repetitions_validation(self) -> None:
        """Number of repetitions property setter should raise ValueError on arguments not one"""
        
        with self.assertRaises(ValueError):
            self.pilot_section.num_repetitions = 0
        
        with self.assertRaises(ValueError):
            self.pilot_section.num_repetitions = -1
            
        with self.assertRaises(ValueError):
            self.pilot_section.num_repetitions = 2
            
    def test_sample_offset(self) -> None:
        """Sample offset property may only be zero"""
        
        with self.assertRaises(ValueError):
            self.pilot_section.sample_offset = 1
            
        with self.assertRaises(ValueError):
            self.pilot_section.sample_offset = -1

    def test_num_samples(self) -> None:
        """The number of samples property should compute the correct sample count"""

        self.assertEqual(4 * self.wave.num_subcarriers, self.pilot_section.num_samples)

    def test_num_references(self) -> None:
        """Number of references property should return the correct value"""
        
        self.assertEqual(0, self.pilot_section.num_references)
        
        self.pilot_section.pilot_elements = None
        self.assertEqual(self.wave.num_subcarriers, self.pilot_section.num_references)

    def test_resource_mask(self) -> None:
        """The pilot section's resource mask should be completely references"""
        
        mask = self.pilot_section.resource_mask
        self.assertEqual(self.wave.num_subcarriers, np.sum(mask[ElementType.REFERENCE.value, ...], axis=(0, 1), keepdims=False))

    def test_pilot_subsymbols_setget(self) -> None:
        """Pilot subsymbol getter should return setter argument"""

        self.pilot_section.pilot_elements = None
        self.assertIs(None, self.pilot_section.pilot_elements)

        expected_subsymbols = Symbols(np.array([-1.0, 1.0]))
        self.pilot_section.pilot_elements = expected_subsymbols
        self.assertIs(expected_subsymbols, self.pilot_section.pilot_elements)

    def test_pilot_subsymbols_validation(self) -> None:
        """Pilot subsymbol setter should raise ValueErrors on invalid arguments"""

        with self.assertRaises(ValueError):
            self.pilot_section.pilot_elements = Symbols(np.array([[1], [1]]))

        with self.assertRaises(ValueError):
            self.pilot_section.pilot_elements = Symbols()

        with self.assertRaises(ValueError):
            self.pilot_section.pilot_elements = Symbols(np.empty((1, 1, 0)))

    def test_pseudorandom_pilot_sequence(self) -> None:
        """Unspecified subsymbols should result in the generation of constant valid pilot sequence"""

        self.pilot_section.pilot_elements = None

        first_pilot_sequence = self.pilot_section._pilot_sequence(self.wave.num_subcarriers)
        second_pilot_sequence = self.pilot_section._pilot_sequence()

        assert_array_equal(first_pilot_sequence, second_pilot_sequence)

    def test_configured_pilot_sequence(self) -> None:
        """Specified subsymbols should result in the generation of a valid pilot sequence"""

        self.pilot_section.pilot_elements = Symbols(np.array([[[1.0, -1.0, 1.0j, -1.0j]]], dtype=np.complex128))
        pilot_sequence = self.pilot_section._pilot_sequence()

        assert_array_equal(self.pilot_section.pilot_elements.raw.flatten(), pilot_sequence[:4])

    def test_generate(self) -> None:
        """Generation should return a valid pilot section"""

        expected_pilot_symbols = np.exp(2j * pi * self.rng.uniform(0, 1, (1, 1, self.wave.num_subcarriers)))
        self.pilot_section.pilot_elements = Symbols(expected_pilot_symbols)

        pilot = self.pilot_section.generate()
        cached_pilot = self.pilot_section.generate()
        assert_array_equal(pilot, cached_pilot)
        
    def test_pick_place_symbols(self) -> None:
        
        placed_symbols = self.pilot_section.place_symbols(np.empty(0, dtype=np.complex128), Mock())
        picked_symbols = self.pilot_section.pick_symbols(placed_symbols)
        
        self.assertEqual(0, picked_symbols.size)

    def test_pick_place_samples(self) -> None:
        
        samples = Mock()
        picked_samples = self.pilot_section.pick_samples(self.pilot_section.place_samples(samples))
        self.assertIs(samples, picked_samples)

    def test_generate_validation(self) -> None:
        """Generate should raise a RuntimeError if no waveform is set"""
        
        self.pilot_section.wave = None
        with self.assertRaises(RuntimeError):
            self.pilot_section.generate()

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        self.pilot_section.wave = None
        test_yaml_roundtrip_serialization(self, self.pilot_section)


class TestCorrelationSynchronization(TestCase):
    """Test OFDM Synchronization via pilot section correlation"""

    def setUp(self) -> None:
        self.rng = default_rng(42)

        test_resource = GridResource(repetitions=1, elements=[GridElement(ElementType.DATA, repetitions=1024)])
        test_payload = SymbolSection(num_repetitions=3, pattern=[0])
        self.wave = OrthogonalWaveformMock(num_subcarriers=1024, oversampling_factor=4, grid_resources=[test_resource], grid_structure=[test_payload])
        self.wave.pilot_section = PilotSection()

        self.synchronization = OFDMCorrelationSynchronization()
        self.wave.synchronization = self.synchronization

        self.num_streams = 3
        self.delays_in_samples = [0, 9, 80]
        self.num_frames = [1, 2, 3]

    def test_synchronize(self) -> None:
        """Test the proper estimation of delays during correlation synchronization"""

        for d, n in product(self.delays_in_samples, self.num_frames):
            symbols = np.exp(2j * pi * self.rng.uniform(0, 1, (n, self.wave.num_data_symbols)))
            frames = [np.outer(np.exp(2j * pi * self.rng.uniform(0, 1, self.num_streams)), self.wave.modulate(self.wave.place(Symbols(symbols[f, :])))) for f in range(n)]

            signal = np.empty((self.num_streams, 0), dtype=np.complex128)
            for frame in frames:
                signal = np.concatenate((signal, np.zeros((self.num_streams, d), dtype=np.complex128), frame), axis=1)

            frame_delays = self.synchronization.synchronize(signal)

            self.assertCountEqual(d + (d + self.wave.samples_per_frame) * np.arange(n), frame_delays)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, OFDMCorrelationSynchronization())


class TestLeastSquaresChannelEstimation(TestCase):
    """Test leat squares channel estimation for orthogonal waveforms."""
    
    def setUp(self) -> None:
        self.subcarrier_spacing = 1e3
        self.num_subcarriers = 100
        self.oversampling_factor = 2

        self.rng = default_rng(42)

        self.modem = Mock()
        self.modem.random_generator = self.rng
        self.modem.carrier_frequency = 100e6

        self.prefix_type_a = PrefixType.CYCLIC
        self.prefix_ratio_a = 0.1
        self.elements_a = [GridElement(ElementType.REFERENCE), GridElement(ElementType.DATA)]
        self.resource_a = GridResource(50, self.prefix_type_a, self.prefix_ratio_a, self.elements_a)

        self.section_a = SymbolSection(2, [0])

        self.resources = [self.resource_a]
        self.sections = [self.section_a]

        self.wave = OrthogonalWaveformMock(modem=self.modem, grid_resources=self.resources, grid_structure=self.sections, num_subcarriers=self.num_subcarriers, oversampling_factor=self.oversampling_factor)

        self.estimation = OrthogonalLeastSquaresChannelEstimation()
        self.wave.channel_estimation = self.estimation

        self.wave.pilot_symbol_sequence = CustomPilotSymbolSequence(np.arange(1, 1 + 2 * self.num_subcarriers))

    def test_estimate_channel_validation(self) -> None:
        """Least squares channel estimation should raise a NotImplementedError on invalid arguments"""

        with self.assertRaises(NotImplementedError):
            self.estimation.estimate_channel(Symbols(np.empty((2, 0, 10), dtype=np.complex128)))

    def test_estimate_channel(self) -> None:
        """Least squares channel estimation should correctly compute the channel estimate"""

        symbols = self.wave.place(self.wave.map(self.rng.integers(0, 2, self.wave.bits_per_frame(self.wave.num_data_symbols))))

        expected_state = np.ones(symbols.num_blocks * symbols.num_symbols).reshape((1, 1, symbols.num_blocks, symbols.num_symbols), order="F")
        propagated_symbols_raw = symbols.raw * expected_state[:, 0, :, :]
        propagated_symbols = Symbols(propagated_symbols_raw)

        stated_symbols = self.estimation.estimate_channel(propagated_symbols)
        assert_array_almost_equal(expected_state, stated_symbols.states)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.estimation)


class TestChannelEqualization(TestCase):
    """Test orthogonal waveform channel equalization."""
    
    def setUp(self) -> None:
        self.channel_equalization = OrthogonalChannelEqualization()

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.channel_equalization)


class TestZeroForcingChannelEqualization(TestCase):
    """Test zero-forcing channel equalization for orthogonal waveforms"""
    
    def setUp(self) -> None:
        self.channel_equalization = OrthogonalZeroForcingChannelEqualization()

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.channel_equalization)
