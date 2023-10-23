# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, patch, PropertyMock

from numpy.random import default_rng

from hermespy.channel import Cost259Type, MultipathFadingCost259
from hermespy.modem import SimplexLink, WaveformGenerator, OFDMWaveform, FrameSymbolSection, FrameGuardSection, FrameResource
from hermespy.modem.symbols import StatedSymbols, Symbols
from hermespy.modem.waveform_ofdm import FrameElement, ElementType, PrefixType, SchmidlCoxPilotSection
from hermespy.simulation.modem.channel_estimation import IdealChannelEstimation, OFDMIdealChannelEstimation, SimulatedDevice, SingleCarrierIdealChannelEstimation
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization
from unit_tests.modem.test_waveform_single_carrier import MockSingleCarrierWaveform

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockIdealChannelEstimation(IdealChannelEstimation[WaveformGenerator]):
    """Mock ideal channel estimation for testing purposes."""
    
    def estimate_channel(self, symbols: Symbols, delay: float = 0) -> StatedSymbols:
        
        csi = self._csi(delay, 1, 10)
        return Mock()
    

class TestIdealChannelEstimation(TestCase):
    
    def setUp(self) -> None:
        
        self.alpha_device = SimulatedDevice()
        self.beta_device = SimulatedDevice()
        self.estimation = MockIdealChannelEstimation(self.alpha_device, self.beta_device)
        
        self.waveform = Mock()
        self.waveform.modem = Mock()
        self.estimation.waveform_generator = self.waveform

    def test_csi_validation(self) -> None:
        """Fetching the channel state should raise RuntimeErrors on invalid states"""

        self.estimation.waveform_generator = None
        with self.assertRaises(RuntimeError):
            self.estimation.estimate_channel(Mock())
        self.estimation.waveform_generator = self.waveform 
            
        with self.assertRaises(RuntimeError):
            self.estimation.estimate_channel(Mock())

        self.waveform.modem = None
        with self.assertRaises(RuntimeError):
            self.estimation.estimate_channel(Mock())


class _TestIdealChannelEstimation(TestCase):
    """Base class for testing ideal channel estimations"""

    estimation: IdealChannelEstimation
    waveform: WaveformGenerator

    def setUp(self) -> None:
        
        self.rng = default_rng(42)
        
        self.carrier_frequency = 1e8
        self.alpha_device = SimulatedDevice(carrier_frequency=self.carrier_frequency)
        self.beta_device = SimulatedDevice(carrier_frequency=self.carrier_frequency)

        self.channel = MultipathFadingCost259(Cost259Type.URBAN, self.alpha_device, self.beta_device)
        self.channel.seed = 42

        self.link = SimplexLink(self.alpha_device, self.beta_device)
        self.link.seed = 42
        
    def test_properties(self) -> None:
        """Test ideal channel estimation properties"""
        
        self.assertIs(self.estimation.transmitter, self.alpha_device)
        self.assertIs(self.estimation.receiver, self.beta_device)
        self.assertIs(self.link.waveform_generator, self.estimation.waveform_generator)

    def test_estimate_channel(self) -> None:
        """Ideal channel estimation should correctly fetch the channel estimate"""
        
        transmission = self.alpha_device.transmit()
        propagation = self.channel.propagate(transmission)
        self.beta_device.receive(propagation)
        
        symbols = self.link.waveform_generator.demodulate(propagation.signal.samples[0, :])
        stated_symbols = self.estimation.estimate_channel(symbols)
        
        # ToDo: How could this be tested?
                        
    def test_yaml_serialization(self) -> None:
        """Test YAML serialization"""
        
        test_yaml_roundtrip_serialization(self, self.estimation)


class TestOFDMIdealChannelEstimation(_TestIdealChannelEstimation):
    """Test ideal channel estimation for OFDM waveforms."""
    
    def setUp(self) -> None:
        
        super().setUp()
        
        self.subcarrier_spacing = 1e3
        self.num_subcarriers = 100
        self.oversampling_factor = 2

        self.repetitions_a = 2
        self.prefix_type_a = PrefixType.CYCLIC
        self.prefix_ratio_a = 0.1
        self.elements_a = [FrameElement(ElementType.DATA, 2),
                           FrameElement(ElementType.REFERENCE, 1),
                           FrameElement(ElementType.NULL, 3)]
        self.resource_a = FrameResource(self.repetitions_a, self.prefix_type_a, self.prefix_ratio_a, self.elements_a)

        self.repetitions_b = 3
        self.prefix_type_b = PrefixType.ZEROPAD
        self.prefix_ratio_b = 0.2
        self.elements_b = [FrameElement(ElementType.REFERENCE, 2),
                           FrameElement(ElementType.DATA, 1),
                           FrameElement(ElementType.NULL, 3)]
        self.resource_b = FrameResource(self.repetitions_b, self.prefix_type_b, self.prefix_ratio_b, self.elements_b)

        self.section_a = FrameSymbolSection(2, [1, 0, 1])
        self.section_b = FrameGuardSection(1e-3)
        self.section_c = FrameSymbolSection(2, [0, 1, 0])

        self.resources = [self.resource_a, self.resource_b]
        self.sections = [self.section_a, self.section_b, self.section_c]

        self.waveform = OFDMWaveform(subcarrier_spacing=self.subcarrier_spacing, modem=self.link,
                                     resources=self.resources, structure=self.sections,
                                     num_subcarriers=self.num_subcarriers,
                                     oversampling_factor=self.oversampling_factor)
        self.waveform.pilot_section = SchmidlCoxPilotSection()
        
        self.estimation = OFDMIdealChannelEstimation(self.alpha_device, self.beta_device)
        self.waveform.channel_estimation = self.estimation


class TestSingleCarrierIdealChannelEstimation(_TestIdealChannelEstimation):
    """Test ideal channel estimation for single carrier waveforms."""
    
    def setUp(self) -> None:
        
        super().setUp()
        
        self.waveform = MockSingleCarrierWaveform(symbol_rate=1e6,
                                                  num_preamble_symbols=3,
                                                  num_postamble_symbols=3,
                                                  num_data_symbols=100,
                                                  pilot_rate=10,
                                                  modem=self.link)
        self.estimation = SingleCarrierIdealChannelEstimation(self.alpha_device, self.beta_device)
        self.waveform.channel_estimation = self.estimation


del _TestIdealChannelEstimation
