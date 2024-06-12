# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
from numpy.random import default_rng

from hermespy.channel import Cost259Type, Cost259
from hermespy.modem import(
    CustomPilotSymbolSequence,
    SimplexLink,
    CommunicationWaveform,
    OFDMWaveform,
    SymbolSection,
    GuardSection,
    GridResource,
    GridElement,
    ElementType,
    PrefixType,
    SchmidlCoxPilotSection,
    StatedSymbols,
    Symbols,
)
from hermespy.simulation.modem.channel_estimation import IdealChannelEstimation, OFDMIdealChannelEstimation, SimulatedDevice, SingleCarrierIdealChannelEstimation
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization
from unit_tests.modem.test_waveform_single_carrier import MockSingleCarrierWaveform

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockIdealChannelEstimation(IdealChannelEstimation[CommunicationWaveform]):
    """Mock ideal channel estimation for testing purposes."""

    def estimate_channel(self, symbols: Symbols, delay: float = 0) -> StatedSymbols:
        csi = self._csi(delay, 1, 10)
        return Mock()


class TestIdealChannelEstimation(TestCase):
    def setUp(self) -> None:
        self.alpha_device = SimulatedDevice()
        self.beta_device = SimulatedDevice()
        self.channel = Mock()
        self.estimation = MockIdealChannelEstimation(self.channel, self.alpha_device, self.beta_device)

        self.waveform = Mock()
        self.waveform.modem = Mock()
        self.estimation.waveform = self.waveform

    def test_csi_validation(self) -> None:
        """Fetching the channel state should raise RuntimeErrors on invalid states"""

        self.estimation.waveform = None
        with self.assertRaises(RuntimeError):
            self.estimation.estimate_channel(Mock())
        self.estimation.waveform = self.waveform

        with self.assertRaises(RuntimeError):
            self.estimation.estimate_channel(Mock())

        self.waveform.modem = None
        with self.assertRaises(RuntimeError):
            self.estimation.estimate_channel(Mock())


class _TestIdealChannelEstimation(TestCase):
    """Base class for testing ideal channel estimations"""

    estimation: IdealChannelEstimation
    waveform: CommunicationWaveform

    def setUp(self) -> None:
        self.rng = default_rng(42)

        self.carrier_frequency = 1e8
        self.alpha_device = SimulatedDevice(carrier_frequency=self.carrier_frequency)
        self.beta_device = SimulatedDevice(carrier_frequency=self.carrier_frequency)

        self.channel = Cost259(Cost259Type.URBAN)
        self.channel.seed = 42

        self.link = SimplexLink(self.alpha_device, self.beta_device)
        self.link.seed = 42

    def test_properties(self) -> None:
        """Test ideal channel estimation properties"""

        self.assertIs(self.link.waveform, self.estimation.waveform)

    def test_estimate_channel(self) -> None:
        """Ideal channel estimation should correctly fetch the channel estimate"""

        transmission = self.alpha_device.transmit()
        propagation = self.channel.propagate(transmission, self.alpha_device, self.beta_device)
        self.beta_device.receive(propagation)

        symbols = self.link.waveform.demodulate(propagation.getitem(0).flatten())
        stated_symbols = self.estimation.estimate_channel(symbols)
        picked_symbols = self.waveform.pick(stated_symbols)

        # ToDo: How could this be tested?
        return

    def test_yaml_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.estimation)


class TestOFDMIdealChannelEstimation(_TestIdealChannelEstimation):
    """Test ideal channel estimation for multicarrier waveforms."""

    def setUp(self) -> None:
        super().setUp()

        self.subcarrier_spacing = 1e3
        self.num_subcarriers = 100
        self.oversampling_factor = 2

        self.repetitions_a = 2
        self.prefix_type_a = PrefixType.CYCLIC
        self.prefix_ratio_a = 0.1
        self.elements_a = [GridElement(ElementType.DATA, 2), GridElement(ElementType.REFERENCE, 1), GridElement(ElementType.NULL, 3)]
        self.resource_a = GridResource(self.repetitions_a, self.prefix_type_a, self.prefix_ratio_a, self.elements_a)

        self.repetitions_b = 3
        self.prefix_type_b = PrefixType.ZEROPAD
        self.prefix_ratio_b = 0.2
        self.elements_b = [GridElement(ElementType.DATA, 2), GridElement(ElementType.REFERENCE, 1), GridElement(ElementType.NULL, 3)]
        self.resource_b = GridResource(self.repetitions_b, self.prefix_type_b, self.prefix_ratio_b, self.elements_b)

        self.section_a = SymbolSection(2, [1, 0, 1])
        self.section_b = GuardSection(1e-3)
        self.section_c = SymbolSection(2, [0, 1, 0])

        self.resources = [self.resource_a, self.resource_b]
        self.sections = [self.section_a, self.section_b, self.section_c]

        self.waveform = OFDMWaveform(
            subcarrier_spacing=self.subcarrier_spacing,
            modem=self.link,
            grid_resources=self.resources,
            grid_structure=self.sections,
            num_subcarriers=self.num_subcarriers,
            oversampling_factor=self.oversampling_factor,
        )
        self.waveform.pilot_symbol_sequence = CustomPilotSymbolSequence(np.arange(1, 200))
        self.waveform.pilot_section = SchmidlCoxPilotSection()

        self.estimation = OFDMIdealChannelEstimation(self.channel, self.alpha_device, self.beta_device)
        self.waveform.channel_estimation = self.estimation


class TestSingleCarrierIdealChannelEstimation(_TestIdealChannelEstimation):
    """Test ideal channel estimation for single carrier waveforms."""

    def setUp(self) -> None:
        super().setUp()

        self.waveform = MockSingleCarrierWaveform(symbol_rate=1e6, num_preamble_symbols=3, num_postamble_symbols=3, num_data_symbols=100, pilot_rate=10, modem=self.link)
        self.estimation = SingleCarrierIdealChannelEstimation(self.channel, self.alpha_device, self.beta_device)
        self.waveform.channel_estimation = self.estimation


del _TestIdealChannelEstimation
