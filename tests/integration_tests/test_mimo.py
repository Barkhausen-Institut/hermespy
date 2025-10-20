# -*- coding: utf-8 -*-
from unittest import TestCase
from typing import Tuple

import numpy as np
from numpy.testing import assert_array_equal
from scipy.constants import speed_of_light, pi

from hermespy.beamforming import ConventionalBeamformer
from hermespy.core import Transformation
from hermespy.modem import Alamouti, Ganesan, OrthogonalLeastSquaresChannelEstimation, SimplexLink, ChannelEqualization, CommunicationReception, CommunicationTransmission, OFDMWaveform, OrthogonalZeroForcingChannelEqualization, PilotSection, GridResource, GridElement, ElementType, SymbolSection, OrthogonalZeroForcingChannelEqualization, OFDMCorrelationSynchronization, ReferencePosition
from hermespy.simulation import DeviceFocus, SimulatedDevice, SimulatedIdealAntenna, SimulatedUniformArray, OFDMIdealChannelEstimation, N0
from hermespy.channel import Channel, DelayNormalization, UrbanMacrocells, O2IState

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class _TestMIMOLink(TestCase):

    def __configure_ofdm_waveform(self, channel: Channel) -> OFDMWaveform:
        """Configure an OFDM wafeform with default parameters.

        Returns: The configured waveform.
        """

        # Mock 5G numerology #1:
        # 120khz subcarrier spacing, 120 subcarriers, 2us guard interval, 1ms subframe duration

        prefix_ratio = 0.0684
        num_subcarriers = 128
        grid_resources = [
            GridResource(num_subcarriers // 5, prefix_ratio=prefix_ratio, elements=[
                GridElement(ElementType.REFERENCE, 1),
                GridElement(ElementType.DATA, 4)
            ]),
            GridResource(num_subcarriers // 5, prefix_ratio=prefix_ratio, elements=[
                GridElement(ElementType.DATA, 2),
                GridElement(ElementType.REFERENCE, 1),
                GridElement(ElementType.DATA, 2),
            ]),
        ]
        grid_structure = [
            SymbolSection(
                num_subcarriers // 2,
                [0, 1],
                5,
            ),
        ]

        waveform = OFDMWaveform(
            num_subcarriers=num_subcarriers,
            dc_suppression=True,
            grid_resources=grid_resources,
            grid_structure=grid_structure,
            modulation_order=4,
        )
        waveform.pilot_section = PilotSection()
        waveform.synchronization = OFDMCorrelationSynchronization()
        waveform.channel_estimation = OFDMIdealChannelEstimation(channel, self.tx_device, self.rx_device, reference_position=ReferencePosition.IDEAL)
        waveform.channel_equalization = OrthogonalZeroForcingChannelEqualization()

        return waveform

    def setUp(
        self,
        num_transmit_antennas: int = 1,
        num_receive_antennas: int = 1,
    ) -> None:
        self.carrier_frequency = 1e8
        self.bandwidth = 128*12e3
        self.oversampling_factor = 1
        self.wavelength = speed_of_light / self.carrier_frequency

        self.tx_device = SimulatedDevice(
            carrier_frequency=self.carrier_frequency,
            bandwidth=self.bandwidth,
            oversampling_factor=self.oversampling_factor,
            pose=Transformation.From_Translation(np.array([0, 0, 10.0])),
            antennas=SimulatedUniformArray(SimulatedIdealAntenna(), 0.5 * self.wavelength, [num_transmit_antennas, 1, 1]),
            noise_level=N0(0.0),
        )
        self.rx_device = SimulatedDevice(
            carrier_frequency=self.carrier_frequency,
            bandwidth=self.bandwidth,
            oversampling_factor=self.oversampling_factor,
            pose=Transformation.From_RPY(np.array([0, 0, pi]), np.array([1000, 0, 2])),
            antennas=SimulatedUniformArray(SimulatedIdealAntenna(), 0.5 * self.wavelength, [num_receive_antennas, 1, 1]),
            noise_level=N0(0.0),
        )
        self.channel = UrbanMacrocells(
            delay_normalization=DelayNormalization.ZERO,
            expected_state=O2IState.LOS,
            seed=42,
        )

        self.link = SimplexLink()
        self.tx_device.transmitters.add(self.link)
        self.rx_device.receivers.add(self.link)
        self.link.waveform = self.__configure_ofdm_waveform(self.channel)

    def __propagate(self) -> Tuple[CommunicationTransmission, CommunicationReception]:

        device_transmission = self.tx_device.transmit()

        channel_realization = self.channel.realize()
        channel_sample = channel_realization.sample(self.tx_device, self.rx_device)
        propagation = channel_sample.propagate(device_transmission)

        device_reception = self.rx_device.receive(propagation)

        return device_transmission.operator_transmissions[0], device_reception.operator_receptions[0]

    def test_propagation(self) -> None:
        """Test valid data transmission"""

        transmission, reception = self.__propagate()
        reception.signal.plot()
        reception.equalized_symbols.plot_constellation()

        assert_array_equal(transmission.bits, reception.bits)


class TestAlamouti(_TestMIMOLink):
    """Test Alamouti space-time block precoding"""

    def setUp(
        self,
        num_transmit_antennas: int = 2,
        num_receive_antennas: int = 1,
    ) -> None:
        super().setUp(num_transmit_antennas, num_receive_antennas)

        self.link.transmit_symbol_coding[0] = Alamouti()
        self.link.receive_symbol_coding[0] = Alamouti()
        self.link.waveform.channel_estimation = OFDMIdealChannelEstimation(self.channel, self.tx_device, self.rx_device)
        self.link.waveform.channel_equalization = ChannelEqualization()


class TestGanesan(_TestMIMOLink):
    """Test Ganesan space-time block precoding"""

    def setUp(
        self,
        num_transmit_antennas: int = 4,
        num_receive_antennas: int = 1,
    ) -> None:
        super().setUp(num_transmit_antennas, num_receive_antennas)

        self.link.transmit_symbol_coding[0] = Ganesan()
        self.link.receive_symbol_coding[0] = Ganesan()
        self.link.waveform.channel_estimation = OFDMIdealChannelEstimation(self.channel, self.tx_device, self.rx_device)
        self.link.waveform.channel_equalization = ChannelEqualization()


class TestConventionalBeamformer(_TestMIMOLink):

    def setUp(self, num_transmit_antennas: int = 4, num_receive_antennas: int = 4) -> None:

        super().setUp(num_transmit_antennas, num_receive_antennas)

        tx_beamformer = ConventionalBeamformer()
        rx_beamformer = ConventionalBeamformer()
        tx_beamformer.transmit_focus = DeviceFocus(self.rx_device)
        rx_beamformer.receive_focus = DeviceFocus(self.tx_device)

        self.tx_device.transmit_coding[0] = tx_beamformer
        self.rx_device.receive_coding[0] = rx_beamformer

        self.link.waveform.channel_estimation = OrthogonalLeastSquaresChannelEstimation()


del _TestMIMOLink
