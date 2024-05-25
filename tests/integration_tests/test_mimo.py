# -*- coding: utf-8 -*-
from unittest import TestCase
from typing import Tuple

import numpy as np
from numpy.testing import assert_array_equal
from scipy.constants import speed_of_light, pi

from hermespy.beamforming import ConventionalBeamformer, DeviceFocus
from hermespy.core import Transformation
from hermespy.modem import Alamouti, Ganesan, SimplexLink, ChannelEqualization, CommunicationReception, CommunicationTransmission, OFDMWaveform, OrthogonalZeroForcingChannelEqualization, PilotSection, GridResource, GridElement, ElementType, SymbolSection, OrthogonalZeroForcingChannelEqualization, OFDMCorrelationSynchronization, ReferencePosition, OrthogonalLeastSquaresChannelEstimation
from hermespy.simulation import SimulatedDevice, SimulatedIdealAntenna, SimulatedUniformArray, OFDMIdealChannelEstimation
from hermespy.channel import Channel, DelayNormalization, UrbanMacrocells, O2IState

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestMIMOLink(TestCase):
    
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

        waveform = OFDMWaveform(subcarrier_spacing=1e6 / num_subcarriers, num_subcarriers=num_subcarriers, dc_suppression=True, grid_resources=grid_resources, grid_structure=grid_structure)
        waveform.oversampling_factor = 2
        waveform.pilot_section = PilotSection()
        waveform.synchronization = OFDMCorrelationSynchronization()
        waveform.channel_estimation = OFDMIdealChannelEstimation(channel, self.tx_device, self.rx_device, reference_position=ReferencePosition.IDEAL)
        waveform.channel_equalization = OrthogonalZeroForcingChannelEqualization()

        return waveform
    
    def setUp(self) -> None:
        self.carrier_frequency = 1e8
        self.wavelength = speed_of_light / self.carrier_frequency

        self.tx_device = SimulatedDevice(
            carrier_frequency=self.carrier_frequency,
            pose=Transformation.From_Translation(np.array([0, 0, 10.0])),
            antennas=SimulatedUniformArray(SimulatedIdealAntenna(), 0.5 * self.wavelength, [2, 2]),
        )
        self.rx_device = SimulatedDevice(
            carrier_frequency=self.carrier_frequency,
            pose=Transformation.From_RPY(np.array([0, 0, pi]), np.array([1000, 0, 2])),
            antennas=SimulatedUniformArray(SimulatedIdealAntenna(), 0.5 * self.wavelength, [3, 3]),
        )
        self.channel = UrbanMacrocells(
            alpha_device=self.tx_device,
            beta_device=self.rx_device,
            delay_normalization=DelayNormalization.ZERO,
            expected_state=O2IState.LOS,
            seed=42,
        )

        self.link = SimplexLink(self.tx_device, self.rx_device)
        self.link.waveform = self.__configure_ofdm_waveform(self.channel)

    def __propagate(self) -> Tuple[CommunicationTransmission, CommunicationReception]:
        
        device_transmission = self.tx_device.transmit()

        channel_realization = self.channel.realize()
        channel_sample = channel_realization.sample(self.tx_device, self.rx_device)
        propagation = channel_sample.propagate(device_transmission)

        device_reception = self.rx_device.receive(propagation)

        return device_transmission.operator_transmissions[0], device_reception.operator_receptions[0]

    # ToDo: Fix duplex operator transmit/receive device assignment
    #def test_conventional_beamforming(self) -> None:
    #    """Test valid data transmission using conventional beamformers"""
    #
    #    tx_beamformer = ConventionalBeamformer()
    #    rx_beamformer = ConventionalBeamformer()
    #    tx_beamformer.transmit_focus = DeviceFocus(self.rx_device)
    #    rx_beamformer.receive_focus = DeviceFocus(self.tx_device)
    #
    #    self.link.transmit_stream_coding[0] = tx_beamformer
    #    self.link.receive_stream_coding[0] = rx_beamformer
    #    self.link.waveform.channel_estimation = OrthogonalLeastSquaresChannelEstimation()
    #    
    #    transmission, reception = self.__propagate()
    #    assert_array_equal(transmission.bits, reception.bits)

    def test_alamouti(self) -> None:
        """Test valid data tansmission via Alamouti precoding"""

        self.tx_device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, 0.5 * self.wavelength, [2])
        self.rx_device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, 0.5 * self.wavelength, [1])

        self.link.precoding[0] = Alamouti()
        self.link.waveform.channel_estimation = OFDMIdealChannelEstimation(self.channel, self.tx_device, self.rx_device)
        self.link.waveform.channel_equalization = ChannelEqualization()

        transmission, reception = self.__propagate()
        assert_array_equal(transmission.bits, reception.bits)

    def test_ganesan(self) -> None:
        """Test valid data transmission via Ganesan precoding"""

        self.tx_device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, 0.5 * self.wavelength, [4])
        self.rx_device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, 0.5 * self.wavelength, [1])

        self.link.precoding[0] = Ganesan()
        self.link.precoding[0] = Ganesan()
        self.link.waveform.channel_estimation = OFDMIdealChannelEstimation(self.channel, self.tx_device, self.rx_device)
        self.link.waveform.channel_equalization = ChannelEqualization()

        transmission, reception = self.__propagate()
        assert_array_equal(transmission.bits, reception.bits)
