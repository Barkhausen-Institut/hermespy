from unittest import TestCase
from typing import Tuple

import numpy as np
from numpy.testing import assert_array_equal
from scipy.constants import speed_of_light, pi

from hermespy.beamforming import ConventionalBeamformer
from hermespy.core.antennas import UniformArray, IdealAntenna
from hermespy.modem import  TransmittingModem, ReceivingModem, ChannelEqualization, CommunicationReception, CommunicationTransmission
from hermespy.modem.waveform_single_carrier import RootRaisedCosineWaveform, SingleCarrierIdealChannelEstimation, SingleCarrierLeastSquaresChannelEstimation, SingleCarrierZeroForcingChannelEqualization
from hermespy.precoding.space_time_block_coding import SpaceTimeBlockCoding
from hermespy.simulation import SimulatedDevice
from hermespy.channel import RuralMacrocellsLineOfSight

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestMIMOLink(TestCase):
    
    def setUp(self) -> None:
        
        self.carrier_frequency = 60e9
        self.wavelength = speed_of_light / self.carrier_frequency
        
        self.tx_device = SimulatedDevice(seed=42)
        self.tx_device.position = np.array([0, 0, 0])
        self.tx_device.orientation = np.array([0, 0, 0])
        self.tx_device.antennas = UniformArray(IdealAntenna(), .5 * self.wavelength, [2, 2])

        self.rx_device = SimulatedDevice(seed=66)
        self.rx_device.position = np.array([0, 0, 1000])
        self.rx_device.orientation = np.array([pi, 0, 0])
        self.rx_device.antennas = UniformArray(IdealAntenna(), .5 * self.wavelength, [3, 3])

        self.channel = RuralMacrocellsLineOfSight(transmitter=self.tx_device, receiver=self.rx_device, seed=42)
        
        self.tx_modem = TransmittingModem()
        self.tx_modem.waveform_generator = RootRaisedCosineWaveform(symbol_rate=1e8, num_preamble_symbols=16, num_data_symbols=51, 
                                                                    pilot_rate=5, oversampling_factor=4, modulation_order=4)
        
        
        self.rx_modem = ReceivingModem()
        self.rx_modem.waveform_generator = RootRaisedCosineWaveform(symbol_rate=1e8, num_preamble_symbols=16, num_data_symbols=51,
                                                                    pilot_rate=5, oversampling_factor=4, modulation_order=4)
        self.rx_modem.waveform_generator.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()
        self.rx_modem.waveform_generator.channel_equalization = SingleCarrierZeroForcingChannelEqualization()
        
        self.tx_device.transmitters.add(self.tx_modem)
        self.rx_device.receivers.add(self.rx_modem)
    
    def __propagate(self) -> Tuple[CommunicationTransmission, CommunicationReception]:
        
        device_transmission = self.tx_device.transmit()
        
        propagation, _, csi = self.channel.propagate(device_transmission)
        propagation[0].samples = propagation[0].samples[:, :self.rx_modem.waveform_generator.samples_in_frame]
        
        device_reception = self.rx_device.receive(propagation, channel_state=csi)
        
        return device_transmission.operator_transmissions[0], device_reception.operator_receptions[0]
        
    def test_conventional_beamforming(self) -> None:
        """Test valid data transmission using conventional beamformers"""

        tx_beamformer = ConventionalBeamformer()
        rx_beamformer = ConventionalBeamformer()

        self.tx_modem.transmit_stream_coding[0] = tx_beamformer
        self.rx_modem.receive_stream_coding[0] = rx_beamformer

        transmission, reception = self.__propagate()
        assert_array_equal(transmission.bits, reception.bits)

    def test_alamouti(self) -> None:
        """Test valid data tansmission via Alamouti precoding"""
        
        self.tx_device.antennas = UniformArray(IdealAntenna(), .5 * self.wavelength, [2])
        self.rx_device.antennas = UniformArray(IdealAntenna(), .5 * self.wavelength, [2])
        
        self.tx_modem.precoding[0] = SpaceTimeBlockCoding()
        self.rx_modem.precoding[0] = SpaceTimeBlockCoding()
        self.rx_modem.waveform_generator.channel_estimation = SingleCarrierIdealChannelEstimation()
        self.rx_modem.waveform_generator.channel_equalization = ChannelEqualization()

        transmission, reception = self.__propagate()
        assert_array_equal(transmission.bits, reception.bits)
