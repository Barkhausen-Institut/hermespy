from unittest import TestCase
from typing import Tuple

import numpy as np
from numpy.testing import assert_array_equal
from scipy.constants import speed_of_light, pi

from hermespy.beamforming import ConventionalBeamformer
from hermespy.core.antennas import UniformArray, IdealAntenna, Signal
from hermespy.modem import  Modem, ChannelEqualization, CommunicationReception, CommunicationTransmission
from hermespy.modem.waveform_single_carrier import RootRaisedCosineWaveform, SingleCarrierCorrelationSynchronization, SingleCarrierIdealChannelEstimation, SingleCarrierLeastSquaresChannelEstimation, SingleCarrierZeroForcingChannelEqualization
from hermespy.precoding.space_time_block_coding import SpaceTimeBlockCoding
from hermespy.simulation import SimulatedDevice
from hermespy.channel import RuralMacrocellsLineOfSight


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
        
        self.tx_modem = Modem()
        self.tx_modem.waveform_generator = RootRaisedCosineWaveform(symbol_rate=1e8, num_preamble_symbols=16, num_data_symbols=50, 
                                                                    pilot_rate=5, oversampling_factor=4, modulation_order=4)
        self.tx_modem.device = self.tx_device
        
        self.rx_modem = Modem()
        self.rx_modem.waveform_generator = RootRaisedCosineWaveform(symbol_rate=1e8, num_preamble_symbols=16, num_data_symbols=50,
                                                                    pilot_rate=5, oversampling_factor=4, modulation_order=4)
        self.rx_modem.device = self.rx_device
        #self.rx_modem.waveform_generator.synchronization = SingleCarrierCorrelationSynchronization()
        self.rx_modem.waveform_generator.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()
        self.rx_modem.waveform_generator.channel_equalization = SingleCarrierZeroForcingChannelEqualization()
            
    def __propagate(self) -> Tuple[CommunicationTransmission, CommunicationReception]:
        
        communication_transmission = self.tx_modem.transmit()
        device_transmission = self.tx_device.transmit()
        
        device_reception, _, csi = self.channel.propagate(device_transmission)
        
        device_reception[0].samples = device_reception[0].samples[:, :self.rx_modem.waveform_generator.samples_in_frame]
        self.rx_device.receive(device_reception)
        
        self.rx_modem._receiver.cache_reception(self.rx_modem._receiver.signal, csi)
        communication_reception = self.rx_modem.receive()
        
        return communication_transmission, communication_reception
        
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
