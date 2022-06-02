from unittest import TestCase

import numpy as np

from hermespy.channel import Channel
from hermespy.core import Scenario, IdealAntenna, UniformArray
from hermespy.simulation import SimulatedDevice
from hermespy.modem import Modem, WaveformGeneratorPskQam, BitErrorEvaluator
from hermespy.precoding import SpatialMultiplexing

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "3.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestLinks(TestCase):

    def setUp(self) -> None:

        # Configure a 2x2 link scenario
        antennas = UniformArray(IdealAntenna(), 5e-3, [2, 1, 1])
        self.tx_device = SimulatedDevice(antennas=antennas)
        self.rx_device = SimulatedDevice(antennas=antennas)

        scenario = Scenario()
        scenario.add_device(self.tx_device)
        scenario.add_device(self.rx_device)

        # Define a transmit operation on the first device
        self.tx_operator = Modem()
        self.tx_operator.precoding[0] = SpatialMultiplexing()
        self.tx_operator.device = self.tx_device

        # Define a receive operation on the second device
        self.rx_operator = Modem()
        self.rx_operator.precoding[0] = SpatialMultiplexing()
        self.rx_operator.device = self.rx_device
        self.rx_operator.reference_transmitter = self.tx_operator

        self.ber = BitErrorEvaluator(self.tx_operator, self.rx_operator)

    def __propagate(self, channel: Channel) -> None:
        """Helper function to propagate a signal from transmitter to receiver.
        
        Args:

            channel (Channel):
                The channel over which to propagate the signal from transmitter to receiver.
        """

        _ = self.tx_operator.transmit()
        tx_signals = self.tx_device.transmit()
        rx_signals, _, channel_state = channel.propagate(tx_signals)
        self.rx_device.receive(np.array([[rx_signals, channel_state]], dtype=object))
        _ = self.rx_operator.receive()

    def test_ideal_channel_psk_qam(self) -> None:
        """Verify a valid MIMO link over an ideal channel with PSK/QAM modulation"""

        self.tx_operator.waveform_generator = WaveformGeneratorPskQam(oversampling_factor=8)
        self.rx_operator.waveform_generator = WaveformGeneratorPskQam(oversampling_factor=8)

        self.__propagate(Channel(self.tx_device, self.rx_device))

        self.assertEqual(0, self.ber.evaluate().to_scalar())

    # def test_cost256_psk_qam(self) -> None:
    #     """Verify a valid MIMO link over a 3GPP COST256 TDL channel with PSK/QAM modulation"""
    # 
    #     self.tx_operator.waveform_generator = WaveformGeneratorPskQam(oversampling_factor=8)
    #     self.rx_operator.waveform_generator = WaveformGeneratorPskQam(oversampling_factor=8)
    # 
    #     self.__propagate(MultipathFadingCost256(MultipathFadingCost256.TYPE.URBAN, transmitter=self.tx_operator.device, receiver=self.rx_operator.device))
    #     self.assertEqual(0, self.ber.evaluate().to_scalar())
