# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from copy import deepcopy
from unittest import TestCase

import numpy as np
from scipy.constants import speed_of_light

from hermespy.beamforming import ConventionalBeamformer
from hermespy.core import Device, Signal
from hermespy.simulation import SimulatedDevice, SimulatedUniformArray, SimulatedIdealAntenna
from hermespy.hardware_loop import PhysicalDeviceDummy,  IQCombiner, IQSplitter
from hermespy.modem import TransmittingModem, ReceivingModem, DuplexModem, RootRaisedCosineWaveform
from hermespy.radar import Radar, FMCW

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestPrecoding(ABC):

    sampling_rate: float
    carrier_frequency: float
    device: Device

    @abstractmethod
    def _setUp_device(self) -> Device:
        ...

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        self.sampling_rate = 1e6
        self.carrier_frequency = 1e8
        self.device = self._setUp_device()

        # Communication operators
        waveform = RootRaisedCosineWaveform(
            symbol_rate=self.sampling_rate//2,
            oversampling_factor=2,
            num_preamble_symbols=16,
            num_data_symbols=48,
        )
        tx_modem = TransmittingModem(waveform=deepcopy(waveform))
        rx_modem = ReceivingModem(waveform=deepcopy(waveform))
        duplex_modem = DuplexModem(waveform=deepcopy(waveform))

        # Radar operator
        radar = Radar()
        radar.waveform = FMCW()

        self.tx_operators = [tx_modem, duplex_modem, radar]
        self.rx_operators = [rx_modem, duplex_modem, radar]

    def __test_transmission(self) -> None:
        """Test all transmit operators for a given precoding"""

        for operator in self.tx_operators:
            with self.subTest(operator=operator.__class__.__name__):
                # Ensure all operators are removed from the device
                for o in self.tx_operators:
                    self.device.transmitters.remove(o)

                # Add the operator under test to the device
                self.device.transmitters.add(operator)

                # Transmit a signal
                transmission = self.device.transmit()

                # Asser the transmitted signal's shape
                self.assertEqual(self.device.num_transmit_antennas, transmission.mixed_signal.num_streams)

    def __test_reception(self) -> None:
        """Test all receive operators ofr a given precoding"""
        for operator in self.rx_operators:
            with self.subTest(operator=operator.__class__.__name__):
                # Ensure all operators are removed from the device
                for o in self.tx_operators:
                    self.device.receivers.remove(o)

                # Add the operator under test to the device
                self.device.receivers.add(operator)

                # Receive a randomly generated signal
                impinging_signal = Signal.Create(np.random.standard_normal((self.device.num_receive_antennas, 1000)) + 1j* np.random.standard_normal((self.device.num_receive_antennas, 1000)), self.sampling_rate, self.carrier_frequency)
                reception = self.device.receive(impinging_signal)

                # Asser the received signal's shape
                self.assertEqual(self.device.num_digital_receive_ports, reception.operator_inputs[0].num_streams)

    def test_transmit_beamforming(self) -> None:
        """Test digital beamforming precoding during transmission"""

        self.device.transmit_coding[0] = ConventionalBeamformer()
        self.__test_transmission()

    def test_receive_beamforming(self) -> None:
        """Test digital beamforming precoding during reception"""

        self.device.receive_coding[0] = ConventionalBeamformer()
        self.__test_reception()

    def test_iq_splitting(self) -> None:
        """Test the IQ splitter"""

        self.device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, .5 * speed_of_light / self.carrier_frequency, [2, 1, 1])
        self.device.transmit_coding[0] = IQSplitter()
        self.__test_transmission()

    def test_iq_combining(self) -> None:
        """Test the IQ combiner"""

        self.device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, .5 * speed_of_light / self.carrier_frequency, [2, 1, 1])
        self.device.receive_coding[0] = IQCombiner()
        self.__test_reception()


class TestSimulationPrecoding(TestPrecoding, TestCase):
    """Test precoding in simulated devices"""

    def _setUp_device(self) -> SimulatedDevice:
        return SimulatedDevice(
            carrier_frequency=self.carrier_frequency,
            sampling_rate=self.sampling_rate,
            antennas=SimulatedUniformArray(SimulatedIdealAntenna, .5 * speed_of_light / self.carrier_frequency, [2, 2, 1]),
        )


class TestHardwareLoopPrecoding(TestPrecoding, TestCase):
    """Test precoding in hardware loop devices"""

    def _setUp_device(self) -> PhysicalDeviceDummy:
        return PhysicalDeviceDummy(
            carrier_frequency=self.carrier_frequency,
            sampling_rate=self.sampling_rate,
            antennas=SimulatedUniformArray(SimulatedIdealAntenna, .5 * speed_of_light / self.carrier_frequency, [2, 2, 1]),
        )
