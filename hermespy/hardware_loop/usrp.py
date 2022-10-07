# -*- coding: utf-8 -*-
"""
====================
USRP Hardware Driver
====================
"""

from functools import cached_property
from math import ceil
from time import sleep
from typing import List, Optional


import numpy as np
from zerorpc import Client

from hermespy.core import Device, Scenario, Signal
from .physical_device import PhysicalDevice
from .scenario import PhysicalScenario
from usrp_client.rpc_client import UsrpClient
from usrp_client.system import System as _UsrpSystem, LabeledUsrp as _LabeledUsrp
from uhd_wrapper.utils.config import MimoSignal, TxStreamingConfig, RxStreamingConfig, RfConfig, txContainsClippedValue

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class UsrpDevice(UsrpClient, PhysicalDevice):

    __ip: str
    __port: int

    def __init__(self,
                 ip: str,
                 port: Optional[int] = 5555,
                 carrier_frequency: float = 7e8,
                 *args, **kwargs) -> None:

        client = Client()
        client.connect(f"tcp://{ip}:{port}")
        self.__ip = ip
        self.__port = port

        UsrpClient.__init__(self, client)
        PhysicalDevice.__init__(self, *args, **kwargs)

        self.carrier_frequency = carrier_frequency
        self.tx_gain = 0.
        self.rx_gain = 0.
        self.__current_configuration = RfConfig(-1)

    def _configure_device(self) -> None:

        tx_filter_bandwidth = 4e8
        rx_filter_bandwidth = 4e8
        tx_sampling_rate = self.sampling_rate
        rx_sampling_rate = self.sampling_rate
        tx_carrier_frequency = self.carrier_frequency
        rx_carrier_frequency = self.carrier_frequency
        tx_gain = self.tx_gain
        rx_gain = self.rx_gain

        # Check if a frontend reconfiguration is required
        if any([
            tx_filter_bandwidth != self.__current_configuration.txAnalogFilterBw,
            rx_filter_bandwidth != self.__current_configuration.rxAnalogFilterBw,
            tx_sampling_rate != self.__current_configuration.txSamplingRate,
            rx_sampling_rate != self.__current_configuration.rxSamplingRate,
            tx_carrier_frequency != self.__current_configuration.txCarrierFrequency,
            rx_carrier_frequency != self.__current_configuration.rxCarrierFrequency,
            tx_gain != self.__current_configuration.txGain,
            rx_gain != self.__current_configuration.rxGain,
        ]):

            config = RfConfig(
                txAnalogFilterBw=tx_filter_bandwidth,
                rxAnalogFilterBw=rx_filter_bandwidth,
                txSamplingRate=tx_sampling_rate,
                rxSamplingRate=rx_sampling_rate,
                txCarrierFrequency=tx_carrier_frequency,
                rxCarrierFrequency=rx_carrier_frequency,
                txGain=tx_gain,
                rxGain=rx_gain,
            )

            self.configureRfConfig(config)
            self.resetStreamingConfigs()
            self.__current_configuration = config

    def configure(self) -> None:

        # Configure device
        self._configure_device()

        # Configure transmission
        baseband_signal = Device.transmit(self)

        if baseband_signal.num_samples > 0:
            baseband_signal.samples /= np.abs(baseband_signal.samples).max()

        if baseband_signal.num_samples % 2 != 0:
            baseband_signal.samples = np.append(baseband_signal.samples, np.zeros((baseband_signal.num_streams, 1), dtype=complex), axis=1)

        mimo_signal = MimoSignal(list(baseband_signal.samples))
        tx_config = TxStreamingConfig(max(0., -self.calibration_delay), mimo_signal)
        self.configureTx(tx_config)

        # Configure reception
        duration = self.receivers.min_frame_duration
        num_receive_samples = int((duration + self.max_receive_delay) * self.sampling_rate)
        num_receive_samples += num_receive_samples % 2  # Workaround for the uneven sample bug

        rx_config = RxStreamingConfig(max(0., self.calibration_delay), num_receive_samples)
        self.configureRx(rx_config)

    def trigger(self) -> None:

        # Configure transmit and receive behaviour
        self.configure()

        # Queue execution command
        self.execute(self.getCurrentFpgaTime() + .2)

        # Fetch resulting samples
        self.fetch()

    def fetch(self) -> None:
        
        mimo_signals = self.collect()
        signal_model = Signal.empty(self.sampling_rate, self.antennas.num_antennas)

        for mimo_signal in mimo_signals:

            streams = np.array(mimo_signal.signals)
            signal_model.samples = np.append(signal_model.samples, streams, axis=1)

        for receiver in self.receivers:
            receiver.cache_reception(signal_model)

    @property
    def ip(self) -> str:
        """Internet protocol address of the remote host.

        Returns:

            IP adress.
        """

        return self.__ip

    @property
    def port(self) -> int:
        """Internet protocol port of the remote host.

        Returns:

            Port.
        """

        return self.__port

    @property
    def tx_gain(self) -> float:

        return self.__tx_gain

    @tx_gain.setter
    def tx_gain(self, value: float) -> None:

        self.__tx_gain = value

    @property
    def rx_gain(self) -> float:

        return self.__rx_gain

    @rx_gain.setter
    def rx_gain(self, value: float) -> None:

        self.__rx_gain = value

    @property
    def sampling_rate(self) -> float:

        ideal_sampling_rate = self.transmitters.max_sampling_rate
        selected_sampling_rate = min(self.__supported_sampling_rates, key=lambda x: abs(x - ideal_sampling_rate))
        
        return selected_sampling_rate

    @property
    def max_sampling_rate(self) -> float:

        return max(self.__supported_sampling_rates)

    @property
    def carrier_frequency(self) -> float:

        return self.__carrier_frequency

    @carrier_frequency.setter
    def carrier_frequency(self, value: float) -> None:

        self.__carrier_frequency = value

    @cached_property
    def __supported_sampling_rates(self) -> List[float]:

        return self.getSupportedSamplingRates()


class UsrpSystem(PhysicalScenario[UsrpDevice]):

    def __init__(self, *args, **kwargs) -> None:

        self.__synchronized = False
        PhysicalScenario.__init__(self, *args, **kwargs)

        # Hacked USRP system (hidden)   
        self.__system = _UsrpSystem()

    def new_device(self, *args, **kwargs) -> UsrpDevice:

        device = UsrpDevice(*args, **kwargs)
        self.add_device(device)

        return device

    def add_device(self, device: UsrpDevice) -> None:

        usrp_uid = str(self.num_devices)
        self.__system._System__usrpClients[usrp_uid] = _LabeledUsrp(usrp_uid, device.ip, device)
        Scenario.add_device(self, device)
     
    def trigger(self) -> None:

        # Configure devices
        for device in self.devices:
            device.configure()

        self.__system.execute()

        # Fetch results from device
        for device in self.devices:
            device.fetch()
