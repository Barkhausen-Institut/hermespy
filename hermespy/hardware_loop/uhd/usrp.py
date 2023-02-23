# -*- coding: utf-8 -*-
"""
==========
UHD Device
==========
"""

from functools import cached_property
from typing import Any, List, Callable

import numpy as np
from zerorpc.exceptions import LostRemote, RemoteError
from usrp_client import UsrpClient, MimoSignal, TxStreamingConfig, RxStreamingConfig, RfConfig

from hermespy.core import Serializable, Signal
from ..physical_device import PhysicalDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class UsrpDevice(PhysicalDevice, Serializable):

    yaml_tag = "USRP"
    """YAML serialization tag"""

    property_blacklist = {"topology", "wavelength", "velocity"}

    __usrp_client: UsrpClient
    __num_rpc_retries = 10
    __collection_enabled: bool

    def __init__(self, ip: str, port: int = 5555, carrier_frequency: float = 7e8, tx_gain: float = 0.0, rx_gain: float = 0.0, *args, **kwargs) -> None:

        self.__usrp_client = UsrpClient.create(ip, port)

        PhysicalDevice.__init__(self, *args, **kwargs)

        self.carrier_frequency = carrier_frequency
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain
        self.__current_configuration = self.__rpc_call_wrapper(self.__usrp_client.getRfConfig)
        self._configure_device()
        self.__collection_enabled = False

    def __rpc_call_wrapper(self, call: Callable, *args, **kwargs) -> Any:
        """Wrapper to RPC client calls to perform multiple calls to hack the timeout bug.

        Returns: The call return.

        Raises:

            RuntimeError: If the call didn't succeed within the configured amount of attempts.
        """

        for _ in range(self.__num_rpc_retries):

            try:
                return call(*args, **kwargs)

            except LostRemote:
                continue

            except RemoteError as e:
                raise RuntimeError(f"Remote exception occured at '{self.ip}:{self.port}': {e.msg}")

        raise RuntimeError(f"Lost connection to  the remote '{self.ip}:{self.port}'")

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
        if any(
            [
                tx_filter_bandwidth != self.__current_configuration.txAnalogFilterBw,
                rx_filter_bandwidth != self.__current_configuration.rxAnalogFilterBw,
                tx_sampling_rate != self.__current_configuration.txSamplingRate,
                rx_sampling_rate != self.__current_configuration.rxSamplingRate,
                tx_carrier_frequency != self.__current_configuration.txCarrierFrequency,
                rx_carrier_frequency != self.__current_configuration.rxCarrierFrequency,
                tx_gain != self.__current_configuration.txGain,
                rx_gain != self.__current_configuration.rxGain,
            ]
        ):

            config = RfConfig(txAnalogFilterBw=tx_filter_bandwidth, rxAnalogFilterBw=rx_filter_bandwidth, txSamplingRate=tx_sampling_rate, rxSamplingRate=rx_sampling_rate, txCarrierFrequency=tx_carrier_frequency, rxCarrierFrequency=rx_carrier_frequency, txGain=tx_gain, rxGain=rx_gain)

            self.__rpc_call_wrapper(self.__usrp_client.configureRfConfig, config)
            self.__current_configuration = config

    def _upload(self, baseband_signal: Signal) -> None:

        # Configure device
        self._configure_device()

        # Reset the streaming config
        self.__rpc_call_wrapper(self.__usrp_client.resetStreamingConfigs)

        # Scale signal to a maximum absolute vlaue of zero to full exploit the DAC range
        if baseband_signal.num_samples > 0:
            maxAmp = np.abs(baseband_signal.samples).max()
            if maxAmp != 0:
                baseband_signal.samples /= maxAmp

        # Hack: Append some zeros to account for the premature transmission stop
        hack_num_samples = 200
        baseband_signal.samples = np.concatenate((np.zeros((baseband_signal.num_streams, hack_num_samples), dtype=complex), baseband_signal.samples, np.zeros((baseband_signal.num_streams, hack_num_samples), dtype=complex)), axis=1)

        if baseband_signal.num_samples % 2 != 0:
            baseband_signal.samples = np.append(baseband_signal.samples, np.zeros((baseband_signal.num_streams, 1), dtype=complex), axis=1)

        mimo_signal = MimoSignal(list(baseband_signal.samples))
        tx_config = TxStreamingConfig(max(0.0, -self.calibration_delay), mimo_signal)
        self.__rpc_call_wrapper(self.__usrp_client.configureTx, tx_config)

        # Configure reception
        duration = self.receivers.min_frame_duration

        if duration >= 0.0:

            num_receive_samples = int((duration + self.max_receive_delay) * self.sampling_rate)
            # Workaround for the uneven sample bug
            num_receive_samples += num_receive_samples % 2

            rx_config = RxStreamingConfig(max(0.0, self.calibration_delay), num_receive_samples)
            self.__rpc_call_wrapper(self.__usrp_client.configureRx, rx_config)

            self.__collection_enabled = True

        else:
            self.__collection_enabled = False

    def trigger(self) -> None:
        # Queue execution command
        self.__usrp_client.executeImmediately()

    def _download(self) -> Signal:

        # Abort if no samples are to be expcted during collection
        if not self.__collection_enabled:
            return Signal.empty(self.sampling_rate, self.antennas.num_antennas)

        mimo_signals = self.__usrp_client.collect()
        signal_model = Signal.empty(self.sampling_rate, self.antennas.num_antennas, carrier_frequency=self.carrier_frequency)

        for mimo_signal in mimo_signals:

            streams = np.array(mimo_signal.signals)
            signal_model.samples = np.append(signal_model.samples, streams, axis=1)

        return signal_model

    @property
    def _client(self) -> UsrpClient:
        """Access to the UHD client.

        Returns: Handle to the client.
        """

        return self.__usrp_client

    @property
    def ip(self) -> str:
        """Internet protocol address of the remote host.

        Returns:

            IP adress.
        """

        return self.__usrp_client.ip

    @property
    def port(self) -> int:
        """Internet protocol port of the remote host.

        Returns:

            Port.
        """

        return self.__usrp_client.port

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

        ideal_sampling_rate = self.transmitters.max_sampling_rate if self.transmitters.num_operators > 0 else self.receivers.max_sampling_rate
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

        return self.__usrp_client.getSupportedSamplingRates()
