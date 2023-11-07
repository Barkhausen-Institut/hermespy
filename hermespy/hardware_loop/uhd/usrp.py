# -*- coding: utf-8 -*-
"""
==========
UHD Device
==========
"""

from __future__ import annotations
from copy import deepcopy
from functools import cached_property
from typing import Any, List, Callable

import numpy as np
from zerorpc.exceptions import LostRemote, RemoteError
from usrp_client import UsrpClient, MimoSignal, TxStreamingConfig, RxStreamingConfig, RfConfig

from hermespy.core import AntennaArrayBase, AntennaArray, AntennaMode, Device, IdealAntenna, Serializable, Signal
from ..physical_device import PhysicalDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class UsrpDevice(PhysicalDevice, Serializable):
    """Bindung to a USRP device via the UHD library."""

    yaml_tag = "USRP"
    """YAML serialization tag"""

    property_blacklist = {"topology", "wavelength", "velocity"}

    __usrp_client: UsrpClient
    __num_rpc_retries = 10
    __collection_enabled: bool
    __scale_transmission: bool
    __sampling_rate: float | None
    __num_prepended_zeros: int
    __num_appended_zeros: int
    __num_transmit_antennas: int
    __num_receive_antennas: int

    def __init__(self, ip: str, port: int = 5555, carrier_frequency: float = 7e8, sampling_rate: float | None = None, tx_gain: float = 0.0, rx_gain: float = 0.0, scale_transmission: bool = True, num_prepended_zeros: int = 200, num_appended_zeros: int = 200, num_transmit_antennas: int = 1, num_receive_antennas: int = 1, antennas: AntennaArrayBase | None = None, *args, **kwargs) -> None:
        """
        Args:

            ip (str):
                The IP address of the USRP device.

            port (int, optional):
                The port of the USRP device.

            carrier_frequency (float, optional):
                Carrier frequency of the USRP device.
                :math:`700~\\mathrm{MHz}` by default.

            sampling_rate (float, optional):
                Sampling rate of the USRP device.
                If not provided, the sampling rate is determined from the configured operators.

            tx_gain (float, optional):
                The transmission gain of the USRP device.
                Zero by default.

            rx_gain (float, optional):
                The reception gain of the USRP device.
                Zero by default.

            scale_transmission (bool, optional):
                If `True`, the transmission signal is scaled to the maximum floating point value of the USRP device.
                This ensures a proper digital to analog conversion.

            num_prepended_zeros (int, optional):
                The number of zeros prepended to the transmission signal.
                :math:`200` by default.

            num_appended_zeros (int, optional):
                The number of zeros appended to the transmission signal.
                :math:`200` by default.

            num_transmit_antennas (int, optional):
                Number of transmit antennas.
                :math:`1` by default.

            num_receive_antennas (int, optional):
                Number of receive antennas.
                :math:`1` by default.

            antennas (AntennaArrayBase, optional):
                Antenna array topology of the USRP device.
                If not provided, an antenna array with ideal antennas matching the number of transmit and receive antennas is created.

            *args, **kwargs:
                Additional arguments passed to the :class:`.PhysicalDevice` parent class.
        """

        self.__usrp_client = UsrpClient.create(ip, port)
        self.__num_transmit_antennas = num_transmit_antennas
        self.__num_receive_antennas = num_receive_antennas

        # Infer antenna array topology from USRP device
        _antennas: AntennaArrayBase
        if antennas is None:
            _antennas = AntennaArray([IdealAntenna(AntennaMode.TX) for _ in range(self.num_transmit_antennas)] + [IdealAntenna(AntennaMode.RX) for _ in range(self.num_receive_antennas)])
        else:
            _antennas = antennas

        PhysicalDevice.__init__(self, *args, antennas=_antennas, **kwargs)

        self.carrier_frequency = carrier_frequency
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain
        self.sampling_rate = sampling_rate
        self.num_prepeneded_zeros = num_prepended_zeros
        self.num_appended_zeros = num_appended_zeros
        self.__current_configuration = self.__rpc_call_wrapper(self.__usrp_client.getRfConfig)
        self._configure_device(force=True)
        self.__collection_enabled = False
        self.__scale_transmission = scale_transmission

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

    def _configure_device(self, force: bool = False) -> None:
        tx_filter_bandwidth = 4e8
        rx_filter_bandwidth = 4e8
        tx_sampling_rate = self.sampling_rate
        rx_sampling_rate = self.sampling_rate
        tx_carrier_frequency = self.carrier_frequency
        rx_carrier_frequency = self.carrier_frequency
        tx_gain = self.tx_gain
        rx_gain = self.rx_gain

        # Check if a frontend reconfiguration is required
        if (
            any(
                [
                    tx_filter_bandwidth != self.__current_configuration.txAnalogFilterBw,
                    rx_filter_bandwidth != self.__current_configuration.rxAnalogFilterBw,
                    tx_sampling_rate != self.__current_configuration.txSamplingRate,
                    rx_sampling_rate != self.__current_configuration.rxSamplingRate,
                    tx_carrier_frequency != self.__current_configuration.txCarrierFrequency,
                    rx_carrier_frequency != self.__current_configuration.rxCarrierFrequency,
                    tx_gain != self.__current_configuration.txGain,
                    rx_gain != self.__current_configuration.rxGain,
                    self.num_transmit_antennas != self.__current_configuration.noTxAntennas,
                    self.num_receive_antennas != self.__current_configuration.noRxAntennas,
                ]
            )
            or force
        ):
            config = RfConfig(txAnalogFilterBw=tx_filter_bandwidth, rxAnalogFilterBw=rx_filter_bandwidth, txSamplingRate=tx_sampling_rate, rxSamplingRate=rx_sampling_rate, txCarrierFrequency=tx_carrier_frequency, rxCarrierFrequency=rx_carrier_frequency, txGain=tx_gain, rxGain=rx_gain, noTxAntennas=self.num_transmit_antennas, noRxAntennas=self.num_receive_antennas)

            self.__rpc_call_wrapper(self.__usrp_client.configureRfConfig, config)
            self.__current_configuration = config

    def _upload(self, baseband_signal: Signal) -> None:
        baseband_signal = baseband_signal.copy()

        # Configure device
        self._configure_device()

        # Reset the streaming config
        self.__rpc_call_wrapper(self.__usrp_client.resetStreamingConfigs)

        # Scale signal to a maximum absolute vlaue of zero to full exploit the DAC range
        if baseband_signal.num_samples > 0 and self.scale_transmission:
            maxAmp = np.abs(baseband_signal.samples).max()
            if maxAmp != 0:
                baseband_signal.samples /= maxAmp

        # Hack: Prepend some zeros to account for the premature transmission stop
        baseband_signal.samples = np.concatenate((np.zeros((baseband_signal.num_streams, self.num_prepeneded_zeros), dtype=np.complex_), baseband_signal.samples, np.zeros((baseband_signal.num_streams, self.num_appended_zeros), dtype=np.complex_)), axis=1)

        if baseband_signal.num_samples % 4 != 0:
            baseband_signal.samples = np.append(baseband_signal.samples, np.zeros((baseband_signal.num_streams, 4 - baseband_signal.num_samples % 4), dtype=complex), axis=1)

        mimo_signal = MimoSignal(list(baseband_signal.samples))
        tx_config = TxStreamingConfig(max(0.0, -self.delay_calibration.delay), mimo_signal)
        self.__rpc_call_wrapper(self.__usrp_client.configureTx, tx_config)

        # Configure reception
        duration = self.receivers.min_frame_duration

        if duration > 0.0:
            num_receive_samples = int((duration + self.max_receive_delay) * self.sampling_rate)

            # Hack
            num_receive_samples += self.num_prepeneded_zeros + self.num_appended_zeros

            # Workaround for the uneven sample bug
            num_receive_samples += 4 - num_receive_samples % 4

            rx_config = RxStreamingConfig(max(0.0, self.delay_calibration.delay), num_receive_samples)
            self.__rpc_call_wrapper(self.__usrp_client.configureRx, rx_config)

            self.__collection_enabled = True

        else:
            num_receive_samples = baseband_signal.num_samples + 4 - baseband_signal.num_samples % 4

            rx_config = RxStreamingConfig(max(0.0, self.delay_calibration.delay), num_receive_samples)
            self.__rpc_call_wrapper(self.__usrp_client.configureRx, rx_config)

            self.__collection_enabled = True

    def trigger(self) -> None:
        # Queue execution command
        self.__usrp_client.executeImmediately()

    def _download(self) -> Signal:
        # Abort if no samples are to be expcted during collection
        if not self.__collection_enabled:
            return Signal.empty(self.sampling_rate, self.antennas.num_receive_antennas)

        mimo_signals = self.__usrp_client.collect()
        signal_model = Signal.empty(self.sampling_rate, self.num_receive_antennas, carrier_frequency=self.carrier_frequency)

        for mimo_signal in mimo_signals:
            streams = np.array(mimo_signal.signals)
            signal_model.samples = np.append(signal_model.samples, streams, axis=1)

        # Remove the zero padding hack
        signal_model.samples = signal_model.samples[:, self.num_prepeneded_zeros : signal_model.num_samples - self.num_appended_zeros]

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
        """Gain of the transmitting front-end in dB."""

        return self.__tx_gain

    @tx_gain.setter
    def tx_gain(self, value: float) -> None:
        self.__tx_gain = value

    @property
    def rx_gain(self) -> float:
        """Gain of the receiving front-end in dB."""

        return self.__rx_gain

    @rx_gain.setter
    def rx_gain(self, value: float) -> None:
        self.__rx_gain = value

    @property
    def num_transmit_antennas(self) -> int:
        """Number of transmit antennas."""

        return self.__num_transmit_antennas

    @property
    def num_receive_antennas(self) -> int:
        """Number of receive antennas."""

        return self.__num_receive_antennas

    @Device.antennas.setter  # type: ignore
    def antennas(self, value: AntennaArrayBase) -> None:
        if value.num_transmit_antennas != self.num_transmit_antennas:
            raise ValueError(f"Number of antenna array's transmit antennas must match the number of USRP's transmit antennas ({value.num_transmit_antennas} != {self.num_transmit_antennas})")

        if value.num_receive_antennas != self.num_receive_antennas:
            raise ValueError(f"Number of antenna array's receive antennas must match the number of USRP's receive antennas ({value.num_receive_antennas} != {self.num_receive_antennas})")

        Device.antennas.fset(self, deepcopy(value))  # type: ignore

    @property
    def sampling_rate(self) -> float:
        if self.__sampling_rate is not None:
            return self.__sampling_rate

        ideal_sampling_rate = self.transmitters.max_sampling_rate if self.transmitters.num_operators > 0 else self.receivers.max_sampling_rate
        selected_sampling_rate = min(self.__supported_sampling_rates, key=lambda x: abs(x - ideal_sampling_rate))

        return selected_sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:
        self.__sampling_rate = value

    @property
    def max_sampling_rate(self) -> float:
        return max(self.__supported_sampling_rates)

    @property
    def carrier_frequency(self) -> float:
        return self.__carrier_frequency

    @carrier_frequency.setter
    def carrier_frequency(self, value: float) -> None:
        self.__carrier_frequency = value

    @property
    def num_prepeneded_zeros(self) -> int:
        """Number of zero padding samples prepended to the transmitted signal.

        Returns: Number of prepended samples.

        Raises:
            ValueError: For negative values.
        """

        return self.__num_prepended_zeros

    @num_prepeneded_zeros.setter
    def num_prepeneded_zeros(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"Number of prepended zeros must be non-negative ({value} < 0)")

        self.__num_prepended_zeros = value

    @property
    def num_appended_zeros(self) -> int:
        """Number of zero padding samples appended to the transmitted signal.

        Returns: Number of appended samples.

        Raises:
            ValueError: For negative values.
        """

        return self.__num_appended_zeros

    @num_appended_zeros.setter
    def num_appended_zeros(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"Number of appended zeros must be non-negative ({value} < 0)")

        self.__num_appended_zeros = value

    @cached_property
    def __supported_sampling_rates(self) -> List[float]:
        return self.__usrp_client.getSupportedSamplingRates()

    @property
    def scale_transmission(self) -> bool:
        """Indicates whether the transmission is scaled to the full DAC range.

        Returns: Boolean enabled flag.
        """

        return self.__scale_transmission

    @scale_transmission.setter
    def scale_transmission(self, value: bool) -> None:
        self.__scale_transmission = value
