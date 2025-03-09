# -*- coding: utf-8 -*-
"""
==========
UHD Device
==========
"""

from __future__ import annotations
from collections.abc import Sequence
from datetime import datetime
from functools import cached_property
from typing import Any, List, Callable
from typing_extensions import override

import numpy as np
from zerorpc.exceptions import LostRemote, RemoteError
from usrp_client import UsrpClient, MimoSignal, TxStreamingConfig, RxStreamingConfig, RfConfig

from hermespy.core import (
    Antenna,
    AntennaArray,
    AntennaMode,
    AntennaPort,
    DeserializationProcess,
    IdealAntenna,
    Serializable,
    SerializationProcess,
    Signal,
    Transformation,
)
from ..physical_device import PhysicalDevice, PhysicalDeviceState

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class UsrpPort(AntennaPort):
    """Antenna port representation of a USRP device."""

    def __init__(self, array: UsrpAntennas, mode: AntennaMode) -> None:
        """
        Args:

            array (UsrpAntennas):
                Antenna array this port belongs to.

            mode (AntennaMode):
                Direction of the port.
        """

        # Init base class with a single ideal antenna
        AntennaPort.__init__(self, array=array, antennas=[IdealAntenna(mode)])


class UsrpAntennas(AntennaArray[AntennaPort, Antenna]):
    """Antenna and port configuration of a USRP device.

    :class:`UsrpAntennas`' port configuration is linked to the attached
    :class`UsrpDevice` and cannot be changed manually.
    """

    def __init__(self, device: UsrpDevice, pose: Transformation | None = None) -> None:
        """
        Args:

            device (UsrpDevice):
                USRP device this antenna array models.

            pose (Transformation, optional):
                Pose of the antenna array with respect to the `device`.
        """

        # Initialize base class
        AntennaArray.__init__(self, pose)

        # Initialize attributes
        self.__device = device
        self.__transmit_ports = [
            self._new_port(AntennaMode.TX) for _ in range(device.num_digital_transmit_ports)
        ]
        self.__receive_ports = [
            self._new_port(AntennaMode.RX) for _ in range(device.num_digital_receive_ports)
        ]

        # Configure kinematic chain
        self.set_base(device)

    @property
    def device(self) -> UsrpDevice:
        """USRP device the antenna array is attached to."""

        return self.__device

    @property
    def transmit_ports(self) -> Sequence[AntennaPort]:
        return self.__transmit_ports

    @property
    def receive_ports(self) -> Sequence[AntennaPort]:
        return self.__receive_ports

    @property
    def ports(self) -> Sequence[AntennaPort]:
        return self.__transmit_ports + self.__receive_ports

    @property
    def num_transmit_ports(self) -> int:
        return self.__device.num_digital_transmit_ports

    @property
    def num_receive_ports(self) -> int:
        return self.__device.num_digital_receive_ports

    @property
    def num_ports(self) -> int:
        return self.num_transmit_ports + self.num_receive_ports

    def _new_port(self, mode: AntennaMode = AntennaMode.DUPLEX) -> AntennaPort:
        return UsrpPort(self, mode)

    @property
    def num_transmit_antennas(self) -> int:
        return self.num_transmit_ports

    @property
    def num_receive_antennas(self) -> int:
        return self.num_receive_ports


class UsrpDevice(PhysicalDevice[PhysicalDeviceState], Serializable):
    """Bindung to a USRP device via the UHD library."""

    __usrp_client: UsrpClient
    __num_rpc_retries = 10
    __collection_enabled: bool
    __scale_transmission: bool
    __sampling_rate: float | None
    __num_prepended_zeros: int
    __num_appended_zeros: int
    __selected_transmit_ports: Sequence[int]
    __selected_receive_ports: Sequence[int]
    __max_selected_receive_port: int
    __max_selected_transmit_port: int
    __current_configuration: RfConfig
    __receive_delay: float  # Configured front-end delay during reception

    def __init__(
        self,
        ip: str,
        port: int = 5555,
        carrier_frequency: float = 7e8,
        sampling_rate: float | None = None,
        tx_gain: float = 0.0,
        rx_gain: float = 0.0,
        scale_transmission: bool = True,
        num_prepended_zeros: int = 200,
        num_appended_zeros: int = 200,
        selected_transmit_ports: Sequence[int] | None = None,
        selected_receive_ports: Sequence[int] | None = None,
        **kwargs,
    ) -> None:
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

            selected_transmit_ports (Sequence[int], optional):
                Indices of the selected transmit antenna ports.
                If not specified, i.e. :py:obj:`None`, only the first antenna port is selected.

            selected_receive_ports (Sequence[int], optional):
                Indices of the selected receive antenna ports.
                If not specified, i.e. :py:obj:`None`, only the first antenna port is selected.

            \**kwargs:
                Additional arguments passed to the :class:`.PhysicalDevice` parent class.
        """

        # Initialize base class
        PhysicalDevice.__init__(self, **kwargs)

        # Initialize attributes and configure RF frontend
        self.__usrp_client = UsrpClient.create(ip, port)

        # Query the available number of antennas ports
        max_num_ports = self.__usrp_client.getNumAntennas()

        # Configure transmit port selection
        if selected_transmit_ports is not None:
            self.__selected_transmit_ports = selected_transmit_ports
        elif max_num_ports > 0:
            self.__selected_transmit_ports = [0]
        else:
            self.__selected_transmit_ports = []
        self.__max_selected_transmit_port = max(self.__selected_transmit_ports, default=0)
        if 1 + self.__max_selected_transmit_port > max_num_ports:

            raise ValueError(
                f"Selected transmit ports exceed the maximum number of ports ({1 + self.__max_selected_transmit_port} > {max_num_ports})"
            )

        # Configure receive port selection
        if selected_receive_ports is not None:
            self.__selected_receive_ports = selected_receive_ports
        elif max_num_ports > 0:
            self.__selected_receive_ports = [0]
        else:
            self.__selected_receive_ports = []
        self.__max_selected_receive_port = max(self.__selected_receive_ports, default=0)

        if 1 + self.__max_selected_receive_port > max_num_ports:
            raise ValueError(
                f"Selected receive ports exceed the maximum number of ports ({1 + self.__max_selected_receive_port} > {max_num_ports})"
            )

        self.__antennas = UsrpAntennas(self)
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
        self.__receive_delay = 0.0

    def state(self) -> PhysicalDeviceState:
        return PhysicalDeviceState(
            id(self),
            datetime.now().timestamp(),
            self.pose,
            self.velocity,
            self.carrier_frequency,
            self.sampling_rate,
            self.num_digital_transmit_ports,
            self.num_digital_receive_ports,
            self.antennas.state(self.pose),
        )

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
                    self.selected_transmit_ports != self.__current_configuration.txAntennaMapping,
                    self.selected_receive_ports != self.__current_configuration.rxAntennaMapping,
                ]
            )
            or force
        ):
            config = RfConfig(
                txAnalogFilterBw=tx_filter_bandwidth,
                rxAnalogFilterBw=rx_filter_bandwidth,
                txSamplingRate=tx_sampling_rate,
                rxSamplingRate=rx_sampling_rate,
                txCarrierFrequency=tx_carrier_frequency,
                rxCarrierFrequency=rx_carrier_frequency,
                txGain=tx_gain,
                rxGain=rx_gain,
                noTxStreams=self.num_digital_transmit_ports,
                noRxStreams=self.num_digital_receive_ports,
                txAntennaMapping=self.selected_transmit_ports,
                rxAntennaMapping=self.selected_receive_ports,
            )

            self.__rpc_call_wrapper(self.__usrp_client.configureRfConfig, config)
            self.__current_configuration = config

    def _upload(self, baseband_signal: Signal) -> Signal:
        baseband_signal = baseband_signal.copy()

        # Configure device
        self._configure_device()

        # Reset the streaming config
        self.__rpc_call_wrapper(self.__usrp_client.resetStreamingConfigs)

        # Make a copy in order to avoid changing baseband_signal
        uploaded_samples = baseband_signal.copy()

        # Apply the antenna array calibration
        for block in uploaded_samples:
            self.antenna_calibration.correct_transmission(block)

        # Scale signal to a maximum absolute vlaue of zero to full exploit the DAC range
        if uploaded_samples.num_samples > 0 and self.scale_transmission:
            maxAmp = np.abs([b for b in uploaded_samples]).max()
            if maxAmp != 0:
                for block in uploaded_samples:
                    block /= maxAmp

        # Make a copy here
        corrected_uploaded_samples = uploaded_samples.copy()

        # Hack: Prepend some zeros to account for the premature transmission stop
        uploaded_samples.set_samples(
            np.concatenate(
                (
                    np.zeros(
                        (uploaded_samples.num_streams, self.num_prepeneded_zeros),
                        dtype=np.complex128,
                    ),
                    uploaded_samples.getitem(),
                    np.zeros(
                        (uploaded_samples.num_streams, self.num_appended_zeros), dtype=np.complex128
                    ),
                ),
                axis=1,
            )
        )

        if uploaded_samples.num_samples % 4 != 0:
            uploaded_samples.set_samples(
                np.append(
                    uploaded_samples.getitem(),
                    np.zeros(
                        (uploaded_samples.num_streams, 4 - uploaded_samples.num_samples % 4),
                        dtype=complex,
                    ),
                    axis=1,
                )
            )

        # Append a zero vector for unselected transmit ports
        # Workaround for the USRP wrapper missing dedicated port selections
        transmit_delay = max(0.0, -self.delay_calibration.delay)
        self.__receive_delay = max(0.0, self.delay_calibration.delay)
        uploaded_samples.delay = transmit_delay
        tx_config = TxStreamingConfig(transmit_delay, MimoSignal(uploaded_samples.getitem()))
        self.__rpc_call_wrapper(self.__usrp_client.configureTx, tx_config)

        # Configure reception
        duration = max(self.receivers.min_frame_duration, uploaded_samples.duration)

        if duration > 0.0:
            num_receive_samples = int((duration + self.max_receive_delay) * self.sampling_rate)

            # Hack
            num_receive_samples += self.num_prepeneded_zeros + self.num_appended_zeros

            # Workaround for the uneven sample bug
            num_receive_samples += 4 - num_receive_samples % 4

            rx_config = RxStreamingConfig(self.__receive_delay, num_receive_samples)
            self.__rpc_call_wrapper(self.__usrp_client.configureRx, rx_config)
            self.__collection_enabled = True

        else:
            num_receive_samples = (
                uploaded_samples.num_samples + 4 - uploaded_samples.num_samples % 4
            )

            rx_config = RxStreamingConfig(self.__receive_delay, num_receive_samples)
            self.__rpc_call_wrapper(self.__usrp_client.configureRx, rx_config)
            self.__collection_enabled = True

        return corrected_uploaded_samples

    def trigger(self) -> None:
        # Queue execution command
        self.__usrp_client.executeImmediately()

    def _download(self) -> Signal:
        # Abort if no samples are to be expcted during collection
        if not self.__collection_enabled:
            return Signal.Empty(self.sampling_rate, self.num_receive_antenna_ports)

        mimo_signals = self.__usrp_client.collect()
        signal_model = Signal.Empty(
            self.sampling_rate,
            self.num_receive_antenna_ports,
            carrier_frequency=self.carrier_frequency,
            delay=self.__receive_delay,
        )

        for mimo_signal in mimo_signals:
            streams = np.array(
                [mimo_signal.signals[i] for i in range(self.num_receive_antenna_ports)]
            )
            signal_model.append_samples(streams)

        # Remove the zero padding hack
        signal_model.set_samples(
            signal_model.getitem(
                (
                    slice(None, None),
                    slice(
                        self.num_prepeneded_zeros,
                        signal_model.num_samples - self.num_appended_zeros,
                    ),
                )
            )
        )

        # Apply the antenna array calibration
        for block in signal_model:
            self.antenna_calibration.correct_reception(block)

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
    def num_transmit_antenna_ports(self) -> int:
        """Number of transmit ports controlled on the USRP device."""

        return len(self.__selected_transmit_ports)

    @property
    def num_receive_antenna_ports(self) -> int:
        """Number of receive ports controlled on the USRP device."""

        return len(self.__selected_receive_ports)

    @property
    def selected_transmit_ports(self) -> Sequence[int]:
        """Indices of the selected transmit ports."""

        return self.__selected_transmit_ports

    @property
    def selected_receive_ports(self) -> Sequence[int]:
        """Indices of the selected receive ports."""

        return self.__selected_receive_ports

    @property
    def antennas(self) -> UsrpAntennas:
        """Antenna array model of the USRP device.

        Allows for the further configuration of the antenna elements,
        however, the number of :attr:`transmit ports<num_transmit_ports>`
        and :attr:`receive ports<num_receive_ports>` is fixed.
        """

        return self.__antennas

    @property
    def sampling_rate(self) -> float:
        if self.__sampling_rate is not None:
            return self.__sampling_rate

        ideal_sampling_rate = (
            self.transmitters.max_sampling_rate
            if self.transmitters.num_operators > 0
            else self.receivers.max_sampling_rate
        )
        selected_sampling_rate = min(
            self.__supported_sampling_rates, key=lambda x: abs(x - ideal_sampling_rate)
        )

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

    @override
    def serialize(self, process: SerializationProcess) -> None:
        PhysicalDevice.serialize(self, process)
        process.serialize_string(self.ip, "ip")
        process.serialize_integer(self.port, "port")
        process.serialize_floating(self.carrier_frequency, "carrier_frequency")
        process.serialize_floating(self.sampling_rate, "sampling_rate")
        process.serialize_floating(self.tx_gain, "tx_gain")
        process.serialize_floating(self.rx_gain, "rx_gain")
        process.serialize_integer(self.scale_transmission, "scale_transmission")
        process.serialize_integer(self.num_prepeneded_zeros, "num_prepeneded_zeros")
        process.serialize_integer(self.num_appended_zeros, "num_appended_zeros")
        process.serialize_array(np.asarray(self.selected_transmit_ports), "selected_transmit_ports")
        process.serialize_array(np.asarray(self.selected_receive_ports), "selected_receive_ports")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> UsrpDevice:
        return cls(
            process.deserialize_string("ip"),
            process.deserialize_integer("port"),
            process.deserialize_floating("carrier_frequency"),
            process.deserialize_floating("sampling_rate"),
            process.deserialize_floating("tx_gain"),
            process.deserialize_floating("rx_gain"),
            bool(process.deserialize_integer("scale_transmission")),
            process.deserialize_integer("num_prepeneded_zeros"),
            process.deserialize_integer("num_appended_zeros"),
            process.deserialize_array("selected_transmit_ports", np.int64).tolist(),
            process.deserialize_array("selected_receive_ports", np.int64).tolist(),
            **cls._DeserializeParameters(process),  # type: ignore[arg-type]
        )
