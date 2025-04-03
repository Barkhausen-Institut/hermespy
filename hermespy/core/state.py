# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Sequence, TypeVar

import numpy as np

from .antennas import AntennaArrayState, AntennaMode, AntennaPort, IdealAntenna
from .transformation import Transformation

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class State(object):
    """Base of all state data containers.

    They describe the information available to different stages of the signal processing pipelines
    and are used to pass information to the corresponding signal processing algorithms.
    """

    __timestamp: float
    __pose: Transformation
    __velocity: np.ndarray
    __carrier_frequency: float
    __sampling_rate: float
    __antennas: AntennaArrayState

    def __init__(
        self,
        timestamp: float,
        pose: Transformation,
        velocity: np.ndarray,
        carrier_frequency: float,
        sampling_rate: float,
        antennas: AntennaArrayState,
    ) -> None:
        """
        Args:
            timestamp: Global timestamp of the device state.
            pose: Pose of the device with respect to the global coordinate system.
            velocity: Velocity of the device in m/s.
            carrier_frequency: Carrier frequency of the device in Hz.
            sampling_rate: Sampling rate of the device in Hz.
            antennas: State of the device's antenna array.
        """

        # Initialize class attributes
        self.__timestamp = timestamp
        self.__pose = pose
        self.__velocity = velocity
        self.__carrier_frequency = carrier_frequency
        self.__sampling_rate = sampling_rate
        self.__antennas = antennas

    @property
    def timestamp(self) -> float:
        """Global timestamp of the device state."""

        return self.__timestamp

    @property
    def pose(self) -> Transformation:
        """Global pose of the device."""

        return self.__pose

    @property
    def position(self) -> np.ndarray:
        """Global position of the device in cartesian coordinates.

        Shorthand to :attr:`translation<hermespy.core.transformation.Transformation.translation>` of :attr:`.pose`.
        """

        return self.pose.translation

    @property
    def velocity(self) -> np.ndarray:
        """Velocity of the device in m/s."""

        return self.__velocity

    @property
    def carrier_frequency(self) -> float:
        """Carrier frequency of the device in Hz."""

        return self.__carrier_frequency

    @property
    def sampling_rate(self) -> float:
        """Sampling rate of the device in Hz."""

        return self.__sampling_rate

    @property
    def antennas(self) -> AntennaArrayState:
        """State of the device's antenna array."""

        return self.__antennas


class TransmitState(State):
    """State of a during transmit signal processing.

    Generated by calling the :meth:`transmit_state<DeviceState.transmit_state>` method of a :class:`DeviceState`.
    """

    __num_digital_transmit_ports: int

    def __init__(
        self,
        timestamp: float,
        pose: Transformation,
        velocity: np.ndarray,
        carrier_frequency: float,
        sampling_rate: float,
        antennas: AntennaArrayState,
        num_digital_transmit_ports: int,
    ) -> None:
        """
        Args:

            timestamp:
                Global timestamp of the device state.

            pose:
                Pose of the device with respect to the global coordinate system.

            velocity:
                Velocity of the device in m/s.

            carrier_frequency:
                Carrier frequency of the device in Hz.

            sampling_rate:
                Sampling rate of the device in Hz.

            antennas:
                State of the device's antenna array.

            num_digital_transmit_ports:
                Number of digital transmit streams to be generated by the signal processing algorithm.
        """

        # Initialize base class
        State.__init__(self, timestamp, pose, velocity, carrier_frequency, sampling_rate, antennas)

        # Initialize class attributes
        self.__num_digital_transmit_ports = num_digital_transmit_ports

    @property
    def num_digital_transmit_ports(self) -> int:
        """Number of digital transmit streams to be generated by the signal processing algorithm."""

        return self.__num_digital_transmit_ports


class ReceiveState(State):
    """State of a device during receive signal processing.

    Generated by calling the :meth:`receive_state<DeviceState.receive_state>` method of a :class:`DeviceState`.
    """

    __num_digital_receive_ports: int

    def __init__(
        self,
        timestamp: float,
        pose: Transformation,
        velocity,
        carrier_frequency: float,
        sampling_rate: float,
        antennas: AntennaArrayState,
        num_digital_receive_ports: int,
    ) -> None:
        """
        Args:

            timestamp:
                Global timestamp of the device state.

            pose:
                Pose of the device with respect to the global coordinate system.

            velocity:
                Velocity of the device in m/s.

            carrier_frequency:
                Carrier frequency of the device in Hz.

            sampling_rate:
                Sampling rate of the device in Hz.

            antennas:
                State of the device's antenna array.

            num_digital_receive_ports:
                Number of digital receive streams to be consumed by the signal processing algorithm.
        """

        # Initialize base class
        State.__init__(self, timestamp, pose, velocity, carrier_frequency, sampling_rate, antennas)

        # Initialize class attributes
        self.__num_digital_receive_ports = num_digital_receive_ports

    @property
    def num_digital_receive_ports(self) -> int:
        """Number of digital receive streams to be consumed by the signal processing algorithm."""

        return self.__num_digital_receive_ports


class DeviceState(State):
    """Data container representing the immutable physical state of a device.

    Generated by calling the :meth:`Device.state<hermespy.core.device.Device.state>` method.
    In order to avoid confusion with the mutable state of the device, this class is immutable and
    only references the represented device by its memory address and time stamp.
    """

    __device_id: int
    __num_digital_transmit_ports: int
    __num_digital_receive_ports: int

    def __init__(
        self,
        device_id: int,
        timestamp: float,
        pose: Transformation,
        velocity: np.ndarray,
        carrier_frequency: float,
        sampling_rate: float,
        num_digital_transmit_ports: int,
        num_digital_receive_ports: int,
        antennas: AntennaArrayState,
    ) -> None:
        """
        Args:

            device_id:
                Unique identifier of the device this state represents.

            timestamp:
                Time stamp of the device state.

            pose:
                Pose of the device with respect to the global coordinate system.

            velocity:
                Velocity of the device in m/s.

            carrier_frequency:
                Carrier frequency of the device in Hz.

            sampling_rate:
                Sampling rate of the device in Hz.

            num_digital_transmit_ports:
                Number of digital transmit streams feeding into the device's signal encoding layer.
                This is the number of streams expected to be generated by transmit DSP algorithms by default.

            num_digital_receive_ports:
                Number of digital receive streams emerging from the device's signal decoding layer.
                This the number of streams expected to be feeding into receive DSP algorithms by default.

            antennas:
                State of the device's antenna array.
        """

        # Initialize base class
        State.__init__(self, timestamp, pose, velocity, carrier_frequency, sampling_rate, antennas)

        # Initialize class attributes
        self.__device_id = device_id
        self.__num_digital_receive_ports = num_digital_receive_ports
        self.__num_digital_transmit_ports = num_digital_transmit_ports

    @property
    def device_id(self) -> int:
        """Unique identifier of the device this state represents."""

        return self.__device_id

    @property
    def position(self) -> np.ndarray:
        """Global position of the device in cartesian coordinates.

        Shorthand to the translation property of the pose property.
        """

        return self.pose.translation

    @property
    def num_digital_transmit_ports(self) -> int:
        """Number of digital transmit streams feeding into the device's signal encoding layer.

        This is the number of streams expected to be generated by transmit DSP algorithms by default.
        """

        return self.__num_digital_transmit_ports

    @property
    def num_digital_receive_ports(self) -> int:
        """Number of digital receive streams emerging from the device's signal decoding layer.

        This the number of streams expected to be feeding into receive DSP algorithms by default.
        """

        return self.__num_digital_receive_ports

    def transmit_state(
        self, selected_digital_transmit_ports: Sequence[int] | None = None
    ) -> TransmitState:
        """Generate a device state for transmission.

        Args:

            selected_digital_transmit_ports:
                Indices of the selected digital transmit ports.
                If not specified, all available transmit ports will be considered.
        """

        _num_digital_transmit_ports = (
            self.num_digital_transmit_ports
            if selected_digital_transmit_ports is None
            else len(selected_digital_transmit_ports)
        )
        _selected_digital_transmit_ports = (
            list(range(self.antennas.num_transmit_ports))
            if selected_digital_transmit_ports is None
            else selected_digital_transmit_ports
        )
        return TransmitState(
            self.timestamp,
            self.pose,
            self.velocity,
            self.carrier_frequency,
            self.sampling_rate,
            self.antennas.select_subarray(_selected_digital_transmit_ports, AntennaMode.TX),
            _num_digital_transmit_ports,
        )

    def receive_state(
        self, selected_digital_receive_ports: Sequence[int] | None = None
    ) -> ReceiveState:
        """Generate a device state for reception.

        Args:
            selected_digital_receive_ports:
                Indices of the selected digital receive ports.
                If not specified, all available receive ports will be considered.
        """

        _num_digital_receive_ports = (
            self.num_digital_receive_ports
            if selected_digital_receive_ports is None
            else len(selected_digital_receive_ports)
        )
        _selected_digital_receive_ports = (
            list(range(self.antennas.num_receive_ports))
            if selected_digital_receive_ports is None
            else selected_digital_receive_ports
        )
        return ReceiveState(
            self.timestamp,
            self.pose,
            self.velocity,
            self.carrier_frequency,
            self.sampling_rate,
            self.antennas.select_subarray(_selected_digital_receive_ports, AntennaMode.RX),
            _num_digital_receive_ports,
        )

    @staticmethod
    def Basic(
        num_digital_transmit_ports: int = 1,
        num_digital_receive_ports: int = 1,
        carrier_frequency: float = 0.0,
        sampling_rate: float = 1.0,
        pose: Transformation | None = None,
    ) -> DeviceState:
        """Create a basic state initilized with default values.

        Useful shorthand for debugging and testing.

        Args:

            num_digital_transmit_ports:
                Number of digital transmit streams feeding into the device's signal encoding layer.
                This is the number of streams expected to be generated by transmit DSP algorithms by default.

            num_digital_receive_ports:
                Number of digital receive streams emerging from the device's signal decoding layer.
                This the number of streams expected to be feeding into receive DSP algorithms by default.

            carrier_frequency:
                Carrier frequency of the device in Hz.
                If not specified, base band is assumed.

            sampling_rate:
                Sampling rate of the device in Hz.
                If not specified, 1 Hz is assumed.

            pose:
                Pose of the device with respect to the global coordinate system.
                If not specified, the identity transformation is assumed, so the device is located in the origin.

        Returns: The initialized state.
        """

        _pose = Transformation.No() if pose is None else pose
        return DeviceState(
            0,  # ID
            0.0,  # Timestamp
            _pose,
            np.zeros(3),  # Velocity
            carrier_frequency,
            sampling_rate,
            num_digital_transmit_ports,
            num_digital_receive_ports,
            AntennaArrayState(
                [
                    AntennaPort([IdealAntenna()])
                    for _ in range(max(num_digital_receive_ports, num_digital_transmit_ports))
                ],
                _pose,
            ),
        )


DST = TypeVar("DST", bound=DeviceState)
"""Type variable for device states."""
