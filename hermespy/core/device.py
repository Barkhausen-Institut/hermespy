# -*- coding: utf-8 -*-
"""


"""

from __future__ import annotations
from abc import ABC, abstractmethod
from itertools import chain
from typing import Generic, List, Optional, TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:

    from scenario import Scenario

from hermespy.signal import Signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class FloatingError(RuntimeError):
    """Exception raised if an operation fails due to a currently being considered floating."""
    pass


SlotType = TypeVar('SlotType', bound='OperatorSlot')
"""Type of slot."""

OperatorType = TypeVar('OperatorType', bound='Operator')
"""Type of operator."""


class Operator(Generic[SlotType]):
    """Base class for operators of devices.

    In HermesPy, operators may configure devices, broadcast signals over them or capture signals from them.
    Each operator is attached to a single device instance it operates on.
    """

    __slots__ = ['__slot']
    __slot: Optional[SlotType]          # Slot within a device this operator 'operates'

    def __init__(self,
                 slot: Optional[SlotType] = None) -> None:
        """
        Args:
            slot (OperatorSlot, optional):
                Device slot this operator is attached to.
                By default, the operator is not attached and considered floating.
        """

        self.slot = slot

    @property
    def slot(self) -> Optional[OperatorSlot]:
        """Device slot this operator operates.

        Returns:
            slot (Optional[OperatorSlot]):
                Handle to the device slot.
                `None` if the operator is currently considered floating.
        """

        return self.__slot

    @slot.setter
    def slot(self, value: Optional[OperatorSlot]) -> None:
        """Set device slot this operator operates on."""

        # A None argument indicates the slot should be unbound
        if value is None:

            if hasattr(self, '_Operator__slot') and self.__slot is not None:

                slot = self.__slot      # This is necessary to prevent event-loops. Just ignore it.
                self.__slot = None
                if slot.registered(self):
                    slot.remove(self)

            else:
                self.__slot = None

        elif not hasattr(self, '_Operator__slot'):

            self.__slot = value

            if not self.__slot.registered(self):
                self.__slot.add(self)

        elif self.__slot is not value:

            if self.__slot is not None and self.__slot.registered(self):
                self.__slot.remove(self)

            self.__slot = value

            if not self.__slot.registered(self):
                self.__slot.add(self)

    @property
    def slot_index(self) -> Optional[int]:
        """Index of the operator within its slot.

        Returns:
            index (Optional[int]):
                Index of the operator.
                `None` if the operator is currently considered floating.
        """

        if self.__slot is None:
            return None

        return self.__slot.operator_index(self)

    @property
    def device(self) -> Optional[Device]:
        """Device this operator is attached to.

        Returns:
            device (Optional[Device]):
                Handle to the operated device.
                `None` if the operator is currently considered floating.
        """

        if self.__slot is None:
            return None

        return self.__slot.device

    @property
    def attached(self) -> bool:
        """Attachment state of the operator.

        Returns:
            bool: Boolean attachment indicator
        """

        return self.__slot is not None

    @property
    @abstractmethod
    def frame_duration(self) -> float:
        """Duration of a single sample frame.

        Returns:
            duration (float): Frame duration in seconds.
        """
        ...


class Receiver(Operator):
    """Operator receiving from a device."""

    def __init__(self,
                 *args,
                 **kwargs) -> None:
        """
        Args:

            *args:
                Operator base class initialization parameters.

            **kwargs:
                Operator base class initialization parameters.
        """

        # Initialize operator base class
        Operator.__init__(*args, **kwargs)


class OperatorSlot(Generic[OperatorType]):
    """Slot list for operators of a single device."""

    __slots__ = ['device', '__operators']

    device: Device
    """Device this operator belongs to."""

    __operators: List[OperatorType]     # List of operators registered at this slot

    def __init__(self,
                 device: Device) -> None:
        """
        Args:
            device (Device):
                Device this slot belongs to.
        """

        self.device = device
        self.__operators = []

    def operator_index(self, operator: OperatorType) -> int:
        """Index of an operator within this slot.

        Returns:
            index (int): The `operator`'s index.

        Raises:
            ValueError: If the `operator` is not registered at this slot.
        """

        return self.__operators.index(operator)

    def add(self,
            operator: OperatorType) -> None:
        """Add a new operator to this slot.

        Args:
            operator (OperatorType):
                Operator to be added to this slot.

        Raises:
            RuntimeError: If the `operator` is already registered at this slot.
        """

        if operator in self.__operators:
            raise RuntimeError("Error trying to re-register operator at slot")

        self.__operators.append(operator)
        operator.slot = self

    def remove(self, operator: OperatorType) -> None:
        """Remove an operator from this slot.

        Args:
            operator (OperatorType):
                Handle to the operator to be removed.

        Raises:
            ValueError: If the `operator` is not registered at this slot.
        """

        self.__operators.remove(operator)
        operator.slot = None

    def registered(self, operator: OperatorType) -> bool:
        """Check if an operator is registered at this slot.

        Returns:
            bool: Boolean indicating the registration status.
        """

        return operator in self.__operators

    @property
    def num_operators(self) -> int:
        """Number of operators on this slot.

        Returns:
            int: Number of oeprators.
        """

        return len(self.__operators)

    def __getitem__(self, item):
        return self.__operators[item]

    def __iter__(self):
        """Iterating over operator slots returns the iterator over the slots."""

        return self.__operators.__iter__()

    def __contains__(self, operator: Operator) -> bool:
        """Contains is just a convenient mask for registered.

        Returns:
            bool: Registration indicator.
        """

        return self.registered(operator)


class Transmitter(Operator[SlotType]):
    """Operator transmitting over a device."""

    def __init__(self, *args, **kwargs) -> None:
        """
        Args:

            *args:
                Operator base class initialization parameters.

            **kwargs:
                Operator base class initialization parameters.
        """

        # Initialize operator base class
        Operator[TransmitterSlot].__init__(self, *args, **kwargs)

    def transmit(self, signal: Signal) -> None:
        """Transmit a signal.

        Registers the signal samples to be transmitted by the underlying device.

        Args:


        Raises:
            FloatingError:
                If the transmitter is currently considered floating.
        """

        if not self.attached:
            raise FloatingError("Error trying to transmit via a floating transmitter")

        self.slot.transmit(self, signal)


class TransmitterSlot(OperatorSlot):
    """Slot for transmitting operators within devices."""

    __slots__ = ['__signals']
    __signals: List[Optional[Signal]]       # Next signals to be transmitted

    def __init__(self, *args, **kwargs) -> None:
        """
        Args:
            *args:
                OperatorSlot base class initialization parameters.

            **kwargs:
                OperatorSlot base class initialization parameters.
        """

        self.__signals = []

        # Init base class
        OperatorSlot.__init__(self, *args, **kwargs)

    def add_transmission(self,
                         transmitter: Transmitter,
                         signal: Signal) -> None:
        """Add a transmitter's most recent transmission.

        Args:

            transmitter(Transmitter):
                The transmitter emitting the signal.

            signal (Signal):
                The signal to be transmitted.

        Raises:
            ValueError: If the transmitter is not registered at this slot.
        """

        if not self.registered(transmitter):
            raise ValueError("Transmitter not registered at this slot")

        self.__signals[transmitter.slot_index] = signal


class ReceiverSlot(OperatorSlot[Receiver]):
    """Slot for receiving operators within devices."""

    __slots__ = ['__signals']
    __signals: List[Optional[Signal]]  # Recently received signals

    def __init__(self, *args, **kwargs) -> None:
        """
        Args:
            *args:
                OperatorSlot base class initialization parameters.

            **kwargs:
                OperatorSlot base class initialization parameters.
        """

        self.__signals = []

        # Init base class
        OperatorSlot.__init__(self, *args, **kwargs)

    def add(self,
            operator: Receiver) -> None:

        self.__signals.append(None)
        OperatorSlot[Receiver].add(self, operator)


class UnsupportedSlot(OperatorSlot):
    """Slot for unsupported operations within devices."""

    def add(self,
            operator: Operator) -> None:

        raise RuntimeError("Slot not supported by this device")


class Device(ABC):
    """Physical device representation within HermesPy.

    It acts as the basis for all transmissions and receptions of sampled electromagnetic signals.
    """

    __slots__ = ['transmitters', 'receivers', '__position', '__orientation', '__topology']

    transmitters: TransmitterSlot
    """"Transmitters broadcasting signals over this device."""

    receivers: ReceiverSlot
    """Receivers capturing signals from this device"""

    __position: Optional[np.ndarray]        # Position of the device within its scenario in cartesian coordinates
    __orientation: Optional[np.ndarray]     # Orientation of the device within its scenario as a quaternion
    __topology: np.ndarray                  # Antenna array topology of the device

    def __init__(self,
                 position: Optional[np.array] = None,
                 orientation: Optional[np.array] = None,
                 topology: Optional[np.ndarray] = None) -> None:
        """
        Args:

            position (np.ndarray, optional):
                Position of the device within its scenario in cartesian coordinates.
                By default, the device position is considered to be unknown (`None`).

            orientation (np.ndarray, optional):
                Orientation of the device within its scenario as a quaternion.
                By default, the device orientation is considered to be unknown (`None`).

            topology (np.ndarray, optional):
                Antenna array topology of the device.
                By default, a single ideal omnidirectional antenna is assumed.
        """

        self.transmitters = TransmitterSlot(self)
        self.receivers = ReceiverSlot(self)
        self.position = position
        self.orientation = orientation
        self.topology = topology

    @property
    def position(self) -> Optional[np.ndarray]:
        """Position of the device within its scenario in cartesian coordinates.

        Returns:
            position (Optional[np.array]):
                The device's position.
                `None` if the position is considered to be unknown.

        Raises:
            ValueError: If the `position` is not a valid cartesian vector.
        """

        return self.__position

    @position.setter
    def position(self, value: Optional[np.ndarray]) -> None:
        """Set position of the device within its scenario in cartesian coordinates."""

        if value is None:
            self.__position = None
            return

        if value.ndim != 1:
            raise ValueError("Modem position must be a vector")

        if len(value) > 3:
            raise ValueError("Modem position may have at most three dimensions")

        # Make vector 3D if less dimensions have been supplied
        if len(value) < 3:
            value = np.append(value, np.zeros(3 - len(value), dtype=float))

        self.__position = value

    @property
    def orientation(self) -> Optional[np.ndarray]:
        """Orientation of the device within its scenario as a quaternion.

        Returns:
            orientation (Optional[np.array]):
                The modem orientation as a normalized quaternion.
                `None` if the position is considered to be unknown.

        Raises:
            ValueError: If `orientation` is not a valid quaternion.
        """

        return self.__orientation

    @orientation.setter
    def orientation(self, value: Optional[np.ndarray]) -> None:
        """Set the orientation of the device within its scenario as a quaternion."""

        if value is None:
            self.__orientation = None
            return

        if value.ndim != 1 or len(value) != 3:
            raise ValueError("Modem orientation must be a three-dimensional vector")

        self.__orientation = value

    @property
    def topology(self) -> np.ndarray:
        """Antenna array topology.

        The topology is represented by an `Mx3` numpy array,
        where the first dimension M is the number of antennas and the second dimension their cartesian position
        within the device's local coordinate system.

        Returns:
            topology (np.ndarray):
                A matrix of `Mx3` entries describing the sensor array topology.

        Raises:
            ValueError:
                If the first dimension `topology` is smaller than 1 or its second dimension is larger than 3.
        """

        return self.__topology

    @topology.setter
    def topology(self, value: Optional[np.ndarray]) -> None:
        """Set the antenna array topology."""

        # Default value
        if value is None:
            value = np.zeros((1, 3), dtype=float)

        if value.ndim > 2:
            raise ValueError("The topology array must be of dimension 2")

        # If the topology is one-dimensional, we assume a ULA and therefore simply expand the second dimension
        if value.ndim == 1:
            value = value.T[:, np.newaxis]

        if value.shape[0] < 1:
            raise ValueError("The topology must contain at least one sensor")

        if value.shape[1] > 3:
            raise ValueError("Second topology dimension may contain at most 3 fields (xyz)")

        # The second dimension may have less than 3 entries, in this case we expand by zeros
        if value.shape[1] < 3:
            value = np.append(value, np.zeros((value.shape[0], 3 - value.shape[1]), dtype=float), axis=1)

        self.__topology = value

        # Automatically detect linearity in default configurations, where all sensor elements
        # are oriented along the local x-axis.
        # axis_sums = np.sum(self.__topology, axis=0)
        # self.__linear_topology = ((axis_sums[1] + axis_sums[2]) < 1e-10)

    @property
    def num_antennas(self) -> int:
        """Number of antennas within this device's topology.

        Returns:
            int: Number of antennas, greater or equal to one.
        """

        return self.__topology.shape[0]

    @property
    def max_frame_duration(self) -> float:
        """Maximum frame duration transmitted by this device.

        Returns:
            max_duration (float): Maximum duration in seconds.
        """

        max_duration = 0.

        for operator in chain(self.transmitters.__iter__(), self.receivers.__iter__()):
            max_duration = max(max_duration, operator.frame_duration)

        return max_duration


DeviceType = TypeVar('DeviceType', bound=Device)
"""Type of device."""


class PhysicalDevice(Device):
    """Base representing any device controlling real hardware."""

    def __init__(self, *args, **kwargs) -> None:
        """
        Args:
            *args:
                Device base class initialization parameters.

            **kwargs:
                Device base class initialization parameters.
        """

        # Init base class
        Device.__init__(self, *args, **kwargs)

    @abstractmethod
    def trigger(self) -> None:
        """Trigger the device."""
        ...


class SimulatedDevice(Device):
    """Representation of a device simulating hardware.

    Simulated devices are required to attach to a scenario in order to simulate proper channel propagation.
    """

    __slots__ = ['__scenario']

    __scenario: Optional[Scenario]          # Scenario this device is attached to

    def __init__(self,
                 scenario: Optional[Scenario] = None,
                 *args,
                 **kwargs) -> None:
        """
        Args:

            scenario (Scenario, optional):
                Scenario this device is attached to.
                By default, the device is considered floating.

            *args:
                Device base class initialization parameters.

            **kwargs:
                Device base class initialization parameters.
        """

        # Init base class
        Device.__init__(self, *args, **kwargs)

        self.scenario = scenario

    @property
    def scenario(self) -> Scenario:
        """Scenario this device is attached to.

        Returns:
            Scenario:
                Handle to the scenario this device is attached to.

        Raises:
            FloatingError: If the device is currently floating.
            RuntimeError: Trying to overwrite the scenario of an already attached device.
        """

        if self.__scenario is None:
            raise FloatingError("Error trying to access the scenario of a floating modem")

        return self.__scenario

    @scenario.setter
    def scenario(self, scenario: Scenario) -> None:
        """Set the scenario this device is attached to. """

        if hasattr(self, '_SimulatedDevice__scenario') and self.__scenario is not None:
            raise RuntimeError("Error trying to modify the scenario of an already attached modem")

        self.__scenario = scenario

    @property
    def attached(self) -> bool:
        """Attachment state of this device.

        Returns:
            bool: `True` if the device is currently attached, `False` otherwise.
        """

        return self.__scenario is not None
