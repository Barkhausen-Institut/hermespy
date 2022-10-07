# -*- coding: utf-8 -*-
"""
===============
Device Modeling
===============

.. mermaid::

   %%{init: {'theme': 'dark'}}%%
   classDiagram-v2

   direction RL

   class Device {

       topology
       carrier_frequency
       transmit()
       receive()
   }

   class SimulatedDevice {

       position
       orientation
       velocity
       acceleration
   }

   class HardwareDevice {

       trigger()
   }

   class Operator {

       transmit()
       receive()
   }

   class Modem {

       tx_bits
       rx_bits
       ...
       waveform
       transmit()
       receive()
   }

   class Radar {

       waveform
       ...
       target_estimates
       transmit()
       receive()
   }

   Operator <|-- Modem
   Operator <|-- Radar
   Device *-- Operator
   SimulatedDevice <|-- Device
   HardwareDevice <|-- Device


.. mermaid::

   %%{init: {'theme': 'dark'}}%%
   flowchart LR

   channel{Real World}

   subgraph devicea[HardwareDevice]

       direction TB
       deviceatx>Tx Slot] --> deviceabinding[Hardware Binding] --> devicearx>Rx Slot]

   end

   subgraph deviceb[HardwareDevice]

       direction TB
       devicebtx>Tx Slot] --> devicebbinding[Hardware Binding] --> devicebrx>Rx Slot]
   end

   deviceabinding <--> channel
   devicebbinding <--> channel

.. mermaid::

   %%{init: {'theme': 'dark'}}%%
   flowchart LR

   channel{Channel Modeling}

   subgraph devicea[SimulatedDevice]

       direction TB
       deviceatx>Tx Slot]
       devicearx>Rx Slot]

   end

   subgraph deviceb[SimulatedDevice]

       direction TB
       devicebtx>Tx Slot]
       devicebrx>Rx Slot]
   end

   deviceatx --> channel --> devicearx
   devicebtx --> channel --> devicebrx

"""

from __future__ import annotations
from abc import ABC, abstractmethod
from itertools import chain
from math import ceil
from typing import Any, Generic, Iterator, List, Optional, Tuple, TypeVar

import numpy as np
from scipy.constants import speed_of_light

from hermespy.tools.math import transform_coordinates
from .antennas import AntennaArrayBase, UniformArray, IdealAntenna
from .channel_state_information import ChannelStateInformation
from .signal_model import Signal
from .random_node import RandomNode

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
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
    
    def bind(self, slot: SlotType) -> None:
        """Bind the operator to a specific slot.
        
        Args:
        
            slot (SlotType):
                The slot to be bound to.
        """
        
        self.__slot = slot

    @slot.setter
    def slot(self, value: Optional[SlotType]) -> None:
        """Set device slot this operator operates on."""

        # A None argument indicates the slot should be unbound
        if value is None:

            if hasattr(self, '_Operator__slot') and self.__slot is not None:

                slot = self.__slot      # This is necessary to prevent event-loops. Just ignore it.
                self.__slot = None
                
                # if slot.registered(self):
                #     slot.remove(self)

            else:
                self.__slot = None

        elif not hasattr(self, '_Operator__slot'):

            self.__slot = value

            if not self.__slot.registered(self):
                self.__slot.add(self)

        elif self.__slot is not value:

            #if self.__slot is not None and self.__slot.registered(self):
            #    self.__slot.remove(self)

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

    @property
    @abstractmethod
    def sampling_rate(self) -> float:
        """The operator's preferred sampling rate.
        
        Returns:

            Sampling rate in Hz.
        """
        ...


SecondSlotType = TypeVar('SecondSlotType', bound='OperatorSlot')
"""Type of slot."""


class MixingOperator(Generic[SlotType], Operator[SlotType], ABC):
    """Base class for operators performing mixing operations."""

    __carrier_frequency: Optional[float]    # Carrier frequency

    def __init__(self,
                 carrier_frequency: Optional[float] = None,
                 slot: Optional[SlotType] = None) -> None:
        """
        Args:

            carrier_frequency (float, optional):
                Central frequency of the mixed signal in radio-frequency transmission band.

            slot (SlotType, optional):
                Device slot this operator operators.
        """

        self.carrier_frequency = carrier_frequency
        Operator.__init__(self, slot)

    @property
    def carrier_frequency(self) -> float:
        """Central frequency of the mixed signal in radio-frequency transmission band.

        By default, the carrier frequency of the operated device is returned.

        Returns:
            float: Carrier frequency in Hz.

        Raises:
            ValueError: If the carrier frequency is smaller than zero.
            RuntimeError: If the operator is considered floating.
        """

        if self.__carrier_frequency is None:

            if self.device is None:
                raise RuntimeError("Error trying to get the carrier frequency of a floating operator")

            return self.device.carrier_frequency

        return self.__carrier_frequency

    @carrier_frequency.setter
    def carrier_frequency(self, value: Optional[float]) -> None:
        """Set the central frequency of the mixed signal in radio-frequency transmission band."""

        if value is None:

            self.__carrier_frequency = None
            return

        if value < 0.:
            raise ValueError("Carrier frequency must be greater or equal to zero")

        self.__carrier_frequency = value


class Reception(object):
    """Information generated by receiving over a receive operator."""
    ...  # pragma no cover


class Receiver(RandomNode, MixingOperator['ReceiverSlot']):
    """Operator receiving from a device."""

    __reference: Optional[Device]
    __signal: Optional[Signal]
    __csi: Optional[ChannelStateInformation]

    def __init__(self,
                 seed: Optional[int] = None,
                 reference: Optional[Device] = None,
                 *args,
                 **kwargs) -> None:
        """
        Args:

            seed (int, optional):
                Random seed used to initialize the pseudo-random number generator.

            reference (Device, optional):
                Reference device to which the channel shall be estimated.

            *args:
                Operator base class initialization parameters.

            **kwargs:
                Operator base class initialization parameters.
        """

        # Initialize base classes
        RandomNode.__init__(self, seed=seed)
        MixingOperator[ReceiverSlot].__init__(self, params=args)

        self.reference = reference
        self.__signal = None
        self.__csi = None

    @Operator.slot.setter
    def slot(self, value: Optional[ReceiverSlot]) -> None:

        Operator.slot.fset(self, value)
        self.random_mother = None if value is None else value.device

    @property
    def reference(self) -> Optional[Device]:
        """Reference transmitter for this receiver.

        Returns:
            A handle to the referenced device.
            `None` if the device was not specified.
        """

        return self.__reference

    @reference.setter
    def reference(self, value: Optional[Device]) -> None:

        self.__reference = value

    @abstractmethod
    def receive(self) -> Reception:
        """Receive a signal.

        Pulls the required signal model and channel state information from the underlying device.

        Returns:
            Information received by this operator.

        Raises:
            FloatingError:
                If the transmitter is currently considered floating.
        """
        ...  # pragma no cover

    @property
    def signal(self) -> Optional[Signal]:
        """Cached recently received signal model.

        Returns:
            signal (Optional[Signal]):
                Signal model.
                `None` if no signal model is cached.
        """

        return self.__signal

    @property
    def csi(self) -> Optional[ChannelStateInformation]:
        """Cached recently received channel state information

        Returns:
            csi (Optional[ChannelStateInformation]):
                Channel state information.
                `None` if no state information is cached.
        """

        return self.__csi

    def cache_reception(self,
                        signal: Optional[Signal],
                        csi: Optional[ChannelStateInformation] = None) -> None:
        """Cache recent reception at this receiver.

        Args:

            signal (Optional[Signal]):
                Recently received signal.

            csi (Optional[ChannelStateInformation]):
                Recently received channel state.
        """

        self.__signal = signal
        self.__csi = csi

    @property
    @abstractmethod
    def energy(self) -> float:
        """Average energy of the received signal.

        Returns:
            float: Energy.
        """
        ...  # pragma no cover


class OperatorSlot(Generic[OperatorType]):
    """Slot list for operators of a single device."""

    __device: Device                   # Device this operator belongs to.
    __operators: List[OperatorType]     # List of operators registered at this slot

    def __init__(self,
                 device: Device) -> None:
        """
        Args:
            device (Device):
                Device this slot belongs to.
        """

        self.__device = device
        self.__operators = []

    @property
    def device(self) -> Device:
        """Device this operator slot belongs to.

        Returns:
            Device: Handle to the device.
        """

        return self.__device

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

        # Do nothing if the operator is already registered
        if operator in self.__operators:
            return
            
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
            int: Number of operators.
        """

        return len(self.__operators)

    @property
    def max_sampling_rate(self) -> float:
        """Maximum requested sampling rate of all operators.

        Returns:

            The maximum sampling rate in Hz.
        """

        sampling_rate = 0.

        for operator in self.__operators:
            sampling_rate = max(sampling_rate, operator.sampling_rate)

        return sampling_rate

    @property
    def min_frame_duration(self) -> float:
        """The mininum frame duration requried to process at least a single frame for each operator.

        Returns:

            The minimum frame duration in seconds.
        """
        
        frame_duration = 0.

        for operator in self.__operators:
            frame_duration = max(frame_duration, operator.frame_duration)

        return frame_duration

    @property
    def min_num_samples_per_frame(self) -> int:
        """The number of samples required to collect at last a single frame for each operator.

        Returns:

            Number of samples.
        """

        sampling_rate = self.max_sampling_rate
        frame_duration = self.min_frame_duration

        return ceil(frame_duration * sampling_rate)

    def __getitem__(self, item):
        return self.__operators[item]

    def __iter__(self) -> Iterator[Operator]:
        """Iterating over operator slots returns the iterator over the slots."""

        return self.__operators.__iter__()

    def __contains__(self, operator: Operator) -> bool:
        """Contains is just a convenient mask for registered.

        Returns:
            bool: Registration indicator.
        """

        return self.registered(operator)
    
    
class Transmission(object):
    """Information generated by transmitting over a transmit operator."""
    ...  # pragma no cover
    

class Transmitter(RandomNode, MixingOperator['TransmitterSlot']):
    """Operator transmitting over a device."""

    def __init__(self,
                 seed: Optional[int] = None,
                 *args, **kwargs) -> None:
        """
        Args:

            seed (int, optional):
                Random seed used to initialize the pseudo-random number generator.

            *args:
                Operator base class initialization parameters.

            **kwargs:
                Operator base class initialization parameters.
        """

        # Initialize operator base class
        RandomNode.__init__(self, seed=seed)
        MixingOperator.__init__(self, *args, **kwargs)

    @abstractmethod
    def transmit(self,
                 duration: float = 0.) -> Transmission:
        """Transmit a signal.

        Registers the signal samples to be transmitted by the underlying device.

        Args:
        
            duration (float, optional):
                Duration of the transmitted signal.
                If not specified, the duration will be inferred by the transmitter.

        Returns:
        
            Information generated while transmitting over this operator.
        
        Raises:
        
            FloatingError: If the transmitter is currently considered floating.
        """
        ...  # pragma no cover
        
    def _configure_transmitted_signal(self, signal: Signal) -> None:
        """Submit a signal to be transmitted by the operated device.
        
        Args:
        
            signal (Signal):
                Model of the signal to be transmitted.
        """
        
        if self.device is None:
            raise FloatingError("Error trying to configure the transmission of a floating device")
        
        self.device.transmitters.add_transmission(self, signal)

    @Operator.slot.setter
    def slot(self, value: Optional[TransmitterSlot]) -> None:

        Operator.slot.fset(self, value)
        self.random_mother = None if value is None else value.device


class TransmitterSlot(OperatorSlot[Transmitter]):
    """Slot for transmitting operators within devices."""

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

    def add(self,
            operator: Receiver) -> None:

        self.__signals.append(None)
        OperatorSlot[Transmitter].add(self, operator)

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

    def get_transmissions(self,
                          clear_cache: bool = True) -> List[Signal]:
        """Get recent transmissions."""

        signals: List[Signal] = []

        for operator_idx, signal in enumerate(self.__signals):
            
            # Ignore operators without registered signal transmissions
            if signal is not None:
            
                # If the signal is a base-band-band signal, assume the operator's carrier frequency requirements
                if signal.carrier_frequency == 0.:
                    signal.carrier_frequency = self[operator_idx].carrier_frequency
                
                signals.append(signal)

        # Clear the cache if the respective flag is enabled
        if clear_cache:
            self.clear_cache()

        return signals

    def clear_cache(self) -> None:
        """Clear the cached transmission of all registered transmit operators."""

        self.__signals = [None for _ in self.__iter__()]


class ReceiverSlot(OperatorSlot[Receiver]):
    """Slot for receiving operators within devices."""

    def __init__(self, *args, **kwargs) -> None:
        """
        Args:
            *args:
                OperatorSlot base class initialization parameters.

            **kwargs:
                OperatorSlot base class initialization parameters.
        """

        # Init base class
        OperatorSlot.__init__(self, *args, **kwargs)

    def get_receptions(self,
                       clear_cache: bool = True) -> List[Tuple[Signal, ChannelStateInformation]]:
        """Get recent receptions."""

        receptions: List[Tuple[Signal, ChannelStateInformation]] = []
        for receiver in self.__operators:

            receptions.append((receiver.signal, receiver.csi))

            if clear_cache:
                receiver.cache_reception(None, None)

        return receptions


class UnsupportedSlot(OperatorSlot):
    """Slot for unsupported operations within devices."""

    def add(self,
            operator: Operator) -> None:

        raise RuntimeError("Slot not supported by this device")


class Device(ABC, RandomNode):
    """Physical device representation within HermesPy.

    It acts as the basis for all transmissions and receptions of sampled electromagnetic signals.
    """
    
    antennas: AntennaArrayBase
    """Model of the device's antenna array."""

    transmitters: TransmitterSlot
    """Transmitters broadcasting signals over this device."""

    receivers: ReceiverSlot
    """Receivers capturing signals from this device"""

    __power: float                          # Average power of the transmitted signals
    __position: Optional[np.ndarray]        # Position of the device within its scenario in cartesian coordinates
    __orientation: Optional[np.ndarray]     # Orientation of the device within its scenario as a quaternion
    __topology: np.ndarray                  # Antenna array topology of the device

    def __init__(self,
                 antennas: Optional[AntennaArrayBase] = None,
                 power: float = 1.0,
                 position: Optional[np.ndarray] = None,
                 orientation: Optional[np.ndarray] = None,
                 topology: Optional[np.ndarray] = None,
                 seed: Optional[int] = None) -> None:
        """
        Args:
        
            antennas (AntennaArrayBase, optional):
                Model of the device's antenna array.
                By default, a :class:`UniformArray` of ideal antennas is assumed.

            power (float, optional):
                Average power of the transmitted signals in Watts.
                1.0 Watt by default.

            position (np.ndarray, optional):
                Position of the device within its scenario in cartesian coordinates.
                By default, the device position is considered to be unknown (`None`).

            orientation (np.ndarray, optional):
                Orientation of the device within its scenario as a quaternion.
                By default, the device orientation is considered to be unknown (`None`).

            topology (np.ndarray, optional):
                Antenna array topology of the device.
                By default, a single ideal omnidirectional antenna is assumed.

            seed (int, optional):
                Random seed used to initialize the pseudo-random number generator.
        """

        RandomNode.__init__(self, seed=seed)

        self.antennas = UniformArray(IdealAntenna(), 5e-3, (1,)) if antennas is None else antennas
        self.transmitters = TransmitterSlot(self)
        self.receivers = ReceiverSlot(self)

        self.power = power
        self.position = position
        self.orientation = orientation
        self.topology = topology

    @property
    def power(self) -> float:
        """Average power of the transmitted signal signal.

        Returns:
            power (float): Power in Watt.

        Raises:
            ValueError: If `value` is smaller than zero.
        """

        return self.__power

    @power.setter
    def power(self, value: float) -> None:
        """Set the average power of the transmitted signal."""

        if value < 0.:
            raise ValueError("Average signal power must be greater or equal to zero")

        self.__power = value
        
    @property
    def snr(self) -> float:
        """Signal to noise ratio of the received signal.
        
        Returns: Linear signal to noise ratio
        """
        
        # By default, we have no knowledge about the snr and assum no noise
        return float('inf')

    @property
    def position(self) -> Optional[np.ndarray]:
        """Position of the device within its scenario in cartesian coordinates.

        Returns:
            position (Optional[np.ndarray]):
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

        return self.topology.shape[0]

    @property
    def antenna_positions(self) -> np.ndarray:
        """Global positions of the antenna array elements.

        Returns:
            np.ndarray:
                Matrix of dimension `num_antennas x 3`.
        """

        return transform_coordinates(self.topology, self.position, self.orientation)

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

    @property
    @abstractmethod
    def carrier_frequency(self) -> float:
        """Central frequency of the device's emissions in the RF-band.

        Returns:

            frequency (float):
                Carrier frequency in Hz.

        Raises:
            ValueError: On negative carrier frequencies.
        """
        ...
        
    @property
    def wavelength(self) -> float:
        """Central wavelength of the device's emmissions in the RF-band.
        
        Returns:
        
            wavelength (float):
                Wavelength in m.
                
        Raises:
            ValueError:
                On wavelengths smaller or equal to zero.
        """
        
        return speed_of_light / self.carrier_frequency
    
    @wavelength.setter
    def wavelength(self, value: float) -> None:
        
        if value <= 0.:
            raise ValueError("Device wavelength must be greater than zero")
        
        self.carrier_frequency = speed_of_light / value

    @property
    @abstractmethod
    def sampling_rate(self) -> float:
        """Sampling rate at which the device's analog-to-digital converters operate.

        Returns:
            sampling_rate (float): Sampling rate in Hz.

        Raises:
            ValueError: If the sampling rate is not greater than zero.
        """
        ...

    @property
    @abstractmethod
    def velocity(self) -> np.ndarray:
        """Cartesian device velocity vector.

        Returns:
            np.ndarray: Velocity vector.

        Raises:
            ValueError: If `velocity` is not three-dimensional.
            NotImplementedError: If `velocity` is unknown.
        """
        ...

    def transmit(self,
                 clear_cache: bool = True) -> Signal:
        """Transmit signal over this device.

        Args:

            clear_cache (bool, optional):
                Clear the cache of operator signals to be transmitted.
                Enabled by default.

        Returns:
            Signal: Signal emitted by this device.
        """

        # Mix transmit signal
        signal = Signal.empty(sampling_rate=self.sampling_rate, num_streams=self.num_antennas,
                              num_samples=0, carrier_frequency=self.carrier_frequency)

        for transmitted_signal in self.transmitters.get_transmissions(clear_cache):
            
            transmitted_signal.carrier_frequency = self.carrier_frequency
            signal.superimpose(transmitted_signal)

        # Trigger actual hardware transmission here.

        return signal


DeviceType = TypeVar('DeviceType', bound=Device)
"""Type of device."""


class DuplexOperator(Transmitter, Receiver):
    """Operator binding to both transmit and receive slots of any device."""

    __device: Optional[Device]

    def __init__(self,
                 device: Optional[Device] = None,
                 reference: Optional[Device] = None,
                 seed: Optional[int] = None):
        """
        Args:

            device (Device, optional):
                Device the duplex operator operates.
        """
        
        Transmitter.__init__(self, seed=seed)
        Receiver.__init__(self, seed=seed, reference=reference)
        
        self.__device = None
        self.device = device
        
    @property
    def device(self) -> Optional[Device]:
        """Device this operator is operating.

        Returns:
            device (Optional[device]):
                Handle to the operated device.
                `None` if the operator is currently operating no device and considered floating.
        """

        return self.__device

    @device.setter
    def device(self, value: Optional[Device]) -> None:
        """Set the device this operator is operating."""

        if value is self.device:
            return
        
        if self.__device is not None:
            
            self.__device.transmitters.remove(self)
            self.__device.receivers.remove(self)

        if value is not None:
            
            value.transmitters.add(self)
            value.receivers.add(self)
            
        self.__device = value
        
    @Transmitter.slot.setter
    def slot(self, value: OperatorSlot[Transmitter]) -> None:
        
        if value is not None:
            self.device = value.device
            
        Transmitter.slot.fset(self, value)
    
    @property
    def csi(self) -> Optional[ChannelStateInformation]:
        
        return Receiver.csi.fget(self)

    @abstractmethod
    def transmit(self, duration: float = 0.) -> Transmission:
        ...  # pragma no cover

    @abstractmethod
    def receive(self) -> Reception:
        ...  # pragma no cover

    @property
    @abstractmethod
    def sampling_rate(self) -> float:
        ...  # pragma no cover

    @property
    @abstractmethod
    def frame_duration(self) -> float:
        ...  # pragma no cover

    @property
    @abstractmethod
    def energy(self) -> float:
        """Average energy of the transmitted and received signal.

        Returns:
            flot: Energy.
        """
        ...  # pragma no cover
