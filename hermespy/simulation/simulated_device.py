# -*- coding: utf-8 -*-
"""
=================
Simulated Devices
=================
"""

from __future__ import annotations
from typing import List, Optional, Type, Union

import numpy as np
from ruamel.yaml import MappingNode, SafeConstructor, SafeRepresenter, ScalarNode
from scipy.constants import pi

from hermespy.core import Device, FloatingError, RandomNode, Scenario, Serializable, Signal
from hermespy.core.statistics import SNRType
from .analog_digital_converter import AnalogDigitalConverter
from .antenna import AntennaArrayBase, UniformArray, IdealAntenna
from .noise import Noise, AWGN
from .rf_chain.rf_chain import RfChain

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SimulatedDevice(Device, RandomNode, Serializable):
    """Representation of a device simulating hardware.

    Simulated devices are required to attach to a scenario in order to simulate proper channel propagation.
    """

    yaml_tag = u'SimulatedDevice'
    """YAML serialization tag."""

    antennas: AntennaArrayBase
    """Model of the device's antenna array."""

    rf_chain: RfChain
    """Model of the device's radio-frequency chain."""

    adc: AnalogDigitalConverter
    """Model of receiver's ADC"""

    operator_separation: bool
    """Separate operators during signal modeling."""

    __noise: Noise                          # Model of the hardware noise
    __scenario: Optional[Scenario]          # Scenario this device is attached to
    __sampling_rate: Optional[float]        # Sampling rate at which this device operate
    __carrier_frequency: float              # Center frequency of the mixed signal in rf-band
    __velocity: np.ndarray                  # Cartesian device velocity vector
    __operator_separation: bool

    def __init__(self,
                 scenario: Optional[Scenario] = None,
                 antennas: Optional[AntennaArrayBase] = None,
                 rf_chain: Optional[RfChain] = None,
                 adc: Optional[AnalogDigitalConverter] = None,
                 sampling_rate: Optional[float] = None,
                 carrier_frequency: float = 0.,
                 *args,
                 **kwargs) -> None:
        """
        Args:

            scenario (Scenario, optional):
                Scenario this device is attached to.
                By default, the device is considered floating.

            antennas (AntennaArrayBase, optional):
                Model of the device's antenna array.
                By default, a :class:`UniformArray` of ideal antennas is assumed.

            rf_chain (RfChain, optional):
                Model of the device's radio frequency amplification chain.

            adc (AnalogDigitalConverter, optional):
                Model of receiver's ADC converter.

            sampling_rate (float, optional):
                Sampling rate at which this device operates.
                By default, the sampling rate of the first operator is assumed.

            carrier_frequency (float, optional):
                Center frequency of the mixed signal in rf-band in Hz.
                Zero by default.

            *args:
                Device base class initialization parameters.

            **kwargs:
                Device base class initialization parameters.
        """

        # Init base class
        Device.__init__(self, *args, **kwargs)

        self.scenario = scenario
        self.antennas = UniformArray(IdealAntenna(), 5e-3, (1,)) if antennas is None else antennas
        self.rf_chain = RfChain() if rf_chain is None else rf_chain
        self.adc = AnalogDigitalConverter() if adc is None else adc
        self.noise = AWGN()
        self.operator_separation = False
        self.sampling_rate = sampling_rate
        self.carrier_frequency = carrier_frequency
        self.velocity = np.zeros(3, dtype=float)

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

    @Device.orientation.getter
    def orientation(self) -> np.ndarray:

        angles: Optional[np.ndarray] = Device.orientation.fget(self)

        # Return the fixed angle configuration if it is specified
        if angles is not None:
            return angles

        # Draw a random orientation if the angle configuration was not specified
        return self._rng.uniform(0, 2 * pi, 3)

    @scenario.setter
    def scenario(self, scenario: Scenario) -> None:
        """Set the scenario this device is attached to. """

        if hasattr(self, '_SimulatedDevice__scenario') and self.__scenario is not None:
            raise RuntimeError("Error trying to modify the scenario of an already attached modem")

        self.__scenario = scenario

    @Device.topology.getter
    def topology(self) -> np.ndarray:

        return self.antennas.topology

    @property
    def attached(self) -> bool:
        """Attachment state of this device.

        Returns:
            bool: `True` if the device is currently attached, `False` otherwise.
        """

        return self.__scenario is not None

    @property
    def noise(self) -> Noise:
        """Model of the hardware noise.

        Returns:
            Noise: Handle to the noise model.
        """

        return self.__noise

    @noise.setter
    def noise(self, value: Noise) -> None:
        """Set the model of the hardware noise."""

        self.__noise = value
        self.__noise.random_mother = self

    @property
    def sampling_rate(self) -> float:
        """Sampling rate at which the device's analog-to-digital converters operate.

        Returns:
            sampling_rate (float): Sampling rate in Hz.

        Raises:
            ValueError: If the sampling rate is not greater than zero.
            RuntimeError: If the sampling rate could not be inferred.
        """

        if self.__sampling_rate is not None:
            return self.__sampling_rate

        if self.transmitters.num_operators > 0:
            return self.transmitters[0].sampling_rate

        if self.receivers.num_operators > 0:
            return self.receivers[0].sampling_rate

        raise RuntimeError("Simulated device's sampling rate is not defined")

    @sampling_rate.setter
    def sampling_rate(self, value: Optional[float]) -> None:
        """Set the sampling rate at which the device's analog-to-digital converters operate."""

        if value is None:
            self.__sampling_rate = None
            return

        if value <= 0.:
            raise ValueError("Sampling rate must be greater than zero")

        self.__sampling_rate = value

    @property
    def carrier_frequency(self) -> float:

        return self.__carrier_frequency

    @carrier_frequency.setter
    def carrier_frequency(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Carrier frequency must be greater or equal to zero")

        self.__carrier_frequency = value

    @property
    def velocity(self) -> np.ndarray:

        return self.__velocity

    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:

        value = value.flatten()

        if len(value) != 3:
            raise ValueError("Velocity vector must be three-dimensional")

        self.__velocity = value

    @property
    def operator_separation(self) -> bool:
        """Separate operators during signal modeling.
        
        Returns:

            Enabled flag.
        """

        return self.__operator_separation

    @operator_separation.setter
    def operator_separation(self, value: bool) -> None:

        self.__operator_separation = value

    def transmit(self,
                 clear_cache: bool = True) -> List[Signal]:

        # Collect transmissions
        signals = self.transmitters.get_transmissions(clear_cache) if self.operator_separation else \
            [Device.transmit(self, clear_cache)]

        # Simulate rf-chain
        transmissions = [self.rf_chain.transmit(signal) for signal in signals]

        # Return result
        return transmissions

    def receive(self,
                device_signals: Union[List[Signal], np.ndarray],
                snr: float = float('inf'),
                snr_type: SNRType = SNRType.EBN0) -> Signal:
        """Receive signals at this device.

        Args:

            device_signals (Union[List[Signal], np.ndarray]):
                List of signal models arriving at the device.
                May also be a two-dimensional numpy object array where the first dimension indicates the link
                and the second dimension contains the transmitted signal as the first element and the link channel
                as the second element.

            snr (float, optional):
                Signal to noise power ratio.
                Infinite by default, meaning no noise will be added to the received signals.

            snr_type (SNRType, optional):
                Type of signal to noise ratio.

        Returns:

            baseband_signal (Signal):
                Baseband signal sampled after hardware-modeling.
        """

        # Mix arriving signals
        mixed_signal = Signal.empty(sampling_rate=self.sampling_rate, num_streams=self.num_antennas,
                                    num_samples=0, carrier_frequency=self.carrier_frequency)

        if isinstance(device_signals, list):

            for signal in device_signals:

                if signal is not None:
                    mixed_signal.superimpose(signal)

        elif isinstance(device_signals, np.ndarray):

            for signals, _ in device_signals:

                if signals is not None:
                    for signal in signals:
                        mixed_signal.superimpose(signal)

        # Model radio-frequency chain during transmission
        baseband_signal = self.rf_chain.receive(mixed_signal)

        baseband_signal = self.adc.convert(baseband_signal)
        
        # Cache received signal at receiver slots
        for receiver in self.receivers:

            # Collect the reference channel if a reference transmitter has been specified
            if receiver.reference_transmitter is not None and self.attached:

                reference_device = receiver.reference_transmitter.device
                reference_device_idx = self.scenario.devices.index(reference_device)

                reference_csi = device_signals[reference_device_idx][1] if isinstance(device_signals[reference_device_idx], tuple) else None

                if self.operator_separation:

                    reference_transmitter_idx = receiver.slot_index
                    receiver_signal = device_signals[reference_device_idx][0][reference_transmitter_idx].copy()

                else:
                    receiver_signal = baseband_signal.copy()

            else:

                reference_csi = None
                receiver_signal = baseband_signal.copy()

            # Add noise to the received signal according to the selected ratio
            noise_power = receiver.energy / snr
            self.__noise.add(receiver_signal, noise_power)

            # Cache reception
            receiver.cache_reception(receiver_signal, reference_csi)

        return baseband_signal

    @classmethod
    def to_yaml(cls: Type[SimulatedDevice], representer: SafeRepresenter, node: SimulatedDevice) -> MappingNode:
        """Serialize a `SimulatedDevice` object to YAML.

        Args:

            representer (SimulatedDevice):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (SimulatedDevice):
                The `Device` instance to be serialized.

        Returns:

            MappingNode:
                The serialized YAML node.
        """

        state = {
            'num_antennas': node.num_antennas,
            'sampling_rate': node.__sampling_rate,
            'carrier_frequency': node.__carrier_frequency,
        }

        return representer.represent_mapping(cls.yaml_tag, state)

    @classmethod
    def from_yaml(cls: Type[SimulatedDevice], constructor: SafeConstructor, node: MappingNode) -> SimulatedDevice:
        """Recall a new `SimulatedDevice` class instance from YAML.

        Args:

            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (MappingNode):
                YAML node representing the `SimulatedDevice` serialization.

        Returns:

            SimulatedDevice:
                Newly created serializable instance.
        """

        if isinstance(node, ScalarNode):
            return cls()

        state = constructor.construct_mapping(node)
        return cls.InitializationWrapper(state)
