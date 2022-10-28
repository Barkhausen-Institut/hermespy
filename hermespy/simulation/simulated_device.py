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

from hermespy.core import Device, FloatingError, RandomNode, Scenario, Serializable, Signal, Receiver, SNRType
from .analog_digital_converter import AnalogDigitalConverter
from .noise import Noise, AWGN
from .rf_chain.rf_chain import RfChain
from .isolation import Isolation, PerfectIsolation
from .coupling import Coupling, PerfectCoupling

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SimulatedDevice(Device, RandomNode, Serializable):
    """Representation of a device simulating hardware.

    Simulated devices are required to attach to a scenario in order to simulate proper channel propagation.
    
    
    .. warning::
    
       When configuring simulated devices within simulation scenarios,
       channel models may ignore spatial properties such as :func:`.position`, :func:`.orientation` or :func:`.velocity`.  
       
    """

    yaml_tag = u'SimulatedDevice'
    """YAML serialization tag."""

    rf_chain: RfChain
    """Model of the device's radio-frequency chain."""

    adc: AnalogDigitalConverter
    """Model of receiver's ADC"""
    
    __isolation: Isolation
    """Model of the device's transmit-receive isolations"""
    
    __coupling: Coupling
    """Model of the device's antenna array mutual coupling"""

    __noise: Noise                          # Model of the hardware noise
    __scenario: Optional[Scenario]          # Scenario this device is attached to
    __sampling_rate: Optional[float]        # Sampling rate at which this device operate
    __carrier_frequency: float              # Center frequency of the mixed signal in rf-band
    __velocity: np.ndarray                  # Cartesian device velocity vector
    __operator_separation: bool

    def __init__(self,
                 scenario: Optional[Scenario] = None,
                 rf_chain: Optional[RfChain] = None,
                 adc: Optional[AnalogDigitalConverter] = None,
                 isolation: Optional[Isolation] = None,
                 coupling: Optional[Coupling] = None,
                 sampling_rate: Optional[float] = None,
                 carrier_frequency: float = 0.,
                 *args,
                 **kwargs) -> None:
        """
        Args:

            scenario (Scenario, optional):
                Scenario this device is attached to.
                By default, the device is considered floating.

            rf_chain (RfChain, optional):
                Model of the device's radio frequency amplification chain.

            adc (AnalogDigitalConverter, optional):
                Model of receiver's ADC converter.
                
            isolation (Isolation, optional):
                Model of the device's transmit-receive isolations.
                By default, perfect isolation is assumed.
                
            coupling (Coupling, optional):
                Model of the device's antenna array mutual coupling.
                By default, ideal coupling behaviour is assumed.

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
        self.rf_chain = RfChain() if rf_chain is None else rf_chain
        self.adc = AnalogDigitalConverter() if adc is None else adc
        self.isolation = PerfectIsolation() if isolation is None else isolation
        self.coupling = PerfectCoupling() if coupling is None else coupling
        self.noise = AWGN()
        self.snr = float('inf')
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
    def isolation(self) -> Isolation:
        """Model of the device's transmit-receive isolation.
        
        Returns: Handle to the isolation model.
        """
        
        return self.__isolation
    
    @isolation.setter
    def isolation(self, value: Isolation) -> None:
        
        self.__isolation = value
        value.device = self
         
    @property
    def coupling(self) -> Coupling:
        """Model of the device's antenna array mutual coupling behaviour.
        
        Returns: Handle to the coupling model.
        """
        
        return self.__coupling
    
    @coupling.setter
    def coupling(self, value: Coupling) -> None:
        
        self.__coupling = value
        value.device = self

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
        signals = [t.signal for t in self.transmitters.get_transmissions(clear_cache)] if self.operator_separation else \
            [Device.transmit(self, clear_cache)]

        # Simulate rf-chain
        transmissions = [self.rf_chain.transmit(signal) for signal in signals]
        
        # Simulate mutual coupling behaviour
        coupled_transmission = [self.coupling.transmit(signal) for signal in transmissions]

        # Return result
        return coupled_transmission

    @property
    def snr(self) -> float:
        """Signal to noise ratio at the receiver side.
        
        Returns:

            Linear ratio of signal to noise power.
        """

        return self.__snr

    @snr.setter
    def snr(self, value: float) -> None:

        if value <= 0:
            raise ValueError("The linear signal to noise ratio must be greater than zero")

        self.__snr = value

    def receive(self,
                device_signals: Union[List[Signal], Signal, np.ndarray],
                snr: Optional[float] = None,
                snr_type: SNRType = SNRType.PN0,
                leaking_signal: Optional[Signal] = None) -> Signal:
        """Receive signals at this device.

        Args:

            device_signals (Union[List[Signal], Signal, np.ndarray]):
                List of signal models arriving at the device.
                May also be a two-dimensional numpy object array where the first dimension indicates the link
                and the second dimension contains the transmitted signal as the first element and the link channel
                as the second element.

            snr (float, optional):
                Signal to noise power ratio.
                Infinite by default, meaning no noise will be added to the received signals.

            snr_type (SNRType, optional):
                Type of signal to noise ratio.
                
            leaking_signal(Signal, optional):
                Signal leaking from transmit to receive chains.

        Returns:

            baseband_signal (Signal):
                Baseband signal sampled after hardware-modeling.
        """

        # Default to the device's configured SNR if no snr argument was provided
        snr = self.snr if snr is None else snr

        # Mix arriving signals
        mixed_signal = Signal.empty(sampling_rate=self.sampling_rate, num_streams=self.num_antennas,
                                    num_samples=0, carrier_frequency=self.carrier_frequency)

        # Transform signal argument to matrix argument
        if isinstance(device_signals, Signal):
            
            propagation_matrix = np.empty(1, dtype=object)
            propagation_matrix[0] = ([device_signals], None)
            device_signals = propagation_matrix

        # Tranform list arguments to matrix arguments
        elif isinstance(device_signals, list):

            if isinstance(device_signals[0], Signal):
                device_signals = np.array([(device_signals, None)], dtype=object)

            elif isinstance(device_signals[0], tuple):
                device_signals = np.array([device_signals], dtype=object)
                
            else:
                raise ValueError("Unsupported propagation matrix")

        # Superimpose receive signals
        for signals, _ in device_signals:

            if signals is not None:
                for signal in signals:
                    mixed_signal.superimpose(signal)
                    
        # Model mutual coupling behaviour
        coupled_signal = self.coupling.receive(mixed_signal)
                    
        # Add leaked signal if provided
        if leaking_signal is not None:
            
            modeled_leakage = self.__isolation.leak(leaking_signal)
            coupled_signal.superimpose(modeled_leakage)

        # Model radio-frequency chain during transmission
        baseband_signal = self.rf_chain.receive(coupled_signal)

        # Model adc conversion during transmission
        baseband_signal = self.adc.convert(baseband_signal)

        # After rf chain transmission, the signal is considered to be converted to base-band
        # However, for now, carrier frequency information will remain to enable beamforming

        # Cache received signal at receiver slots
        receiver: Receiver
        for receiver in self.receivers:

            # Collect the reference channel if a reference transmitter has been specified
            if receiver.reference is not None and self.attached:

                reference_device_idx = self.scenario.devices.index(receiver.reference)
                reference_csi = device_signals[reference_device_idx][1] if isinstance(device_signals[reference_device_idx], (tuple, list, np.ndarray)) else None

                if self.operator_separation:

                    reference_transmitter_idx = receiver.slot_index
                    receiver_signal = device_signals[reference_device_idx][0][reference_transmitter_idx].copy()

                else:
                    receiver_signal = baseband_signal.copy()

            else:

                reference_csi = None
                receiver_signal = baseband_signal.copy()

            # Add noise to the received signal according to the selected ratio
            noise_power = receiver.noise_power(snr, snr_type)
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
