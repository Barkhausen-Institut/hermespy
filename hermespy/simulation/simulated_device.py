# -*- coding: utf-8 -*-
"""
=================
Simulated Devices
=================
"""

from __future__ import annotations
from typing import List, Optional, Union, Tuple, Type

import numpy as np
from scipy.constants import pi

from hermespy.core import ChannelStateInformation, Device, RandomNode, Scenario, Serializable, Signal, Receiver, SNRType
from .analog_digital_converter import AnalogDigitalConverter
from .noise import Noise, NoiseRealization, AWGN
from .rf_chain.rf_chain import RfChain
from .isolation import Isolation, PerfectIsolation
from .coupling import Coupling, PerfectCoupling

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SimulatedDeviceTransmission(object):
    """Information transmitted by a simulated device"""

    __signal: Union[Signal, List[Signal]]
    __operator_separation: bool

    def __init__(self,
                 signal: Union[Signal, List[Signal]],
                 operator_separation: bool) -> None:
        """
        Args:


        """
        self.__signal = signal
        self.__operator_separation = operator_separation


    @property
    def signal(self) -> Union[Signal, List[Signal]]:

        return self.__signal

    @property
    def operator_separation(self) -> bool:

        return self.__operator_separation


class SimulatedDeviceReceiveRealization(object):
    """Realization of a simulated device reception random process"""
    
    __operator_separation: bool
    __noise_realizations: List[NoiseRealization]
    
    def __init__(self,
                 operator_separation: bool,
                 noise_realizations: List[NoiseRealization]) -> None:
        """
        Args:
        
            operator_separation (bool): Is the operator separation flag enabled?
            noise_realization (List[NoiseRealization]): Noise realizations for each receive operator.
        """
        
        self.__operator_separation = operator_separation
        self.__noise_realizations = noise_realizations
    
    @property
    def operator_separation(self) -> bool:
        """Is operator separation mode enabled?
        
        Returns: Boolean indicator.
        """
        
        return self.__operator_separation
    
    @property
    def noise_realizations(self) -> List[NoiseRealization]:
        """Receive operator noise realizations.
        
        Returns: List of noise realizations corresponding to the number of registerd receive operators.
        """
        
        return self.__noise_realizations


class SimulatedDeviceReception(SimulatedDeviceReceiveRealization):
    """Information received by a simulated device"""

    __impinging_signals: np.ndarray
    __leaking_signal: Optional[Signal]

    def __init__(self,
                 impinging_signals: np.ndarray,
                 leaking_signal: Signal,
                 operator_separation: bool,
                 operator_inputs: List[Tuple[Signal, ChannelStateInformation]],
                 noise_realizations: List[NoiseRealization]) -> None:
        """
        Args:

            impinging_signals (np.ndarray): Numpy vector containing lists of signals impinging onto the device.
            leaking_signal (Optional[Signal]): Signal leaking from transmit to receive chains.
            operator_separation (bool): Is the operator separation flag enabled?
            operator_inputs (bool): Information cached by the device operators.
            noise_realization (List[NoiseRealization]): Noise realizations for each receive operator. 
        """

        self.__impinging_signals = impinging_signals
        self.__leaking_signal = leaking_signal
        self.__operator_inputs = operator_inputs
        
        SimulatedDeviceReceiveRealization.__init__(self, operator_separation, noise_realizations)
   
    @classmethod
    def From_Realization(cls: Type[SimulatedDeviceReception],
                         impinging_signals: np.ndarray,
                         leaking_signal: Optional[Signal],
                         operator_inputs: List[Tuple[Signal, ChannelStateInformation]],
                         realization: SimulatedDeviceReceiveRealization) -> SimulatedDeviceReception:
        """Initialize a simulated device reception from its realization.
        
        Returns: Initialized object.
        """
        
        return cls(impinging_signals,
                   leaking_signal,
                   realization.operator_separation,
                   operator_inputs,
                   realization.noise_realizations)

    @property
    def impinging_signals(self) -> np.ndarray:
        
        return self.__impinging_signals
    
    @property
    def leaking_signal(self) -> Optional[Signal]:
        
        return self.__leaking_signal
    
    @property
    def operator_inputs(self) -> List[Tuple[Signal, ChannelStateInformation]]:
        
        return self.__operator_inputs


class SimulatedDevice(Device, RandomNode, Serializable):
    """Representation of a device simulating hardware.

    Simulated devices are required to attach to a scenario in order to simulate proper channel propagation.


    .. warning::

       When configuring simulated devices within simulation scenarios,
       channel models may ignore spatial properties such as :func:`.position`, :func:`.orientation` or :func:`.velocity`.
    """

    yaml_tag = "SimulatedDevice"
    property_blacklist = {"num_antennas", "orientation", "topology", "velocity", "wavelength"}
    serialized_attribute = {"rf_chain", "adc"}

    rf_chain: RfChain
    """Model of the device's radio-frequency chain."""

    adc: AnalogDigitalConverter
    """Model of receiver's ADC"""

    __isolation: Isolation
    """Model of the device's transmit-receive isolations"""

    __coupling: Coupling
    """Model of the device's antenna array mutual coupling"""

    __transmission: Optional[SimulatedDeviceTransmission]
    """Recent transmission of the device. None if nothing has been transmitted"""

    __noise: Noise  # Model of the hardware noise
    # Scenario this device is attached to
    __scenario: Optional[Scenario]
    # Sampling rate at which this device operate
    __sampling_rate: Optional[float]
    # Center frequency of the mixed signal in rf-band
    __carrier_frequency: float
    __velocity: np.ndarray  # Cartesian device velocity vector
    __operator_separation: bool
    __realization: Optional[SimulatedDeviceReceiveRealization]
    __reception: Optional[SimulatedDeviceReception]

    def __init__(self, scenario: Optional[Scenario] = None, rf_chain: Optional[RfChain] = None, adc: Optional[AnalogDigitalConverter] = None, isolation: Optional[Isolation] = None, coupling: Optional[Coupling] = None, sampling_rate: Optional[float] = None, carrier_frequency: float = 0.0, *args, **kwargs) -> None:
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

        self.__transmission = None

        self.scenario = scenario
        self.rf_chain = RfChain() if rf_chain is None else rf_chain
        self.adc = AnalogDigitalConverter() if adc is None else adc
        self.isolation = PerfectIsolation() if isolation is None else isolation
        self.coupling = PerfectCoupling() if coupling is None else coupling
        self.noise = AWGN()
        self.snr = float("inf")
        self.operator_separation = False
        self.sampling_rate = sampling_rate
        self.carrier_frequency = carrier_frequency
        self.velocity = np.zeros(3, dtype=float)
        self.__realization = None
        self.__reception = None

    @property
    def scenario(self) -> Optional[Scenario]:
        """Scenario this device is attached to.

        Returns: Handle to the scenario this device is attached to.
        """

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
        """Set the scenario this device is attached to."""

        if hasattr(self, "_SimulatedDevice__scenario") and self.__scenario is not None:
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
    def sampling_rate(self) -> Optional[float]:
        """Sampling rate at which the device's analog-to-digital converters operate.

        Returns:
            Sampling rate in Hz.
            `None` if the sampling rate is unknown.

        Raises:
            ValueError: If the sampling rate is not greater than zero.
        """

        if self.__sampling_rate is not None:
            return self.__sampling_rate

        if self.transmitters.num_operators > 0:
            return self.transmitters[0].sampling_rate

        if self.receivers.num_operators > 0:
            return self.receivers[0].sampling_rate

        return None

    @sampling_rate.setter
    def sampling_rate(self, value: Optional[float]) -> None:
        """Set the sampling rate at which the device's analog-to-digital converters operate."""

        if value is None:
            self.__sampling_rate = None
            return

        if value <= 0.0:
            raise ValueError("Sampling rate must be greater than zero")

        self.__sampling_rate = value

    @property
    def carrier_frequency(self) -> float:

        return self.__carrier_frequency

    @carrier_frequency.setter
    def carrier_frequency(self, value: float) -> None:

        if value < 0.0:
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

    def transmit(self, clear_cache: bool = True) -> SimulatedDeviceTransmission:

        # Collect transmissions
        operator_signals = [t.signal for t in self.transmitters.get_transmissions(clear_cache)] if self.operator_separation else [Device.transmit(self, clear_cache)]

        transmitted_signals = []
        for operator_signal in operator_signals:

            if operator_signal is None:

                transmitted_signals.append(Signal.empty(self.sampling_rate, self.num_antennas, carrier_frequency=self.carrier_frequency))
                continue

            # Simulate rf-chain
            rf_signal = self.rf_chain.transmit(operator_signal)

            # Simulate mutual coupling behaviour
            coupled_signal = self.coupling.transmit(rf_signal)

            transmitted_signals.append(coupled_signal)

        # Cache and return resulting transmission
        device_transmission = SimulatedDeviceTransmission(transmitted_signals, self.operator_separation)
        
        self.__transmission = device_transmission
        return device_transmission

    @property
    def transmission(self) -> Optional[SimulatedDeviceTransmission]:
        """Recent transmission of the simulated device.

        Updated during the :meth:`.transmit` routine.

        Returns:
            The recent device transmission.
            `None` if the deice has not yet transmitted.
        """

        return self.__transmission

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
        
    @property
    def realization(self) -> Optional[SimulatedDeviceReceiveRealization]:
        """Most recent random realization of a receive process.
        
        Updated during :meth:`.realize_reception`.
        
        Returns:
            The realization.
            `None` if :meth:`.realize_reception` has not been called yet.
        """
        
        return self.__realization
    
    @property
    def reception(self) -> Optional[SimulatedDeviceReception]:
        """Most recent reception of this device.
        
        Updated during :meth:`.receive` and :meth:`.receive_from_realization`.
        
        Returns:
            The reception.
            `None` if :meth:`.receive` or :meth:`.receive_from_realization` has not been called yet.
        """
        
        return self.__reception
        
    def realize_reception(self,
                          snr: Optional[float] = None,
                          snr_type: SNRType = SNRType.PN0) -> SimulatedDeviceReceiveRealization:
        """Generate a random realization for receiving over the simulated device.

        Args:

            snr (float, optional):
                Signal to noise power ratio.
                Infinite by default, meaning no noise will be added to the received signals.

            snr_type (SNRType, optional):
                Type of signal to noise ratio.

        Returns: The generated realization.
        """
        
        # Generate noise realizations for each registered receive operator
        noise_realizations = [self.noise.realize(r.noise_power(snr, snr_type)) for r in self.receivers]
        
        return SimulatedDeviceReceiveRealization(self.operator_separation,
                                                 noise_realizations)

    def receive_from_realization(self,
                                 impinging_signals: Union[List[Signal], Signal, np.ndarray],
                                 realization: SimulatedDeviceReceiveRealization,
                                 leaking_signal: Optional[Signal] = None,
                                 cache: bool = True) -> SimulatedDeviceReception:
        """Simulate a signal reception for this device model.

        Args:
        
            impinging_signals (Union[List[Signal], Signal, np.ndarray]):
                List of signal models arriving at the device.
                May also be a two-dimensional numpy object array where the first dimension indicates the link
                and the second dimension contains the transmitted signal as the first element and the link channel
                as the second element.

            realization (SimulatedDeviceRealization):
                Random realization of the device reception process.

            leaking_signal(Signal, optional):
                Signal leaking from transmit to receive chains.
                If not specified, no leakage is considered during signal reception.
                
            cache (bool, optional):
                Cache the resulting device reception and operator inputs.
                Enabled by default.

        Returns:
            SimulatedDeviceReception: _description_
            
        Raises:
            
            ValueError: If `device_signals` is constructed improperly.
        """
        
        # Transform signal argument to matrix argument
        if isinstance(impinging_signals, Signal):

            propagation_matrix = np.empty(1, dtype=object)
            propagation_matrix[0] = ([impinging_signals], None)
            impinging_signals = propagation_matrix

        # Tranform list arguments to matrix arguments
        elif isinstance(impinging_signals, list):

            if isinstance(impinging_signals[0], Signal):
                impinging_signals = np.array([(impinging_signals, None)], dtype=object)

            elif isinstance(impinging_signals[0], tuple):
                impinging_signals = np.array([impinging_signals], dtype=object)

            else:
                raise ValueError("Unsupported propagation matrix")
            
        # Mix arriving signals
        mixed_signal = Signal.empty(sampling_rate=self.sampling_rate, num_streams=self.num_antennas, num_samples=0, carrier_frequency=self.carrier_frequency)

        # Transform signal argument to matrix argument
        if isinstance(impinging_signals, Signal):

            propagation_matrix = np.empty(1, dtype=object)
            propagation_matrix[0] = ([impinging_signals], None)
            impinging_signals = propagation_matrix

        # Tranform list arguments to matrix arguments
        elif isinstance(impinging_signals, list):

            if isinstance(impinging_signals[0], Signal):
                impinging_signals = np.array([(impinging_signals, None)], dtype=object)

            elif isinstance(impinging_signals[0], tuple):
                impinging_signals = np.array([impinging_signals], dtype=object)

            else:
                raise ValueError("Unsupported propagation matrix")

        # Superimpose receive signals
        for signals, _ in impinging_signals:

            if signals is not None:
                for signal in signals:
                    mixed_signal.superimpose(signal)

        # Call base class reception routine
        Device.receive(self, mixed_signal)

        # Model mutual coupling behaviour
        coupled_signal = self.coupling.receive(mixed_signal)

        # Add leaked signal if provided
        if leaking_signal is not None:

            modeled_leakage = self.__isolation.leak(leaking_signal)
            coupled_signal.superimpose(modeled_leakage)

        # Model radio-frequency chain during reception
        baseband_signal = self.rf_chain.receive(coupled_signal)

        # Model adc conversion during reception
        baseband_signal = self.adc.convert(baseband_signal)

        # After rf chain reception, the signal is considered to be converted to base-band
        # However, for now, carrier frequency information will remain to enable beamforming

        # Cache received signal at receiver slots
        operator_inputs: List[Tuple[Signal, ChannelStateInformation]] = []
        receiver: Receiver
        for receiver, noise_realization in zip(self.receivers, realization.noise_realizations):

            # Collect the reference channel if a reference transmitter has been specified
            if receiver.reference is not None and self.attached:

                reference_device_idx = self.scenario.devices.index(receiver.reference)
                reference_csi = impinging_signals[reference_device_idx][1] if isinstance(impinging_signals[reference_device_idx], (tuple, list, np.ndarray)) else None

                if self.operator_separation:

                    reference_transmitter_idx = receiver.slot_index
                    receiver_signal = impinging_signals[reference_device_idx][0][reference_transmitter_idx].copy()

                else:
                    receiver_signal = baseband_signal.copy()

            else:

                reference_csi = None
                receiver_signal = baseband_signal.copy()

            # Add noise to the received signal
            noisy_signal = self.__noise.add(receiver_signal, noise_realization)
            
            # Cache reception
            if cache:
                receiver.cache_reception(noisy_signal, reference_csi)
                
            operator_inputs.append((noisy_signal, reference_csi))

        # Generate and cache recent reception
        device_reception = SimulatedDeviceReception.From_Realization(impinging_signals, leaking_signal, operator_inputs, realization)
        if cache:
            self.__reception = device_reception
        
        # Return final result
        return device_reception

    def receive(self,
                impinging_signals: Union[List[Signal], Signal, np.ndarray],
                snr: Optional[float] = None,
                snr_type: SNRType = SNRType.PN0,
                leaking_signal: Optional[Signal] = None,
                cache: bool = True) -> SimulatedDeviceReception:
        """Receive signals at this device.

        Args:

            impinging_signals (Union[List[Signal], Signal, np.ndarray]):
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
                
            cache (bool, optional):
                Cache the resulting device reception and operator inputs.
                Enabled by default.
                
        Returns: The device reception.
        """
        
        # Realize the random process
        realization = self.realize_reception(snr, snr_type)
        
        # Receive the signal
        reception = self.receive_from_realization(impinging_signals, realization, leaking_signal, cache)
        
        # Return result
        return reception
