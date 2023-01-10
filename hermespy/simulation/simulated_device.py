# -*- coding: utf-8 -*-
"""
=================
Simulated Devices
=================
"""

from __future__ import annotations
from typing import Iterable, List, Optional, Union, Tuple, Type

import numpy as np
from h5py import Group
from scipy.constants import pi

from hermespy.core import ChannelStateInformation, Device, DeviceInput, DeviceOutput, DeviceReception, DeviceReception, DeviceTransmission, ProcessedDeviceInput, Transmission, RandomNode, Reception, Scenario, Serializable, Signal, Receiver, SNRType
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


class SimulatedDeviceOutput(DeviceOutput):
    """Information transmitted by a simulated device"""

    __emerging_signals: List[Signal]

    def __init__(self,
                 emerging_signals: Union[Signal, List[Signal]],
                 sampling_rate: float,
                 num_antennas: int,
                 carrier_frequency: float) -> None:
        """
        Args:

            emerging_signals (Union[Signal, List[Signal]]):
                Signal models emerging from the device.

            sampling_rate (float):
                Device sampling rate in Hz during the transmission.

            num_antennas (int):
                Number of transmitting device antennas.

            carrier_frequency (float):
                Device carrier frequency in Hz.

        Raises:

            ValueError: If `sampling_rate` is greater or equal to zero.
            ValueError: If `num_antennas` is smaller than one.
        """
        
        emerging_signals: List[Signal] = [emerging_signals] if isinstance(emerging_signals, Signal) else emerging_signals
        superimposed_signal = Signal.empty(sampling_rate, num_antennas, carrier_frequency=carrier_frequency)

        # Assert emerging signal's validity and superimpose the signals
        for signal in emerging_signals:
            
            if signal.sampling_rate != sampling_rate:
                raise ValueError(f"Emerging signal has unexpected sampling rate ({signal.sampling_rate} instad of {sampling_rate})")
                
            if signal.num_streams != num_antennas:
                raise ValueError(f"Emerging signal has unexpected number of transmit antennas ({signal.num_antennas} instead of {num_antennas})")

            if signal.carrier_frequency != carrier_frequency:
                raise ValueError(f"Emerging signal has unexpected carrier frequency ({signal.carrier_frequency} instead of {carrier_frequency})")

            superimposed_signal.superimpose(signal)

        # Initialize attributes
        self.__emerging_signals = emerging_signals

        # Initialize base class
        DeviceOutput.__init__(self, superimposed_signal)

    @classmethod
    def From_DeviceOutput(cls: Type[SimulatedDeviceOutput],
                          device_output: DeviceOutput,
                          emerging_signals: Union[Signal, List[Signal]]) -> SimulatedDeviceOutput:
        """Initialize a simulated device output from its base class.
        
        Args:
        
            device_output (DeviceOutput): Device output.
            emerging_signals (Union[Signal, List[Signal]]): Signal models emerging from the device.
            
        Returns: The initialized object.
        """
        
        return cls(device_output.operator_transmissions, emerging_signals, device_output.sampling_rate, device_output.num_antennas, device_output.carrier_frequency)
                
    @property
    def operator_separation(self) -> bool:
        """Operator separation enabled?
        
        Returns: Operator separation indicator.
        """
        
        return len(self.__emerging_signals) > 1
        
    @property
    def emerging_signals(self) -> List[Signal]:

        return self.__emerging_signals
        
    @classmethod
    def from_HDF(cls: Type[SimulatedDeviceOutput], group: Group) -> SimulatedDeviceOutput:
 
        # Recall base class
        device_output = DeviceOutput.from_HDF(group)
        
        # Recall emerging signals
        num_emerging_signals = group.attrs.get("num_emerging_signals", 0)
        emerging_signals = [Signal.from_HDF(group[f"emerging_signal_{s:02d}"]) for s in range(num_emerging_signals)]

        # Initialize object
        return cls.From_DeviceOutput(device_output, emerging_signals)
    
    def to_HDF(self, group: Group) -> None:
        
        # Serialize base class
        DeviceOutput.to_HDF(self, group)
        
        # Serialize emerging signals
        group.attrs['num_emerging_signals'] = self.num_emerging_signals

        for e, emerging_signal in enumerate(self.emerging_signals):
            emerging_signal.to_HDF(group.create_group(f"emerging_signal_{e:02d}"))


class SimulatedDeviceTransmission(DeviceTransmission, SimulatedDeviceOutput):
    """Information generated by transmitting over a simulated device."""

    def __init__(self, 
                 operator_transmissions: List[Transmission],
                 emerging_signals: Union[Signal, List[Signal]],
                 sampling_rate: float,
                 num_antennas: int,
                 carrier_frequency: float) -> None:
        """
        Args:

            operator_transmissions (List[Transmission]):
                Information generated by transmitting over transmit operators.

            emerging_signals (Union[Signal, List[Signal]]):
                Signal models emerging from the device.

            sampling_rate (float):
                Device sampling rate in Hz during the transmission.

            num_antennas (int):
                Number of transmitting device antennas.

            carrier_frequency (float):
                Device carrier frequency in Hz.

        Raises:

            ValueError: If `sampling_rate` is greater or equal to zero.
            ValueError: If `num_antennas` is smaller than one.
        """

        # Initialize base classes
        SimulatedDeviceOutput.__init__(self, emerging_signals, sampling_rate, num_antennas, carrier_frequency)
        DeviceTransmission.__init__(self, operator_transmissions, SimulatedDeviceOutput.mixed_signal.fget(self))

    @classmethod
    def From_DeviceOutput(cls: Type[SimulatedDeviceTransmission],
                    output: SimulatedDeviceOutput,
                    operator_transmissions: List[Transmission]) -> SimulatedDeviceTransmission:

        return cls(operator_transmissions, output.emerging_signals, output.sampling_rate, output.num_antennas, output.carrier_frequency)


class SimulatedDeviceReceiveRealization(object):
    """Realization of a simulated device reception random process"""
    
    __noise_realizations: List[NoiseRealization]
    
    def __init__(self,
                 noise_realizations: List[NoiseRealization]) -> None:
        """
        Args:
        
            noise_realization (List[NoiseRealization]): Noise realizations for each receive operator.
        """
        
        self.__noise_realizations = noise_realizations
    
    @property
    def noise_realizations(self) -> List[NoiseRealization]:
        """Receive operator noise realizations.
        
        Returns: List of noise realizations corresponding to the number of registerd receive operators.
        """
        
        return self.__noise_realizations


class ProcessedSimulatedDeviceInput(SimulatedDeviceReceiveRealization, ProcessedDeviceInput):
    """Information generated by receiving over a simulated device."""

    __leaking_signal: Optional[Signal]
    __operator_separation: bool

    def __init__(self,
                 impinging_signals: List[Signal],
                 leaking_signal: Signal,
                 operator_separation: bool,
                 operator_inputs: List[Tuple[Signal, ChannelStateInformation]],
                 noise_realizations: List[NoiseRealization]) -> None:
        """
        Args:

            impinging_signals (List[Signal]): Numpy vector containing lists of signals impinging onto the device.
            leaking_signal (Optional[Signal]): Signal leaking from transmit to receive chains.
            operator_separation (bool): Is the operator separation flag enabled?
            operator_inputs (bool): Information cached by the device operators.
            noise_realization (List[NoiseRealization]): Noise realizations for each receive operator. 
        """

        # Initialize base classes
        SimulatedDeviceReceiveRealization.__init__(self, noise_realizations)
        ProcessedDeviceInput.__init__(self, impinging_signals, operator_inputs)

        # Initialize attributes
        self.__leaking_signal = leaking_signal
        self.__operator_separation = operator_separation
    
    @property
    def leaking_signal(self) -> Optional[Signal]:
        """Signal leaking from transmit to receive chains.
        
        Returns:
            Model if the leaking signal.
            `None` if no leakage was considered.
        """
        
        return self.__leaking_signal

    @property
    def operator_separation(self) -> bool:
        """Operator separation flag.

        Returns: Boolean flag.
        """

        return self.__operator_separation

class SimulatedDeviceReception(ProcessedSimulatedDeviceInput, DeviceReception):
    """Information generated by receiving over a simulated device and its operators."""

    def __init__(self,
                 impinging_signals: np.ndarray,
                 leaking_signal: Signal,
                 operator_separation: bool,
                 operator_inputs: List[Tuple[Signal, ChannelStateInformation]],
                 noise_realizations: List[NoiseRealization],
                 operator_receptions: List[Reception]) -> None:
        """
        Args:

            impinging_signals (np.ndarray):
                Numpy vector containing lists of signals impinging onto the device.

            leaking_signal (Optional[Signal]):
                Signal leaking from transmit to receive chains.

            operator_separation (bool):
                Is the operator separation flag enabled?

            operator_inputs (bool):
                Information cached by the device operators.

            noise_realization (List[NoiseRealization]):
                Noise realizations for each receive operator.

            operator_receptions (List[Reception]):
                Information inferred from receive operators.
        """

        ProcessedSimulatedDeviceInput.__init__(self, impinging_signals, leaking_signal, operator_separation, operator_inputs, noise_realizations)
        DeviceReception.__init__(self, impinging_signals, operator_inputs, operator_receptions)
    
    @classmethod
    def From_DeviceInput(cls: Type[SimulatedDeviceReception],
                         device_input: ProcessedSimulatedDeviceInput,
                         operator_receptions: List[Reception]) -> SimulatedDeviceReception:
        """Initialize a simulated device reception from a device input.
        
        Args:

            device (ProcessedSimulatedDeviceInput): The simulated device input.
            operator_receptions (Reception): Information received by device operators.

        Returns: The initialized object.
        """

        return cls(device_input.impinging_signals, device_input.leaking_signal, device_input.operator_separation,
                   device_input.operator_inputs, device_input.noise_realizations, operator_receptions)


class SimulatedDevice(Device, RandomNode, Serializable):
    """Representation of a device simulating hardware.

    Simulated devices are required to attach to a scenario in order to simulate proper channel propagation.


    .. warning::

       When configuring simulated devices within simulation scenarios,
       channel models may ignore spatial properties such as :func:`.position`, :func:`.orientation` or :func:`.velocity`.
    """

    yaml_tag = "SimulatedDevice"
    property_blacklist = {"num_antennas", "orientation", "random_mother", "scenario", "topology", "velocity", "wavelength"}
    serialized_attribute = {"rf_chain", "adc"}

    rf_chain: RfChain
    """Model of the device's radio-frequency chain."""

    adc: AnalogDigitalConverter
    """Model of receiver's ADC"""

    __isolation: Isolation
    """Model of the device's transmit-receive isolations"""

    __coupling: Coupling
    """Model of the device's antenna array mutual coupling"""

    __output: Optional[SimulatedDeviceOutput]                       # Most recent device output
    __input: Optional[ProcessedSimulatedDeviceInput]                # Most recent device input
    __noise: Noise                                                  # Model of the hardware noise
    __scenario: Optional[Scenario]                                  # Scenario this device is attached to
    __sampling_rate: Optional[float]                                # Sampling rate at which this device operate
    __carrier_frequency: float                                      # Center frequency of the mixed signal in rf-band
    __velocity: np.ndarray                                          # Cartesian device velocity vector
    __operator_separation: bool                                     # Operator separation flag
    __realization: Optional[SimulatedDeviceReceiveRealization]      # Most recent device receive realization
    __reception: Optional[SimulatedDeviceReception]                 # Most recent device reception

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

        self.__scenario = None
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
        self.__output = None
        self.__realization = None
        self.__reception = None

    @property
    def scenario(self) -> Optional[Scenario]:
        """Scenario this device is attached to.

        Returns:
            Handle to the scenario this device is attached to.
            `None` if the device is considered floating.
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

        #if hasattr(self, "_SimulatedDevice__scenario") and self.__scenario is not None:
        #    raise RuntimeError("Error trying to modify the scenario of an already attached modem")

        if self.__scenario is not scenario:

            # Pop the device from the old scenario
            if self.__scenario is not None:
                ...  # ToDo

            self.__scenario = scenario
            self.random_mother = scenario

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
            Sampling rate in Hz.
            If no operator has been specified and the sampling rate was not set,
            a sampling rate of :math:`1` Hz will be assumed by default.

        Raises:
            ValueError: If the sampling rate is not greater than zero.
        """

        if self.__sampling_rate is not None:
            return self.__sampling_rate

        if self.transmitters.num_operators > 0:
            return self.transmitters[0].sampling_rate

        if self.receivers.num_operators > 0:
            return self.receivers[0].sampling_rate

        return 1.

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
        
    def _simulate_output(self, signal: Signal) -> Signal:
        """Simulate a device output over the device's hardware model.
        
        Args:
        
            signal (Signal): Signal feeding into the hardware chain.
        
        Returns: Signal emerging from the hardware chain.
        """
        
        # Simulate rf-chain
        rf_signal = self.rf_chain.transmit(signal)

        # Simulate mutual coupling behaviour
        coupled_signal = self.coupling.transmit(rf_signal)
        
        # Return result
        return coupled_signal

    def generate_output(self,
                        operator_transmissions: Optional[List[Transmission]] = None,
                        cache: bool = True) -> SimulatedDeviceOutput:

        operator_transmissions = self.transmit_operators() if operator_transmissions is None else operator_transmissions

        if len(operator_transmissions) != self.transmitters.num_operators:
            raise ValueError(f"Unexpcted amount of operator transmissions provided ({len(operator_transmissions)} instead of {self.transmitters.num_operators})")

        # Generate emerging signals
        emerging_signals: List[Signal] = []
        
        # If operator separation is enabled, each operator transmission is processed independetly
        if self.operator_separation:
            emerging_signals = [self._simulate_output(t.signal) for t in operator_transmissions]
            
        # If operator separation is disable, the transmissions are superimposed to a single signal model
        else:
            
            superimposed_signal = Signal.empty(self.sampling_rate, self.num_antennas, carrier_frequency=self.carrier_frequency)

            for transmission in operator_transmissions:
                superimposed_signal.superimpose(transmission.signal)

            emerging_signals = [self._simulate_output(superimposed_signal)]

        # Genreate the output data object
        output = SimulatedDeviceOutput(emerging_signals, self.sampling_rate, self.num_antennas, self.carrier_frequency)

        # Cache the output if the respective flag is enabled
        if cache:
            self.__output = output

        # Return result
        return output

    def transmit(self, cache: bool = True) -> SimulatedDeviceTransmission:

        # Generate operator transmissions
        transmissions = self.transmit_operators()

        # Generate base device output
        output = self.generate_output(transmissions, cache)

        # Cache and return resulting transmission
        simulated_device_output = SimulatedDeviceTransmission.From_DeviceOutput(output, transmissions)

        return simulated_device_output

    @property
    def output(self) -> Optional[SimulatedDeviceOutput]:
        """Recent output of the simulated device.

        Updated during the :meth:`.transmit` routine.

        Returns:
            The recent device output.
            `None` if the device has not yet transmitted.
        """

        return self.__output

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
    def output(self) -> Optional[SimulatedDeviceOutput]:
        """Most recent output of this device.
        
        Updated during :meth:`.transmit`.
        
        Returns:
            The output information.
            `None` if :meth:`.transmit` has not been called yet.
        """
        
        return self.__output

    @property
    def input(self) -> Optional[ProcessedSimulatedDeviceInput]:
        """Most recent input of this device.
        
        Updated during :meth:`.receive` and :meth:`.receive_from_realization`.

        Returns:
            The input information.
            `None` if :meth:`.receive` or :meth:`.receive_from_realization` has not been called yet.
        """

        return self.__input
        
    def realize_reception(self,
                          snr: float = float('inf'),
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
        
        # Return device receive realization
        return SimulatedDeviceReceiveRealization(noise_realizations)

    def process_from_realization(self,
                                 impinging_signals: Union[List[Signal], Signal, np.ndarray, SimulatedDeviceOutput],
                                 realization: SimulatedDeviceReceiveRealization,
                                 leaking_signal: Optional[Signal] = None,
                                 cache: bool = True,
                                 channel_state: Optional[ChannelStateInformation] = None) -> ProcessedSimulatedDeviceInput:
        """Simulate a signal reception for this device model.

        Args:
        
            impinging_signals (Union[List[Signal], Signal, np.ndarray, SimulatedDeviceOutput]):
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

            channel_state (ChannelStateInformation, optional):
                Assumed state of the channel over which the impinging signals were propagated.

        Returns: The received information.
            
        Raises:
            
            ValueError: If `device_signals` is constructed improperly.
        """
        
        if isinstance(impinging_signals, SimulatedDeviceOutput):
            impinging_signals = impinging_signals.emerging_signals
            
        if isinstance(impinging_signals, DeviceInput):
            impinging_signals = impinging_signals.impinging_signals
        
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

        # Superimpose receive signals
        for signals, _ in impinging_signals:

            if signals is not None:
                for signal in signals:
                    mixed_signal.superimpose(signal)

        # Model mutual coupling behaviour
        coupled_signal = self.coupling.receive(mixed_signal)

        # If no leaking signal has been specified, assume the most recent transmission to be leaking
        if leaking_signal is None and self.output is not None:
            leaking_signal = self.output.mixed_signal
        
        # Simulate signal transmit-receive isolation leakage
        if leaking_signal is not None:

            modeled_leakage = self.__isolation.leak(leaking_signal)
            coupled_signal.superimpose(modeled_leakage)

        # Model radio-frequency chain during reception
        baseband_signal = self.rf_chain.receive(coupled_signal)

        # Model adc conversion during reception
        # ToDo: Move ADC after noise addition.
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

            # Overwrite the reference csi if channel_state was specified
            if channel_state is not None:
                reference_csi = channel_state

            # Add noise to the received signal
            noisy_signal = self.__noise.add(receiver_signal, noise_realization)
            
            # Cache reception
            if cache:
                receiver.cache_reception(noisy_signal, reference_csi)
                
            operator_inputs.append((noisy_signal, reference_csi))

        # Generate output information
        stored_impinging_signals = [s[0][0] for s in impinging_signals]  # Hack, ToDo: Find better solution
        processed_input = ProcessedSimulatedDeviceInput(stored_impinging_signals, leaking_signal, self.operator_separation, operator_inputs, realization.noise_realizations)
        
        # Cache information if respective flag is enabled
        if cache:
            self.__input = processed_input
        
        # Return final result
        return processed_input

    def process_input(self,
                      impinging_signals: Union[List[Signal], Signal, np.ndarray, SimulatedDeviceOutput],
                      snr: float = float('inf'),
                      snr_type: SNRType = SNRType.PN0,
                      leaking_signal: Optional[Signal] = None,
                      cache: bool = True,
                      channel_state: Optional[ChannelStateInformation] = None) -> ProcessedSimulatedDeviceInput:
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
        processed_input = self.process_from_realization(impinging_signals, realization, leaking_signal, cache, channel_state)
        
        # Return result
        return processed_input

    def receive(self,
                impinging_signals: Union[DeviceInput, Signal, Iterable[Signal]],
                cache: bool = True,
                channel_state: Optional[ChannelStateInformation] = None) -> DeviceReception:

        # Process input
        processed_input = self.process_input(impinging_signals, cache=cache, channel_state=channel_state)

        # Generate receptions
        receptions = self.receive_operators(processed_input.operator_inputs, cache)

        # Generate device reception
        return DeviceReception.From_ProcessedDeviceInput(processed_input, receptions)
    