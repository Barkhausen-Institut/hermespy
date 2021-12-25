# -*- coding: utf-8 -*-
"""
=================
Simulated Devices
=================
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np
from scipy.constants import speed_of_light

from hermespy.channel import ChannelStateInformation
from hermespy.core import Device, FloatingError
from hermespy.core.scenario import Scenario
from hermespy.core.signal_model import Signal
from .rf_chain.rf_chain import RfChain

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SimulatedDevice(Device):
    """Representation of a device simulating hardware.

    Simulated devices are required to attach to a scenario in order to simulate proper channel propagation.
    """

    __slots__ = ['__scenario', 'rf_chain', '__sampling_rate']

    rf_chain: RfChain
    """Model of the device's radio-frequency chain."""

    __scenario: Optional[Scenario]          # Scenario this device is attached to
    __sampling_rate: Optional[float]        # Sampling rate at which this device operate

    def __init__(self,
                 scenario: Optional[Scenario] = None,
                 num_antennas: Optional[int] = None,
                 rf_chain: Optional[RfChain] = None,
                 sampling_rate: Optional[float] = None,
                 *args,
                 **kwargs) -> None:
        """
        Args:

            scenario (Scenario, optional):
                Scenario this device is attached to.
                By default, the device is considered floating.

            num_antennas (int, optional):
                Number of antennas.
                The information is used to initialize the simulated device as a Uniform Linear Array with
                half-wavelength antenna spacing.

            rf_chain (RfChain, optional):
                Model of the device's radio frequency amplification chain.

            sampling_rate (float, optional):
                Sampling rate at which this device operates.
                By default, the sampling rate of the first operator is assumed.

            *args:
                Device base class initialization parameters.

            **kwargs:
                Device base class initialization parameters.
        """

        # Init base class
        Device.__init__(self, *args, **kwargs)

        self.scenario = scenario
        self.rf_chain = RfChain() if rf_chain is None else rf_chain
        self.sampling_rate = sampling_rate

        # If num_antennas is configured initialize the modem as a Uniform Linear Array
        # with half wavelength element spacing
        if num_antennas is not None:

            if not np.array_equal(self.topology, np.zeros((1, 3))):
                raise ValueError("The num_antennas and topology parameters are mutually exclusive")

            # For a carrier frequency of 0.0 we will initialize all antennas at the same position.
            half_wavelength = 0.0
            if self.__carrier_frequency > 0.0:
                half_wavelength = .5 * speed_of_light / self.__carrier_frequency

            self.topology = half_wavelength * np.outer(np.arange(num_antennas), np.array([1., 0., 0.]))

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

    def transmit(self,
                 clear_cache: bool = True) -> Signal:

        # Capture transmitted signal
        transmitted_signal = Device.transmit(self, clear_cache)

        # Simulate rf-chain
        rf_signal = self.rf_chain.transmit(transmitted_signal)

        # Return result
        return rf_signal

    def receive(self, signals: List[Tuple[Signal, ChannelStateInformation]]) -> Signal:
        """Receive signals at this device.

        Args:
            signals (List[Tuple[Signal, ChannelStateInformation]]):
                List of signal models arriving at the device.

        Returns:
            baseband_signal (Signal):
                Baseband signal sampled after hardware-modeling.
        """

        # Mix arriving signals
        mixed_signal = Signal.empty(sampling_rate=self.sampling_rate, num_streams=self.num_antennas,
                                    num_samples=0, carrier_frequency=self.carrier_frequency)

        for signal, _ in signals:
            mixed_signal.superimpose(signal)

        # Model radio-frequency chain during transmission
        baseband_signal = self.rf_chain.receive(mixed_signal)

        # Cache received signal at receiver slots
        for receiver in self.receivers:

            # Collect the reference channel if a reference transmitter has been specified
            if receiver.reference_transmitter is not None:

                reference_device = receiver.reference_transmitter.device
                reference_idx = self.scenario.devices.index(reference_device)
                reference_csi = signals[reference_idx][1]

            else:
                reference_csi = None

            # Cache reception
            receiver.cache_reception(baseband_signal, reference_csi)

        return baseband_signal
