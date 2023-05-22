# -*- coding: utf-8 -*-
"""
=================
Delay Calibration
=================
"""

from __future__ import annotations
from time import sleep
from typing import List, Type

from h5py import Group
import numpy as np

from hermespy.core import Serializable, Signal
from ..physical_device import DelayCalibrationBase, PhysicalDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DelayCalibration(DelayCalibrationBase, Serializable):
    """Static delay calibration"""

    yaml_tag = "DelayCalibration"

    __delay: float

    def __init__(self, delay: float, physical_device: PhysicalDevice | None = None) -> None:
        # Initialize base class
        DelayCalibrationBase.__init__(self, physical_device)

        # Initialize class attributes
        self.delay = delay

    @property
    def delay(self) -> float:
        return self.__delay

    @delay.setter
    def delay(self, value: float) -> None:
        self.__delay = value

    def to_HDF(self, group: Group) -> None:
        # Serialize attributes
        group.attrs["delay"] = self.delay

    @classmethod
    def from_HDF(cls: Type[DelayCalibration], group: Group) -> DelayCalibration:
        # Deserialize attributes
        delay = group.attrs["delay"]

        # Return new instance
        return cls(delay=delay)

    @staticmethod
    def Estimate(device: PhysicalDevice, max_delay: float, num_iterations: int = 10, wait: float = 0.0) -> DelayCalibration:
        """Estimate a physical device's inherent transmit-receive delay.

        Ideally, the transmit and receive channels of the device should be connected by a patch cable.
        WARNING: An attenuator element may be required! Be careful!!!!

        Args:

            device (PhysicalDevice):
                The physical device to calibrate, i.e. the device of which a delay is to be estimated.

            max_delay (float):
                The maximum expected delay which the calibration should compensate for in seconds.

            num_iterations (int, optional):
                Number of calibration iterations.
                Default is 10.

            wait (float, optional):
                Idle time between iteration transmissions in seconds.
                Zero by default.

        Returns: An initialized delay calibration instance.
        """

        if num_iterations < 1:
            raise ValueError("The number of iterations must be greater or equal to one")

        if wait < 0.0:
            raise ValueError("The waiting time must be greater or equal to zero")

        sampling_rate = device.max_sampling_rate
        num_samples = int(2 * max_delay * device.max_sampling_rate)
        if num_samples <= 1:
            raise ValueError("The assumed maximum delay is not resolvable by the configured sampling rate")

        dirac_index = int(max_delay * sampling_rate)
        waveform = np.zeros((device.num_antennas, num_samples), dtype=complex)
        waveform[:, dirac_index] = 1.0
        calibration_signal = Signal(waveform, sampling_rate, device.carrier_frequency)

        propagated_signals: List[Signal] = []
        propagated_dirac_indices = np.empty(num_iterations, dtype=int)

        # Make multiple iteration calls for calibration
        for n in range(num_iterations):
            propagated_signal = device.trigger_direct(calibration_signal)

            # Infer the implicit delay by estimating the sample index of the propagated dirac
            propagated_signals.append(propagated_signal)
            propagated_dirac_indices[n] = np.argmax(np.abs(propagated_signal.samples[0, :]))

            # Wait the configured amount of time between iterations
            sleep(wait)

        # Compute calibration delay
        # This is just a ML estimation
        mean_dirac_index = np.mean(propagated_dirac_indices)
        calibration_delay = (mean_dirac_index - dirac_index) / propagated_signal.sampling_rate

        # Visualize the results
        # Feature currently deactivated
        # fig, axes = plt.subplots(2, 1)
        # fig.suptitle("Device Delay Calibration")

        # axes[0].plot(calibration_signal.timestamps, abs(calibration_signal.samples[0, :]))
        # for sig in propagated_signals:
        #    axes[1].plot(sig.timestamps, abs(sig.samples[0, :]), color="blue")
        # axes[1].axvline(x=(dirac_index / sampling_rate - calibration_delay), color="red")

        return DelayCalibration(calibration_delay)
