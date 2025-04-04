# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Type, TYPE_CHECKING
from typing_extensions import override

import numpy as np
from scipy.signal import convolve
from scipy.fft import ifft

from hermespy.core import Serializable, Signal, SerializationProcess, DeserializationProcess
from .isolation import Isolation

if TYPE_CHECKING:
    from ..simulated_device import SimulatedDevice  # pragma: no cover


__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SelectiveLeakage(Isolation):
    """Model of frequency-selective transmit-receive leakage."""

    __leakage_response: np.ndarray  # Impulse response of the leakage model

    def __init__(self, leakage_response: np.ndarray, *args, **kwargs) -> None:
        """
        Args:

            leakage_response:
                Three-dimensional leakge impulse response matrix :math:`\\mathbf{H}` of dimensions :math:`M \\times N \\times L`,
                where :math:`M` is the number of receive streams and
                :math:`N` is the number of transmit streams and
                :math:`L` is the number of samples in the impulse response.

        Raises:

            ValueError: If the leakage response matrix has invalid dimensions.
        """

        if leakage_response.ndim != 3:
            raise ValueError(
                f"Leakage response matrix must be a three-dimensional array (has {leakage_response.ndim} dimensions)"
            )

        # Initialize base classes
        Serializable.__init__(self)
        Isolation.__init__(self, *args, **kwargs)

        # Initialize class attributes
        self.__leakage_response = leakage_response

    @classmethod
    def Normal(
        cls: Type[SelectiveLeakage],
        device: SimulatedDevice,
        num_samples: int = 100,
        mean: float = 0.0,
        variance: float = 1.0,
    ) -> SelectiveLeakage:
        """Initialize a frequency-selective leakage model with a normally distributed frequency response.

        Args:

            mean:
                Mean of the frequency response in real and imaginary parts.
                One by default.

            variance:
                Variance of the frequency response in real and imaginary parts.
                One by default.

        Returns: An initialized selective frequency model.
        """

        frequency_response = np.random.normal(
            np.sqrt(0.5) * mean,
            variance,
            (
                device.antennas.num_receive_antennas,
                device.antennas.num_transmit_antennas,
                num_samples,
            ),
        ) + 1j * np.random.normal(
            np.sqrt(0.5) * mean,
            variance,
            (
                device.antennas.num_receive_antennas,
                device.antennas.num_transmit_antennas,
                num_samples,
            ),
        )
        leakage_response = ifft(frequency_response, axis=2, norm="backward")

        return cls(leakage_response=leakage_response)

    @property
    def leakage_response(self) -> np.ndarray:
        """Leakage impulse response matrix.

        Numpy matrix of dimensions :math:`M \\times N \\times L`,
        where :math:`M` is the number of receive streams and :math:`N` is the number of transmit streams and
        :math:`L` is the number of samples in the impulse response.
        """

        return self.__leakage_response

    def _leak(self, signal: Signal) -> Signal:
        num_leaked_samples = self.leakage_response.shape[2] + signal.num_samples - 1
        leaking_samples = np.zeros(
            (self.leakage_response.shape[0], num_leaked_samples), dtype=np.complex128
        )

        for m, n in np.ndindex(self.leakage_response.shape[0], signal.num_streams):
            # The leaked signal is the convolution of the transmitted signal with the leakage response
            leaking_samples[m, :] += convolve(
                self.leakage_response[m, n, :], signal.getitem(n).flatten(), "full"
            )[:num_leaked_samples]

        return signal.from_ndarray(leaking_samples)

    @override
    def serialize(self, serialization_process: SerializationProcess) -> None:
        serialization_process.serialize_array(self.__leakage_response, "leakage_response")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> SelectiveLeakage:
        return cls(process.deserialize_array("leakage_response", np.complex128))
