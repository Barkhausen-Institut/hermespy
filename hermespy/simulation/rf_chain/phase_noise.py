# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import abstractmethod
from typing_extensions import override

import numpy as np
from scipy.signal import lfilter
from scipy.constants import pi

from hermespy.core import (
    RandomNode,
    Serializable,
    Signal,
    SerializationProcess,
    DeserializationProcess,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PhaseNoise(RandomNode, Serializable):
    """Base class of phase noise models."""

    @abstractmethod
    def add_noise(self, signal: Signal) -> Signal:
        """Add phase noise to a signal model.

        Args:

            signal (Signal):
                The signal model to which phase noise is to be added.

        Returns: Noise signal model.
        """
        ...  # pragma: no cover


class NoPhaseNoise(PhaseNoise):
    """No phase noise considered within the device model."""

    def add_noise(self, signal: Signal) -> Signal:
        # It's just a stub
        return signal

    @override
    def serialize(self, process: SerializationProcess) -> None:
        return

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> NoPhaseNoise:
        return NoPhaseNoise()


class OscillatorPhaseNoise(PhaseNoise, Serializable):
    """Oscillator phase noise model defined in frequency domain.

    Refer to :footcite:t:`2014:Khanzadi` for addtional information.
    """

    __K0: float
    __K2: float
    __K3: float

    def __init__(
        self,
        K0: float = 10 ** (-110 / 10),
        K2: float = 10,
        K3: float = 10**4,
        seed: int | None = None,
    ) -> None:
        """
        Args:

            K0 (float):
                White noise floor power level, denoted as :math:`K_0` :footcite:p:`2014:Khanzadi`.

            K2 (float):
                Power level of the 2nd order flicker noise component, denoted as :math:`K_2` :footcite:p:`2014:Khanzadi`.

            K3 (float):
                Power level of the 3rd order flicker noise component, denoted as :math:`K_3` :footcite:p:`2014:Khanzadi`.
        """

        # Base class initialization
        PhaseNoise.__init__(self, seed=seed)

        self.K0 = K0
        self.K2 = K2
        self.K3 = K3

    @property
    def K0(self) -> float:
        """White noise floor power level, denoted as :math:`K_0`.

        Raises:

            ValueError: If the value is negative.
        """

        return self.__K0

    @K0.setter
    def K0(self, value: float) -> None:
        if value < 0:
            raise ValueError("K0 must be non-negative")
        self.__K0 = value

    @property
    def K2(self) -> float:
        """Power level of the 2nd order flicker noise component, denoted as :math:`K_2`.

        Raises:

            ValueError: If the value is negative.
        """

        return self.__K2

    @K2.setter
    def K2(self, value: float) -> None:
        if value < 0:
            raise ValueError("K2 must be non-negative")
        self.__K2 = value

    @property
    def K3(self) -> float:
        """Power level of the 3rd order flicker noise component, denoted as :math:`K_3`.

        Raises:

            ValueError: If the value is negative.
        """

        return self.__K3

    @K3.setter
    def K3(self, value: float) -> None:
        if value < 0:
            raise ValueError("K3 must be non-negative")
        self.__K3 = value

    def _get_noise_samples(
        self, num_samples: int, num_streams: int, sampling_rate: float
    ) -> np.ndarray:
        """Generate phase noise samples.

        Subroutine of :meth:`add_noise`.

        Args:

            num_samples (int):
                Number of samples to generate.

            num_streams (int):
                Number of streams to generate.

            sampling_rate (float):
                Sampling rate of the generated samples.

        Returns: Phase noise samples.
        """

        sampling_interval = 1 / sampling_rate

        var_w0 = self.K0 / sampling_interval
        var_w2 = 4 * self.K2 * sampling_interval * pi**2
        var_w3 = 8 * self.K3 * sampling_interval**2 * pi**3

        # phi0
        w0 = self._rng.normal(0.0, var_w0**0.5, (num_streams, num_samples))
        phi0 = w0

        # phi2
        w2 = self._rng.normal(0.0, var_w2**0.5, (num_streams, num_samples))
        phi2 = lfilter([1, 0], [1, -1], w2)

        # phi3
        w3 = self._rng.normal(0.0, var_w3**0.5, (num_streams, num_samples))

        w3_filter_order = num_samples
        h = np.ones(w3_filter_order, dtype=float)
        for k in range(1, w3_filter_order):
            h[k] = (1.5 + k - 1) * h[k - 1] / k
        phi3 = lfilter(h, 1, w3)

        pn = phi0 + phi2 + phi3
        return pn

    def add_noise(self, signal: Signal) -> Signal:
        pn = self._get_noise_samples(signal.num_samples, signal.num_streams, signal.sampling_rate)
        for b in signal:
            b *= np.exp(1j * pn[:, b.offset : b.end])

        return signal

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_floating(self.__K0, "K0")
        process.serialize_floating(self.__K2, "K2")
        process.serialize_floating(self.__K3, "K3")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> OscillatorPhaseNoise:
        K0 = process.deserialize_floating("K0")
        K2 = process.deserialize_floating("K2")
        K3 = process.deserialize_floating("K3")
        return OscillatorPhaseNoise(K0, K2, K3)
