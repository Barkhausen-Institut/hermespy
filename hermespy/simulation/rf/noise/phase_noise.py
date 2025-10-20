# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from typing_extensions import override

import numpy as np
from numba import jit
from numpy.random import default_rng
from scipy.signal import lfilter
from scipy.constants import pi
from scipy.optimize import lsq_linear

from hermespy.core import RandomNode, Serializable, SerializationProcess, DeserializationProcess
from ..signal import RFSignal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PhaseNoiseRealization(ABC):
    """Realization of a phase noise model."""

    @abstractmethod
    def add_noise(self, signal: RFSignal) -> RFSignal:
        """Add phase noise to an existing signal stream.

        Args:
            signal: Signal to which phase noise is to be added.

        Returns:
            Signal with added phase noise.
        """
        ...  # pragma: no cover


class PhaseNoise(RandomNode, Serializable):
    """Base class of phase noise models."""

    @abstractmethod
    def realize(
        self, bandwidth: float, oversampling_factor: int
    ) -> PhaseNoiseRealization: ...  # pragma: no cover

    def add_noise(self, signal: RFSignal) -> RFSignal:
        """Add phase noise to a signal model.

        Args:

            signal: The signal model to which phase noise is to be added.

        Returns:
            Noisy signal model.
        """

        return self.realize(signal.sampling_rate, 1).add_noise(signal)


class NoPhaseNoiseRealization(PhaseNoiseRealization):
    """Realization of a phase noise model that does not add any noise."""

    def add_noise(self, signal: RFSignal) -> RFSignal:
        return signal


class NoPhaseNoise(PhaseNoise):
    """No phase noise considered within the device model."""

    @override
    def realize(self, bandwidth: float, oversampling_factor: int) -> NoPhaseNoiseRealization:
        return NoPhaseNoiseRealization()

    @override
    def serialize(self, process: SerializationProcess) -> None:
        return

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> NoPhaseNoise:
        return NoPhaseNoise()


class OscillatorPhaseNoiseRealization(PhaseNoiseRealization):
    """Realization of an oscillator phase noise model defined in frequency domain

    Refer to :footcite:t:`2014:Khanzadi` for addtional information.
    """

    __K0: float
    __K2: float
    __K3: float
    __seed: int

    def __init__(self, K0: float, K2: float, K3: float, seed: int) -> None:
        """
        Args:
            K0: White noise floor power level, denoted as :math:`K_0` :footcite:p:`2014:Khanzadi`.
            K2: Power level of the 2nd order flicker noise component, denoted as :math:`K_2` :footcite:p:`2014:Khanzadi`.
            K3: Power level of the 3rd order flicker noise component, denoted as :math:`K_3` :footcite:p:`2014:Khanzadi`.
            seed: Seed with which to initialize the random state of the model.
        """

        self.__K0 = K0
        self.__K2 = K2
        self.__K3 = K3
        self.__seed = seed

    @staticmethod
    @jit(nopython=True, cache=True)
    def __w3_filter_coefficients(filter_order: int) -> np.ndarray:
        """Generate coefficients for the 3rd order flicker noise filter.

        Args:

            filter_order: Number of filter coefficients to generate.

        Returns: Coefficients for the 3rd order flicker noise filter.
        """
        h = np.ones(filter_order, dtype=float)
        for k in range(1, filter_order):
            h[k] = (1.5 + k - 1) * h[k - 1] / k
        return h

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

        if num_samples < 1 or num_streams < 1:
            return np.empty((num_streams, num_samples), dtype=np.float64)

        sampling_interval = 1 / sampling_rate
        rng = default_rng(self.__seed)

        var_w0 = self.__K0 / sampling_interval
        var_w2 = 4 * self.__K2 * sampling_interval * pi**2
        var_w3 = 8 * self.__K3 * sampling_interval**2 * pi**3

        # phi0
        if var_w0 > 0:
            w0 = rng.normal(0.0, var_w0**0.5, (num_streams, num_samples))
            phi0 = w0
        else:
            phi0 = np.zeros((num_streams, num_samples), dtype=np.float64)

        # phi2
        if var_w2 > 0:
            w2 = rng.normal(0.0, var_w2**0.5, (num_streams, num_samples))
            phi2 = lfilter([1, 0], [1, -1], w2)
        else:
            phi2 = np.zeros((num_streams, num_samples), dtype=np.float64)

        # phi3
        if var_w3 > 0:
            w3 = rng.normal(0.0, var_w3**0.5, (num_streams, num_samples))

            w3_filter_order = num_samples
            w3_filter_coefficients = OscillatorPhaseNoiseRealization.__w3_filter_coefficients(
                w3_filter_order
            )
            phi3 = lfilter(w3_filter_coefficients, 1, w3)
        else:
            phi3 = np.zeros((num_streams, num_samples), dtype=np.float64)

        pn = phi0 + phi2 + phi3
        return pn

    @override
    def add_noise(self, signal: RFSignal) -> RFSignal:
        # Abort of signal is empty
        if signal.size == 0:
            return signal

        # Generate phase noise samples and multiply them with the signal's phase
        pn = self._get_noise_samples(signal.num_samples, signal.num_streams, signal.sampling_rate)
        noisy_signal = signal * np.exp(1j * pn)

        return noisy_signal


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

            K0: White noise floor power level, denoted as :math:`K_0` :footcite:p:`2014:Khanzadi`.
            K2: Power level of the 2nd order flicker noise component, denoted as :math:`K_2` :footcite:p:`2014:Khanzadi`.
            K3: Power level of the 3rd order flicker noise component, denoted as :math:`K_3` :footcite:p:`2014:Khanzadi`.
            seed: Seed with which to initialize the random state of the model.
        """

        # Base class initialization
        PhaseNoise.__init__(self, seed=seed)

        self.K0 = K0
        self.K2 = K2
        self.K3 = K3

    @classmethod
    def FromPSD(
        cls: type[OscillatorPhaseNoise],
        frequency_offsets: np.typing.ArrayLike,
        noise_levels: np.typing.ArrayLike,
        precision: float = 1e-20,
        seed: int | None = None,
    ) -> OscillatorPhaseNoise:
        """Create an oscillator phase noise model from a given power spectral density (PSD).

        Args:
            frequency_offsets: Frequency offset to the assumed carrier frequency in Hz.
            noise_levels: Noise levels corresponding to the frequency offsets (linear scale).
            precision: Precision of the resulting model fit.
            seed: Seed with which to initialize the random state of the model.

        Returns:
            An instance of :class:`OscillatorPhaseNoise` fitted to the provided PSD.
        """

        _frequency_offsets = np.asarray(frequency_offsets, dtype=np.float64)

        K3, K2, K0 = lsq_linear(
            np.array(
                [_frequency_offsets**-3, _frequency_offsets**-2, np.ones_like(_frequency_offsets)]
            ).T,
            noise_levels,
            bounds=(0, np.inf),
            tol=precision,
        ).x

        return OscillatorPhaseNoise(K0, K2, K3, seed)

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

    @override
    def realize(
        self, bandwidth: float, oversampling_factor: int
    ) -> OscillatorPhaseNoiseRealization:
        return OscillatorPhaseNoiseRealization(
            self.K0, self.K2, self.K3, int(self._rng.integers(0, 2**31 - 1))
        )

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
