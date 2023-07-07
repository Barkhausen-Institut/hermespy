# -*- coding: utf-8 -*-
"""
====================
Phase Noise Modeling
====================
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import lfilter
from scipy.constants import pi

from hermespy.core import RandomNode, Serializable, Signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PhaseNoise(RandomNode, ABC):
    """Base class of phase noise models."""

    @abstractmethod
    def add_noise(self, signal: Signal) -> Signal:
        """Add phase noise to a signal model.

        Args:

            signal (Signal):
                The signal model to which phase noise is to be added.

        Returns: Noise signal model.
        """
        ...  # pragma no cover


class NoPhaseNoise(PhaseNoise, Serializable):
    """No phase noise considered within the device model."""

    yaml_tag = "NoPhaseNoise"
    """YAML serialization tag"""

    def add_noise(self, signal: Signal) -> Signal:
        # It's just a stub
        return signal


class OscillatorPhaseNoise(PhaseNoise, Serializable):
    """Oscillator phase noise model according to :cite:p:`2014:Khanzadi`.

    Phase noise is modeled as a superposition of three noise power spectral densities (PSDs)

    .. math::

        S_{\\phi}(\\Delta f) = S_{\\phi_0}(\\Delta f) + S_{\\phi_2}(\\Delta f) + S_{\\varphi_3}(\\Delta f)

    where

    .. math::

        S_{\\phi_0}(\\Delta f) = K_0

    denotes the white noise floor PSD of power :math:`K_0`,

    .. math::

        S_{\\phi_2}(\\Delta f) = \\frac{K_2}{f^2}

    denotes the flicker noise PSD of power :math:`K_2` following a square law decay with distance to the carrier frequency :math:`\\Delta f`
    and

    .. math::

        S_{\\phi_3}(\\Delta f) = \\frac{K_3}{f^3}

    denotes the flicker noise PSD of power :math:`K_3` following a cubic law decay with distance to the carrier frequency :math:`\\Delta f`.

    A starting point for the parameter values is given in :cite:p:`2014:Khanzadi` as

    .. math::

        K_0 &= -110~\\mathrm{dB} = 10^{-110/10} \\\\
        K_2 &= 10 \\\\
        K_3 &= 10^4 \\quad \\text{.} \\\\
    """

    __K0: float
    __K2: float
    __K3: float

    yaml_tag = "OscillatorPhaseNoise"

    def __init__(self, K0: float = 10 ** (-110 / 10), K2: float = 10, K3: float = 10**4, seed: int | None = None) -> None:
        """
        Args:

            K0 (float):
                White noise floor power level, denoted as :math:`K_0` in :cite:p:`2014:Khanzadi`.

            K2 (float):
                Power level of the 2nd order flicker noise component, denoted as :math:`K_2` in :cite:p:`2014:Khanzadi`.

            K3 (float):
                Power level of the 3rd order flicker noise component, denoted as :math:`K_3` in :cite:p:`2014:Khanzadi`.
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

    def _get_noise_samples(self, num_samples: int, num_streams: int, sampling_rate: float) -> np.ndarray:
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
        signal.samples *= np.exp(1j * pn)

        return signal
