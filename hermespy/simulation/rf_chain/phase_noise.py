# -*- coding: utf-8 -*-
"""
====================
Phase Noise Modeling
====================
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from math import sqrt, sin

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import pi
from scipy.fft import ifft, fft, fftfreq
from scipy.special import gamma

from hermespy.core import Executable, RandomNode, Serializable, Signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
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

    yaml_tag = 'NoPhaseNoise'
    """YAML serialization tag"""

    def add_noise(self, signal: Signal) -> Signal:

        # It's just a stub
        return signal


class PowerLawPhaseNoise(PhaseNoise, Serializable):
    """Implementation of the power law phase noise model.

    Refer to the details within the respective publications :footcite:p:`2006:chorti,2007:chorti`.

    The model proposes to divide the phase noise power spectral density into dedicated regions,
    each region scaling with increasing powers of the inverse frequency distance to the central carrier :math:`\\tfrac{1}{f}`.

    The convolution of all individual power law spectral density

    .. math::

       S(f) = S_{\\omega_1}(f) \\ast S_{\\omega_2}(f) \\ast S_{\\omega_3}(f) \\ast S_{\\omega_4}(f)

    denotes the overal power spectral density of the phase noise's random process.
    The implemented model parameters configuring each sub-PSD and therefore the whole model are listed
    in the following table:

    .. list-table:: Model Parameters
       :header-rows: 1

       * - Power
         - Name
         - Color
         - Parameters

       * - :math:`\\tfrac{1}{f}`
         - Flicker
         - Blue
         - :py:attr:`.flicker_scale`\ :py:attr:`.flicker_num_subterms`\ :py:attr:`.vicinity`

       * - :math:`\\tfrac{1}{f^2}`
         - White FM
         - White
         - :py:attr:`.white_fm_scale`

       * - :math:`\\tfrac{1}{f^3}`
         - Flicker FM
         - Pink
         - :py:attr:`.flicker_fm_scale`\ :py:attr:`.vicinity`

       * - :math:`\\tfrac{1}{f^4}`
         - Random Walk FM
         - Brown
         - :py:attr:`.random_walk_fm_scale`\ :py:attr:`.gauss_cutoff`

    For a parametrization of

    .. math::

       k_1 = 10^{-3},\\ k_2 = 10^{-2},\\ k_3 = 10^{-1},\\ k_4 = 1,\\ \\omega_4 = 10^{-3},\\ \\nu = 10^{-3}

    the model generates the following symmetrical PSD around the assumed carrier frequency:

    .. plot::
       :align: center

       import matplotlib.pyplot as plt

       from hermespy.core import Signal
       from hermespy.simulation.rf_chain import PowerLawPhaseNoise

       k1 = 1e-3
       k2 = 1e-2
       k3 = 1e-1
       k4 = 1
       vicinity = 1e-3
       gauss_cutoff = 1e-3

       pn = PowerLawPhaseNoise(k1, k2, k3, k4, gauss_cutoff, vicinity)

       frequencies = np.linspace(0, 1e6, 50)
       pn.plot_psds(1e4, 100)

       plt.show()


    Considering a :math:`25` Hz sinusoidal signal being processed by a noisy oscillator,
    the signal model will be distorted according to

    .. plot::
       :align: center

       import matplotlib.pyplot as plt

       from hermespy.core import Signal
       from hermespy.simulation.rf_chain import PowerLawPhaseNoise

       k1 = 1e-3
       k2 = 1e-2
       k3 = 1e-1
       k4 = 1
       vicinity = 1e-3
       gauss_cutoff = 1e-3

       pn = PowerLawPhaseNoise(k1, k2, k3, k4, gauss_cutoff, vicinity)

       frequencies = np.linspace(0, 1e6, 50)

       signal = Signal(np.exp(.5j*np.pi*np.arange(200)), 1)
       signal.plot(title='Perfect Signal')

       noisy_signal = pn.add_noise(signal)
       noisy_signal.plot(title='Noisy Signal')

       plt.show()
    """

    yaml_tag = u'PowerLaw'
    """YAML serialization tag"""

    def __init__(self,
                 flicker_scale: float,
                 white_fm_scale: float,
                 flicker_fm_scale: float,
                 random_walk_fm_scale: float,
                 gauss_cutoff: float,
                 vicinity: float = 0.,
                 num_flicker_subterms: int = 10) -> None:
        """
        Args:

            flicker_scale (float): Linear power law spectral density scale.
            white_fm_scale (float): Square power law spectral density scale.
            flicker_fm_scale (float): Cubic power law spectral density scale.
            random_walk_fm_scale (float): Quartic power law spectral density scale.
            gauss_cutoff (float): Transition frequency from flat to quartic region in Hz.
            num_flicker_subterms (int): Number of subterms within the flicker fm term.
        """

        self.flicker_scale = flicker_scale
        self.white_fm_scale = white_fm_scale
        self.flicker_fm_scale = flicker_fm_scale
        self.random_walk_fm_scale = random_walk_fm_scale
        self.gauss_cutoff = gauss_cutoff
        self.vicinity = vicinity
        self.flicker_num_subterms = num_flicker_subterms

        PhaseNoise.__init__(self)

    @property
    def vicinity(self) -> float:
        """Power law vicinity.

        Denoted by :math:`\\nu` within :footcite:t:`2006:chorti`.

        Returns: Power law vicinity.

        Raises:
            ValueError: For vicinities smaller than zero.
        """

        return self.__vicinity

    @vicinity.setter
    def vicinity(self, value: float) -> None:

        if value < 0.:
            raise ValueError(
                "Power law vicinity must be greater or equal to zero")

        self._psd.cache_clear()
        self.__vicinity = value

    @property
    def flicker_scale(self) -> float:
        """Linear power law spectral density scale.

        Denoted by :math:`k_1` within :footcite:t:`2006:chorti`.

        Returns: Linear scale factor.

        Raises:
            ValueError: For scales smaller than zero.
        """

        return self.__flicker_scale

    @flicker_scale.setter
    def flicker_scale(self, value: float) -> None:

        if value < 0.:
            raise ValueError(
                "Spectral density scale must be greater or equal to zero")

        self._psd.cache_clear()
        self.__flicker_scale = value

    @property
    def flicker_num_subterms(self) -> int:
        """Number of power law subterms.

        Returns: Number of terms.

        Raises:
            ValueError: On numbers smaller than one.
        """

        return self.__flicker_num_subterms

    @flicker_num_subterms.setter
    def flicker_num_subterms(self, value: int) -> None:

        if value < 1:
            raise ValueError(
                "Number of flicker subterms must be greater than zero")

        self._psd.cache_clear()
        self.__flicker_num_subterms = value

    def flicker_psd(self, frequencies: np.ndarray) -> np.ndarray:
        """Power spectral density of the linear power law noise component.

        Denoted by :math:`S_{\\omega_1}` within :footcite:t:`2006:chorti`.
        The implementation adopts the algebraic approximation of assuming a high frequency distance to the carrier.

        Args:

            frequencies (np.ndarray):
                Frequencies in Hz at which the power spectral density should be sampled.

        Returns: Sampled power spectral density.
        """

        psd = np.zeros(len(frequencies), dtype=float)

        nonzero_frequency_indices = np.where(frequencies != 0.)
        nonzero_frequencies = frequencies[nonzero_frequency_indices]
        psd[np.where(frequencies == 0.)] = 2*pi

        y = 1  # Approximation for small vicinities

        for n in range(1, 1 + self.flicker_num_subterms):

            coefficient = (-1) ** (n + 1) * 2 * sin(.5 * pi * n * self.vicinity) * self.vicinity * (
                2 * y * self.flicker_scale / self.vicinity) ** n * gamma(n * self.vicinity) / gamma(n)
            psd[nonzero_frequency_indices] += coefficient * \
                np.abs((2 * pi * nonzero_frequencies)
                       ) ** (-1 - n * self.vicinity)

        return psd

    @property
    def white_fm_scale(self) -> float:
        """Square power law spectral density scale.

        Denoted by :math:`k_2` within :footcite:t:`2006:chorti`.

        Returns: Linear scale factor.

        Raises:
            ValueError: For scales smaller than zero.
        """

        return self.__white_fm_scale

    @white_fm_scale.setter
    def white_fm_scale(self, value: float) -> None:

        if value < 0.:
            raise ValueError(
                "Spectral density scale must be greater or equal to zero")

        self._psd.cache_clear()
        self.__white_fm_scale = value

    def white_fm_psd(self, frequencies: np.ndarray) -> np.ndarray:
        """Power spectral density of the square power law noise component.

        Denoted by :math:`S_{\\omega_2}` within :footcite:t:`2006:chorti`.

        Args:

            frequencies (np.ndarray):
                Frequencies in Hz at which the power spectral density should be sampled.

        Returns: Sampled power spectral density.
        """

        return (4 * pi ** 2 + self.white_fm_scale) / (4 * pi ** 4 * self.white_fm_scale ** 2 + (2 * pi * frequencies ** 2))

    @property
    def flicker_fm_scale(self) -> float:
        """Cube power law spectral density scale.

        Denoted by :math:`k_3` within :footcite:t:`2006:chorti`.

        Returns: Linear scale factor.

        Raises:
            ValueError: For scales smaller than zero.
        """

        return self.__flicker_fm_scale

    @flicker_fm_scale.setter
    def flicker_fm_scale(self, value: float) -> None:

        if value < 0.:
            raise ValueError(
                "Spectral density scale must be greater or equal to zero")

        self._psd.cache_clear()
        self.__flicker_fm_scale = value

    def flicker_fm_psd(self,
                       frequencies: np.ndarray) -> np.ndarray:
        """Power spectral density of the cube power law noise component.

        Denoted by :math:`S_{\omega_3}` within :footcite:t:`2006:chorti`.

        Args:

            frequencies (np.ndarray):
                Frequencies in Hz at which the power spectral density should be sampled.
                Not that the PSD computation assumes the frequencies to be sampled equidistantly.

        Returns: Sampled power spectral density.
        """

        K = 16 * pi ** 2 * sqrt(pi) * (2 * pi) ** (-self.vicinity) * gamma(.5 * self.vicinity) * \
            self.flicker_fm_scale / (2 - self.vicinity) * \
            gamma(1.5 - .5 * self.vicinity)

        # Recover sampling rate
        sampling_rate = np.max(np.abs(frequencies))
        num_samples = len(frequencies)
        delays = np.arange(1, num_samples) / sampling_rate

        autocorrelation = np.empty(num_samples, float)
        autocorrelation[0] = 0
        autocorrelation[1:] = .5 * self.vicinity * K * \
            np.exp(-.5 * K * delays ** 2) * delays ** 2 * np.log(delays)
        autocorrelation_fft = fft(autocorrelation)

        gaussian = sqrt(2 * pi / K) * \
            np.exp(- (2 * pi * frequencies) ** 2 / (2 * K))

        return gaussian + autocorrelation_fft

    @property
    def gauss_cutoff(self) -> float:
        """Transition point of Gaussian to quartic power law PSD region.

        Denoted by :math:`\\omega_4` by :footcite:t:`2006:chorti`.

        Returns: Transition point in Hz.

        Raises:
            ValueError: For cutoffs smaller than zero.
        """

        return self.__gauss_cutoff

    @gauss_cutoff.setter
    def gauss_cutoff(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Gauss cutoff must be greater or equal to zero")

        self._psd.cache_clear()
        self.__gauss_cutoff = value

    @property
    def random_walk_fm_scale(self) -> float:
        """Quartic power law spectral density scale.

        Denoted by :math:`k_4` within :footcite:t:`2006:chorti`.

        Returns: Linear scale factor.

        Raises:
            ValueError: For scales smaller than zero.
        """

        return self.__random_walk_fm

    @random_walk_fm_scale.setter
    def random_walk_fm_scale(self, value: float) -> None:

        if value < 0.:
            raise ValueError(
                "Spectral density scale must be greater or equal to zero")

        self._psd.cache_clear()
        self.__random_walk_fm = value

    def random_walk_fm_psd(self, frequencies: np.ndarray) -> np.ndarray:
        """Power spectral density of the quartic power law noise component.

        Denoted by :math:`S_{\\omega_4}` within :footcite:t:`2006:chorti`.

        Args:

            frequencies (np.ndarray):
                Frequencies in Hz at which the power spectral density should be sampled.

        Returns: Sampled power spectral density.
        """

        ro = 10
        psd = np.empty(len(frequencies), dtype=float)

        nonzero_frequency_indices = np.where(frequencies != 0.)
        nonzero_frequencies = frequencies[nonzero_frequency_indices]

        psd[np.where(frequencies == 0.)] = sqrt(ro) / \
            (2 * pi * sqrt(pi * self.random_walk_fm_scale))
        psd[nonzero_frequency_indices] = sqrt(ro) / (2 * pi * sqrt(pi * self.random_walk_fm_scale)) * np.exp(- ro * (
            2 * pi * nonzero_frequencies) ** 2 / (16 * pi ** 4 * self.random_walk_fm_scale))

        quartic_frequency_indices = np.where(
            np.abs(frequencies) > self.gauss_cutoff)
        quartic_frequencies = frequencies[quartic_frequency_indices]
        psd[quartic_frequency_indices] += self.random_walk_fm_scale * \
            quartic_frequencies ** -4

        return psd

    @lru_cache(maxsize=1)
    def _psd(self,
             sampling_rate: float,
             num_samples: int) -> np.ndarray:
        """Internal power spectral density computation.

        Cachable for faster simulaion execution speeds.

        Args:

            sampling_rate (float): Assumed signal sampling rate / PSD modeling bandwidth.
            num_samples (int): Number of discrete PSD frequency bins.

        Returns: A numpy vector containing the power spectral density matching the current parameterization.
        """

        if num_samples < 2:
            raise ValueError("Number of samples must be greater than one")

        # Determin PSD frequency bins
        frequencies = fftfreq(num_samples, 1 / sampling_rate)

        # Convolve all power spectral densities
        psd = np.zeros(len(frequencies), dtype=complex)
        psd[0] = 1.

        for sub_psd in (self.flicker_psd, self.white_fm_psd, self.flicker_fm_psd, self.random_walk_fm_psd):
            psd = np.convolve(psd, sub_psd(frequencies), 'valid')

        return psd

    def plot_psds(self, max_frequency: float, num_samples: int) -> plt.Figure:
        """Plot power spectral densities.

        Args:

            frequencies (np.ndarray): Frequencies at which the PSDs should be evaluated.
        """

        frequencies = fftfreq(num_samples, 1 / (2 * max_frequency))
        nonneg_frequency_indices = np.where(frequencies >= 0)
        nonneg_frequencies = frequencies[nonneg_frequency_indices]

        with Executable.style_context():

            figure, axes = plt.subplots(2)
            figure.suptitle('Power Law Phase Noise Model PSDs')

            axes[0].semilogy(nonneg_frequencies, self.flicker_psd(frequencies)[
                             nonneg_frequency_indices], color='blue', linestyle='dashed', label='Flicker (Power 1)')
            axes[0].semilogy(nonneg_frequencies, self.white_fm_psd(frequencies)[
                             nonneg_frequency_indices], color='white', linestyle='dashed', label='White FM (Power 2)')
            axes[0].semilogy(nonneg_frequencies, abs(self.flicker_fm_psd(frequencies)[
                             nonneg_frequency_indices]), color='pink', linestyle='dashed', label='Flicker FM (Power 3)')
            axes[0].semilogy(nonneg_frequencies, self.random_walk_fm_psd(frequencies)[
                             nonneg_frequency_indices], color='brown', linestyle='dashed', label='Random Walk FM (Power 4)')

            axes[0].legend()

            f4 = self.gauss_cutoff / self.random_walk_fm_scale
            f3 = self.random_walk_fm_scale / self.flicker_fm_scale
            f2 = self.flicker_fm_scale / self.white_fm_scale
            f1 = self.white_fm_scale / self.flicker_fm_scale

            axes[1].semilogy(nonneg_frequencies, abs(
                self._psd(max_frequency, num_samples)[nonneg_frequency_indices]), label="PSD")
            axes[1].axvline(x=f1, color='blue',
                            linestyle='dashed', label='Flicker')
            axes[1].axvline(x=f2, color='white',
                            linestyle='dashed', label='White')
            axes[1].axvline(x=f3, color='pink',
                            linestyle='dashed', label='Flicker FM')
            axes[1].axvline(x=f4, color='brown',
                            linestyle='dashed', label='Random Walk FM')
            axes[1].legend()

            return figure

    def add_noise(self, signal: Signal) -> Signal:

        psd = self._psd(signal.sampling_rate, signal.num_samples)

        white_noise_spectrum = np.exp(2j * self._rng.uniform(0, pi, size=(signal.num_streams, signal.num_samples)))
        coloured_noise_spectrum = white_noise_spectrum * psd

        phase_noise = ifft(coloured_noise_spectrum)

        # The phase noise is, as the name suggests, only a phase shift, so we need to normalize each bin
        # This is clearly noted in equation (4)
        phase_noise /= np.abs(phase_noise)

        noisy_signal = signal.copy()
        noisy_signal.samples *= phase_noise

        return noisy_signal
