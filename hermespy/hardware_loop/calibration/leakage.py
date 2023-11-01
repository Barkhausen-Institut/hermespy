# -*- coding: utf-8 -*-
"""
===================
Leakage Calibration
===================
"""

from __future__ import annotations
from math import ceil
from typing import Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
from h5py import Group
from numpy.linalg import svd
from scipy.fft import fft, fftfreq, fftshift, ifft
from scipy.signal import convolve

from hermespy.core import Serializable, Signal, VAT, Visualizable
from ..physical_device import LeakageCalibrationBase, PhysicalDevice
from .delay import DelayCalibration

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SelectiveLeakageCalibration(LeakageCalibrationBase, Visualizable, Serializable):
    """Calibration of a frequency-selective leakage model."""

    yaml_tag = "SelectiveLeakageCalibration"

    __leakage_response: np.ndarray  # Impulse response of the leakage model
    __sampling_rate: float  # Sampling rate of the leakage model
    __delay: float  # Implicit delay of the leakage model

    def __init__(self, leakage_response: np.ndarray, sampling_rate: float, delay: float = 0.0, physical_device: PhysicalDevice | None = None) -> None:
        """
        Args:

            leakage_response (np.ndarray):
                The leakage impulse response matrix.

            sampling_rate (float):
                The sampling rate of the leakage model in Hz.

            delay (float, optional):
                The implicit delay of the leakage model in seconds.
                Defaults to zero.
        """

        if leakage_response.ndim != 3:
            raise ValueError(f"Leakage response matrix must be a three-dimensional array (has {leakage_response.ndim} dimensions)")

        if sampling_rate <= 0.0:
            raise ValueError(f"Sampling rate must be non-negative (not {sampling_rate} Hz)")

        if delay < 0.0:
            raise ValueError(f"Delay must be non-negative (not {delay} seconds)")

        # Initialize base class
        LeakageCalibrationBase.__init__(self)

        # Initialize class attributes
        self.__leakage_response = leakage_response
        self.__sampling_rate = sampling_rate
        self.__delay = delay

    @property
    def leakage_response(self) -> np.ndarray:
        """Leakage impulse response matrix.

        Returns:
            Numpy matrix of dimensions :math:`M \\times N \\times L`,
            where :math:`M` is the number of receive streams and :math:`N` is the number of transmit streams and :math:`L` is the number of samples in the impulse response.
        """

        return self.__leakage_response

    @property
    def sampling_rate(self) -> float:
        """Sampling rate of the leakage model in Hz."""

        return self.__sampling_rate

    @property
    def delay(self) -> float:
        """Implicit delay of the leakage model in seconds."""

        return self.__delay

    def remove_leakage(self, transmitted_signal: Signal, received_signal: Signal, delay_correction: float = 0.0) -> Signal:
        if transmitted_signal.num_streams != self.leakage_response.shape[1]:
            raise ValueError(f"Transmitted signal has unxpected number of streams ({transmitted_signal.num_streams} instead of {self.leakage_response.shape[1]})")

        if received_signal.num_streams != self.leakage_response.shape[0]:
            raise ValueError(f"Received signal has unxpected number of streams ({received_signal.num_streams} instead of {self.leakage_response.shape[0]})")

        if transmitted_signal.sampling_rate != received_signal.sampling_rate:
            raise ValueError(f"Transmitted and received signal must have the same sampling rate ({transmitted_signal.sampling_rate} != {received_signal.sampling_rate})")

        if transmitted_signal.carrier_frequency != received_signal.carrier_frequency:
            raise ValueError(f"Transmitted and received signal must have the same carrier frequency ({transmitted_signal.carrier_frequency} != {received_signal.carrier_frequency})")

        # The received signal is corrected by subtracting the leaked samples
        corrected_signal = received_signal.copy()

        # Compute the implicit delay shift in samples
        delay_sample_shift = round((self.delay - delay_correction) * received_signal.sampling_rate)

        for m, n in np.ndindex(received_signal.num_streams, transmitted_signal.num_streams):
            # The leaked signal is the convolution of the transmitted signal with the leakage response
            predicted_siso_signal = convolve(self.__leakage_response[m, n, :], transmitted_signal.samples[n, :], "full")

            # The correction is achieved by subtracting the leaked signal from the received signal
            if delay_sample_shift >= 0:
                corrected_signal.samples[m, delay_sample_shift : min(delay_sample_shift + len(predicted_siso_signal), corrected_signal.num_samples)] -= predicted_siso_signal[: min(corrected_signal.num_samples - delay_sample_shift, len(predicted_siso_signal))]

            else:
                corrected_signal.samples[m, 0 : min(delay_sample_shift + len(predicted_siso_signal), corrected_signal.num_samples + delay_sample_shift)] -= predicted_siso_signal[-delay_sample_shift : min(corrected_signal.num_samples, len(predicted_siso_signal))]

        return corrected_signal

    @property
    def title(self) -> str:
        return "Frequency Selective Leakage Calibration"

    def _new_axes(self, **kwargs) -> Tuple[plt.Figure, VAT]:
        figure, axes = plt.subplots(1, 2, squeeze=False)
        return figure, axes

    def _plot(self, axes: VAT) -> None:
        time_axes: plt.Axes = axes[0, 0]
        freq_axes: plt.Axes = axes[0, 1]

        for m, n in np.ndindex(self.__leakage_response.shape[0], self.__leakage_response.shape[1]):
            sample_instances = np.arange(self.__leakage_response.shape[2]) / self.__sampling_rate
            frequency_bins = fftshift(fftfreq(self.__leakage_response.shape[2]))

            time_axes.plot(sample_instances, self.__leakage_response[m, n, :].real, label=f"Tx: {n} Rx{m}")
            freq_axes.plot(frequency_bins, fftshift(abs(fft(self.__leakage_response[m, n, :]))), label=f"Tx: {n} Rx{m}")

            time_axes.set_xlabel("Time [s]")
            freq_axes.set_xlabel("Frequency [Hz]")

    def to_HDF(self, group: Group) -> None:
        self._write_dataset(group, "leakage_response", self.leakage_response)
        group.attrs["sampling_rate"] = self.sampling_rate
        group.attrs["delay"] = self.delay

    @classmethod
    def from_HDF(cls: Type[SelectiveLeakageCalibration], group: Group) -> SelectiveLeakageCalibration:
        leakage_response = np.asarray(group.get("leakage_response"), dtype=np.complex_)
        sampling_rate = group.attrs.get("sampling_rate")
        delay = group.attrs.get("delay")

        return SelectiveLeakageCalibration(leakage_response, sampling_rate, delay)

    def estimate_delay(self) -> DelayCalibration:
        """Estimate the delay of the leakage model.

        Returns:
            The delay of the leakage model in seconds.
        """

        # The delay is estimated by finding the maximum of the absolute value of the leakage response
        delay = float(np.argmax(np.abs(self.leakage_response)) / self.sampling_rate + self.delay)

        return DelayCalibration(delay)

    @staticmethod
    def MMSEEstimate(device: PhysicalDevice, num_probes: int = 7, num_wavelet_samples: int = 127, noise_power: np.ndarray | None = None, configure_device: bool = True) -> SelectiveLeakageCalibration:
        """Estimate the transmit receive leakage for a physical device using Minimum Mean Square Error (MMSE) estimation.

        Args:

            device (PhysicalDevice):
                Physical device to estimate the covariance matrix for.

            num_probes (int, optional):
                Number of probings transmitted to estimate the covariance matrix.
                :math:`7` by default.

            num_wavelet_samples (int, optional):
                Number of samples transmitted per probing to estimate the covariance matrix.
                :math:`127` by default.

            noise_power (np.ndarray, optional):
                Noise power at the receiving antennas.
                If not specified, the device's noise power configuration will be assumed or estimated on-the-fly.

            configure_device (bool, optional):
                Configure the specified device by the estimated leakage calibration.
                Enabled by default.

        Returns: The initialized :class:`SelectiveLeakageCalibration` instance.

        Raises:

            ValueError: If the number of probes is not strictly positive.
            ValueError: If the number of samples is not strictly positive.
            ValueError: If the noise power is negative.

        """

        if num_probes < 1:
            raise ValueError(f"Number of probes must be greater than zero (not {num_probes})")

        if num_wavelet_samples < 1:
            raise ValueError(f"Number of samples must be greater than zero (not {num_wavelet_samples}")

        # Estimate noise power if not specified
        if noise_power is None:
            if device.noise_power is None:
                noise_power = device.estimate_noise_power()

            else:
                noise_power = device.noise_power

        if np.any(noise_power < 0.0):
            raise ValueError(f"Noise power must be non-negative (not {noise_power})")

        if noise_power.ndim != 1 or noise_power.shape[0] != device.antennas.num_receive_antennas:
            raise ValueError("Noise power has invalid dimensions")

        # Define estimation parameters
        num_prefix_samples = 0
        num_padding_samples = ceil(device.max_receive_delay * device.sampling_rate)
        num_samples = num_wavelet_samples + num_prefix_samples + num_padding_samples

        # Generate zadoff-chu sequences to probe the device leakage
        cf = num_wavelet_samples % 2
        q = 0
        sample_indices = np.arange(num_wavelet_samples)
        probe_indices = 1 + 2 * np.arange(0, num_probes)
        probing_waveforms = np.exp(-1j * np.pi * np.outer(probe_indices, sample_indices * (sample_indices + cf + 2 * q)) / num_wavelet_samples)

        # Add zero padding to waveforms to account for possible transmit delays
        probing_waveforms = np.append(probing_waveforms, np.zeros((num_probes, num_padding_samples)), axis=1)

        # Compute probing frequency spectra
        probing_frequencies = fft(probing_waveforms, axis=1)

        # Collect received samples
        received_waveforms = np.zeros((num_probes, device.antennas.num_receive_antennas, device.antennas.num_transmit_antennas, num_samples), dtype=np.complex_)
        for p, n in np.ndindex(num_probes, device.antennas.num_transmit_antennas):
            tx_samples = np.zeros((device.antennas.num_transmit_antennas, num_samples), dtype=np.complex_)
            tx_samples[n, :] = probing_waveforms[p, :]
            tx_signal = Signal(tx_samples, sampling_rate=device.sampling_rate, carrier_frequency=device.carrier_frequency)

            rx_signal = device.trigger_direct(tx_signal, calibrate=False)
            rx_signal.samples = rx_signal.samples[:, :num_samples]

            received_waveforms[p, :, n, :] = rx_signal.samples

        # Compute received frequency spectra
        received_frequencies = fft(received_waveforms, axis=3)

        # Estimate frequency spectra via MMSE estimation
        # https://nowak.ece.wisc.edu/ece830/ece830_fall11_lecture20.pdf
        # Page 2 Example 1

        h = np.zeros((num_samples * num_probes, num_samples), dtype=np.complex_)
        for p, probe in enumerate(probing_frequencies):
            h[p * num_samples : (1 + p) * num_samples, :] = np.diag(probe)

        # For now, the noise power is assumed to be the mean over all receive chains
        mean_noise_power = np.mean(noise_power)

        # Compute the pseudo-inverse of the received frequency spectra by singular value decomposition
        u, s, vh = svd(h @ h.T.conj() + mean_noise_power * np.eye(num_samples * num_probes, num_samples * num_probes), full_matrices=False, hermitian=True)
        u_select = u[:, :num_samples]
        s_select = s[:num_samples]
        vh_select = vh[:num_samples, :]

        mmse_estimator = h.T.conj() @ vh_select.T.conj() @ np.diag(1 / s_select) @ u_select.T.conj()

        # Estimate the frequency spectra for each antenna probing independently
        mmse_frequency_selectivity_estimation = np.zeros((device.antennas.num_receive_antennas, device.antennas.num_transmit_antennas, num_samples), dtype=np.complex_)
        for m, n in np.ndindex(device.antennas.num_receive_antennas, device.antennas.num_transmit_antennas):
            probing_estimation = mmse_estimator @ received_frequencies[:, m, n, :].flatten()
            mmse_frequency_selectivity_estimation[m, n, :] = probing_estimation

        # Initialize the calibration from the estimated frequency spectra
        leakage_response = ifft(mmse_frequency_selectivity_estimation, axis=2)[:, :, :num_wavelet_samples]
        calibration = SelectiveLeakageCalibration(leakage_response, device.sampling_rate, device.delay_calibration.delay)

        # Configure the device with the estimated calibration if the respective flag is enabled
        if configure_device:
            device.leakage_calibration = calibration

        return calibration
