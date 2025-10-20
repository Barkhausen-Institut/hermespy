# -*- coding: utf-8 -*-
"""
===================
Leakage Calibration
===================
"""

from __future__ import annotations
from typing import Type
from typing_extensions import override

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd
from scipy.fft import fft, ifft
from scipy.signal import convolve, find_peaks, peak_widths

from hermespy.core import DenseSignal, Signal, VAT, SerializationProcess, DeserializationProcess
from ..physical_device import LeakageCalibrationBase, PhysicalDevice
from .delay import DelayCalibration

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SelectiveLeakageCalibration(LeakageCalibrationBase):
    """Calibration of a frequency-selective leakage model."""

    __DEFAULT_DELAY: float = 0.0

    __leakage_response: np.ndarray  # Impulse response of the leakage model
    __sampling_rate: float  # Sampling rate of the leakage model
    __delay: float  # Implicit delay of the leakage model

    def __init__(
        self,
        leakage_response: np.ndarray,
        sampling_rate: float,
        delay: float = __DEFAULT_DELAY,
        physical_device: PhysicalDevice | None = None,
    ) -> None:
        """
        Args:

            leakage_response:
                The leakage impulse response matrix.

            sampling_rate:
                The sampling rate of the leakage model in Hz.

            delay:
                The implicit delay of the leakage model in seconds.
                Defaults to zero.
        """

        if leakage_response.ndim != 3:
            raise ValueError(
                f"Leakage response matrix must be a three-dimensional array (has {leakage_response.ndim} dimensions)"
            )

        if sampling_rate <= 0.0:
            raise ValueError(f"Sampling rate must be non-negative (not {sampling_rate} Hz)")

        if delay < 0.0:
            raise ValueError(f"Delay must be non-negative (not {delay} seconds)")

        # Initialize base class
        LeakageCalibrationBase.__init__(self, physical_device)

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

    @override
    def predict_leakage(self, transmitted_signal: Signal) -> Signal:
        if transmitted_signal.num_streams != self.leakage_response.shape[1]:
            raise ValueError(
                f"Transmitted signal has unxpected number of streams ({transmitted_signal.num_streams} instead of {self.leakage_response.shape[1]})"
            )

        predicted_signal_samples = np.zeros(
            (
                self.leakage_response.shape[0],
                transmitted_signal.num_samples + self.leakage_response.shape[2] - 1,
            ),
            dtype=np.complex128,
        )
        for m, n in np.ndindex(self.leakage_response.shape[0], transmitted_signal.num_streams):
            # The leaked signal is the convolution of the transmitted signal with the leakage response
            predicted_signal_samples[m, :] += convolve(
                self.__leakage_response[m, n, :],
                transmitted_signal[n : n + 1, :].view(np.ndarray).flatten(),
            )

        return DenseSignal.FromNDArray(
            predicted_signal_samples,
            transmitted_signal.sampling_rate,
            transmitted_signal.carrier_frequency,
            transmitted_signal.delay + self.delay,
        )

    @override
    def remove_leakage(self, transmitted_signal: Signal, received_signal: Signal) -> Signal:
        if transmitted_signal.num_streams != self.leakage_response.shape[1]:
            raise ValueError(
                f"Transmitted signal has unxpected number of streams ({transmitted_signal.num_streams} instead of {self.leakage_response.shape[1]})"
            )

        if received_signal.num_streams != self.leakage_response.shape[0]:
            raise ValueError(
                f"Received signal has unxpected number of streams ({received_signal.num_streams} instead of {self.leakage_response.shape[0]})"
            )

        if transmitted_signal.sampling_rate != received_signal.sampling_rate:
            raise ValueError(
                f"Transmitted and received signal must have the same sampling rate ({transmitted_signal.sampling_rate} != {received_signal.sampling_rate})"
            )

        if transmitted_signal.carrier_frequency != received_signal.carrier_frequency:
            raise ValueError(
                f"Transmitted and received signal must have the same carrier frequency ({transmitted_signal.carrier_frequency} != {received_signal.carrier_frequency})"
            )

        # Compute the delay correction
        delay_correction = transmitted_signal.delay + received_signal.delay

        # The received signal is corrected by subtracting the leaked samples
        corrected_signal = received_signal.copy()

        # If nothing has been transmitted, abort
        if transmitted_signal.num_samples < 1:
            return corrected_signal

        # Compute the implicit delay shift in samples
        delay_sample_shift = round((self.delay - delay_correction) * received_signal.sampling_rate)

        for m, n in np.ndindex(received_signal.num_streams, transmitted_signal.num_streams):
            # The leaked signal is the convolution of the transmitted signal with the leakage response
            predicted_siso_signal = convolve(
                self.__leakage_response[m, n, :],
                transmitted_signal[n : n + 1, :].view(np.ndarray).flatten(),
            )

            # The correction is achieved by subtracting the leaked signal from the received signal
            if delay_sample_shift >= 0:
                leak_range = min(
                    delay_sample_shift + len(predicted_siso_signal), corrected_signal.num_samples
                )
                min_num_samples = min(
                    corrected_signal.num_samples - delay_sample_shift, len(predicted_siso_signal)
                )
                corrected_signal[m : m + 1, delay_sample_shift:leak_range] = (
                    corrected_signal[m : m + 1, delay_sample_shift:leak_range]
                    - predicted_siso_signal[:min_num_samples]
                )

            else:
                leak_range = min(
                    delay_sample_shift + len(predicted_siso_signal),
                    corrected_signal.num_samples + delay_sample_shift,
                )
                min_num_samples = min(corrected_signal.num_samples, len(predicted_siso_signal))
                corrected_signal[m : m + 1, 0:leak_range] = (
                    corrected_signal[m : m + 1, :leak_range]
                    - predicted_siso_signal[-delay_sample_shift:min_num_samples]
                )

        return corrected_signal

    def plot(self) -> tuple[plt.Figure, VAT]:
        """Plot the leakage response in the time and frequency domain."""

        num_tx = self.__leakage_response.shape[1]
        num_rx = self.__leakage_response.shape[0]
        figure, axes = plt.subplots(num_rx, num_tx, squeeze=False)

        max_amplitude = 0.0
        sample_instances = np.arange(self.__leakage_response.shape[2]) / self.__sampling_rate
        for m, n in np.ndindex(num_rx, num_tx):
            # frequency_bins = fftshift(fftfreq(self.__leakage_response.shape[2]))

            samples = np.abs(self.__leakage_response[m, n, :])
            max_amplitude = max(samples.max(), max_amplitude)
            axes[m, n].plot(sample_instances, samples)
            # freq_axes.plot(
            #    frequency_bins,
            #    fftshift(abs(fft(self.__leakage_response[m, n, :]))),
            #    label=f"Tx: {n} Rx{m}",
            # )

            axes[m, n].set_xlabel("Time [s]")
            axes[m, n].set_ylabel("Absolute Amplitude")
            # freq_axes.set_xlabel("Frequency [Hz]")

        # Set the y-axis limits to the maximum amplitude
        for ax in axes.flat:
            ax.set_ylim(0, max_amplitude)

        return figure, axes

    def estimate_delay(self) -> DelayCalibration:
        """Estimate the delay of the leakage model.

        Returns:
            The delay of the leakage model in seconds.
        """

        # The delay is estimated by finding the maximum of the absolute value of the leakage response
        delay = float(np.argmax(np.abs(self.leakage_response)) / self.sampling_rate + self.delay)

        return DelayCalibration(delay)

    @staticmethod
    def __probe_leakage(
        device: PhysicalDevice, num_probes: int, num_wavelet_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if num_probes < 1:
            raise ValueError(f"Number of probes must be greater than zero (not {num_probes})")

        if num_wavelet_samples < 1:
            raise ValueError(
                f"Number of samples must be greater than zero (not {num_wavelet_samples})"
            )

        # Generate zadoff-chu sequences to probe the device leakage
        cf = num_wavelet_samples % 2
        q = 1
        sample_indices = np.arange(num_wavelet_samples)
        probe_indices = np.arange(1, 1 + num_probes)
        zadoff_chu_sequences = np.exp(
            -1j
            * np.pi
            * np.outer(probe_indices, sample_indices * (sample_indices + cf + 2 * q))
            / num_wavelet_samples
        )

        # Replicate the (periodic) ZC waveform such that any window of
        # num_wavelet_samples will always contain an entire ZC sequence
        # (time-shifted). Essentially, build a huge CP and CS around the center
        # ZC sequence. At the receiver, we will focus on receiving the second
        # (i.e. center) ZC waveform.
        probing_waveforms = np.tile(zadoff_chu_sequences, (1, 3))
        probing_frequencies = np.asarray(
            fft(zadoff_chu_sequences, axis=1, norm="ortho"), np.complex128
        )
        num_samples = probing_waveforms.shape[1]

        # Collect received samples
        received_waveforms = np.zeros(
            (
                num_probes,
                device.num_receive_rf_ports,
                device.num_transmit_rf_ports,
                num_wavelet_samples,
            ),
            dtype=np.complex128,
        )
        for p, n in np.ndindex(num_probes, device.num_transmit_rf_ports):
            tx_samples = np.zeros((device.num_transmit_rf_ports, num_samples), dtype=np.complex128)
            tx_samples[n, :] = probing_waveforms[p, :]
            tx_signal = Signal.Create(
                tx_samples,
                sampling_rate=device.sampling_rate,
                carrier_frequency=device.carrier_frequency,
            )

            rx_signal = device.trigger_direct(tx_signal, calibrate=False)
            # From the received signal, collect the middle num_wavelet_samples of the transmitted
            # ZC sequences. This should account for any delays of the transmission, as long as the
            # sequence is long enough. If it's not long enough, the leakage calculation will fail.
            # TODO: look at the estimated delay and its reliability. If the delay cannot be estimated
            # reliably, most probably, the TX signal was not received in the window decided here
            start = num_wavelet_samples
            received_waveforms[p, :, n, :] = rx_signal[
                :, start : start + num_wavelet_samples
            ].to_dense()

        # Compute received frequency spectra
        received_frequencies = np.asarray(
            fft(received_waveforms, axis=3, norm="ortho"), np.complex128
        )

        # Return the collected probing and received frequency spectra
        return probing_frequencies, received_frequencies

    @staticmethod
    def LeastSquaresEstimate(
        device: PhysicalDevice,
        num_probes: int = 7,
        num_wavelet_samples: int = 4673,
        configure_device: bool = True,
        filter_calibration: bool = True,
    ) -> SelectiveLeakageCalibration:
        """Estimate the transmit-receive leakage for a physical device using Leat-Squares estimation.

        Args:

            device:
                Physical device to estimate the covariance matrix for.

            num_probes:
                Number of probings transmitted to estimate the covariance matrix.
                :math:`7` by default.

            num_wavelet_samples:
                Number of samples transmitted per probing to estimate the covariance matrix.
                :math:`4673` by default.

            configure_device:
                Configure the specified device by the estimated leakage calibration.
                Enabled by default.

            filter_calibration:
                Filter the estimated calibration to consider only prominent peaks.
                Enabled by default.

        Returns: The initialized :class:`SelectiveLeakageCalibration` instance.

        Raises:

            ValueError: If the number of probes is not strictly positive.
            ValueError: If the number of samples is not strictly positive.
        """

        # Probe the device leakage
        probing_frequencies, received_frequencies = SelectiveLeakageCalibration.__probe_leakage(
            device, num_probes, num_wavelet_samples
        )
        num_samples = probing_frequencies.shape[1]

        estimated_frequency_response = np.zeros(
            (device.num_receive_rf_ports, device.num_transmit_rf_ports, num_samples),
            dtype=np.complex128,
        )
        for m, n in np.ndindex(device.num_receive_rf_ports, device.num_transmit_rf_ports):
            # Select the transmitted and received frequency spectra for the current antenna pairs
            rx_frequencies = received_frequencies[:, m, n, :]
            tx_frequencies = probing_frequencies[:, :]

            # Estimate the frequency-selectivity by least-squares estimation
            Rx = rx_frequencies
            Tx = tx_frequencies

            # Solve for X (i.e. the channel frequency response) in the least-squares sense:
            # Minimize \|Rx - X*Tx\|^2 under the constraint that X is diagonal.
            # See https://math.stackexchange.com/a/3502842/397295
            # results in this expression:
            # x_ls = np.diag(Tx.conj().T.dot(Rx)) / np.diag(Tx.conj().T.dot(Tx))

            # optimized version:
            x_ls = np.sum(Tx.conj() * Rx, axis=0) / np.sum(Tx.conj() * Tx, axis=0)

            estimated_frequency_response[m, n, :] = x_ls

        # Convert the estimated leakage-response into the time-domain
        leakage_response = np.asarray(
            ifft(estimated_frequency_response, norm="backward"), np.complex128
        )

        # Only consider the leakage response around the highest peak
        if filter_calibration:
            summed_response = np.sum(abs(leakage_response), axis=(0, 1), keepdims=False)
            peaks: np.ndarray = find_peaks(summed_response, 0.25 * summed_response.max())[0]
            if peaks.size != 0:
                widths: np.ndarray = np.ceil(
                    peak_widths(summed_response, peaks, rel_height=0.75)[0]
                )
                new_leakage_response = np.zeros_like(leakage_response)
                for p, w in zip(peaks, widths):
                    p_start = max(0, int(p - w // 2))
                    p_end = min(leakage_response.shape[2], 1 + int(p + w // 2))
                    new_leakage_response[:, :, p_start:p_end] = leakage_response[
                        :, :, p_start:p_end
                    ]
                leakage_response = new_leakage_response

        calibration = SelectiveLeakageCalibration(
            leakage_response, device.sampling_rate, device.delay_calibration.delay
        )

        # Configure the device with the estimated calibration if the respective flag is enabled
        if configure_device:
            device.leakage_calibration = calibration

        return calibration

    @staticmethod
    def MMSEEstimate(
        device: PhysicalDevice,
        num_probes: int = 7,
        num_wavelet_samples: int = 127,
        noise_power: np.ndarray | None = None,
        configure_device: bool = True,
    ) -> SelectiveLeakageCalibration:
        """Estimate the transmit receive leakage for a physical device using Minimum Mean Square Error (MMSE) estimation.

        Args:

            device:
                Physical device to estimate the covariance matrix for.

            num_probes:
                Number of probings transmitted to estimate the covariance matrix.
                :math:`7` by default.

            num_wavelet_samples:
                Number of samples transmitted per probing to estimate the covariance matrix.
                :math:`127` by default.

            noise_power:
                Noise power at the receiving antennas.
                If not specified, the device's noise power configuration will be assumed or estimated on-the-fly.

            configure_device:
                Configure the specified device by the estimated leakage calibration.
                Enabled by default.

        Returns: The initialized :class:`SelectiveLeakageCalibration` instance.

        Raises:

            ValueError: If the number of probes is not strictly positive.
            ValueError: If the number of samples is not strictly positive.
            ValueError: If the noise power is negative.
        """

        # Probe the device leakage
        probing_frequencies, received_frequencies = SelectiveLeakageCalibration.__probe_leakage(
            device, num_probes, num_wavelet_samples
        )
        num_samples = probing_frequencies.shape[1]

        # Estimate noise power if not specified
        if noise_power is None:
            if device.noise_power is None:
                noise_power = device.estimate_noise_power()

            else:
                noise_power = device.noise_power

        if np.any(noise_power < 0.0):
            raise ValueError(f"Noise power must be non-negative (not {noise_power})")

        if noise_power.ndim != 1 or noise_power.shape[0] != device.num_transmit_rf_ports:
            raise ValueError("Noise power has invalid dimensions")

        # Estimate frequency spectra via MMSE estimation
        # https://nowak.ece.wisc.edu/ece830/ece830_fall11_lecture20.pdf
        # Page 2 Example 1

        h = np.zeros((num_samples * num_probes, num_samples), dtype=np.complex128)
        for p, probe in enumerate(probing_frequencies):
            h[p * num_samples : (1 + p) * num_samples, :] = np.diag(probe)

        # For now, the noise power is assumed to be the mean over all receive chains
        mean_noise_power = np.mean(noise_power)

        # Compute the pseudo-inverse of the received frequency spectra by singular value decomposition
        u, s, vh = svd(
            h @ h.T.conj()
            + mean_noise_power * np.eye(num_samples * num_probes, num_samples * num_probes),
            full_matrices=False,
            hermitian=True,
        )
        u_select = u[:, :num_samples]
        s_select = s[:num_samples]
        vh_select = vh[:num_samples, :]

        mmse_estimator = h.T.conj() @ vh_select.T.conj() @ np.diag(1 / s_select) @ u_select.T.conj()

        # Estimate the frequency spectra for each antenna probing independently
        mmse_frequency_selectivity_estimation = np.zeros(
            (device.num_receive_rf_ports, device.num_transmit_rf_ports, num_samples),
            dtype=np.complex128,
        )
        for m, n in np.ndindex(device.num_receive_rf_ports, device.num_transmit_rf_ports):
            probing_estimation = mmse_estimator @ received_frequencies[:, m, n, :].flatten()
            mmse_frequency_selectivity_estimation[m, n, :] = probing_estimation

        # Initialize the calibration from the estimated frequency spectra
        leakage_response = np.asarray(
            ifft(mmse_frequency_selectivity_estimation, axis=2), np.complex128
        )[:, :, :num_wavelet_samples]
        calibration = SelectiveLeakageCalibration(
            leakage_response, device.sampling_rate, device.delay_calibration.delay
        )

        # Configure the device with the estimated calibration if the respective flag is enabled
        if configure_device:
            device.leakage_calibration = calibration

        return calibration

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_array(self.leakage_response, "leakage_response")
        process.serialize_floating(self.sampling_rate, "sampling_rate")
        process.serialize_floating(self.delay, "delay")

    @override
    @classmethod
    def Deserialize(
        cls: Type[SelectiveLeakageCalibration], process: DeserializationProcess
    ) -> SelectiveLeakageCalibration:
        return cls(
            process.deserialize_array("leakage_response", np.complex128),
            process.deserialize_floating("sampling_rate"),
            process.deserialize_floating("delay", cls.__DEFAULT_DELAY),
        )
