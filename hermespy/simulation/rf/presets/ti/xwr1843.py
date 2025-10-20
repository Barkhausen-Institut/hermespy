# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

import numpy as np
from scipy.constants import speed_of_light
from scipy.fft import fft2, fftshift
from scipy.signal import resample

from hermespy.core import TransmitState, ReceiveState, DenseSignal, Signal
from hermespy.radar.fmcw import RadarWaveform
from ... import (
    RampGenerator,
    Mixer,
    MixerType,
    OscillatorPhaseNoise,
    Shift,
    Source,
    ADC,
    PowerAmplifier,
    ClippingPowerAmplifier,
    N0,
    RFChain,
    HPF,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TIXWR1843(RadarWaveform, RFChain):
    """Model of the Texas Instruments AWR1843 / IWR1843 radar SoC.

    .. list-table:: Modeled Parameters
       :header-rows: 1
       * - Parameter
         - Value
       * - Phase noise at :math:`10~\\mathrm{MHz}` offset
         - :math:`-93~\\mathrm{dBc/Hz}`
       * - Output Power
         - :math:`12~\\mathrm{dBm}`
       * - ADC sampling rate
         - :math:`25~\\mathrm{MHz}`
       * - ADC resolution
         - :math:`12~\\mathrm{bit}`

    https://www.ti.com/product/AWR1843
    https://www.ti.com/product/IWR1843
    """

    __min_carrier_frequency = 76e9
    __max_carrier_frequency = 81e9
    __max_chirp_slope = 1e8 / 1e-6  # 100 MHz/us
    __num_adc_quantization_bits = 12
    __adc_sampling_rate = 12.5e6
    __tx_power = 12  # dBm per channel
    __rx_if_bandwidth = 10e6  # Hz
    __rx_max_gain = 48  # dB
    __rx_compression_point = -8  # dBm

    __default_num_chirps = 16
    __default_chirp_bandwidth = 3e9  # Hz
    __default_chirp_interval = __default_chirp_bandwidth / __max_chirp_slope  # s
    __default_hpf_cutoff = 175e3  # Hz

    __ramp: RampGenerator

    def __init__(
        self,
        carrier_frequency: float = 77.5e9,
        num_chirps: int = __default_num_chirps,
        chirp_bandwidth: float = __default_chirp_bandwidth,
        chirp_slope: float = __max_chirp_slope,
        chirp_interval: float = __default_chirp_interval,
        hpf_cutoff: float = __default_hpf_cutoff,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            carrier_frequency: Center frequency of the generated signal in Hz.
            num_chirps: Number of chirps in the FMCW ramp.
            chirp_bandwidth: Bandwidth of the chirp in Hz.
            chirp_slope: Chirp slope in Hz/s.
            chirp_interval: Time between two consecutive chirps in seconds.
            hpf_cutoff: Cutoff frequency of the high-pass filter applied to the received signal in Hz.
            seed: Seed with which to initialize the random state of the model.
        """

        # Initialize base class
        RFChain.__init__(self, seed)

        # Initialize parameters
        self.ramp = RampGenerator(
            num_chirps=num_chirps,
            chirp_bandwidth=chirp_bandwidth,
            chirp_slope=chirp_slope,
            chirp_interval=chirp_interval,
            noise_level=N0(0.0),  # No thermal noise in the ramp generator
        )

        output_voltage = (
            50 / 1000 * 10 ** (self.__tx_power / 10)
        ) ** 0.5  # dBm to V_rms given 50 Ohm load
        pa_amplification_factor = 10 ** (30 / 20)  # 30 dB gain of the power amplifier
        source_voltage = output_voltage / pa_amplification_factor

        # Frequency ramp generation
        ramp_ref = self.add_block(self.ramp)
        ramp_mixer_ref = self.new_block(Mixer, mixer_type=MixerType.UP, noise_level=N0(0.0))
        source_ref = self.new_block(
            Source,
            carrier_frequency=carrier_frequency,
            phase_noise=OscillatorPhaseNoise(
                K0=0.0,
                K2=10 ** (-95 / 10) * 1e12,  # 1 MHz offset phase noise, assumed square decay
                K3=0.0,
            ),
            amplitude=source_voltage,
            noise_level=N0(0.0),  # No thermal noise in the frequency source
        )
        self.connect(ramp_ref.port("o"), ramp_mixer_ref.port("i"))
        self.connect(source_ref.port("o"), ramp_mixer_ref.port("lo"))

        # Tx RF chains
        shifter_refs = self.new_blocks(
            Shift, 3, num_quantization_bits=self.__num_adc_quantization_bits, noise_level=N0(0.0)
        )
        pa_refs = self.new_blocks(
            PowerAmplifier, 3, gain=pa_amplification_factor, noise_level=N0(10 ** (-14.5))
        )
        for i in range(3):
            self.connect(ramp_mixer_ref.port("o"), shifter_refs[i].port("i"))
            self.connect(shifter_refs[i].port("o"), pa_refs[i].port("i"))

        # Rx RF chains
        lna_refs = self.new_blocks(ClippingPowerAmplifier, 4, noise_level=N0(0.0))
        mixer_refs = self.new_blocks(Mixer, 4, mixer_type=MixerType.DOWN, noise_level=N0(0.0))
        hpf_refs = self.new_blocks(HPF, 4, cutoff_frequency=hpf_cutoff, noise_level=N0(0.0))
        adc_refs = self.new_blocks(
            block=ADC,
            count=4,
            num_quantization_bits=self.__num_adc_quantization_bits,
            noise_level=N0(0.0),
        )
        for i in range(4):
            self.connect(lna_refs[i].port("o"), mixer_refs[i].port("i"))
            self.connect(mixer_refs[i].port("o"), hpf_refs[i].port("i"))
            self.connect(hpf_refs[i].port("o"), adc_refs[i].port("i"))
            self.connect(mixer_refs[i].port("lo"), ramp_mixer_ref.port("o"))

    @property
    def ramp(self) -> RampGenerator:
        """FMCW ramp generator block model."""

        return self.__ramp

    @ramp.setter
    def ramp(self, value: RampGenerator) -> None:
        self.__ramp = value
        self.__ramp.random_mother = self

    ###########################################################
    # Radar waveform implementation

    @override
    def energy(self, bandwidth: float, oversampling_factor: int) -> float:
        return self.ramp.num_chirps * bandwidth**2 / self.ramp.chirp_slope * oversampling_factor

    @property
    @override
    def power(self) -> float:
        return 1.0

    @override
    def ping(self, state: TransmitState) -> DenseSignal:
        # Since the TI frontend does not posess any DACs, the generated waveform
        # has zero streams.
        return DenseSignal.FromNDArray(
            np.empty(
                (0, int(self.frame_duration(state.bandwidth) * state.sampling_rate)), np.complex128
            ),
            state.sampling_rate,
            0.0,
        )

    @override
    def estimate(self, signal: Signal, state: ReceiveState) -> np.ndarray:

        # Coarsly downsample the given signal to the expected ADC sampling rate
        if signal.sampling_rate > self.__adc_sampling_rate:
            decimation_factor = int(signal.sampling_rate / self.__adc_sampling_rate)
            signal = signal[:, ::decimation_factor]
            signal.sampling_rate = signal.sampling_rate / decimation_factor

        num_chirp_samples = int(self.ramp.chirp_interval * signal.sampling_rate)
        num_frame_samples = num_chirp_samples * self.ramp.num_chirps

        # Resample to match the ADC's specifications
        # ToDo: Decimate here
        resampled_signal = np.asarray(resample(signal, num_frame_samples, axis=1), np.complex128)

        # Jointly perform an FFT over the chirp and time domain
        chirp_stack = resampled_signal.reshape((self.ramp.num_chirps, num_chirp_samples))
        transform = fft2(chirp_stack, norm="forward", overwrite_x=True)
        transform = np.fliplr(fftshift(transform, axes=(0, 1))[:, : num_chirp_samples // 2 + 1])
        return np.abs(transform) ** 2

    @override
    def range_bins(self, bandwidth: float) -> np.ndarray:
        num_range_bins = int(self.ramp.chirp_interval * self.__adc_sampling_rate / 2) + 1
        return np.linspace(0, self.max_range(bandwidth), num_range_bins)

    @override
    def frame_duration(self, bandwidth: float) -> float:
        return self.ramp.num_chirps * self.ramp.chirp_interval

    @override
    def max_range(self, bandwidth: float) -> float:
        return self.__adc_sampling_rate * speed_of_light / (4 * self.ramp.chirp_slope)

    @override
    def range_resolution(self, bandwidth: float) -> float:
        return 2 * self.max_range(bandwidth) / (self.ramp.chirp_interval * bandwidth)

    @property
    @override
    def max_relative_doppler(self) -> float:
        max_doppler = 1 / (4 * self.ramp.chirp_interval)
        return max_doppler

    @property
    @override
    def relative_doppler_resolution(self) -> float:
        resolution = 1 / (2 * self.ramp.num_chirps * self.ramp.chirp_interval)
        return resolution

    @property
    @override
    def relative_doppler_bins(self) -> np.ndarray:
        return (
            np.arange(self.ramp.num_chirps) * self.relative_doppler_resolution
            - self.max_relative_doppler
        )
