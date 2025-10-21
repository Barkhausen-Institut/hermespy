# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Literal, Sequence
from typing_extensions import override

import numpy as np

from hermespy.core import DenseSignal, SerializationProcess, DeserializationProcess
from hermespy.core.signal_model import Buffer

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RFSignal(DenseSignal):
    """An extension of general dense signals for propagation through RF chains.

    It allows for diverging carrier frequencies and noise powers for each represented stream.
    """

    __carrier_frequencies: np.ndarray[tuple[int], np.dtype[np.float64]]
    __noise_powers: np.ndarray[tuple[int], np.dtype[np.float64]]

    def __new__(
        cls,
        num_streams: int,
        num_samples: int,
        sampling_rate: float,
        carrier_frequencies: np.ndarray | None = None,
        noise_powers: np.ndarray | None = None,
        delay: float = 0.0,
        buffer: Buffer | None = None,
    ):
        """
        Args:
            num_streams: Number of streams in the signal.
            num_samples: Number of samples per stream.
            sampling_rate: Sampling rate of the signal in Hz.
            carrier_frequencies:
                Carrier frequencies for each stream in Hz.
                If specified, must be of size `num_streams`.
            noise_powers:
                Noise power for each stream in Watts.
                If specified, must be of size `num_streams`.
            delay: Delay of the signal in seconds.
            buffer: Optional buffer to use for the signal data.
        """

        _carrier_frequencies = (
            np.zeros(num_streams, dtype=np.float64)
            if carrier_frequencies is None
            else carrier_frequencies.flatten()
        )
        _noise_powers = (
            np.zeros(num_streams, dtype=np.float64)
            if noise_powers is None
            else noise_powers.flatten()
        )

        # Initialize base class
        rf = DenseSignal.__new__(
            cls,
            num_streams,
            num_samples,
            sampling_rate,
            0.0 if _carrier_frequencies.size == 0 else _carrier_frequencies[0],
            0.0 if _noise_powers.size == 0 else _noise_powers[0],
            delay,
            buffer,
        )

        # Initialize additional RFSignal attributes
        rf.carrier_frequencies = _carrier_frequencies
        rf.noise_powers = _noise_powers

        return rf

    @property
    def carrier_frequencies(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Carrier frequencies for each stream in Hz.

        Raises:
            ValueError: If the carrier frequencies array does not match the number of streams.
        """
        return self.__carrier_frequencies

    @carrier_frequencies.setter
    def carrier_frequencies(self, value: np.ndarray[tuple[int], np.dtype[np.float64]]) -> None:
        if value.size != self.num_streams:
            raise ValueError(
                f"Expected carrier frequencies array of size {self.num_streams}, but got {value.size}"
            )
        self.__carrier_frequencies = value

    @property
    def noise_powers(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Noise power for each stream.

        Raises:
            ValueError: If the noise powers array does not match the number of streams.
        """
        return self.__noise_powers

    @noise_powers.setter
    def noise_powers(self, value: np.ndarray[tuple[int], np.dtype[np.float64]]) -> None:
        if value.size != self.num_streams:
            raise ValueError(
                f"Expected noise powers array of size {self.num_streams}, but got {value.size}"
            )
        self.__noise_powers = value

    @override
    def __array_finalize__(self, obj: np.ndarray | RFSignal | None) -> None:
        if obj is None:
            return  # pragma: no cover

        # Finalize base class
        DenseSignal.__array_finalize__(self, obj)

        # Recover additional attributes lost in numpy view/copy operations
        if hasattr(obj, "carrier_frequencies"):
            self.carrier_frequencies = obj.carrier_frequencies[:self.num_streams].copy()  # type: ignore
        else:
            self.carrier_frequencies = np.full(
                (self.num_streams,), self.carrier_frequency, np.float64
            )
        if hasattr(obj, "noise_powers"):
            self.noise_powers = obj.noise_powers[:self.num_streams].copy()  # type: ignore
        else:
            self.noise_powers = np.full((self.num_streams,), self.noise_power, np.float64)

    @override
    def __getitem__(self, key) -> RFSignal:

        # Slice the data using the base class method
        key = self._correct_key_slicing(key)
        sliced_rf_signal = DenseSignal.__getitem__(self, key).view(RFSignal)

        # Properly udpdate the additional attributes by applying the first dimension
        # slicing to the carrier frequencies and noise powers
        if isinstance(key, Sequence):
            stream_selector = key[0]

            sliced_rf_signal.carrier_frequencies = self.carrier_frequencies[stream_selector].flatten().copy()
            sliced_rf_signal.noise_powers = self.noise_powers[stream_selector].flatten().copy()

        return sliced_rf_signal

    @classmethod
    @override
    def FromNDArray(
        cls,
        array: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
        sampling_rate: float,
        carrier_frequency: float | np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        noise_power: float | np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        delay: float = 0.0,
    ) -> RFSignal:
        """Create a new RF signal from a given numpy ndarray.

        Args:
            array: Array containing the signal samples.
                Must be of shape (num_streams, num_samples).
            sampling_rate: Sampling rate of the signal in Hz.
            carrier_frequency:
                Carrier frequencies for each stream in Hz.
                If specified, must be of size `num_streams`.
            noise_power:
                Noise power for each stream in Watts.
                If specified, must be of size `num_streams`.
            delay: Delay of the signal in seconds.

        Returns:
            The created RF signal.
        """

        # Assert array is 2D
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D ndarray, but got {array.ndim}D")

        # Cast ndarray to RFSignal
        rf_signal = array.astype(np.complex128).view(RFSignal)

        # Set additional attributes
        rf_signal.sampling_rate = sampling_rate
        rf_signal.delay = delay

        if carrier_frequency is not None:
            if isinstance(carrier_frequency, float):
                rf_signal.carrier_frequencies = np.full(
                    (rf_signal.num_streams,), carrier_frequency, dtype=np.float64
                )
            elif carrier_frequency.size == rf_signal.num_streams:
                rf_signal.carrier_frequencies = carrier_frequency.flatten()

        if noise_power is not None:
            if isinstance(noise_power, float):
                rf_signal.noise_powers = np.full(
                    (rf_signal.num_streams,), noise_power, dtype=np.float64
                )
            elif noise_power.size == rf_signal.num_streams:
                rf_signal.noise_powers = noise_power.flatten()

        return rf_signal

    @staticmethod
    def FromDense(signal: DenseSignal) -> RFSignal:
        """Create a new RF signal from a given dense signal.

        Args:
            signal: Dense signal to convert.

        Returns:
            The created RF signal.
        """

        # Cast DenseSignal to RFSignal
        rf_signal = signal.view(RFSignal)

        # Set additional attributes
        rf_signal.carrier_frequencies = np.full(
            rf_signal.num_streams, signal.carrier_frequency, np.float64
        )
        rf_signal.noise_powers = np.full(rf_signal.num_streams, signal.noise_power, np.float64)

        return rf_signal

    @override
    def __setitem__(self, key, value) -> None:
        _key = (key, slice(None)) if isinstance(key, (int, slice)) else key

        # Set the sliced data
        DenseSignal.__setitem__(self, _key, value.view(np.ndarray))

        # Update the carrier frequencies and noise powers
        if isinstance(value, RFSignal):
            selector = [_key[0]] if isinstance(_key[0], int) else _key[0]
            self.carrier_frequencies[selector] = value.carrier_frequencies
            self.noise_powers[selector] = value.noise_powers

            if np.all(self.carrier_frequencies == self.carrier_frequencies[0]):
                self.carrier_frequency = self.carrier_frequencies[0]
            if np.all(self.noise_powers == self.noise_powers[0]):
                self.noise_power = self.noise_powers[0]

    @override
    def resample(self, sampling_rate: float, aliasing_filter: bool = True) -> RFSignal:
        # Do nothing if the expected sampling rate is already available
        if self.sampling_rate == sampling_rate:
            return self

        resampled_block = self.resample_block(self.sampling_rate, sampling_rate, aliasing_filter)
        return RFSignal(
            self.num_streams,
            resampled_block.num_samples,
            sampling_rate,
            self.carrier_frequencies.copy(),
            self.noise_powers.copy(),
            self.delay,
            resampled_block.tobytes(),
        )

    @override
    def copy(self, order: Literal["K", "A", "C", "F"] | None = None) -> RFSignal:
        return RFSignal(
            self.num_streams,
            self.num_samples,
            self.sampling_rate,
            self.carrier_frequencies.copy(),
            self.noise_powers.copy(),
            self.delay,
            bytearray(self),
        )

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_array(self.view(np.ndarray), "samples")
        process.serialize_floating(self.sampling_rate, "sampling_rate")
        process.serialize_array(self.carrier_frequencies, "carrier_frequencies")
        process.serialize_array(self.noise_powers, "noise_powers")
        process.serialize_floating(self.delay, "delay")

    @classmethod
    @override
    def Deserialize(cls: type[RFSignal], process: DeserializationProcess) -> RFSignal:
        samples = process.deserialize_array("samples", np.complex128)
        sampling_rate = process.deserialize_floating("sampling_rate")
        carrier_frequencies = process.deserialize_array("carrier_frequencies", dtype=np.float64)
        noise_powers = process.deserialize_array("noise_powers", dtype=np.float64)
        delay = process.deserialize_floating("delay", 0.0)

        return cls(
            samples.shape[0],
            samples.shape[1],
            sampling_rate,
            carrier_frequencies,
            noise_powers,
            delay,
            buffer=samples.tobytes(),
        )
