# -*- coding: utf-8 -*-
"""HermesPy Signal Model."""

from __future__ import annotations

import numpy as np

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Signal:
    """Base class of signal models in HermesPy.

    Attributes:

        __samples (np.ndarray):
            An MxT matrix containing uniformly sampled base-band-samples of the modeled signal.
            M is the number of individual streams, T the number of available samples.

        __sampling_rate (float):
            Sampling rate of the modeled signal in Hz (in the base-band).

        __carrier_frequency (float):
            Carrier-frequency of the modeled signal in the radio-frequency band,
            i.e. the central frequency in Hz.

        delay (float):
            Delay of the signal in seconds.
    """

    __samples: np.ndarray
    __sampling_rate: float
    __carrier_frequency: float
    delay: float

    def __init__(self,
                 samples: np.ndarray,
                 sampling_rate: float,
                 carrier_frequency: float = 0.,
                 delay: float = 0.) -> None:
        """Signal model initialization.

        Args:
            samples (np.ndarray):
                An MxT matrix containing uniformly sampled base-band-samples of the modeled signal.
                M is the number of individual streams, T the number of available samples.

            sampling_rate (float):
                Sampling rate of the modeled signal in Hz (in the base-band).

            carrier_frequency (float, optional):
                Carrier-frequency of the modeled signal in the radio-frequency band,
                i.e. the central frequency in Hz.
                Zero by default.

            delay (float, optional):
                Delay of the signal in seconds.
                Zero by default.
        """

        self.samples = samples
        self.sampling_rate = sampling_rate
        self.carrier_frequency = carrier_frequency
        self.delay = delay

    @property
    def samples(self) -> np.ndarray:
        """Uniformly sampled c

        Returns:
            np.ndarray:
                An MxT matrix of samples,
                where M is the number of individual streams and T the number of samples.
        """

        return self.__samples

    @samples.setter
    def samples(self, value: np.ndarray) -> None:
        """Modify the base-band samples of the modeled signal.

        Args:
            value (np.ndarray):
                An MxT matrix of samples,
                where M is the number of individual streams and T the number of samples.

        Raises:
            ValueError: If `value` can't be interpreted as a matrix.
        """

        if value.ndim != 2:
            raise ValueError("Signal model samples must be a matrix (a two-dimensional array)")

        self.__samples = value

    @property
    def num_streams(self) -> int:
        """The number of streams within this signal model.

        Returns:
            int: The number of streams.
        """

        return self.__samples.shape[0]

    @property
    def num_samples(self) -> int:
        """The number of samples within this signal model.

        Returns:
            int: The number of samples.
        """

        return self.__samples.shape[1]

    @property
    def sampling_rate(self) -> float:
        """The rate at which the modeled signal was sampled.

        Returns:
            float: The sampling rate in Hz.
        """

        return self.__sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:
        """Modify the rate at which the modeled signal was sampled.

        Args:
            value (float): The sampling rate in Hz.

        Raises:
            ValueError: If `value` is smaller or equal to zero.
        """

        if value <= 0.:
            raise ValueError("The sampling rate of modeled signals must be greater than zero")

        self.__sampling_rate = value

    @property
    def carrier_frequency(self) -> float:
        """The center frequency of the modeled signal in the radio-frequency transmit band.

        Returns:
            float: The carrier frequency in Hz.
        """

        return self.__carrier_frequency

    @carrier_frequency.setter
    def carrier_frequency(self, value: float) -> None:
        """Modify the center frequency of the modeled signal in the radio-frequency transmit band.

        Args:
            value (float): he carrier frequency in Hz.

        Raises:
            ValueError: If `value` is smaller than zero.
        """

        if value < 0.:
            raise ValueError("The carrier frequency of modeled signals must be greater or equal to zero")

        self.__carrier_frequency = value
