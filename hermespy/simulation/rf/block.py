# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from functools import cache
from typing import Generic, Iterable, SupportsInt, SupportsIndex, TypeVar

import numpy as np
from scipy.signal import butter, sosfiltfilt

from hermespy.core import RandomNode, Serializable, SerializableEnum
from .signal import RFSignal
from .noise import AWGN, NoiseLevel, NoiseModel, NoiseRealization, N0

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RFBlockRealization(object):
    """Base class for the realiztation of a radio-frequency block."""

    __bandwidth: float
    __oversampling_factor: int
    __noise_realization: NoiseRealization

    def __init__(
        self, bandwidth: float, oversampling_factor: int, noise_realization: NoiseRealization
    ) -> None:
        """
        Args:
            sampling_rate: Sampling rate of the block in Hz.
            oversampling_factor: Oversampling factor of the modeling in Hz.
            noise_realization: Noise realization of the block.
        """

        # Store attributes
        self.__bandwidth = bandwidth
        self.__oversampling_factor = oversampling_factor
        self.__sampling_rate = bandwidth * oversampling_factor
        self.__noise_realization = noise_realization

    @property
    def bandwidth(self) -> float:
        """Bandwidth of the simulated signals in Hz."""

        return self.__bandwidth

    @property
    def oversampling_factor(self) -> int:
        """Oversampling factor of the modeling in Hz."""
        return self.__oversampling_factor

    @property
    def sampling_rate(self) -> float:
        """Sampling rate of the block in Hz.

        Equivalent to the bandwidth multiplied by the oversampling factor.
        """

        return self.__sampling_rate

    @property
    def noise_realization(self) -> NoiseRealization:
        """Noise realization of the block."""

        return self.__noise_realization


RFBRT = TypeVar("RFBRT", bound=RFBlockRealization)


class RFBlock(ABC, Generic[RFBRT], RandomNode, Serializable):
    """Base class of a single block within physical models of radio-frequency chains."""

    __noise_model: NoiseModel | None
    __noise_level: NoiseLevel

    def __init__(
        self,
        noise_model: NoiseModel | None = None,
        noise_level: NoiseLevel | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            noise_model:
                Assumed noise model of the block.
                If not specified, i.e. :py:obj:`None`, additive white Gaussian noise will be assumed.
            noise_level:
                Assumed noise level of the block.
                If not specified, i.e. :py:obj:`None`, no noise is assumed.
            seed: Seed with which to initialize the block's random state.
        """

        # Init base class
        RandomNode.__init__(self, seed=seed)

        # Initialize class attributes
        self.__noise_model = noise_model if noise_model is not None else AWGN()
        self.__noise_model.random_mother = self
        self.__noise_level = noise_level if noise_level is not None else N0(0.0)

    @property
    def noise_model(self) -> NoiseModel:
        """This block's assumed noise model."""

        return self.__noise_model

    @property
    def noise_level(self) -> NoiseLevel:
        """This block's assumed noise level."""

        return self.__noise_level

    @property
    @abstractmethod
    def num_input_ports(self) -> int:
        """Number of physical ports feeding into this radio-frequency block.

        If the returned number is negative, the block can accept an arbitrary number of input ports.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def num_output_ports(self) -> int:
        """Number of physical ports emerging from this radio-frequency block."""
        ...  # pragma: no cover

    @abstractmethod
    def realize(
        self, bandwidth: float, oversampling_factor: int, carrier_frequency: float
    ) -> RFBRT:
        """Return the current state of the radio-frequency block.

        Args:
            bandwidth: Bandwith of the proecessed signals in Hz.
            oversampling_factor: Oversampling factor of the modeling.
            carrier_frequency: Target carrier frequency of the modeled radio front-end in Hz.

        Returns:
            Current state of the radio-frequency block.
        """
        ...  # pragma: no cover

    @cache
    @staticmethod
    def _antialiasing_filter(bandwidth: float, oversampling_factor: int) -> np.ndarray:
        """Return the coefficients of an anti-aliasing filter for the given bandwidth and oversampling factor.

        Args:
            bandwidth: Bandwidth of the block in Hz.
            oversampling_factor: Oversampling factor of the block.

        Returns:
            Second-order sections of the anti-aliasing filter.
        """

        min_cutoff = 1 / oversampling_factor
        max_cutoff = 0.5
        ideal_cutoff = 0.5 * (min_cutoff + max_cutoff)
        cutoff = max(min_cutoff, min(max_cutoff, ideal_cutoff))

        return butter(16, cutoff, output="sos")

    def propagate(self, realization: RFBRT, input: RFSignal, filter: bool = True) -> RFSignal:
        """Propagate the input signals through the radio-frequency block.

        Args:
            realization: Current state of the radio-frequency block.
            inputs: List of input signals to propagate through the block.
            filter:
                Whether to apply an anti-aliasing filter after propagation.
                Only relevant if the propagated signal is oversampled, i.e. `oversampling_factor` > 1.
                Enabled by default.

        Raises:
            ValueError: If the number of input signals does not match the :attr:`.num_input_ports`.

        Returns: Propagated signal block containing :attr:`.num_output_ports` individual signal streams.
        """

        if input.num_streams != self.num_input_ports:
            raise ValueError(
                f"Expected {self.num_input_ports} input streams, got {input.num_streams}"
            )

        # Propagate the input signals through the block
        propagated_signal = self._propagate(realization, input)

        # Add noise to the propagated signal
        noisy_propagated_signal = realization.noise_realization.add_to(propagated_signal)

        # Apply an anti-aliasing filter if oversampling is used
        if (
            filter
            and realization.oversampling_factor > 1
            and noisy_propagated_signal.num_samples > 51
        ):
            filtered_output = sosfiltfilt(
                RFBlock._antialiasing_filter(
                    realization.bandwidth, realization.oversampling_factor
                ),
                noisy_propagated_signal,
                axis=1,
            )
            noisy_propagated_signal = RFSignal(
                propagated_signal.num_streams,
                propagated_signal.num_samples,
                realization.sampling_rate,
                noisy_propagated_signal.carrier_frequencies,
                noisy_propagated_signal.noise_powers,
                noisy_propagated_signal.delay,
                filtered_output.tobytes(),
            )

        return noisy_propagated_signal

    @abstractmethod
    def _propagate(self, realization: RFBRT, input: RFSignal) -> RFSignal:
        """Propagate the input signals through the radio-frequency block.

        Args:
            realization: Current state of the radio-frequency block.
            input: Signal block containing input signals to propagate through the block.

        Returns:
            Propagated signal block containing :attr:`.num_output_ports` individual signal streams.
        """
        ...  # pragma: no cover


RFBT = TypeVar("RFBT", bound=RFBlock)
"""Type variable for radio-frequency blocks."""


class DSPInputBlock(RFBlock):
    """Base class for radio-frequency blocks representing ports connecting transmitting DSP layers to the radio-frequency chain.

    RF blocks inheriting from this class will automatically be represented as ports in the RF chain.
    """

    ...  # pragma: no cover


class DSPOutputBlock(RFBlock):
    """Base class for radio-frequency blocks representing ports connecting receiving DSP layers to the radio-frequency chain.

    RF blocks inheriting from this class will automatically be represented as ports in the RF chain.
    """

    ...  # pragma: no cover


class RFBlockPortType(SerializableEnum):
    """Enumeration of the types of ports in a radio-frequency block."""

    IN = 0
    """Input port of a radio-frequency block."""

    OUT = 1
    """Output port of a radio-frequency block."""


class RFBlockPort(Generic[RFBT]):
    """Representation of a single port of a radio-frequency block."""

    __block: RFBT
    __port_inidices: list[int]

    def __init__(
        self, block: RFBT, port_indices: SupportsInt | Iterable[int], port_type: RFBlockPortType
    ) -> None:
        """
        Args:
            block: Block instance this port belongs to.
            port_indices: Integer indices of the represented port or sequence of ports.
            port_type: Type of the port, either input or output.
        """

        # Store attributes
        self.__port = block
        self.__port_indices = (
            [int(port_indices)] if isinstance(port_indices, SupportsInt) else list(port_indices)
        )
        self.__port_type = port_type

    @property
    def block(self) -> RFBT:
        """Reference to the block instance this port belongs to."""

        return self.__port

    @property
    def port_indices(self) -> list[int]:
        """Indices of the represented port or sequence of ports."""

        return self.__port_indices

    @property
    def num_ports(self) -> int:
        """Number of ports represented by this port instance."""

        return len(self.__port_indices)

    @property
    def port_type(self) -> RFBlockPortType:
        """Type of the port, either input or output."""

        return self.__port_type

    def __getitem__(self, index: SupportsIndex | slice, /) -> RFBlockPort[RFBT]:
        """Select a single port or a range of ports from the block.

        Args:
            index: Index of the port to select or a slice of ports to select.

        Returns: A new port object representing the selected port(s).
        """

        # Select the port index subset
        selected_port_indices = self.port_indices[index]

        # Return a new port instance representing the selected port(s)
        return RFBlockPort(self.block, selected_port_indices, self.port_type)
