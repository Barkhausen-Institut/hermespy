# -*- coding: utf-8 -*-
"""Shaping Filter for Communcation Links."""

from __future__ import annotations
import numpy as np
from enum import Enum
from typing import Type, Union
from ruamel.yaml import SafeConstructor, SafeRepresenter, MappingNode

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronauer@barkhauseninstitut.org"
__status__ = "Prototype"


class ShapingFilter:
    """Implements a shaping/reception filter for a communications link.

    Currently, raised-cosine, root-raised-cosine, rectangular and FMCW
    (frequency modulated continuous wave) filters are implemented.
    An FIR filter with truncated impulse response is created.
    The filter is normalized, i.e., the impulse response has unit energy.

    Attributes:
        samples_per_symbol (int): samples per modulation symbol
        number_of_samples (int): filter length in samples
        delay_in_samples (int): delay introduced by filter
        impulse_response (numpy.array): filter impulse response
    """

    class FilterType(Enum):
        """Type of filter applied to the signal."""

        NONE = 0
        ROOT_RAISED_COSINE = 1
        RAISED_COSINE = 2
        RECTANGULAR = 3
        FMCW = 4

    yaml_tag = u'ShapingFilter'

    def __init__(
            self,
            filter_type: Union[str, FilterType],
            samples_per_symbol: int,
            length_in_symbols: float = 16,
            bandwidth_factor: float = 1.0,
            roll_off: float = 0,
            is_matched: bool = False):
        """
        Creates an object for a transmission/reception filter.

        Args:
            filter_type (str): Determines filter, currently supported:
                - RAISED_COSINE
                - ROOT_RAISED_COSINE
                - RECTANGULAR
                - FMCW
                - NONE
            samples_per_symbol (int): number of samples per modulation symbol.
            length_in_symbols (int): filter length in modulation symbols.
            bandwidth_factor (float): filter bandwidth can be expanded/reduced by this factor
                                     (default = 1), relatively to the symbol rate.
                                     For (root)-raised cosine, the Nyquist symbol rate will be
                                     multiplied by this factor
                                     For rectangular pulses, the pulse width in time will be divided by this factor.
                                     For FMCW, the sweep bandwidth will be given by the symbol rate multiplied by this
                                     factor.
            roll_off (float): Roll off factor between 0 and 1. Only relevant for (root)-raised cosine filters.
            is_matched (bool): if True, then a matched filter is considered.
        """

        if isinstance(filter_type, str):
            filter_type = ShapingFilter.FilterType[filter_type]

        self.filter_type = filter_type
        self.samples_per_symbol = samples_per_symbol
        self.length_in_symbols = length_in_symbols
        self.bandwidth_factor = bandwidth_factor
        self.roll_off = roll_off
        self.is_matched = is_matched

        self.samples_per_symbol = samples_per_symbol
        self.number_of_samples = None
        self.delay_in_samples = None
        self.impulse_response = None

        if filter_type == ShapingFilter.FilterType.NONE:
            self.impulse_response = 1.0
            self.delay_in_samples = 0
            self.number_of_samples = 1

        elif filter_type == ShapingFilter.FilterType.RECTANGULAR:
            self.number_of_samples = np.round(self.samples_per_symbol / bandwidth_factor).astype(int)
            self.delay_in_samples = int(self.number_of_samples / 2)
            self.impulse_response = np.ones(self.number_of_samples)

            if is_matched:
                self.delay_in_samples -= 1

        elif filter_type == ShapingFilter.FilterType.RAISED_COSINE or filter_type == ShapingFilter.FilterType.ROOT_RAISED_COSINE:
            self.number_of_samples = int(
                self.samples_per_symbol * length_in_symbols)
            delay_in_symbols = int(
                np.floor(
                    self.number_of_samples /
                    2) /
                samples_per_symbol)
            self.delay_in_samples = delay_in_symbols * samples_per_symbol

            self.impulse_response = self._get_raised_cosine(
                filter_type, roll_off, bandwidth_factor)

        elif filter_type == ShapingFilter.FilterType.FMCW:
            self.number_of_samples = int(
                np.ceil(
                    self.samples_per_symbol *
                    length_in_symbols))
            self.delay_in_samples = int(self.number_of_samples / 2)

            chirp_slope = bandwidth_factor / length_in_symbols
            self.impulse_response = self._get_fmcw(
                length_in_symbols, chirp_slope)

            if is_matched:
                self.impulse_response = np.flip(np.conj(self.impulse_response))
                self.delay_in_samples -= 1

        else:
            raise ValueError(f"Shaping filter {filter_type} not supported")

        # normalization (filter energy should be equal to one)
        self.impulse_response = self.impulse_response / \
            np.linalg.norm(self.impulse_response)

    def filter(self, input_signal: np.ndarray) -> np.ndarray:
        """Filters the input signal with the shaping filter.

        Args:
            input_signal (np.array): Input signal with N samples to filter.

        Returns:
            np.array:
                Filtered signal with  `N + samples_per_symbol*length_in_symbols - 1` samples.
        """

        # Convolve over a vector by default
        if input_signal.ndim == 1:
            return np.convolve(input_signal, self.impulse_response)

        # For multidimensional arrays, convolve over the first axis
        return np.apply_along_axis(lambda v: np.convolve(v, self.impulse_response), axis=0, arr=input_signal)

    def _get_raised_cosine(self, filter_type: str, roll_off: float,
                           bandwidth_expansion: float) -> np.ndarray:
        """Returns a raised-cosine or root-raised-cosine impulse response

        Args:
            filter_type (str): either 'RAISED_COSINE' or 'ROOT_RAISED_COSINE'
            roll_off (float): filter roll-off factor, between 0 and 1
            bandwidth_expansion (float): bandwidth scaling factor, relative to the symbol rate. If equal to one, then
                the filter is  built for a symbol rate 1/'self.samples_per_symbol'

        Returns:
            impulse_response (np.array): filter impulse response
        """
        delay_in_symbols = self.delay_in_samples / self.samples_per_symbol

        impulse_response = np.zeros(self.number_of_samples)

        # create time reference
        t_min = -delay_in_symbols
        t_max = self.number_of_samples / self.samples_per_symbol + t_min
        time = np.arange(
            t_min,
            t_max,
            1 / self.samples_per_symbol) * bandwidth_expansion

        if filter_type == "RAISED_COSINE":
            if roll_off != 0:
                # indices with division of zero by zero
                idx_0_by_0 = (abs(time) == 1 / (2 * roll_off))
            else:
                idx_0_by_0 = np.zeros_like(time, dtype=bool)
            idx = ~idx_0_by_0
            impulse_response[idx] = (np.sinc(time[idx]) * np.cos(np.pi * roll_off * time[idx])
                                     / (1 - (2 * roll_off * time[idx]) ** 2))
            if np.any(idx_0_by_0):
                impulse_response[idx_0_by_0] = np.pi / \
                    4 * np.sinc(1 / (2 * roll_off))
        else:  # ROOT_RAISED_COSINE
            idx_0_by_0 = (time == 0)  # indices with division of zero by zero

            if roll_off != 0:
                # indices with division by zero
                idx_x_by_0 = (abs(time) == 1 / (4 * roll_off))
            else:
                idx_x_by_0 = np.zeros_like(time, dtype=bool)
            idx = (~idx_0_by_0) & (~idx_x_by_0)

            impulse_response[idx] = ((np.sin(np.pi * time[idx] * (1 - roll_off)) +
                                      4 * roll_off * time[idx] * np.cos(np.pi * time[idx] * (1 + roll_off))) /
                                     (np.pi * time[idx] * (1 - (4 * roll_off * time[idx])**2)))
            if np.any(idx_x_by_0):
                impulse_response[idx_x_by_0] = roll_off / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(
                    np.pi / (4 * roll_off)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * roll_off)))
            impulse_response[idx_0_by_0] = 1 + roll_off * (4 / np.pi - 1)

        return impulse_response

    def _get_fmcw(self, chirp_duration_in_symbols: float, chirp_slope: float):
        """Returns an FMCW impulse response

        Args:
            chirp_duration_in_symbols (float):
            chirp_slope (float): chirp bandwidth / chirp duration

        Returns:
            impulse_response (np.array): filter impulse response
        """
        time = np.arange(self.number_of_samples) / self.samples_per_symbol

        bandwidth = np.abs(chirp_duration_in_symbols * chirp_slope)
        sign = np.sign(chirp_slope)

        impulse_response = np.exp(
            1j * np.pi * (-sign * bandwidth * time + chirp_slope * time**2))

        return impulse_response

    @classmethod
    def to_yaml(cls: Type[ShapingFilter],
                representer: SafeRepresenter,
                node: ShapingFilter) -> MappingNode:
        """Serialize an `ShapingFilter` object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (ShapingFilter):
                The `ShapingFilter` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node
        """

        state = {
            "filter_type": node.filter_type.name,
            "samples_per_symbol": node.samples_per_symbol,
            "length_in_symbols": node.length_in_symbols,
            "bandwidth_factor": node.bandwidth_factor,
            "roll_off": node.roll_off,
            "is_matched": node.is_matched,
        }
        return representer.represent_mapping(cls.yaml_tag, state)

    @classmethod
    def from_yaml(cls: Type[ShapingFilter],
                  constructor: SafeConstructor,
                  node: MappingNode) -> ShapingFilter:
        """Recall a new `ShapingFilter` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `ShapingFilter` serialization.

        Returns:
            ShapingFilter:
                Newly created `ShapingFilter` instance.
        """

        state = constructor.construct_mapping(node)
        return cls(**state)
