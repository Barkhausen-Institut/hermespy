# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Generic, Set, TypeVar, TYPE_CHECKING

import numpy as np
from h5py import Group
from scipy.constants import speed_of_light

from hermespy.core import AntennaMode, ChannelStateInformation, ChannelStateFormat, SignalBlock
from hermespy.tools import amplitude_path_loss
from ..channel import (
    Channel,
    ChannelSample,
    LinkState,
    ChannelSampleHook,
    ChannelRealization,
    InterpolationMode,
)

if TYPE_CHECKING:
    from hermespy.simulation import SimulatedDevice  # pragma: no cover

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


DCRT = TypeVar("DCRT", bound="DelayChannelRealization")
"""Type of delay channel realization"""


class DelayChannelSample(ChannelSample):
    """Sample of a delay channel."""

    def __init__(
        self, delay: float, model_propagation_loss: bool, gain: float, state: LinkState
    ) -> None:

        # Initialize base class
        ChannelSample.__init__(self, state)

        # Store attributes
        self.__delay = delay
        self.__model_propagation_loss = model_propagation_loss
        self.__gain = gain

    @property
    def delay(self) -> float:
        """Propagation delay in seconds."""

        return self.__delay

    @property
    def model_propagation_loss(self) -> bool:
        """Should free space propagation loss be modeled?"""

        return self.__model_propagation_loss

    @property
    def gain(self) -> float:
        """Linear power gain factor a signal experiences when being propagated over this realization."""

        return self.__gain

    def __spatial_response(self) -> np.ndarray:

        receiver_position = self.receiver_state.position
        transmitter_position = self.transmitter_state.position

        if np.all(receiver_position == transmitter_position):
            return np.ones(
                (
                    self.receiver_state.antennas.num_receive_antennas,
                    self.transmitter_state.antennas.num_transmit_antennas,
                ),
                dtype=np.complex_,
            )

        transmit_response = self.transmitter_state.antennas.cartesian_array_response(
            self.carrier_frequency, self.receiver_state.position, "global", AntennaMode.TX
        )
        receive_response = self.receiver_state.antennas.cartesian_array_response(
            self.carrier_frequency, self.transmitter_state.position, "global", AntennaMode.RX
        )

        return receive_response @ transmit_response.T

    def __path_gain(self, carrier_frequency: float) -> float:
        """Compute the realization's linear path gain factor.

            carrier_frequency (float):
                Central frequency of the propagation in :math:`\\mathrm{Hz}`.

        Returns: Linear path gain scaling the amplitude of a propagated signal.
        """

        gain_factor = self.gain

        if self.model_propagation_loss:
            if carrier_frequency == 0.0:
                raise RuntimeError(
                    "Transmitting device's carrier frequency may not be zero, disable propagation path loss modeling"
                )

            if self.delay > 0:
                gain_factor *= amplitude_path_loss(carrier_frequency, self.delay * speed_of_light)

        return gain_factor

    def _propagate(self, signal: SignalBlock, interpolation: InterpolationMode) -> SignalBlock:

        delay_samples = round(self.delay * self.bandwidth)
        spatial_response = self.__spatial_response()
        gain_factor = self.__path_gain(self.carrier_frequency)

        propagated_samples = np.append(
            np.zeros(
                (self.receiver_state.antennas.num_receive_antennas, delay_samples), np.complex_
            ),
            gain_factor * spatial_response @ signal,
        )

        propagated_signal = SignalBlock(propagated_samples, signal._offset)
        return propagated_signal

    def state(
        self,
        num_samples: int,
        max_num_taps: int,
        interpolation_mode: InterpolationMode = InterpolationMode.NEAREST,
    ) -> ChannelStateInformation:

        delay_samples = round(self.delay * self.bandwidth)
        spatial_response = self.__spatial_response()
        gain_factor = self.__path_gain(self.carrier_frequency)

        cir = np.zeros(
            (
                self.receiver_state.antennas.num_receive_antennas,
                self.transmitter_state.antennas.num_transmit_antennas,
                num_samples,
                min(1 + delay_samples, max_num_taps),
            ),
            dtype=np.complex_,
        )
        if delay_samples < max_num_taps:
            cir[:, :, :, delay_samples] = gain_factor * spatial_response[:, :, np.newaxis]

        state = ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE, cir)
        return state


class DelayChannelRealization(ChannelRealization[DelayChannelSample]):
    """Base class for delay channel realizations."""

    __model_propagation_loss: bool

    def __init__(
        self,
        model_propagation_loss: bool,
        sample_hooks: Set[ChannelSampleHook[DelayChannelSample]],
        gain: float,
    ) -> None:
        """
        Args:
            model_propagation_loss (bool):
                Should free space propagation loss be modeled?

            gain (float):
                Linear power gain factor a signal experiences when being propagated over this realization.
        """

        # Initialize base class
        ChannelRealization.__init__(self, sample_hooks, gain)

        # Initialize class attributes
        self.__model_propagation_loss = model_propagation_loss

    @property
    def model_propagation_loss(self) -> bool:
        """Should free-space propagation losses be modeled?"""

        return self.__model_propagation_loss

    def _reciprocal_sample(
        self, sample: DelayChannelSample, state: LinkState
    ) -> DelayChannelSample:
        return DelayChannelSample(sample.delay, self.model_propagation_loss, self.gain, state)

    def to_HDF(self, group: Group) -> None:
        group.attrs["model_propagation_loss"] = self.model_propagation_loss
        group.attrs["gain"] = self.gain


class DelayChannelBase(Generic[DCRT], Channel[DCRT, DelayChannelSample]):
    """Base of delay channel models."""

    __model_propagation_loss: bool

    def __init__(
        self,
        alpha_device: SimulatedDevice | None = None,
        beta_device: SimulatedDevice | None = None,
        gain: float = 1.0,
        model_propagation_loss: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:

            alpha_device (SimulatedDevice, optional):
                First device linked by the :class:`.DelayChannelBase` instance that generated this realization.

            beta_device (SimulatedDevice, optional):
                Second device linked by the :class:`.DelayChannelBase` instance that generated this realization.

            gain (float, optional):
                Linear power gain factor a signal experiences when being propagated over this realization.
                :math:`1.0` by default.

            model_propagation_loss (bool, optional):
                Should free space propagation loss be modeled?
                Enabled by default.

            **kawrgs:
                :class:`Channel` base class initialization arguments.
        """

        # Initialize base class
        Channel.__init__(self, alpha_device, beta_device, gain, **kwargs)

        # Initialize class attributes
        self.__model_propagation_loss = model_propagation_loss

    @property
    def model_propagation_loss(self) -> bool:
        """Should free space propagation loss be modeled?"""

        return self.__model_propagation_loss

    @model_propagation_loss.setter
    def model_propagation_loss(self, value: bool) -> None:
        self.__model_propagation_loss = value
