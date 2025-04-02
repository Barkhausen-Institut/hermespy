# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Set
from typing_extensions import override


import numpy as np

from hermespy.core import (
    ChannelStateInformation,
    ChannelStateFormat,
    DeserializationProcess,
    SerializationProcess,
    SignalBlock,
)
from ..channel import (
    Channel,
    ChannelRealization,
    ChannelSample,
    ChannelSampleHook,
    LinkState,
    InterpolationMode,
)
from .matlab import MatlabEngine
from .octave import Oct2Py

if MatlabEngine is not None:  # pragma: no cover
    from .matlab import QuadrigaMatlabInterface as QuadrigaInterface  # type: ignore
elif Oct2Py is not None:  # pragma: no cover
    from .octave import QuadrigaOctaveInterface as QuadrigaInterface  # type: ignore
else:  # pragma: no cover
    from .interface import QuadrigaInterface  # type: ignore

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class QuadrigaChannelSample(ChannelSample):
    """Sample of a quadriga channel model."""

    __path_gains: np.ndarray
    __path_delays: np.ndarray

    def __init__(
        self, path_gains: np.ndarray, path_delays: np.ndarray, gain: float, state: LinkState
    ) -> None:
        """
        Args:
            path_gains: Path gains.
            path_delays: Path delays.
            gain: Channel gain.
            state: Channel state at which the sample was generated.
        """

        # Initialize base class
        ChannelSample.__init__(self, state)

        # Initialize class attributes
        self.__gain = gain
        self.__path_gains = path_gains
        self.__path_delays = path_delays

    @property
    def path_gains(self) -> np.ndarray:
        """Path gains."""

        return self.__path_gains

    @property
    def path_delays(self) -> np.ndarray:
        """Path delays."""

        return self.__path_delays

    @property
    def expected_energy_scale(self) -> float:
        return self.__gain * float(np.sum(self.__path_gains))

    def _propagate(self, signal: SignalBlock, interpolation: InterpolationMode) -> SignalBlock:
        max_delay_in_samples = int(np.round(np.max(self.path_delays) * self.bandwidth))
        propagated_signal = np.zeros(
            (
                self.transmitter_state.antennas.num_receive_antennas,
                signal.num_samples + max_delay_in_samples,
            ),
            dtype=np.complex128,
        )

        for channel, delay in zip(
            self.path_gains.transpose((2, 0, 1)), self.path_delays.transpose((2, 0, 1))
        ):
            time_delay = int(np.round(delay * self.bandwidth))
            propagated_signal[:, time_delay : time_delay + signal.num_samples] += channel @ signal

        propagated_signal *= np.sqrt(self.__gain)
        return SignalBlock(propagated_signal, signal._offset)

    def state(
        self,
        num_samples: int,
        max_num_taps: int,
        interpolation_mode: InterpolationMode = InterpolationMode.NEAREST,
    ) -> ChannelStateInformation:
        max_delay_in_samples = int(np.round(np.max(self.path_delays) * self.bandwidth))
        num_taps = min(max_num_taps, max_delay_in_samples + 1)

        impulse_response = np.zeros(
            (
                self.receiver_state.antennas.num_receive_antennas,
                self.transmitter_state.antennas.num_transmit_antennas,
                num_samples,
                num_taps,
            ),
            dtype=np.complex128,
        )

        for channel, delay in zip(
            self.path_gains.transpose((2, 0, 1)), self.path_delays.transpose((2, 0, 1))
        ):
            time_delay = int(np.round(delay * self.bandwidth))
            impulse_response[:, :, :, time_delay] += channel

        impulse_response *= np.sqrt(self.__gain)
        return ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE, impulse_response)


class QuadrigaChannelRealization(ChannelRealization[QuadrigaChannelSample]):
    """Realization of a quadriga channel model."""

    def __init__(
        self,
        interface: QuadrigaInterface,
        sample_hooks: Set[ChannelSampleHook[QuadrigaChannelSample]],
        gain: float,
    ) -> None:
        """
        Args:

            quadriga_interface: Interface to the Quadriga channel model.
            sample_hooks: Hooks to be called when a new sample is generated.
            gain: Linear channel power gain factor.
        """

        # Initialize base class
        ChannelRealization.__init__(self, sample_hooks, gain)

        # Save interface settings
        self.__interface = interface

    def _sample(self, state: LinkState) -> QuadrigaChannelSample:
        # Execute the matlab backend to fetch a channel impulse response
        cirs = self.__interface.sample_quadriga(state)

        # Return the sample
        return QuadrigaChannelSample(cirs[0, 0].coefficients, cirs[0, 0].delays, self.gain, state)

    def _reciprocal_sample(
        self, sample: QuadrigaChannelSample, state: LinkState
    ) -> QuadrigaChannelSample:  # pragma: no cover
        return self._sample(state)

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> QuadrigaChannelRealization:
        return cls(
            QuadrigaInterface(),
            set(),
            **ChannelRealization._DeserializeParameters(process),  # type: ignore[arg-type]
        )


class QuadrigaChannel(Channel[QuadrigaChannelRealization, QuadrigaChannelSample]):
    """Quadriga Channel Model.

    Maps the output of the selected interface to fit into Hermes' software architecture.
    """

    __interface: QuadrigaInterface | None  # Reference to the interface class

    def __init__(
        self,
        interface: QuadrigaInterface | None = None,
        gain: float = Channel._DEFAULT_GAIN,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            interface:
                Specifies the consisdered Quadriga interface.
            gain:
                Linear channel power gain factor.
                :math:`1.0` by default.
            seed:
                Seed used to initialize the pseudo-random number generator.
        """

        # Init base channel class
        Channel.__init__(self, gain, seed)

        # Save interface settings
        self.__interface = QuadrigaInterface() if interface is None else interface  # type: ignore

    def _realize(self) -> QuadrigaChannelRealization:
        return QuadrigaChannelRealization(self.__interface, self.sample_hooks, self.gain)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        Channel.serialize(self, process)

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> QuadrigaChannel:
        return cls(**Channel._DeserializeParameters(process))  # type: ignore[arg-type]
