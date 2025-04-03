# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

import numpy as np
from scipy.constants import speed_of_light

from hermespy.core import DeserializationProcess
from ..channel import LinkState
from .delay import DelayChannelBase, DelayChannelRealization, DelayChannelSample

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SpatialDelayChannelRealization(DelayChannelRealization):
    """Realization of a spatial delay channel.

    Generated from :class:`SpatialDelayChannel<SpatialDelayChannel>`.
    """

    def _sample(self, state: LinkState) -> DelayChannelSample:
        delay = (
            np.linalg.norm(state.transmitter.position - state.receiver.position) / speed_of_light
        )
        return DelayChannelSample(delay, self.model_propagation_loss, self.gain, state)

    @classmethod
    @override
    def Deserialize(cls, group: DeserializationProcess) -> SpatialDelayChannelRealization:
        return SpatialDelayChannelRealization(
            sample_hooks=set(),
            **DelayChannelRealization._DeserializeParameters(group),  # type: ignore[arg-type]
        )


class SpatialDelayChannel(DelayChannelBase[SpatialDelayChannelRealization]):
    """Delay channel based on spatial relations between the linked devices."""

    def _realize(self) -> SpatialDelayChannelRealization:
        return SpatialDelayChannelRealization(
            self.model_propagation_loss, self.sample_hooks, self.gain
        )

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> SpatialDelayChannel:
        return SpatialDelayChannel(**DelayChannelBase._DeserializeParameters(process))  # type: ignore[arg-type]
