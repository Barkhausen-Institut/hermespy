# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Set

import numpy as np
from h5py import Group
from scipy.constants import speed_of_light

from ..channel import ChannelSampleHook, LinkState
from .delay import DelayChannelBase, DelayChannelRealization, DelayChannelSample

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SpatialDelayChannelRealization(DelayChannelRealization):
    """Realization of a spatial delay channel.

    Generated from :class:`SpatialDelayChannel's<SpatialDelayChannel>` :meth:`realize<SpatialDelayChannel.realize>` routine.
    """

    def _sample(self, state: LinkState) -> DelayChannelSample:
        delay = (
            np.linalg.norm(state.transmitter.position - state.receiver.position) / speed_of_light
        )
        return DelayChannelSample(delay, self.model_propagation_loss, self.gain, state)

    @staticmethod
    def From_HDF(
        group: Group, sample_hooks: Set[ChannelSampleHook[DelayChannelSample]]
    ) -> SpatialDelayChannelRealization:
        return SpatialDelayChannelRealization(
            group.attrs["model_propagation_loss"], sample_hooks, group.attrs["gain"]
        )


class SpatialDelayChannel(DelayChannelBase[SpatialDelayChannelRealization]):
    """Delay channel based on spatial relations between the linked devices."""

    yaml_tag: str = "SpatialDelay"

    def _realize(self) -> SpatialDelayChannelRealization:
        return SpatialDelayChannelRealization(
            self.model_propagation_loss, self.sample_hooks, self.gain
        )

    def recall_realization(self, group: Group) -> SpatialDelayChannelRealization:
        return SpatialDelayChannelRealization.From_HDF(group, self.sample_hooks)
