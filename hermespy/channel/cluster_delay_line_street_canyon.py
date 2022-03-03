# -*- coding: utf-8 -*-
"""
===========================================
3GPP Street Canyon (Urban Microcells) Model
===========================================

Implements several parameter sets defined within the 3GPP standard modeling specific scenarios.
"""

from abc import ABCMeta
from math import log10

import numpy as np

from ..core.factory import Serializable
from .cluster_delay_lines import ClusterDelayLineBase

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class StreetCanyonLineOfSight(ClusterDelayLineBase, Serializable):
    """Parameter Preset for the 3GPP Cluster Delay Line Urban Microcells Street Canyon Model."""

    yaml_tag = u'StreetCanyonLOS'
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return True

    @property
    def delay_spread_mean(self) -> float:
        return -.24 * log10(1 + self._center_frequency * 1e-9) - 7.14

    @property
    def delay_spread_std(self) -> float:
        return .38

    @property
    def aod_spread_mean(self) -> float:
        return -.05 * log10(1 + self._center_frequency * 1e-9) + 1.21

    @property
    def aod_spread_std(self) -> float:
        return .41

    @property
    def aoa_spread_mean(self) -> float:
        return -.08 * log10(1 + self._center_frequency * 1e-9) + 1.73

    @property
    def aoa_spread_std(self) -> float:
        return -.014 * log10(1 + self._center_frequency * 1e-9) + .28

    @property
    def zoa_spread_mean(self) -> float:
        return -.1 * log10(1 + self._center_frequency * 1e-9) + .73

    @property
    def zoa_spread_std(self) -> float:
        return -.04 * log10(1 + self._center_frequency * 1e-9) + .34

    @property
    def zod_spread_mean(self) -> float:

        device_distance = np.linalg.norm(self.receiver.position - self.transmitter.position, 2)
        terminal_height = abs(self.transmitter.position[2] - self.receiver.position[2])

        return max(-.21, -148e-4 * device_distance + .01 * terminal_height + .83)

    @property
    def zod_spread_std(self) -> float:
        return .35

    @property
    def zod_offset(self) -> float:
        return 0.

    @property
    def rice_factor_mean(self) -> float:
        return 9.

    @property
    def rice_factor_std(self) -> float:
        return 5.

    @property
    def delay_scaling(self) -> float:
        return 3.

    @property
    def cross_polarization_power_mean(self) -> float:
        return 9.

    @property
    def cross_polarization_power_std(self) -> float:
        return 3.

    @property
    def num_clusters(self) -> int:
        return 12

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 5.

    @property
    def cluster_aod_spread(self) -> float:
        return 3.

    @property
    def cluster_aoa_spread(self) -> float:
        return 17

    @property
    def cluster_zoa_spread(self) -> float:
        return 7.

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.

    @property
    def _center_frequency(self) -> float:
        return max(2e9, ClusterDelayLineBase._center_frequency.fget(self))


class UrbanMicroCellsNoLineOfSight(ClusterDelayLineBase, metaclass=ABCMeta):
    """Shared Parameters for all Urban Microcells No Line of Sight Models."""

    @property
    def zod_spread_mean(self) -> float:

        device_distance = np.linalg.norm(self.receiver.position - self.transmitter.position, 2)
        terminal_height = abs(self.transmitter.position[2] - self.receiver.position[2])

        return max(-.5, -31e-4 * device_distance + .01 * terminal_height + .2)

    @property
    def zod_spread_std(self) -> float:
        return .35

    @property
    def zod_offset(self) -> float:

        device_distance = np.linalg.norm(self.receiver.position - self.transmitter.position, 2)
        return -10 ** (-1.5 * log10(max(10, device_distance)) + 3.3)


class StreetCanyonNoLineOfSight(UrbanMicroCellsNoLineOfSight, Serializable):
    """Parameter Preset for the 3GPP Cluster Delay Line Urban Microcells Street Canyon Model."""

    yaml_tag = u'StreetCanyonNLOS'
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return False

    @property
    def delay_spread_mean(self) -> float:
        return -.24 * log10(1 + self._center_frequency * 1e-9) - 6.83

    @property
    def delay_spread_std(self) -> float:
        return .16 * log10(1 + self._center_frequency * 1e-9) + .28

    @property
    def aod_spread_mean(self) -> float:
        return -.23 * log10(1 + self._center_frequency * 1e-9) + 1.53

    @property
    def aod_spread_std(self) -> float:
        return .11 * log10(1 + self._center_frequency * 1e-9) - .33

    @property
    def aoa_spread_mean(self) -> float:
        return -.08 * log10(1 + self._center_frequency * 1e-9) + 1.81

    @property
    def aoa_spread_std(self) -> float:
        return -.05 * log10(1 + self._center_frequency * 1e-9) + .3

    @property
    def zoa_spread_mean(self) -> float:
        return -.04 * log10(1 + self._center_frequency * 1e-9) + .92

    @property
    def zoa_spread_std(self) -> float:
        return -.07 * log10(1 + self._center_frequency * 1e-9) + .41

    @property
    def rice_factor_mean(self) -> float:
        return 0.

    @property
    def rice_factor_std(self) -> float:
        return 0.

    @property
    def delay_scaling(self) -> float:
        return 2.1

    @property
    def cross_polarization_power_mean(self) -> float:
        return 8.

    @property
    def cross_polarization_power_std(self) -> float:
        return 3.

    @property
    def num_clusters(self) -> int:
        return 19

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 11.

    @property
    def cluster_aod_spread(self) -> float:
        return 10.

    @property
    def cluster_aoa_spread(self) -> float:
        return 22.

    @property
    def cluster_zoa_spread(self) -> float:
        return 7.

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.

    @property
    def _center_frequency(self) -> float:
        return max(2e9, ClusterDelayLineBase._center_frequency.fget(self))


class StreetCanyonOutsideToInside(UrbanMicroCellsNoLineOfSight, Serializable):
    """Parameter Preset for the 3GPP Cluster Delay Line Urban Microcells Street Canyon Model."""

    yaml_tag = u'StreetCanyonO2I'
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return False

    @property
    def delay_spread_mean(self) -> float:
        return -6.62

    @property
    def delay_spread_std(self) -> float:
        return .32

    @property
    def aod_spread_mean(self) -> float:
        return 1.25

    @property
    def aod_spread_std(self) -> float:
        return .42

    @property
    def aoa_spread_mean(self) -> float:
        return 1.76

    @property
    def aoa_spread_std(self) -> float:
        return .16

    @property
    def zoa_spread_mean(self) -> float:
        return 1.01

    @property
    def zoa_spread_std(self) -> float:
        return .43

    @property
    def rice_factor_mean(self) -> float:
        return 0.

    @property
    def rice_factor_std(self) -> float:
        return 0.

    @property
    def delay_scaling(self) -> float:
        return 2.2

    @property
    def cross_polarization_power_mean(self) -> float:
        return 9.

    @property
    def cross_polarization_power_std(self) -> float:
        return 5.

    @property
    def num_clusters(self) -> int:
        return 12

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 11.

    @property
    def cluster_aod_spread(self) -> float:
        return 5.

    @property
    def cluster_aoa_spread(self) -> float:
        return 8.

    @property
    def cluster_zoa_spread(self) -> float:
        return 3.

    @property
    def cluster_shadowing_std(self) -> float:
        return 4.

    @property
    def _center_frequency(self) -> float:
        return max(2e9, ClusterDelayLineBase._center_frequency.fget(self))
