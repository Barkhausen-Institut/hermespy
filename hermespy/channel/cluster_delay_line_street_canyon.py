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
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class StreetCanyonLineOfSight(ClusterDelayLineBase, Serializable):
    """Parameter Preset for the 3GPP Cluster Delay Line Urban Microcells Street Canyon Model."""

    yaml_tag = "StreetCanyonLOS"
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return True

    @property
    def delay_spread_mean(self) -> float:
        return -0.24 * log10(1 + self._center_frequency * 1e-9) - 7.14

    @property
    def delay_spread_std(self) -> float:
        return 0.38

    @property
    def aod_spread_mean(self) -> float:
        return -0.05 * log10(1 + self._center_frequency * 1e-9) + 1.21

    @property
    def aod_spread_std(self) -> float:
        return 0.41

    @property
    def aoa_spread_mean(self) -> float:
        return -0.08 * log10(1 + self._center_frequency * 1e-9) + 1.73

    @property
    def aoa_spread_std(self) -> float:
        return 0.014 * log10(1 + self._center_frequency * 1e-9) + 0.28

    @property
    def zoa_spread_mean(self) -> float:
        return -0.1 * log10(1 + self._center_frequency * 1e-9) + 0.73

    @property
    def zoa_spread_std(self) -> float:
        return -0.04 * log10(1 + self._center_frequency * 1e-9) + 0.34

    @property
    def zod_spread_mean(self) -> float:
        device_distance = np.linalg.norm(self.receiver.position - self.transmitter.position, 2)
        terminal_height = abs(self.transmitter.position[2] - self.receiver.position[2])

        return max(-0.21, -148e-4 * device_distance + 0.01 * terminal_height + 0.83)

    @property
    def zod_spread_std(self) -> float:
        return 0.35

    @property
    def zod_offset(self) -> float:
        return 0.0

    @property
    def rice_factor_mean(self) -> float:
        return 9.0

    @property
    def rice_factor_std(self) -> float:
        return 5.0

    @property
    def delay_scaling(self) -> float:
        return 3.0

    @property
    def cross_polarization_power_mean(self) -> float:
        return 9.0

    @property
    def cross_polarization_power_std(self) -> float:
        return 3.0

    @property
    def num_clusters(self) -> int:
        return 12

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 5.0

    @property
    def cluster_aod_spread(self) -> float:
        return 3.0

    @property
    def cluster_aoa_spread(self) -> float:
        return 17

    @property
    def cluster_zoa_spread(self) -> float:
        return 7.0

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.0

    @property
    def _center_frequency(self) -> float:
        return max(2e9, ClusterDelayLineBase._center_frequency.fget(self))  # type: ignore


class UrbanMicroCellsNoLineOfSight(ClusterDelayLineBase, metaclass=ABCMeta):
    """Shared Parameters for all Urban Microcells No Line of Sight Models."""

    @property
    def zod_spread_mean(self) -> float:
        device_distance = np.linalg.norm(self.receiver.position - self.transmitter.position, 2)
        terminal_height = abs(self.transmitter.position[2] - self.receiver.position[2])

        return max(-0.5, -31e-4 * device_distance + 0.01 * terminal_height + 0.2)

    @property
    def zod_spread_std(self) -> float:
        return 0.35

    @property
    def zod_offset(self) -> float:
        device_distance = float(np.linalg.norm(self.receiver.position - self.transmitter.position, 2))
        return -(10 ** (-1.5 * log10(max(10, device_distance)) + 3.3))


class StreetCanyonNoLineOfSight(UrbanMicroCellsNoLineOfSight, Serializable):
    """Parameter Preset for the 3GPP Cluster Delay Line Urban Microcells Street Canyon Model."""

    yaml_tag = "StreetCanyonNLOS"
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return False

    @property
    def delay_spread_mean(self) -> float:
        return -0.24 * log10(1 + self._center_frequency * 1e-9) - 6.83

    @property
    def delay_spread_std(self) -> float:
        return 0.16 * log10(1 + self._center_frequency * 1e-9) + 0.28

    @property
    def aod_spread_mean(self) -> float:
        return -0.23 * log10(1 + self._center_frequency * 1e-9) + 1.53

    @property
    def aod_spread_std(self) -> float:
        return 0.11 * log10(1 + self._center_frequency * 1e-9) + 0.33

    @property
    def aoa_spread_mean(self) -> float:
        return -0.08 * log10(1 + self._center_frequency * 1e-9) + 1.81

    @property
    def aoa_spread_std(self) -> float:
        return -0.05 * log10(1 + self._center_frequency * 1e-9) + 0.3

    @property
    def zoa_spread_mean(self) -> float:
        return -0.04 * log10(1 + self._center_frequency * 1e-9) + 0.92

    @property
    def zoa_spread_std(self) -> float:
        return -0.07 * log10(1 + self._center_frequency * 1e-9) + 0.41

    @property
    def rice_factor_mean(self) -> float:
        return 0.0

    @property
    def rice_factor_std(self) -> float:
        return 0.0

    @property
    def delay_scaling(self) -> float:
        return 2.1

    @property
    def cross_polarization_power_mean(self) -> float:
        return 8.0

    @property
    def cross_polarization_power_std(self) -> float:
        return 3.0

    @property
    def num_clusters(self) -> int:
        return 19

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 11.0

    @property
    def cluster_aod_spread(self) -> float:
        return 10.0

    @property
    def cluster_aoa_spread(self) -> float:
        return 22.0

    @property
    def cluster_zoa_spread(self) -> float:
        return 7.0

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.0

    @property
    def _center_frequency(self) -> float:
        return max(2e9, ClusterDelayLineBase._center_frequency.fget(self))  # type: ignore


class StreetCanyonOutsideToInside(UrbanMicroCellsNoLineOfSight, Serializable):
    """Parameter Preset for the 3GPP Cluster Delay Line Urban Microcells Street Canyon Model."""

    yaml_tag = "StreetCanyonO2I"
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return False

    @property
    def delay_spread_mean(self) -> float:
        return -6.62

    @property
    def delay_spread_std(self) -> float:
        return 0.32

    @property
    def aod_spread_mean(self) -> float:
        return 1.25

    @property
    def aod_spread_std(self) -> float:
        return 0.42

    @property
    def aoa_spread_mean(self) -> float:
        return 1.76

    @property
    def aoa_spread_std(self) -> float:
        return 0.16

    @property
    def zoa_spread_mean(self) -> float:
        return 1.01

    @property
    def zoa_spread_std(self) -> float:
        return 0.43

    @property
    def rice_factor_mean(self) -> float:
        return 0.0

    @property
    def rice_factor_std(self) -> float:
        return 0.0

    @property
    def delay_scaling(self) -> float:
        return 2.2

    @property
    def cross_polarization_power_mean(self) -> float:
        return 9.0

    @property
    def cross_polarization_power_std(self) -> float:
        return 5.0

    @property
    def num_clusters(self) -> int:
        return 12

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 11.0

    @property
    def cluster_aod_spread(self) -> float:
        return 5.0

    @property
    def cluster_aoa_spread(self) -> float:
        return 8.0

    @property
    def cluster_zoa_spread(self) -> float:
        return 3.0

    @property
    def cluster_shadowing_std(self) -> float:
        return 4.0

    @property
    def _center_frequency(self) -> float:  # pragma: no cover
        return max(2e9, ClusterDelayLineBase._center_frequency.fget(self))  # type: ignore
