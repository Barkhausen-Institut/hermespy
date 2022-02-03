# -*- coding: utf-8 -*-
"""
========================================
3GPP Cluster Delay Line Model Templates
========================================

Implements several parameter sets defined within the 3GPP standard modeling specific scenarios.
"""

from math import log10


from ..core.factory import Serializable
from .cluster_delay_lines import ClusterDelayLineBase


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
        return -.025 * log10(1 + self._center_frequency * 1e-9) + .28

    @property
    def zoa_spread_mean(self) -> float:
        return -.1 * log10(1 + self._center_frequency * 1e-9) + .73

    @property
    def zoa_spread_std(self) -> float:
        return -.04 * log10(1 + self._center_frequency * 1e-9) + .34

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

        return 2e9
