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
        return -.014 * log10(1 + self._center_frequency * 1e-9) + .28

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
        return max(2e9, ClusterDelayLineBase._center_frequency.fget(self))


class StreetCanyonNonLineOfSight(ClusterDelayLineBase, Serializable):
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
        return 9.

    @property
    def rice_factor_std(self) -> float:
        return 5.

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


class StreetCanyonOutsideToInside(ClusterDelayLineBase, Serializable):
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


class UrbanMacrocellsLineOfSight(ClusterDelayLineBase, Serializable):
    """Parameter Preset for the 3GPP Cluster Delay Line Urban Macrocells Model."""

    yaml_tag = u'UMaLOS'
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return True

    @property
    def delay_spread_mean(self) -> float:
        return -6.995 - .0963 * log10(self._center_frequency * 1e-9)

    @property
    def delay_spread_std(self) -> float:
        return .66

    @property
    def aod_spread_mean(self) -> float:
        return 1.06 + .1114 * log10(self._center_frequency * 1e-9)

    @property
    def aod_spread_std(self) -> float:
        return .28

    @property
    def aoa_spread_mean(self) -> float:
        return 1.81

    @property
    def aoa_spread_std(self) -> float:
        return .2

    @property
    def zoa_spread_mean(self) -> float:
        return .95

    @property
    def zoa_spread_std(self) -> float:
        return .16

    @property
    def rice_factor_mean(self) -> float:
        return 9.

    @property
    def rice_factor_std(self) -> float:
        return 3.5

    @property
    def delay_scaling(self) -> float:
        return 2.5

    @property
    def cross_polarization_power_mean(self) -> float:
        return 8.

    @property
    def cross_polarization_power_std(self) -> float:
        return 4.

    @property
    def num_clusters(self) -> int:
        return 12

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return max(.25, 6.5622 - 3.4084 * log10(self._center_frequency * 1e-9))

    @property
    def cluster_aod_spread(self) -> float:
        return 5.

    @property
    def cluster_aoa_spread(self) -> float:
        return 11.

    @property
    def cluster_zoa_spread(self) -> float:
        return 7.

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.

    @property
    def _center_frequency(self) -> float:
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))


class UrbanMacrocellsNoLineOfSight(ClusterDelayLineBase, Serializable):
    """Parameter Preset for the 3GPP Cluster Delay Line Urban Macrocells Model."""

    yaml_tag = u'UMaNLOS'
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return False

    @property
    def delay_spread_mean(self) -> float:
        return -6.28 - .204 * log10(self._center_frequency * 1e-9)

    @property
    def delay_spread_std(self) -> float:
        return .39

    @property
    def aod_spread_mean(self) -> float:
        return 1.5 + .1114 * log10(self._center_frequency * 1e-9)

    @property
    def aod_spread_std(self) -> float:
        return .28

    @property
    def aoa_spread_mean(self) -> float:
        return 2.08 - .27 * log10(self._center_frequency * 1e-9)

    @property
    def aoa_spread_std(self) -> float:
        return .11

    @property
    def zoa_spread_mean(self) -> float:
        return -.3236 * log10(self._center_frequency * 1e-9) + 1.512

    @property
    def zoa_spread_std(self) -> float:
        return .16

    @property
    def rice_factor_mean(self) -> float:
        return 0.

    @property
    def rice_factor_std(self) -> float:
        return 0.

    @property
    def delay_scaling(self) -> float:
        return 2.3

    @property
    def cross_polarization_power_mean(self) -> float:
        return 7.

    @property
    def cross_polarization_power_std(self) -> float:
        return 3.

    @property
    def num_clusters(self) -> int:
        return 20

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return max(.25, 6.5622 - 3.4084 * log10(self._center_frequency * 1e-9))

    @property
    def cluster_aod_spread(self) -> float:
        return 2.

    @property
    def cluster_aoa_spread(self) -> float:
        return 15.

    @property
    def cluster_zoa_spread(self) -> float:
        return 7.

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.

    @property
    def _center_frequency(self) -> float:
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))


class UrbanMacrocellsOutsideToInside(ClusterDelayLineBase, Serializable):
    """Parameter Preset for the 3GPP Cluster Delay Line Urban Macrocells Model."""

    yaml_tag = u'UMa02I'
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
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))


class RuralMacrocellsLineOfSight(ClusterDelayLineBase, Serializable):
    """Parameter Preset for the 3GPP Cluster Delay Line Urban Macrocells Model."""

    yaml_tag = u'RMaLOS'
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return True

    @property
    def delay_spread_mean(self) -> float:
        return -7.49

    @property
    def delay_spread_std(self) -> float:
        return .55

    @property
    def aod_spread_mean(self) -> float:
        return .9

    @property
    def aod_spread_std(self) -> float:
        return .38

    @property
    def aoa_spread_mean(self) -> float:
        return 1.52

    @property
    def aoa_spread_std(self) -> float:
        return .24

    @property
    def zoa_spread_mean(self) -> float:
        return .47

    @property
    def zoa_spread_std(self) -> float:
        return .4

    @property
    def rice_factor_mean(self) -> float:
        return 7.

    @property
    def rice_factor_std(self) -> float:
        return 4.

    @property
    def delay_scaling(self) -> float:
        return 3.8

    @property
    def cross_polarization_power_mean(self) -> float:
        return 12.

    @property
    def cross_polarization_power_std(self) -> float:
        return 4.

    @property
    def num_clusters(self) -> int:
        return 11

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 0.

    @property
    def cluster_aod_spread(self) -> float:
        return 2.

    @property
    def cluster_aoa_spread(self) -> float:
        return 3.

    @property
    def cluster_zoa_spread(self) -> float:
        return 3.

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.

    @property
    def _center_frequency(self) -> float:
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))


class RuralMacrocellsNoLineOfSight(ClusterDelayLineBase, Serializable):
    """Parameter Preset for the 3GPP Cluster Delay Line Urban Macrocells Model."""

    yaml_tag = u'RMaLOS'
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return False

    @property
    def delay_spread_mean(self) -> float:
        return -7.43

    @property
    def delay_spread_std(self) -> float:
        return .48

    @property
    def aod_spread_mean(self) -> float:
        return .95

    @property
    def aod_spread_std(self) -> float:
        return .45

    @property
    def aoa_spread_mean(self) -> float:
        return 1.52

    @property
    def aoa_spread_std(self) -> float:
        return .13

    @property
    def zoa_spread_mean(self) -> float:
        return .58

    @property
    def zoa_spread_std(self) -> float:
        return .37

    @property
    def rice_factor_mean(self) -> float:
        return 0.

    @property
    def rice_factor_std(self) -> float:
        return 0.

    @property
    def delay_scaling(self) -> float:
        return 1.7

    @property
    def cross_polarization_power_mean(self) -> float:
        return 7.

    @property
    def cross_polarization_power_std(self) -> float:
        return 3.

    @property
    def num_clusters(self) -> int:
        return 10

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 0.

    @property
    def cluster_aod_spread(self) -> float:
        return 2.

    @property
    def cluster_aoa_spread(self) -> float:
        return 3.

    @property
    def cluster_zoa_spread(self) -> float:
        return 3.

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.

    @property
    def _center_frequency(self) -> float:
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))


class RuralMacrocellsOutsideToInside(ClusterDelayLineBase, Serializable):
    """Parameter Preset for the 3GPP Cluster Delay Line Urban Macrocells Model."""

    yaml_tag = u'RMaO2I'
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return False

    @property
    def delay_spread_mean(self) -> float:
        return -7.47

    @property
    def delay_spread_std(self) -> float:
        return .24

    @property
    def aod_spread_mean(self) -> float:
        return .67

    @property
    def aod_spread_std(self) -> float:
        return .18

    @property
    def aoa_spread_mean(self) -> float:
        return 1.66

    @property
    def aoa_spread_std(self) -> float:
        return .21

    @property
    def zoa_spread_mean(self) -> float:
        return .93

    @property
    def zoa_spread_std(self) -> float:
        return .22

    @property
    def rice_factor_mean(self) -> float:
        return 0.

    @property
    def rice_factor_std(self) -> float:
        return 0.

    @property
    def delay_scaling(self) -> float:
        return 1.7

    @property
    def cross_polarization_power_mean(self) -> float:
        return 7.

    @property
    def cross_polarization_power_std(self) -> float:
        return 3.

    @property
    def num_clusters(self) -> int:
        return 10

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 0.

    @property
    def cluster_aod_spread(self) -> float:
        return 2.

    @property
    def cluster_aoa_spread(self) -> float:
        return 3.

    @property
    def cluster_zoa_spread(self) -> float:
        return 3.

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.

    @property
    def _center_frequency(self) -> float:
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))


class IndoorOfficeLineOfSight(ClusterDelayLineBase, Serializable):
    """Parameter Preset for the 3GPP Cluster Indoor-Office Model."""

    yaml_tag = u'IndoorOfficeLOS'
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return True

    @property
    def delay_spread_mean(self) -> float:
        return -.01 * log10(1 + self._center_frequency * 1e-9) - 7.692

    @property
    def delay_spread_std(self) -> float:
        return .18

    @property
    def aod_spread_mean(self) -> float:
        return 1.6

    @property
    def aod_spread_std(self) -> float:
        return .18

    @property
    def aoa_spread_mean(self) -> float:
        return -.19 * log10(1 + self._center_frequency * 1e-9) + 1.781

    @property
    def aoa_spread_std(self) -> float:
        return .12 * log10(1 + self._center_frequency * 1e-9) + .119

    @property
    def zoa_spread_mean(self) -> float:
        return -.26 * log10(1 + self._center_frequency * 1e-9) + 1.44

    @property
    def zoa_spread_std(self) -> float:
        return -.04 * log10(1 + self._center_frequency * 1e-9) + .264

    @property
    def rice_factor_mean(self) -> float:
        return 7.

    @property
    def rice_factor_std(self) -> float:
        return 4.

    @property
    def delay_scaling(self) -> float:
        return 3.6

    @property
    def cross_polarization_power_mean(self) -> float:
        return 11.

    @property
    def cross_polarization_power_std(self) -> float:
        return 4.

    @property
    def num_clusters(self) -> int:
        return 15

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 0.

    @property
    def cluster_aod_spread(self) -> float:
        return 5.

    @property
    def cluster_aoa_spread(self) -> float:
        return 8.

    @property
    def cluster_zoa_spread(self) -> float:
        return 9.

    @property
    def cluster_shadowing_std(self) -> float:
        return 6.

    @property
    def _center_frequency(self) -> float:
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))


class IndoorOfficeNoLineOfSight(ClusterDelayLineBase, Serializable):
    """Parameter Preset for the 3GPP Cluster Indoor-Office Model."""

    yaml_tag = u'IndoorOfficeNLOS'
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return False

    @property
    def delay_spread_mean(self) -> float:
        return -.28 * log10(1 + self._center_frequency * 1e-9) - 7.173

    @property
    def delay_spread_std(self) -> float:
        return .1 * log10(1 + self._center_frequency * 1e-9) - .055

    @property
    def aod_spread_mean(self) -> float:
        return 1.62

    @property
    def aod_spread_std(self) -> float:
        return .25

    @property
    def aoa_spread_mean(self) -> float:
        return -.11 * log10(1 + self._center_frequency * 1e-9) + 1.863

    @property
    def aoa_spread_std(self) -> float:
        return .12 * log10(1 + self._center_frequency * 1e-9) + .119

    @property
    def zoa_spread_mean(self) -> float:
        return -.15 * log10(1 + self._center_frequency * 1e-9) + 1.387

    @property
    def zoa_spread_std(self) -> float:
        return -.09 * log10(1 + self._center_frequency * 1e-9) + .746

    @property
    def rice_factor_mean(self) -> float:
        return 0.

    @property
    def rice_factor_std(self) -> float:
        return 0.

    @property
    def delay_scaling(self) -> float:
        return 3

    @property
    def cross_polarization_power_mean(self) -> float:
        return 10.

    @property
    def cross_polarization_power_std(self) -> float:
        return 4.

    @property
    def num_clusters(self) -> int:
        return 19

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 0.

    @property
    def cluster_aod_spread(self) -> float:
        return 5.

    @property
    def cluster_aoa_spread(self) -> float:
        return 11.

    @property
    def cluster_zoa_spread(self) -> float:
        return 9.

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.

    @property
    def _center_frequency(self) -> float:
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))