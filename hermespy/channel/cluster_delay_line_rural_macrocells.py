# -*- coding: utf-8 -*-
"""
===========================
3GPP Rural Macrocells Model
===========================

Implements several parameter sets defined within the 3GPP standard modeling specific scenarios.
"""

from math import atan

import numpy as np

from ..core.factory import Serializable
from .cluster_delay_lines import ClusterDelayLineBase

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RuralMacrocellsLineOfSight(ClusterDelayLineBase, Serializable):
    """Parameter Preset for the 3GPP Cluster Delay Line Rural Macrocells Model."""

    yaml_tag = "RMaLOS"
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return True

    @property
    def delay_spread_mean(self) -> float:
        return -7.49

    @property
    def delay_spread_std(self) -> float:
        return 0.55

    @property
    def aod_spread_mean(self) -> float:
        return 0.9

    @property
    def aod_spread_std(self) -> float:
        return 0.38

    @property
    def aoa_spread_mean(self) -> float:
        return 1.52

    @property
    def aoa_spread_std(self) -> float:
        return 0.24

    @property
    def zoa_spread_mean(self) -> float:
        return 0.47

    @property
    def zoa_spread_std(self) -> float:
        return 0.4

    @property
    def zod_spread_mean(self) -> float:
        device_distance = np.linalg.norm(self.receiver.position - self.transmitter.position, 2)
        terminal_height = abs(self.transmitter.position[2] - self.receiver.position[2])

        return max(-1, -17e-5 * device_distance - 0.01 * (terminal_height - 1.5) + 0.22)

    @property
    def zod_spread_std(self) -> float:
        return 0.34

    @property
    def zod_offset(self) -> float:
        return 0.0

    @property
    def rice_factor_mean(self) -> float:
        return 7.0

    @property
    def rice_factor_std(self) -> float:
        return 4.0

    @property
    def delay_scaling(self) -> float:
        return 3.8

    @property
    def cross_polarization_power_mean(self) -> float:
        return 12.0

    @property
    def cross_polarization_power_std(self) -> float:
        return 4.0

    @property
    def num_clusters(self) -> int:
        return 11

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 0.0

    @property
    def cluster_aod_spread(self) -> float:
        return 2.0

    @property
    def cluster_aoa_spread(self) -> float:
        return 3.0

    @property
    def cluster_zoa_spread(self) -> float:
        return 3.0

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.0

    @property
    def _center_frequency(self) -> float:  # pragma: no cover
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))  # type: ignore


class RuralMacrocellsNoLineOfSight(ClusterDelayLineBase, Serializable):
    """Parameter Preset for the 3GPP Cluster Delay Line Urban Macrocells Model."""

    yaml_tag = "RMaLOS"
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return False

    @property
    def delay_spread_mean(self) -> float:
        return -7.43

    @property
    def delay_spread_std(self) -> float:
        return 0.48

    @property
    def aod_spread_mean(self) -> float:
        return 0.95

    @property
    def aod_spread_std(self) -> float:
        return 0.45

    @property
    def aoa_spread_mean(self) -> float:
        return 1.52

    @property
    def aoa_spread_std(self) -> float:
        return 0.13

    @property
    def zoa_spread_mean(self) -> float:
        return 0.58

    @property
    def zoa_spread_std(self) -> float:
        return 0.37

    @property
    def zod_spread_mean(self) -> float:
        device_distance = np.linalg.norm(self.receiver.position - self.transmitter.position, 2)
        terminal_height = abs(self.transmitter.position[2] - self.receiver.position[2])

        return max(-1, -19e-5 * device_distance - 0.01 * (terminal_height - 1.5) + 0.28)

    @property
    def zod_spread_std(self) -> float:
        return 0.3

    @property
    def zod_offset(self) -> float:
        device_distance = np.linalg.norm(self.receiver.position - self.transmitter.position, 2)

        return atan((35 - 0.35) / device_distance) - atan((35 - 1.5) / device_distance)

    @property
    def rice_factor_mean(self) -> float:
        return 0.0

    @property
    def rice_factor_std(self) -> float:
        return 0.0

    @property
    def delay_scaling(self) -> float:
        return 1.7

    @property
    def cross_polarization_power_mean(self) -> float:
        return 7.0

    @property
    def cross_polarization_power_std(self) -> float:
        return 3.0

    @property
    def num_clusters(self) -> int:
        return 10

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 0.0

    @property
    def cluster_aod_spread(self) -> float:
        return 2.0

    @property
    def cluster_aoa_spread(self) -> float:
        return 3.0

    @property
    def cluster_zoa_spread(self) -> float:
        return 3.0

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.0

    @property
    def _center_frequency(self) -> float:  # pragma: no cover
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))  # type: ignore


class RuralMacrocellsOutsideToInside(ClusterDelayLineBase, Serializable):
    """Parameter Preset for the 3GPP Cluster Delay Line Urban Macrocells Model."""

    yaml_tag = "RMaO2I"
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return False

    @property
    def delay_spread_mean(self) -> float:
        return -7.47

    @property
    def delay_spread_std(self) -> float:
        return 0.24

    @property
    def aod_spread_mean(self) -> float:
        return 0.67

    @property
    def aod_spread_std(self) -> float:
        return 0.18

    @property
    def aoa_spread_mean(self) -> float:
        return 1.66

    @property
    def aoa_spread_std(self) -> float:
        return 0.21

    @property
    def zoa_spread_mean(self) -> float:
        return 0.93

    @property
    def zoa_spread_std(self) -> float:
        return 0.22

    @property
    def zod_spread_mean(self) -> float:
        device_distance = np.linalg.norm(self.receiver.position - self.transmitter.position, 2)
        terminal_height = abs(self.transmitter.position[2] - self.receiver.position[2])

        return max(-1, -19e-5 * device_distance - 0.01 * (terminal_height - 1.5) + 0.28)

    @property
    def zod_spread_std(self) -> float:
        return 0.3

    @property
    def zod_offset(self) -> float:
        device_distance = np.linalg.norm(self.receiver.position - self.transmitter.position, 2)

        return atan((35 - 0.35) / device_distance) - atan((35 - 1.5) / device_distance)

    @property
    def rice_factor_mean(self) -> float:
        return 0.0

    @property
    def rice_factor_std(self) -> float:
        return 0.0

    @property
    def delay_scaling(self) -> float:
        return 1.7

    @property
    def cross_polarization_power_mean(self) -> float:
        return 7.0

    @property
    def cross_polarization_power_std(self) -> float:
        return 3.0

    @property
    def num_clusters(self) -> int:
        return 10

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 0.0

    @property
    def cluster_aod_spread(self) -> float:
        return 2.0

    @property
    def cluster_aoa_spread(self) -> float:
        return 3.0

    @property
    def cluster_zoa_spread(self) -> float:
        return 3.0

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.0

    @property
    def _center_frequency(self) -> float:  # pragma: no cover
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))  # type: ignore
