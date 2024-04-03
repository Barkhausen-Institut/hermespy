# -*- coding: utf-8 -*-

from abc import ABCMeta
from math import log10

import numpy as np

from hermespy.core.factory import Serializable
from .cluster_delay_lines import ClusterDelayLineBase

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class UrbanMacrocellsLineOfSight(ClusterDelayLineBase, Serializable):
    """3GPP cluster delay line preset modeling an urban macrocell scenario with direct
    line of sight between the linked wireless devices.

    Refer to the :footcite:t:`3GPP:TR38901` for detailed information.

    The following minimal example outlines how to configure the channel model
    within the context of a :doc:`simulation.simulation.Simulation`:

    .. literalinclude:: ../scripts/examples/channel_cdl_urban_macrocells_los.py
       :language: python
       :linenos:
       :lines: 12-40
    """

    yaml_tag = "UMaLOS"
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return True

    @property
    def delay_spread_mean(self) -> float:
        return -6.995 - 0.0963 * log10(self._center_frequency * 1e-9)

    @property
    def delay_spread_std(self) -> float:
        return 0.66

    @property
    def aod_spread_mean(self) -> float:
        return 1.06 + 0.1114 * log10(self._center_frequency * 1e-9)

    @property
    def aod_spread_std(self) -> float:
        return 0.28

    @property
    def aoa_spread_mean(self) -> float:
        return 1.81

    @property
    def aoa_spread_std(self) -> float:
        return 0.2

    @property
    def zoa_spread_mean(self) -> float:
        return 0.95

    @property
    def zoa_spread_std(self) -> float:
        return 0.16

    @property
    def zod_spread_mean(self) -> float:
        device_distance = np.linalg.norm(self.beta_device.position - self.alpha_device.position, 2)
        terminal_height = min(self.alpha_device.position[2], self.beta_device.position[2])

        return max(-0.5, -2.1e-3 * device_distance - 1e-2 * (terminal_height - 1.5) + 0.75)

    @property
    def zod_spread_std(self) -> float:
        return 0.4

    @property
    def zod_offset(self) -> float:
        return 0.0

    @property
    def rice_factor_mean(self) -> float:
        return 9.0

    @property
    def rice_factor_std(self) -> float:
        return 3.5

    @property
    def delay_scaling(self) -> float:
        return 2.5

    @property
    def cross_polarization_power_mean(self) -> float:
        return 8.0

    @property
    def cross_polarization_power_std(self) -> float:
        return 4.0

    @property
    def num_clusters(self) -> int:
        return 12

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return max(0.25, 6.5622 - 3.4084 * log10(self._center_frequency * 1e-9)) * 1e-9

    @property
    def cluster_aod_spread(self) -> float:
        return 5.0

    @property
    def cluster_aoa_spread(self) -> float:
        return 11.0

    @property
    def cluster_zoa_spread(self) -> float:
        return 7.0

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.0

    @property
    def _center_frequency(self) -> float:
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))  # type: ignore


class UrbanMacrocellsNoLineOfSightBase(ClusterDelayLineBase, metaclass=ABCMeta):
    """Shared Parameters for all Urban Macrocells No Line of Sight Models."""

    @property
    def zod_spread_mean(self) -> float:
        device_distance = np.linalg.norm(self.beta_device.position - self.alpha_device.position, 2)
        terminal_height = min(self.alpha_device.position[2], self.beta_device.position[2])

        return max(-0.5, -2.1e-3 * device_distance - 1e-2 * (terminal_height - 1.5) + 0.9)

    @property
    def zod_spread_std(self) -> float:
        return 0.49

    @property
    def zod_offset(self) -> float:
        device_distance = float(
            np.linalg.norm(self.beta_device.position - self.alpha_device.position, 2)
        )
        terminal_height = min(self.alpha_device.position[2], self.beta_device.position[2])
        fc = log10(self._center_frequency)

        a = 0.208 * fc - 0.782
        b = 25
        c = -0.13 * fc + 2.03
        e = 7.66 * fc - 5.96

        return e - 10 ** (a * log10(max(b, device_distance)) + c - 0.07 * (terminal_height - 1.5))


class UrbanMacrocellsNoLineOfSight(UrbanMacrocellsNoLineOfSightBase, Serializable):
    """3GPP cluster delay line preset modeling an urban macrocell scenario without direct
    line of sight between the linked wireless devices.

    Refer to the :footcite:t:`3GPP:TR38901` for detailed information.

    The following minimal example outlines how to configure the channel model
    within the context of a :doc:`simulation.simulation.Simulation`:

    .. literalinclude:: ../scripts/examples/channel_cdl_urban_macrocells_nlos.py
       :language: python
       :linenos:
       :lines: 12-40
    """

    yaml_tag = "UMaNLOS"
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return False

    @property
    def delay_spread_mean(self) -> float:
        return -6.28 - 0.204 * log10(self._center_frequency * 1e-9)

    @property
    def delay_spread_std(self) -> float:
        return 0.39

    @property
    def aod_spread_mean(self) -> float:
        return 1.5 + 0.1114 * log10(self._center_frequency * 1e-9)

    @property
    def aod_spread_std(self) -> float:
        return 0.28

    @property
    def aoa_spread_mean(self) -> float:
        return 2.08 - 0.27 * log10(self._center_frequency * 1e-9)

    @property
    def aoa_spread_std(self) -> float:
        return 0.11

    @property
    def zoa_spread_mean(self) -> float:
        return -0.3236 * log10(self._center_frequency * 1e-9) + 1.512

    @property
    def zoa_spread_std(self) -> float:
        return 0.16

    @property
    def rice_factor_mean(self) -> float:
        return 0.0

    @property
    def rice_factor_std(self) -> float:
        return 0.0

    @property
    def delay_scaling(self) -> float:
        return 2.3

    @property
    def cross_polarization_power_mean(self) -> float:
        return 7.0

    @property
    def cross_polarization_power_std(self) -> float:
        return 3.0

    @property
    def num_clusters(self) -> int:
        return 20

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return max(0.25, 6.5622 - 3.4084 * log10(self._center_frequency * 1e-9)) * 1e-9

    @property
    def cluster_aod_spread(self) -> float:
        return 2.0

    @property
    def cluster_aoa_spread(self) -> float:
        return 15.0

    @property
    def cluster_zoa_spread(self) -> float:
        return 7.0

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.0

    @property
    def _center_frequency(self) -> float:
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))  # type: ignore


class UrbanMacrocellsOutsideToInside(UrbanMacrocellsNoLineOfSightBase, Serializable):
    """3GPP cluster delay line preset modeling an urban macrocell scenario with
    the linked wireless devices being outside and inside a building, respectively.

    Refer to the :footcite:t:`3GPP:TR38901` for detailed information.

    The following minimal example outlines how to configure the channel model
    within the context of a :doc:`simulation.simulation.Simulation`:

    .. literalinclude:: ../scripts/examples/channel_cdl_urban_macrocells_o2i.py
       :language: python
       :linenos:
       :lines: 12-40
    """

    yaml_tag = "UMa02I"
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
        return 11e-9

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
    def _center_frequency(self) -> float:
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))  # type: ignore
