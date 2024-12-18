# -*- coding: utf-8 -*-

from __future__ import annotations
from enum import Enum
from typing import Mapping, Set, Tuple, Type
from math import exp, log10

import numpy as np
from h5py import Group

from hermespy.core.factory import Serializable
from .cluster_delay_lines import (
    ClusterDelayLineBase,
    ClusterDelayLineSample,
    ClusterDelayLineSampleParameters,
    ClusterDelayLineRealizationParameters,
    ClusterDelayLineRealization,
    LOSState,
)
from ..channel import ChannelSampleHook
from ..consistent import ConsistentGenerator, ConsistentRealization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class OfficeType(Enum):
    """Type of office."""

    MIXED = 0
    """Mixed office"""

    OPEN = 1
    """Open office"""


class IndoorOfficeRealization(ClusterDelayLineRealization[LOSState]):
    """Realization of an indoor office cluster delay line model."""

    __los_realization: ConsistentRealization
    __nlos_realization: ConsistentRealization
    __office_type: OfficeType

    def __init__(
        self,
        expected_state: LOSState | None,
        state_realization: ConsistentRealization,
        los_realization: ConsistentRealization,
        nlos_realization: ConsistentRealization,
        parameters: ClusterDelayLineRealizationParameters,
        office_type: OfficeType,
        sample_hooks: Set[ChannelSampleHook[ClusterDelayLineSample]],
        gain: float = 1.0,
    ) -> None:
        """
        Args:

            expected_state (O2IState | None):
                Expected large-scale state of the channel.
                If not specified, the large-scale state is randomly generated.

            state_realization (ConsistentRealization):
                Realization of a spatially consistent random number generator for the large-scale state.

            los_realization (ConsistentRealization):
                Realization of a spatially consistent random number generator for small-scale parameters in the LOS state.

            nlos_realization (ConsistentRealization):
                Realization of a spatially consistent random number generator for small-scale parameters in the NLOS state.

            parameters (ClusterDelayLineRealizationParameters):
                General parameters of the cluster delay line realization.

            office_type (OfficeType):
                Type of the modeled office.

            sample_hooks (Set[ChannelSampleHook[ClusterDelayLineSample]]):
                Hooks to be called when a channel sample is generated.

            gain (float, optional):
                Linear amplitude scaling factor if signals propagated over the channel.
        """

        # Initialize base class
        ClusterDelayLineRealization.__init__(
            self, expected_state, state_realization, parameters, sample_hooks, gain
        )

        # Initialize class attributes
        self.__los_realization = los_realization
        self.__nlos_realization = nlos_realization
        self.__office_type = office_type

    # Table 7.4.4-1 in TR 138.901 v17.0.0
    @staticmethod
    def _pathloss_dB(state: LOSState, parameters: ClusterDelayLineSampleParameters) -> float:
        PL_LOS = (
            32.4
            + 17.3 * log10(parameters.distance_3d)
            + 20 * log10(parameters.carrier_frequency * 1e-9)
        )

        if state == LOSState.LOS:
            return PL_LOS

        PL_NLOS = (
            38.3 * log10(parameters.distance_3d)
            + 17.3
            + 24.9 * log10(parameters.carrier_frequency * 1e-9)
        )
        return max(PL_LOS, PL_NLOS)

    def _small_scale_realization(self, state: LOSState) -> ConsistentRealization:
        if state == LOSState.LOS:
            return self.__los_realization
        else:
            return self.__nlos_realization

    def _sample_large_scale_state(
        self, state_variable_sample: float, parameters: ClusterDelayLineSampleParameters
    ) -> LOSState:  # pragma: no cover
        # Implementation of TR 138.901 v17.0.0 Table 7.4.2-1
        if self.__office_type == OfficeType.MIXED:
            if parameters.distance_2d <= 1.2:
                los_probability = 1.0
            elif parameters.distance_2d < 6.5:
                los_probability = exp(-(parameters.distance_2d - 1.2) / 4.7)
            else:
                los_probability = exp(-(parameters.distance_2d - 6.5) / 32.6) * 0.32
        elif self.__office_type == OfficeType.OPEN:
            if parameters.distance_2d <= 5:
                los_probability = 1.0
            elif parameters.distance_2d <= 49:
                los_probability = exp(-(parameters.distance_2d - 5) / 70.8)
            else:
                los_probability = exp(-(parameters.distance_2d - 49) / 211.7) * 0.54

        if state_variable_sample < los_probability:
            return LOSState.LOS
        else:
            return LOSState.NLOS

    @staticmethod
    def __parameter_dependency(carrier_frequency: float, factor: float, summand: float) -> float:
        """An implementation of the frequently used equation

        .. math::

           y = a \\log_{10}(1 + f_c) + b

        Args:
            carrier_frequency (float): Carrier frequency
            factor (float): Factor scaling the logarithmic frequency dependency.
            summand (float): Added constant.

        Returns: The result.
        """

        # Note that the standard does not lower-bound the frequency for the indoor-office scenario
        # This might be an error!!!!
        fc = carrier_frequency * 1e-9
        return factor * log10(1 + fc) + summand

    # Parameters for computing the mean delay spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __delay_spread_mean: Mapping[LOSState, Tuple[float, float]] = {
        LOSState.LOS: (-0.01, -7.629),
        LOSState.NLOS: (-0.28, -7.173),
    }

    @staticmethod
    def _delay_spread_mean(state: LOSState, carrier_frequency: float) -> float:
        mean_parameters = IndoorOfficeRealization.__delay_spread_mean[state]
        return IndoorOfficeRealization.__parameter_dependency(carrier_frequency, *mean_parameters)

    # Parameters for computing the standard deviation of the delay spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __delay_spread_std: Mapping[LOSState, Tuple[float, float]] = {
        LOSState.LOS: (0.0, 0.18),
        LOSState.NLOS: (0.1, 0.055),
    }

    @staticmethod
    def _delay_spread_std(state: LOSState, carrier_frequency: float) -> float:
        std_parameters = IndoorOfficeRealization.__delay_spread_std[state]
        return IndoorOfficeRealization.__parameter_dependency(carrier_frequency, *std_parameters)

    # Parameters for computing the mean angle of departure spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __aod_spread_mean: Mapping[LOSState, float] = {LOSState.LOS: 1.60, LOSState.NLOS: 1.62}

    @staticmethod
    def _aod_spread_mean(state: LOSState, carrier_frequency: float) -> float:
        return IndoorOfficeRealization.__aod_spread_mean[state]

    # Parameters for computing the standard deviation of the angle of departure spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __aod_spread_std: Mapping[LOSState, float] = {LOSState.LOS: 0.18, LOSState.NLOS: 0.25}

    @staticmethod
    def _aod_spread_std(state: LOSState, carrier_frequency: float) -> float:
        return IndoorOfficeRealization.__aod_spread_std[state]

    # Parameters for computing the mean angle of arrival spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __aoa_spread_mean: Mapping[LOSState, Tuple[float, float]] = {
        LOSState.LOS: (-0.19, 1.781),
        LOSState.NLOS: (-0.11, 1.863),
    }

    @staticmethod
    def _aoa_spread_mean(state: LOSState, carrier_frequency: float) -> float:
        mean_parameters = IndoorOfficeRealization.__aoa_spread_mean[state]
        return IndoorOfficeRealization.__parameter_dependency(carrier_frequency, *mean_parameters)

    # Parameters for computing the standard deviation of the angle of arrival spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __aoa_spread_std: Mapping[LOSState, Tuple[float, float]] = {
        LOSState.LOS: (0.12, 0.119),
        LOSState.NLOS: (0.12, 0.059),
    }

    @staticmethod
    def _aoa_spread_std(state: LOSState, carrier_frequency: float) -> float:
        std_parameters = IndoorOfficeRealization.__aoa_spread_std[state]
        return IndoorOfficeRealization.__parameter_dependency(carrier_frequency, *std_parameters)

    # Parameters for computing the mean zenith of arrival spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __zoa_spread_mean: Mapping[LOSState, Tuple[float, float]] = {
        LOSState.LOS: (-0.26, 1.44),
        LOSState.NLOS: (-0.15, 1.387),
    }

    @staticmethod
    def _zoa_spread_mean(state: LOSState, carrier_frequency: float) -> float:
        mean_parameters = IndoorOfficeRealization.__zoa_spread_mean[state]
        return IndoorOfficeRealization.__parameter_dependency(carrier_frequency, *mean_parameters)

    # Parameters for computing the standard deviation of the zenith of arrival spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __zoa_spread_std: Mapping[LOSState, Tuple[float, float]] = {
        LOSState.LOS: (-0.04, 0.264),
        LOSState.NLOS: (-0.09, 0.746),
    }

    @staticmethod
    def _zoa_spread_std(state: LOSState, carrier_frequency: float) -> float:
        std_parameters = IndoorOfficeRealization.__zoa_spread_std[state]
        return IndoorOfficeRealization.__parameter_dependency(carrier_frequency, *std_parameters)

    @staticmethod
    def _rice_factor_mean() -> float:
        return 7.0

    @staticmethod
    def _rice_factor_std() -> float:
        return 4.0

    # Delay scaling factors for different LOS states
    # TR 138.901 v17.0.0 Table 7.5-6
    __delay_scaling: Mapping[LOSState, float] = {LOSState.LOS: 3.6, LOSState.NLOS: 3.0}

    @staticmethod
    def _delay_scaling(state: LOSState) -> float:
        return IndoorOfficeRealization.__delay_scaling[state]

    # Mean cross-polarization power ratio for different LOS states
    # TR 138.901 v17.0.0 Table 7.5-6
    __cross_polarization_power_mean: Mapping[LOSState, float] = {
        LOSState.LOS: 11.0,
        LOSState.NLOS: 10.0,
    }

    @staticmethod
    def _cross_polarization_power_mean(state: LOSState) -> float:
        return IndoorOfficeRealization.__cross_polarization_power_mean[state]

    @staticmethod
    def _cross_polarization_power_std(state: LOSState) -> float:
        # TR 138.901 v17.0.0 Table 7.5-6
        return 4.0

    # Number of clusters for different LOS states
    # TR 138.901 v17.0.0 Table 7.5-6
    __num_clusters: Mapping[LOSState, int] = {LOSState.LOS: 15, LOSState.NLOS: 19}

    @staticmethod
    def _num_clusters(state: LOSState) -> int:
        return IndoorOfficeRealization.__num_clusters[state]

    @staticmethod
    def _cluster_aod_spread(state: LOSState) -> float:
        return 5.0

    # RMS cluster azimuth of arrival spread for different LOS states in degrees
    # TR 138.901 v17.0.0 Table 7.5-6
    __cluster_aoa_spread: Mapping[LOSState, float] = {LOSState.LOS: 8.0, LOSState.NLOS: 11.0}

    @staticmethod
    def _cluster_aoa_spread(state: LOSState) -> float:
        return IndoorOfficeRealization.__cluster_aoa_spread[state]

    @staticmethod
    def _cluster_zoa_spread(state: LOSState) -> float:
        return 9.0

    # Standard deviation of the shadowing for different LOS states in dB
    # TR 138.901 v17.0.0 Table 7.5-6
    __cluster_shadowing_std: Mapping[LOSState, float] = {LOSState.LOS: 6.0, LOSState.NLOS: 3.0}

    @staticmethod
    def _cluster_shadowing_std(state: LOSState) -> float:
        return IndoorOfficeRealization.__cluster_shadowing_std[state]

    @staticmethod
    def _zod_spread_mean(state: LOSState, parameters: ClusterDelayLineSampleParameters) -> float:
        # TR 138.901 v17.0.0 Table 7.5-10
        # ToDo: Check if f_c is the carrier frequency in Hz or GHz
        if state == LOSState.LOS:
            fc = max(
                6.0, parameters.carrier_frequency * 1e-9
            )  # See note 4 in TR 138.901 v17.0.0 Table 7.5-10
            return -1.43 * log10(1 + fc) + 2.228
        else:
            return 1.08

    @staticmethod
    def _zod_spread_std(state: LOSState, parameters: ClusterDelayLineSampleParameters) -> float:
        # TR 138.901 v17.0.0 Table 7.5-10
        # ToDo: Check if f_c is the carrier frequency in Hz or GHz
        if state == LOSState.LOS:
            fc = max(
                6.0, parameters.carrier_frequency * 1e-9
            )  # See note 4 in TR 138.901 v17.0.0 Table 7.5-10
            return 0.13 * log10(1 + fc) + 0.30
        else:
            return 0.36

    @staticmethod
    def _zod_offset(state: LOSState, parameters: ClusterDelayLineSampleParameters) -> float:
        # TR 138.901 v17.0.0 Table 7.5-10
        return 0.0

    def to_HDF(self, group: Group) -> None:
        ClusterDelayLineRealization.to_HDF(self, group)

        self.__los_realization.to_HDF(group.create_group("los_realization"))
        self.__nlos_realization.to_HDF(group.create_group("nlos_realization"))
        group.attrs["office_type"] = self.__office_type.value

        if self.expected_state is not None:
            group.attrs["expected_state"] = self.expected_state.value

    @classmethod
    def From_HDF(
        cls: Type[IndoorOfficeRealization],
        group: Group,
        parameters: ClusterDelayLineRealizationParameters,
        sample_hooks: Set[ChannelSampleHook[ClusterDelayLineSample]],
    ) -> IndoorOfficeRealization:

        state_realization = ConsistentRealization.from_HDF(group["state_realization"])
        los_realization = ConsistentRealization.from_HDF(group["los_realization"])
        nlos_realization = ConsistentRealization.from_HDF(group["nlos_realization"])
        gain = group.attrs["gain"] if "gain" in group.attrs else 1.0
        if "expected_state" in group.attrs:
            expected_state = LOSState(group.attrs["expected_state"])
        else:
            expected_state = None
        office_type = OfficeType(group.attrs["office_type"])

        return IndoorOfficeRealization(
            expected_state,
            state_realization,
            los_realization,
            nlos_realization,
            parameters,
            office_type,
            sample_hooks,
            gain,
        )


class IndoorOffice(ClusterDelayLineBase[IndoorOfficeRealization, LOSState], Serializable):
    """3GPP cluster delay line preset modeling an indoor office scenario."""

    yaml_tag = "IndoorOffice"
    """YAML serialization tag."""

    __office_type: OfficeType

    def __init__(self, office_type: OfficeType = OfficeType.MIXED, **kwargs) -> None:
        """

        Args:

            office_type (OfficeType, optional):
                Type of the modeled office.
                If not specified, a mixed office is assumed.

            \**kwargs:
                Additional arguments passed to the base class.
        """

        # Initialize base class
        ClusterDelayLineBase.__init__(self, **kwargs)

        # Initialize class attributes
        self.__office_type = office_type

    @property
    def max_num_clusters(self) -> int:
        return 19

    @property
    def max_num_rays(self) -> int:
        return 20

    @property
    def office_type(self) -> OfficeType:
        """Type of the modeled office."""

        return self.__office_type

    @office_type.setter
    def office_type(self, value: OfficeType) -> None:
        self.__office_type = value

    @property
    def _large_scale_correlations(self) -> np.ndarray:
        # Large scale cross correlations
        # TR 138.901 v17.0.0 Table 7.5-6
        return np.array(
            [
                #    LOS     NLOS
                [+0.60, +0.40],  # 0: ASD vs DS
                [+0.80, +0.00],  # 1: ASA vs DS
                [-0.50, -0.40],  # 2: ASA VS SF
                [-0.40, +0.00],  # 3: ASD vs SF
                [-0.80, -0.50],  # 4: DS vs SF
                [+0.40, +0.00],  # 5: ASD vs ASA
                [+0.00, +0.00],  # 6: ASD vs K
                [+0.00, +0.00],  # 7: ASA vs K
                [-0.50, +0.00],  # 8: DS vs K
                [+0.50, +0.00],  # 9: SF vs K
                [+0.20, +0.00],  # 10: ZSD vs SF
                [+0.30, +0.00],  # 11: ZSA vs SF
                [+0.00, +0.00],  # 12: ZSD vs K
                [+0.10, +0.00],  # 13: ZSA vs K
                [+0.10, -0.27],  # 14: ZSD vs DS
                [+0.20, -0.06],  # 15: ZSA vs DS
                [+0.50, +0.35],  # 16: ZSD vs ASD
                [+0.00, +0.23],  # 17: ZSA vs ASD
                [+0.00, -0.08],  # 18: ZSD vs ASA
                [+0.50, +0.43],  # 19: ZSA vs ASA
                [+0.00, +0.42],  # 20: ZSD vs ZSA
            ],
            dtype=np.float64,
        ).T

    def _initialize_realization(
        self,
        state_generator: ConsistentGenerator,
        parameter_generator: ConsistentGenerator,
        parameters: ClusterDelayLineRealizationParameters,
    ) -> IndoorOfficeRealization:

        # Generate realizations for each large scale state
        # TR 138.901 v17.0.0 Table 7.6.3.1-2
        state_realization = state_generator.realize(10.0)
        los_realization = parameter_generator.realize(10.0)
        nlos_realization = parameter_generator.realize(10.0)

        return IndoorOfficeRealization(
            self.expected_state,
            state_realization,
            los_realization,
            nlos_realization,
            parameters,
            self.__office_type,
            self.sample_hooks,
            self.gain,
        )

    def _recall_specific_realization(
        self, group: Group, parameters: ClusterDelayLineRealizationParameters
    ) -> IndoorOfficeRealization:
        return IndoorOfficeRealization.From_HDF(group, parameters, self.sample_hooks)
