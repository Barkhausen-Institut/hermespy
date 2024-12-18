# -*- coding: utf-8 -*-

from __future__ import annotations
from math import atan
from typing import Mapping, Set, Type

import numpy as np
from h5py import Group

from hermespy.core.factory import Serializable
from .cluster_delay_lines import (
    ClusterDelayLineBase,
    ClusterDelayLineRealizationParameters,
    ClusterDelayLineSample,
    ClusterDelayLineSampleParameters,
    ClusterDelayLineRealization,
    O2IState,
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


class RuralMacrocellsRealization(ClusterDelayLineRealization[O2IState]):
    """Realization of a rural cluster delay line channel model."""

    def __init__(
        self,
        expected_state: O2IState | None,
        state_realization: ConsistentRealization,
        los_realization: ConsistentRealization,
        nlos_realization: ConsistentRealization,
        o2i_realization: ConsistentRealization,
        parameters: ClusterDelayLineRealizationParameters,
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

            o2i_realization (ConsistentRealization):
                Realization of a spatially consistent random number generator for small-scale parameters in the O2I state.

            parameters (ClusterDelayLineRealizationParameters):
                General parameters of the cluster delay line realization.

            sample_hooks (Set[ChannelSampleHook[ClusterDelayLineSample]]):
                Set of hooks to be called when a channel sample is generated.

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
        self.__o2i_realization = o2i_realization

    # Table 7.4.4-1 in TR 138.901 v17.0.0
    @staticmethod
    def _pathloss_dB(state: O2IState, parameters: ClusterDelayLineSampleParameters) -> float:

        if state == O2IState.O2I:
            return 0.0

        h_BS = min(150, max(5, parameters.base_height))
        h_UT = min(10, max(1, parameters.terminal_height))
        h = 5  # Average building height in meters
        PL1 = (
            20
            * np.log10(
                40 * np.pi * parameters.distance_3d * parameters.carrier_frequency * 1e-9 / 3
            )
            + min(0.03 * h**1.72, 10) * np.log10(parameters.distance_3d)
            - min(0.044 * (h**1.72), 14.77)
            + 0.002 * np.log10(h) * parameters.distance_3d
        )

        # Compute breakpoint distance according to Note 5 in TR 138.901 v17.0.0
        # Note that the carrier frequency is in Hz here
        breakpoint_distance = 2 * np.pi * h_BS * h_UT * parameters.carrier_frequency * 1e-8 / 3
        if parameters.distance_2d <= breakpoint_distance:
            PL_LOS = PL1
        else:
            PL_LOS = PL1 + 40 * np.log10(parameters.distance_3d / breakpoint_distance)

        if state == O2IState.LOS:
            return PL_LOS

        W = 20
        P_NLOS = (
            161.04
            - 7.1 * np.log10(W)
            + 7.5 * np.log10(h)
            - (24.37 - 3.7 * (h / h_BS) ** 2) * np.log10(h_BS)
            + (43.42 - 3.1 * np.log10(h_BS)) * (np.log10(parameters.distance_3d) - 3)
            + 20 * np.log10(parameters.carrier_frequency * 1e-9)
            - (3.2 * (np.log10(11.75 * h_UT)) ** 2 - 4.97)
        )
        return max(PL_LOS, P_NLOS)

    def _small_scale_realization(self, state: O2IState) -> ConsistentRealization:
        if state == O2IState.LOS:
            return self.__los_realization
        elif state == O2IState.NLOS:
            return self.__nlos_realization
        else:
            return self.__o2i_realization

    def _sample_large_scale_state(
        self, state_variable_sample: float, parameters: ClusterDelayLineSampleParameters
    ) -> O2IState:
        # Implementation of the state probabilities according to TR 138.901 v17.0.0 Table 7.4.2-1
        los_probability = np.exp(-(parameters.distance_2d - 10.0) / 1000.0)

        if state_variable_sample < los_probability:
            return O2IState.LOS
        else:
            return O2IState.NLOS

    # Parameters for computing the mean delay spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __delay_spread_mean: Mapping[O2IState, float] = {
        O2IState.LOS: -7.49,
        O2IState.NLOS: -7.43,
        O2IState.O2I: -7.47,
    }

    @staticmethod
    def _delay_spread_mean(state: O2IState, carrier_frequency: float) -> float:
        return RuralMacrocellsRealization.__delay_spread_mean[state]

    # Parameters for computing the standard deviation of the delay spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __delay_spread_std: Mapping[O2IState, float] = {
        O2IState.LOS: 0.55,
        O2IState.NLOS: 0.48,
        O2IState.O2I: 0.24,
    }

    @staticmethod
    def _delay_spread_std(state: O2IState, carrier_frequency: float) -> float:
        return RuralMacrocellsRealization.__delay_spread_std[state]

    # Parameters for computing the mean angle of departure spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __aod_spread_mean: Mapping[O2IState, float] = {
        O2IState.LOS: 0.9,
        O2IState.NLOS: 0.95,
        O2IState.O2I: 0.67,
    }

    @staticmethod
    def _aod_spread_mean(state: O2IState, carrier_frequency: float) -> float:
        return RuralMacrocellsRealization.__aod_spread_mean[state]

    # Parameters for computing the standard deviation of the angle of departure spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __aod_spread_std: Mapping[O2IState, float] = {
        O2IState.LOS: 0.38,
        O2IState.NLOS: 0.45,
        O2IState.O2I: 0.18,
    }

    @staticmethod
    def _aod_spread_std(state: O2IState, carrier_frequency: float) -> float:
        return RuralMacrocellsRealization.__aod_spread_std[state]

    # Parameters for computing the mean angle of arrival spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __aoa_spread_mean: Mapping[O2IState, float] = {
        O2IState.LOS: 1.52,
        O2IState.NLOS: 1.52,
        O2IState.O2I: 1.66,
    }

    @staticmethod
    def _aoa_spread_mean(state: O2IState, carrier_frequency: float) -> float:
        return RuralMacrocellsRealization.__aoa_spread_mean[state]

    # Parameters for computing the standard deviation of the angle of arrival spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __aoa_spread_std: Mapping[O2IState, float] = {
        O2IState.LOS: 0.24,
        O2IState.NLOS: 0.13,
        O2IState.O2I: 0.21,
    }

    @staticmethod
    def _aoa_spread_std(state: O2IState, carrier_frequency: float) -> float:
        return RuralMacrocellsRealization.__aoa_spread_std[state]

    # Parameters for computing the mean zenith of arrival spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __zoa_spread_mean: Mapping[O2IState, float] = {
        O2IState.LOS: 0.47,
        O2IState.NLOS: 0.58,
        O2IState.O2I: 0.93,
    }

    @staticmethod
    def _zoa_spread_mean(state: O2IState, carrier_frequency: float) -> float:
        return RuralMacrocellsRealization.__zoa_spread_mean[state]

    # Parameters for computing the standard deviation of the zenith of arrival spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __zoa_spread_std: Mapping[O2IState, float] = {
        O2IState.LOS: 0.40,
        O2IState.NLOS: 0.37,
        O2IState.O2I: 0.22,
    }

    @staticmethod
    def _zoa_spread_std(state: O2IState, carrier_frequency: float) -> float:
        return RuralMacrocellsRealization.__zoa_spread_std[state]

    @staticmethod
    def _rice_factor_mean() -> float:
        return 7.0

    @staticmethod
    def _rice_factor_std() -> float:
        return 4.0

    # Delay scaling factors for different LOS states
    # TR 138.901 v17.0.0 Table 7.5-6
    __delay_scaling: Mapping[O2IState, float] = {
        O2IState.LOS: 3.8,
        O2IState.NLOS: 1.7,
        O2IState.O2I: 1.7,
    }

    @staticmethod
    def _delay_scaling(state: O2IState) -> float:
        return RuralMacrocellsRealization.__delay_scaling[state]

    # Mean cross-polarization power ratio for different LOS states
    # TR 138.901 v17.0.0 Table 7.5-6
    __cross_polarization_power_mean: Mapping[O2IState, float] = {
        O2IState.LOS: 12.0,
        O2IState.NLOS: 7.0,
        O2IState.O2I: 7.0,
    }

    @staticmethod
    def _cross_polarization_power_mean(state: O2IState) -> float:
        return RuralMacrocellsRealization.__cross_polarization_power_mean[state]

    # Standard deviation of the cross-polarization power ratio for different LOS states
    # TR 138.901 v17.0.0 Table 7.5-6
    __cross_polarization_power_std: Mapping[O2IState, float] = {
        O2IState.LOS: 4.0,
        O2IState.NLOS: 3.0,
        O2IState.O2I: 3.0,
    }

    @staticmethod
    def _cross_polarization_power_std(state: O2IState) -> float:
        return RuralMacrocellsRealization.__cross_polarization_power_std[state]

    # Number of clusters for different LOS states
    # TR 138.901 v17.0.0 Table 7.5-6
    __num_clusters: Mapping[O2IState, int] = {O2IState.LOS: 11, O2IState.NLOS: 10, O2IState.O2I: 10}

    @staticmethod
    def _num_clusters(state: O2IState) -> int:
        return RuralMacrocellsRealization.__num_clusters[state]

    @staticmethod
    def _cluster_delay_spread(state: O2IState, carrier_frequency: float) -> float:
        return 0.0  # pragma: no cover

    @staticmethod
    def _cluster_aod_spread(state: O2IState) -> float:
        return 2.0  # TR 138.901 v17.0.0 Table 7.5-6

    @staticmethod
    def _cluster_aoa_spread(state: O2IState) -> float:
        return 3.0  # TR 138.901 v17.0.0 Table 7.5-6

    @staticmethod
    def _cluster_zoa_spread(state: O2IState) -> float:
        return 3.0  # TR 138.901 v17.0.0 Table 7.5-6

    @staticmethod
    def _cluster_shadowing_std(state: O2IState) -> float:
        return 3.0  # TR 138.901 v17.0.0 Table 7.5-6

    @staticmethod
    def _zod_spread_mean(state: O2IState, parameters: ClusterDelayLineSampleParameters) -> float:
        # Implementation of TR 138.901 v17.0.0 Table 7.5-9
        if state == O2IState.LOS:
            return max(
                -1,
                -0.17 * parameters.distance_2d / 1000
                - 0.01 * (parameters.terminal_height - 1.5)
                + 0.22,
            )
        else:
            return max(
                -1,
                -0.19 * parameters.distance_2d / 1000
                - 0.01 * (parameters.terminal_height - 1.5)
                + 0.28,
            )

    # Standard deviation of the zenith of departure spread
    # TR 138.901 v17.0.0 Table 7.5-9
    __zod_spread_std: Mapping[O2IState, float] = {
        O2IState.LOS: 0.34,
        O2IState.NLOS: 0.30,
        O2IState.O2I: 0.30,
    }

    @staticmethod
    def _zod_spread_std(state: O2IState, parameters: ClusterDelayLineSampleParameters) -> float:
        return RuralMacrocellsRealization.__zod_spread_std[state]

    @staticmethod
    def _zod_offset(state: O2IState, parameters: ClusterDelayLineSampleParameters) -> float:
        # Implementation of TR 138.901 v17.0.0 Table 7.5-9
        if state == O2IState.LOS:
            return 0.0
        else:
            return atan((35 - 3.5) / parameters.distance_2d) - atan(
                (35 - 1.5) / parameters.distance_2d
            )

    def to_HDF(self, group: Group) -> None:
        ClusterDelayLineRealization.to_HDF(self, group)

        self.__los_realization.to_HDF(group.create_group("los_realization"))
        self.__nlos_realization.to_HDF(group.create_group("nlos_realization"))
        self.__o2i_realization.to_HDF(group.create_group("o2i_realization"))

        if self.expected_state is not None:
            group.attrs["expected_state"] = self.expected_state.value

    @classmethod
    def From_HDF(
        cls: Type[RuralMacrocellsRealization],
        group: Group,
        parameters: ClusterDelayLineRealizationParameters,
        sample_hooks: Set[ChannelSampleHook[ClusterDelayLineSample]],
    ) -> RuralMacrocellsRealization:

        state_realization = ConsistentRealization.from_HDF(group["state_realization"])
        los_realization = ConsistentRealization.from_HDF(group["los_realization"])
        nlos_realization = ConsistentRealization.from_HDF(group["nlos_realization"])
        o2i_realization = ConsistentRealization.from_HDF(group["o2i_realization"])
        gain = group.attrs["gain"] if "gain" in group.attrs else 1.0
        if "expected_state" in group.attrs:
            expected_state = O2IState(group.attrs["expected_state"])
        else:
            expected_state = None

        return RuralMacrocellsRealization(
            expected_state,
            state_realization,
            los_realization,
            nlos_realization,
            o2i_realization,
            parameters,
            sample_hooks,
            gain,
        )


class RuralMacrocells(ClusterDelayLineBase[RuralMacrocellsRealization, O2IState], Serializable):
    """3GPP cluster delay line preset modeling a rural scenario."""

    yaml_tag = "RuralMacrocells"
    """YAML serialization tag."""

    @property
    def max_num_clusters(self) -> int:
        return 11

    @property
    def max_num_rays(self) -> int:
        return 20

    @property
    def _large_scale_correlations(self) -> np.ndarray:
        # Large scale cross correlations
        # TR 138.901 v17.0.0 Table 7.5-6
        return np.array(
            [
                #    LOS     NLOS   O2I
                [+0.00, -0.40, +0.00],  # 0: ASD vs DS
                [+0.00, +0.00, +0.00],  # 1: ASA vs DS
                [+0.00, +0.00, +0.00],  # 2: ASA VS SF
                [+0.00, +0.60, +0.00],  # 3: ASD vs SF
                [-0.50, -0.50, +0.00],  # 4: DS vs SF
                [+0.00, +0.00, -+0.70],  # 5: ASD vs ASA
                [+0.00, +0.00, +0.00],  # 6: ASD vs K
                [+0.00, +0.00, +0.00],  # 7: ASA vs K
                [+0.00, +0.00, +0.00],  # 8: DS vs K
                [+0.00, +0.00, +0.00],  # 9: SF vs K
                [+0.01, -0.04, +0.00],  # 10: ZSD vs SF
                [-0.17, -0.25, +0.00],  # 11: ZSA vs SF
                [+0.00, +0.00, +0.00],  # 12: ZSD vs K
                [-0.02, +0.00, +0.00],  # 13: ZSA vs K
                [-0.05, -0.10, -0.60],  # 14: ZSD vs DS
                [+0.27, -0.40, +0.00],  # 15: ZSA vs DS
                [+0.73, +0.42, +0.66],  # 16: ZSD vs ASD
                [-0.14, -0.27, +0.47],  # 17: ZSA vs ASD
                [-0.20, -0.18, -0.55],  # 18: ZSD vs ASA
                [+0.24, +0.26, -0.22],  # 19: ZSA vs ASA
                [-0.07, -0.27, +0.00],  # 20: ZSD vs ZSA
            ],
            dtype=np.float64,
        ).T

    def _initialize_realization(
        self,
        state_generator: ConsistentGenerator,
        parameter_generator: ConsistentGenerator,
        parameters: ClusterDelayLineRealizationParameters,
    ) -> RuralMacrocellsRealization:

        # Generate realizations for each large scale state
        # TR 138.901 v17.0.0 Table 7.6.3.1-2
        state_realization = state_generator.realize(50.0)
        los_realization = parameter_generator.realize(50.0)
        nlos_realization = parameter_generator.realize(60.0)
        o2i_realization = parameter_generator.realize(15.0)

        return RuralMacrocellsRealization(
            self.expected_state,
            state_realization,
            los_realization,
            nlos_realization,
            o2i_realization,
            parameters,
            self.sample_hooks,
            self.gain,
        )

    def _recall_specific_realization(
        self, group: Group, parameters: ClusterDelayLineRealizationParameters
    ) -> RuralMacrocellsRealization:
        return RuralMacrocellsRealization.From_HDF(group, parameters, self.sample_hooks)
