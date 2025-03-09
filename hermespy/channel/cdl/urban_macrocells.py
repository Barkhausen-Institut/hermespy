# -*- coding: utf-8 -*-

from __future__ import annotations
from math import log10
from typing import Mapping, Set, Tuple
from typing_extensions import override

import numpy as np

from hermespy.core import DeserializationProcess, SerializationProcess
from .cluster_delay_lines import (
    ClusterDelayLineBase,
    ClusterDelayLineRealization,
    ClusterDelayLineRealizationParameters,
    ClusterDelayLineSample,
    ClusterDelayLineSampleParameters,
    O2IState,
)
from ..channel import ChannelSampleHook
from ..consistent import ConsistentGenerator, ConsistentRealization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class UrbanMacrocellsRealization(ClusterDelayLineRealization[O2IState]):
    """Realization of an urban street canyon cluster delay line model."""

    def __init__(
        self,
        expected_state: O2IState | None,
        state_realization: ConsistentRealization,
        los_realization: ConsistentRealization,
        nlos_realization: ConsistentRealization,
        o2i_realization: ConsistentRealization,
        parameters: ClusterDelayLineRealizationParameters,
        sample_hooks: Set[ChannelSampleHook[ClusterDelayLineSample]],
        gain=1.0,
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
        self.__o2i_realization = o2i_realization

    # Table 7.4.4-1 in TR 138.901 v17.0.0
    @staticmethod
    def _pathloss_dB(state: O2IState, parameters: ClusterDelayLineSampleParameters) -> float:

        if state == O2IState.O2I:
            return 0.0

        h_BS = 25.0  # Height of the base station in meters
        h_UT = max(1.5, min(22.5, parameters.terminal_height))  # Height of the terminal in meters

        # Note 1 in Table 7.4.4-1 of TR 138.901 v17.0.0
        breakpoint_distance = (
            4 * (h_BS - 1) * (h_UT - 1) * parameters.carrier_frequency * 1e-8 / 3.0
        )

        if parameters.distance_2d < breakpoint_distance:
            PL_LOS = (
                28.0
                + 22.0 * log10(parameters.distance_3d)
                + 20.0 * log10(parameters.carrier_frequency * 1e-9)
            )

        else:
            PL_LOS = (
                28.0
                + 40.0 * log10(parameters.distance_3d)
                + 20.0 * log10(parameters.carrier_frequency * 1e-9)
                - 9.0 * log10((breakpoint_distance) ** 2 + (h_BS - h_UT) ** 2)
            )

        if state == O2IState.LOS:
            return PL_LOS

        PL_NLOS = (
            13.54
            + 39.08 * log10(parameters.distance_3d)
            + 20.0 * log10(parameters.carrier_frequency * 1e-9)
            - 0.6 * (h_UT - 1.5)
        )
        return max(PL_LOS, PL_NLOS)

    def _small_scale_realization(self, state: O2IState) -> ConsistentRealization:
        if state == O2IState.LOS:
            return self.__los_realization
        elif state == O2IState.NLOS:
            return self.__nlos_realization
        else:
            return self.__o2i_realization

    def _sample_large_scale_state(
        self, state_variable_sample: float, parameters: ClusterDelayLineSampleParameters
    ) -> O2IState:  # pragma: no cover
        los_probability = 18 / parameters.distance_3d + np.exp(-parameters.distance_3d / 36.0) * (
            1 - 18 / parameters.distance_3d
        )

        if state_variable_sample < los_probability:
            return O2IState.LOS
        else:
            return O2IState.NLOS

    @staticmethod
    def __parameter_dependency(carrier_frequency: float, summand: float, factor: float) -> float:
        """An implementation of the frequently used equation

        .. math::

           y = a + b * log_{10}(f)

        Args:
            carrier_frequency (float): Carrier frequency
            summand (float): Added constant.
            factor (float): Factor scaling the logarithmic frequency dependency.

        Returns: The result.
        """

        fc = (
            max(6e9, carrier_frequency) * 1e-9
        )  # Frequency is lower-bounded by 2 GHz, according to Note 7 in table 7.5-6 of TR 138.901 v17.0.0
        return summand + factor * log10(fc)

    # Parameters for computing the mean delay spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __delay_spread_mean: Mapping[O2IState, Tuple[float, float]] = {
        O2IState.LOS: (-6.955, -0.0963),
        O2IState.NLOS: (-6.28, -0.204),
        O2IState.O2I: (-6.62, 0),
    }

    @staticmethod
    def _delay_spread_mean(state: O2IState, carrier_frequency: float) -> float:
        mean_parameters = UrbanMacrocellsRealization.__delay_spread_mean[state]
        return UrbanMacrocellsRealization.__parameter_dependency(
            carrier_frequency, *mean_parameters
        )

    # Parameters for computing the standard deviation of the delay spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __delay_spread_std: Mapping[O2IState, float] = {
        O2IState.LOS: 0.66,
        O2IState.NLOS: 0.39,
        O2IState.O2I: 0.32,
    }

    @staticmethod
    def _delay_spread_std(state: O2IState, carrier_frequency: float) -> float:
        return UrbanMacrocellsRealization.__delay_spread_std[state]

    # Parameters for computing the mean angle of departure spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __aod_spread_mean: Mapping[O2IState, Tuple[float, float]] = {
        O2IState.LOS: (1.06, 0.1114),
        O2IState.NLOS: (1.5, -0.1114),
        O2IState.O2I: (1.25, 0),
    }

    @staticmethod
    def _aod_spread_mean(state: O2IState, carrier_frequency: float) -> float:
        mean_parameters = UrbanMacrocellsRealization.__aod_spread_mean[state]
        return UrbanMacrocellsRealization.__parameter_dependency(
            carrier_frequency, *mean_parameters
        )

    # Parameters for computing the standard deviation of the angle of departure spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __aod_spread_std: Mapping[O2IState, float] = {
        O2IState.LOS: 0.28,
        O2IState.NLOS: 0.28,
        O2IState.O2I: 0.42,
    }

    @staticmethod
    def _aod_spread_std(state: O2IState, carrier_frequency: float) -> float:
        return UrbanMacrocellsRealization.__aod_spread_std[state]

    # Parameters for computing the mean angle of arrival spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __aoa_spread_mean: Mapping[O2IState, Tuple[float, float]] = {
        O2IState.LOS: (1.81, 0.0),
        O2IState.NLOS: (2.08, -0.27),
        O2IState.O2I: (1.76, 0.0),
    }

    @staticmethod
    def _aoa_spread_mean(state: O2IState, carrier_frequency: float) -> float:
        mean_parameters = UrbanMacrocellsRealization.__aoa_spread_mean[state]
        return UrbanMacrocellsRealization.__parameter_dependency(
            carrier_frequency, *mean_parameters
        )

    # Parameters for computing the standard deviation of the angle of arrival spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __aoa_spread_std: Mapping[O2IState, float] = {
        O2IState.LOS: 0.20,
        O2IState.NLOS: 0.11,
        O2IState.O2I: 0.16,
    }

    @staticmethod
    def _aoa_spread_std(state: O2IState, carrier_frequency: float) -> float:
        return UrbanMacrocellsRealization.__aoa_spread_std[state]

    # Parameters for computing the mean zenith of arrival spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __zoa_spread_mean: Mapping[O2IState, Tuple[float, float]] = {
        O2IState.LOS: (0.95, 0.0),
        O2IState.NLOS: (1.1512, -0.3236),
        O2IState.O2I: (1.76, 0.0),
    }

    @staticmethod
    def _zoa_spread_mean(state: O2IState, carrier_frequency: float) -> float:
        mean_parameters = UrbanMacrocellsRealization.__zoa_spread_mean[state]
        return UrbanMacrocellsRealization.__parameter_dependency(
            carrier_frequency, *mean_parameters
        )

    # Parameters for computing the standard deviation of the zenith of arrival spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __zoa_spread_std: Mapping[O2IState, float] = {
        O2IState.LOS: 0.16,
        O2IState.NLOS: 0.16,
        O2IState.O2I: 0.43,
    }

    @staticmethod
    def _zoa_spread_std(state: O2IState, carrier_frequency: float) -> float:
        return UrbanMacrocellsRealization.__zoa_spread_std[state]

    @staticmethod
    def _rice_factor_mean() -> float:
        return 9.0

    @staticmethod
    def _rice_factor_std() -> float:
        return 3.5

    # Delay scaling factors for different LOS states
    # TR 138.901 v17.0.0 Table 7.5-6
    __delay_scaling: Mapping[O2IState, float] = {
        O2IState.LOS: 2.5,
        O2IState.NLOS: 2.3,
        O2IState.O2I: 2.2,
    }

    @staticmethod
    def _delay_scaling(state: O2IState) -> float:
        return UrbanMacrocellsRealization.__delay_scaling[state]

    # Mean cross-polarization power ratio for different LOS states
    # TR 138.901 v17.0.0 Table 7.5-6
    __cross_polarization_power_mean: Mapping[O2IState, float] = {
        O2IState.LOS: 8.0,
        O2IState.NLOS: 7.0,
        O2IState.O2I: 9.0,
    }

    @staticmethod
    def _cross_polarization_power_mean(state: O2IState) -> float:
        return UrbanMacrocellsRealization.__cross_polarization_power_mean[state]

    # Standard deviation of the cross-polarization power ratio for different LOS states
    # TR 138.901 v17.0.0 Table 7.5-6
    __cross_polarization_power_std: Mapping[O2IState, float] = {
        O2IState.LOS: 4.0,
        O2IState.NLOS: 3.0,
        O2IState.O2I: 3.0,
    }

    @staticmethod
    def _cross_polarization_power_std(state: O2IState) -> float:
        return UrbanMacrocellsRealization.__cross_polarization_power_std[state]

    # Number of clusters for different LOS states
    # TR 138.901 v17.0.0 Table 7.5-6
    __num_clusters: Mapping[O2IState, int] = {O2IState.LOS: 12, O2IState.NLOS: 20, O2IState.O2I: 12}

    @staticmethod
    def _num_clusters(state: O2IState) -> int:
        return UrbanMacrocellsRealization.__num_clusters[state]

    @staticmethod
    def _cluster_delay_spread(
        state: O2IState, carrier_frequency: float
    ) -> float:  # pragma: no cover
        # Implementation of TR 138.901 v17.0.0 Table 7.5-6
        if state == O2IState.O2I:
            return 11.0
        else:
            return max(0.25, 6.5622 - 3.4084 * log10(carrier_frequency * 1e-9))

    # RMS cluster azimuth of departure spread for different LOS states in degrees
    # TR 138.901 v17.0.0 Table 7.5-6
    __cluster_aod_spread: Mapping[O2IState, float] = {
        O2IState.LOS: 5.0,
        O2IState.NLOS: 2.0,
        O2IState.O2I: 5.0,
    }

    @staticmethod
    def _cluster_aod_spread(state: O2IState) -> float:
        return UrbanMacrocellsRealization.__cluster_aod_spread[state]

    # RMS cluster azimuth of arrival spread for different LOS states in degrees
    # TR 138.901 v17.0.0 Table 7.5-6
    __cluster_aoa_spread: Mapping[O2IState, float] = {
        O2IState.LOS: 11.0,
        O2IState.NLOS: 15.0,
        O2IState.O2I: 8.0,
    }

    @staticmethod
    def _cluster_aoa_spread(state: O2IState) -> float:
        return UrbanMacrocellsRealization.__cluster_aoa_spread[state]

    # RMS cluster zenith of arrival spread for different LOS states in degrees
    # TR 138.901 v17.0.0 Table 7.5-6
    __cluster_zoa_spread: Mapping[O2IState, float] = {
        O2IState.LOS: 3.0,
        O2IState.NLOS: 3.0,
        O2IState.O2I: 4.0,
    }

    @staticmethod
    def _cluster_zoa_spread(state: O2IState) -> float:
        return UrbanMacrocellsRealization.__cluster_zoa_spread[state]

    # Standard deviation of the shadowing for different LOS states in dB
    # TR 138.901 v17.0.0 Table 7.5-6
    __cluster_shadowing_std: Mapping[O2IState, float] = {
        O2IState.LOS: 3.0,
        O2IState.NLOS: 3.0,
        O2IState.O2I: 4.0,
    }

    @staticmethod
    def _cluster_shadowing_std(state: O2IState) -> float:
        return UrbanMacrocellsRealization.__cluster_shadowing_std[state]

    @staticmethod
    def _zod_spread_mean(state: O2IState, parameters: ClusterDelayLineSampleParameters) -> float:
        # Implementation of TR 138.901 v17.0.0 Table 7.5-7
        if state == O2IState.LOS:
            return (
                max(-0.5, -2.1 * parameters.distance_2d / 1000)
                - 0.01 * (parameters.terminal_height - 1.5)
                + 0.75
            )
        else:
            return (
                max(-0.5, -2.1 * parameters.distance_2d / 1000)
                - 0.01 * (parameters.terminal_height - 1.5)
                + 0.9
            )

    # Standard deviation of the zenith of departure spread for different LOS states
    # TR 138.901 v17.0.0 Table 7.5-7
    __zod_spread_std: Mapping[O2IState, float] = {
        O2IState.LOS: 0.4,
        O2IState.NLOS: 0.49,
        O2IState.O2I: 0.49,
    }

    @staticmethod
    def _zod_spread_std(state: O2IState, parameters: ClusterDelayLineSampleParameters) -> float:
        return UrbanMacrocellsRealization.__zod_spread_std[state]

    @staticmethod
    def _zod_offset(state: O2IState, parameters: ClusterDelayLineSampleParameters) -> float:
        # Implementation of TR 138.901 v17.0.0 Table 7.5-7
        if state == O2IState.LOS:
            return 0.0
        else:
            fc = log10(max(6.0, parameters.carrier_frequency * 1e-9))

            a = 0.208 * fc - 0.782
            b = 25
            c = -0.13 * fc + 2.03
            e = 7.66 * fc - 5.96

            return e - 10 ** (
                a * log10(max(b, parameters.distance_2d))
                + c
                - 0.07 * (parameters.terminal_height - 1.5)
            )

    @override
    def serialize(self, process: SerializationProcess) -> None:
        ClusterDelayLineRealization.serialize(self, process)
        process.serialize_object(self.__los_realization, "los_realization")
        process.serialize_object(self.__nlos_realization, "nlos_realization")
        process.serialize_object(self.__o2i_realization, "o2i_realization")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> UrbanMacrocellsRealization:
        return UrbanMacrocellsRealization(
            expected_state=process.deserialize_object("expected_state", O2IState, None),
            los_realization=process.deserialize_object("los_realization", ConsistentRealization),
            nlos_realization=process.deserialize_object("nlos_realization", ConsistentRealization),
            o2i_realization=process.deserialize_object("o2i_realization", ConsistentRealization),
            sample_hooks=set(),
            **ClusterDelayLineRealization._DeserializeParameters(process),  # type: ignore[arg-type]
        )


class UrbanMacrocells(ClusterDelayLineBase[UrbanMacrocellsRealization, O2IState]):
    """3GPP cluster delay line preset modeling an urban macrocell scenario."""

    @property
    def _large_scale_correlations(self) -> np.ndarray:
        # Large scale cross correlations
        # TR 138.901 v17.0.0 Table 7.5-6
        return np.array(
            [
                #    LOS   NLOS  O2I
                [+0.4, +0.4, +0.4],  # 0: ASD vs DS
                [+0.8, +0.6, +0.4],  # 1: ASA vs DS
                [-0.5, +0.0, +0.0],  # 2: ASA VS SF
                [-0.5, -0.6, +0.2],  # 3: ASD vs SF
                [-0.4, -0.4, -0.5],  # 4: DS vs SF
                [+0.0, +0.4, +0.0],  # 5: ASD vs ASA
                [+0.0, +0.0, +0.0],  # 6: ASD vs K
                [-0.2, +0.0, +0.0],  # 7: ASA vs K
                [-0.4, +0.0, +0.0],  # 8: DS vs K
                [+0.0, +0.0, +0.0],  # 9: SF vs K
                [+0.0, +0.0, +0.0],  # 10: ZSD vs SF
                [-0.8, -0.4, +0.0],  # 11: ZSA vs SF
                [+0.0, +0.0, +0.0],  # 12: ZSD vs K
                [+0.0, +0.0, +0.0],  # 13: ZSA vs K
                [-0.2, -0.5, -0.6],  # 14: ZSD vs DS
                [+0.0, +0.0, -0.2],  # 15: ZSA vs DS
                [+0.5, +0.5, -0.2],  # 16: ZSD vs ASD
                [+0.0, -0.1, +0.0],  # 17: ZSA vs ASD
                [-0.3, +0.0, +0.0],  # 18: ZSD vs ASA
                [+0.4, +0.0, +0.5],  # 19: ZSA vs ASA
                [+0.0, +0.0, +0.5],  # 20: ZSD vs ZSA
            ],
            dtype=np.float64,
        ).T

    @property
    def max_num_clusters(self) -> int:
        return 19

    @property
    def max_num_rays(self) -> int:
        return 20

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> UrbanMacrocells:
        return UrbanMacrocells(**ClusterDelayLineBase._DeserializeParameters(process))  # type: ignore[arg-type]

    def _initialize_realization(
        self,
        state_generator: ConsistentGenerator,
        parameter_generator: ConsistentGenerator,
        parameters: ClusterDelayLineRealizationParameters,
    ) -> UrbanMacrocellsRealization:

        # Generate realizations for each large scale state
        # TR 138.901 v17.0.0 Table 7.6.3.1-2
        state_realization = state_generator.realize(50.0)
        los_realization = parameter_generator.realize(12.0)
        nlos_realization = parameter_generator.realize(15.0)
        o2i_realization = parameter_generator.realize(15.0)

        return UrbanMacrocellsRealization(
            self.expected_state,
            state_realization,
            los_realization,
            nlos_realization,
            o2i_realization,
            parameters,
            self.sample_hooks,
            self.gain,
        )
