# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Any, Optional, Type

import numpy as np
from ruamel.yaml import SafeRepresenter, MappingNode

from hermespy.core import SerializableEnum
from .fading import MultipathFadingChannel

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Cost259Type(SerializableEnum):
    """Supported model types of the Cost256 channel model"""

    URBAN = 0
    """Urban area"""

    RURAL = 1
    """Rural area"""

    HILLY = 2
    """Hilly terrain"""


class Cost259(MultipathFadingChannel):
    """Cost action 259 multipath fading channel model."""

    yaml_tag = "COST259"
    __model_type: Cost259Type

    def __init__(
        self,
        model_type: Cost259Type = Cost259Type.URBAN,
        gain: float = 1.0,
        los_angle: Optional[float] = None,
        doppler_frequency: Optional[float] = None,
        los_doppler_frequency: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:

            model_type (Cost259Type):
                The model type.

            gain (float, optional):
                Linear power gain factor a signal experiences when being propagated over this realization.
                :math:`1.0` by default.

            los_angle (float, optional):
                Angle phase of the line of sight component within the statistical distribution.

            doppler_frequency (float, optional):
                Doppler frequency shift of the statistical distribution.

            \**kwargs (Any):
                `MultipathFadingChannel` initialization parameters.

        Raises:
           ValueError:
                If `model_type` is not supported.
                If `los_angle` is defined in HILLY model type.
        """

        if model_type == Cost259Type.URBAN:
            delays = 1e-6 * np.array(
                [
                    0,
                    0.217,
                    0.512,
                    0.514,
                    0.517,
                    0.674,
                    0.882,
                    1.230,
                    1.287,
                    1.311,
                    1.349,
                    1.533,
                    1.535,
                    1.622,
                    1.818,
                    1.836,
                    1.884,
                    1.943,
                    2.048,
                    2.140,
                ]
            )
            power_db = np.array(
                [
                    -5.7,
                    -7.6,
                    -10.1,
                    -10.2,
                    -10.2,
                    -11.5,
                    -13.4,
                    -16.3,
                    -16.9,
                    -17.1,
                    -17.4,
                    -19.0,
                    -19.0,
                    -19.8,
                    -21.5,
                    -21.6,
                    -22.1,
                    -22.6,
                    -23.5,
                    -24.3,
                ]
            )
            rice_factors = np.zeros(delays.shape)

        elif model_type == Cost259Type.RURAL:
            delays = 1e-6 * np.array(
                [0, 0.042, 0.101, 0.129, 0.149, 0.245, 0.312, 0.410, 0.469, 0.528]
            )
            power_db = np.array([-5.2, -6.4, -8.4, -9.3, -10.0, -13.1, -15.3, -18.5, -20.4, -22.4])
            rice_factors = np.zeros(delays.shape)

        elif model_type == Cost259Type.HILLY:
            if los_angle is not None:
                raise ValueError(
                    "Model type HILLY does not support line of sight angle configuration"
                )

            delays = 1e-6 * np.array(
                [
                    0,
                    0.356,
                    0.441,
                    0.528,
                    0.546,
                    0.609,
                    0.625,
                    0.842,
                    0.916,
                    0.941,
                    15.0,
                    16.172,
                    16.492,
                    16.876,
                    16.882,
                    16.978,
                    17.615,
                    17.827,
                    17.849,
                    18.016,
                ]
            )
            power_db = np.array(
                [
                    -3.6,
                    -8.9,
                    -10.2,
                    -11.5,
                    -11.8,
                    -12.7,
                    -13.0,
                    -16.2,
                    -17.3,
                    -17.7,
                    -17.6,
                    -22.7,
                    -24.1,
                    -25.8,
                    -25.8,
                    -26.2,
                    -29.0,
                    -29.9,
                    -30.0,
                    -30.7,
                ]
            )
            rice_factors = np.hstack([np.array([np.inf]), np.zeros(delays.size - 1)])
            los_angle = np.arccos(0.7)

        else:
            raise ValueError("Requested model type not supported")

        self.__model_type = Cost259Type(model_type)

        # Convert power and normalize
        power_profile = 10 ** (power_db / 10)
        power_profile /= sum(power_profile)

        # Init base class with pre-defined model parameters
        MultipathFadingChannel.__init__(
            self,
            gain=gain,
            delays=delays,
            power_profile=power_profile,
            rice_factors=rice_factors,
            los_angle=los_angle,
            doppler_frequency=doppler_frequency,
            los_doppler_frequency=los_doppler_frequency,
            **kwargs,
        )

    @property
    def model_type(self) -> Cost259Type:
        """Access the configured model type.

        Returns: The configured model type.
        """

        return self.__model_type

    @classmethod
    def to_yaml(cls: Type[Cost259], representer: SafeRepresenter, node: Cost259) -> MappingNode:
        """Serialize a serializable object to YAML.

        Args:

            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (Serializable):
                The MultipathFadingCost256 instance to be serialized.

        Returns: The serialized YAML node.
        """

        blacklist = set()

        if node.model_type == Cost259Type.HILLY:
            blacklist.add("los_angle")

        return node._mapping_serialization_wrapper(representer, blacklist=blacklist)
