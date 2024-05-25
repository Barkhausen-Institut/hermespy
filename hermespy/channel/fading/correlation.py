# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Union

import numpy as np

from hermespy.core import FloatingError, Serializable, SerializableEnum
from .fading import AntennaCorrelation

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DeviceType(SerializableEnum):
    """3GPP device type"""

    BASE_STATION = 0
    """Base station"""

    TERMINAL = 1
    """Mobile terminal"""


class CorrelationType(SerializableEnum):
    """3GPP correlation type"""

    LOW = 0.0, 0.0
    """Low antenna correlation"""

    MEDIUM = 0.3, 0.3
    """Medium antenna correlation"""

    MEDIUM_A = 0.3, 0.3874
    """Medium antenna correlation"""

    HIGH = 0.9, 0.9
    """High antenna correlation"""


class StandardAntennaCorrelation(Serializable, AntennaCorrelation):
    """3GPP 5G Multipath fading standardized antenna correlations"""

    yaml_tag = "StandardCorrelation"
    """YAML serialization tag"""

    __device_type: DeviceType  # The assumed device
    __correlation: CorrelationType  # The assumed correlation

    def __init__(
        self,
        device_type: DeviceType | int | str,
        correlation: Union[CorrelationType, str],
        **kwargs,
    ) -> None:
        """
        Args:

            device_type (Union[DeviceType, int, str]):
                The assumed device.

            correlation (Union[CorrelationType, str]):
                The assumed correlation.
        """

        self.device_type = DeviceType.from_parameters(device_type)
        self.correlation = CorrelationType.from_parameters(correlation)

        AntennaCorrelation.__init__(self, **kwargs)

    @property
    def device_type(self) -> DeviceType:
        """Assumed 3GPP device type."""

        return self.__device_type

    @device_type.setter
    def device_type(self, value: DeviceType) -> None:
        self.__device_type = value

    @property
    def correlation(self) -> CorrelationType:
        """Assumed 3GPP standard correlation type."""

        return self.__correlation

    @correlation.setter
    def correlation(self, value: CorrelationType) -> None:
        self.__correlation = value

    @property
    def covariance(self) -> np.ndarray:
        if self.device is None:
            raise FloatingError(
                "Error trying to compute the covariance matrix of an unknown device"
            )

        f = self.__correlation.value[self.__device_type.value]
        n = self.device.num_antennas

        if n == 1:
            return np.ones((1, 1), dtype=complex)

        if n == 2:
            return np.array([[1, f], [f, 1]], dtype=complex)

        if n == 4:
            return np.array(
                [
                    [1, f ** (1 / 9), f ** (4 / 9), f],
                    [f ** (1 / 9), 1, f ** (1 / 9), f ** (4 / 9)],
                    [f ** (4 / 9), f ** (1 / 9), 1, f ** (1 / 9)],
                    [f, f ** (4 / 9), f ** (1 / 9), 1],
                ],
                dtype=complex,
            )

        raise RuntimeError(
            f"3GPP standard antenna covariance is only defined for 1, 2 and 4 antennas, device has {n} antennas"
        )
