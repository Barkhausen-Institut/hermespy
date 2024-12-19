# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Set, Sequence, Tuple

import numpy as np
from h5py import Group

from hermespy.core import Direction, HDFSerializable, Serializable
from ..channel import ChannelSampleHook, LinkState
from ..consistent import ConsistentGenerator, ConsistentRealization, ConsistentUniform
from .radar import RadarChannelBase, RadarTargetPath, RadarChannelRealization, RadarChannelSample

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SingleTargetRadarChannelRealization(RadarChannelRealization):
    """Realization of a single target radar channel.

    Generated by the :meth:`realize<SingleTargetRadarChannel.realize>` method of :class:`SingleTargetRadarChannel`.
    """

    def __init__(
        self,
        consistent_realization: ConsistentRealization,
        target_range_variable: ConsistentUniform,
        target_azimuth_variable: ConsistentUniform,
        target_zenith_variable: ConsistentUniform,
        target_velocity_variable: ConsistentUniform,
        target_phase_variable: ConsistentUniform,
        target_range: float | Tuple[float, float] | None,
        target_azimuth: float | Tuple[float, float],
        target_zenith: float | Tuple[float, float],
        target_cross_section: float,
        target_velocity: float | np.ndarray | Tuple[float, float],
        attenuate: bool,
        sample_hooks: Set[ChannelSampleHook[RadarChannelSample]],
        gain: float,
    ) -> None:
        """
        Args:

            gain (float):
                Linear power gain factor a signal experiences when being propagated over this realization.
        """

        # Initialize the base class
        RadarChannelRealization.__init__(self, sample_hooks, gain)

        # Initialize class attributes
        self.__consistent_realization = consistent_realization
        self.__target_range_variable = target_range_variable
        self.__target_azimuth_variable = target_azimuth_variable
        self.__target_zenith_variable = target_zenith_variable
        self.__target_velocity_variable = target_velocity_variable
        self.__target_phase_variable = target_phase_variable
        self.__target_range = target_range
        self.__target_azimuth = target_azimuth
        self.__target_zenith = target_zenith
        self.__target_cross_section = target_cross_section
        self.__target_velocity = target_velocity
        self.__attenuate = attenuate

    def _generate_paths(self, state: LinkState) -> Sequence[RadarTargetPath]:
        # If the targe rante is None, then the target is not considered
        if self.__target_range is None:
            return []

        consistent_sample = self.__consistent_realization.sample(
            state.transmitter.position, state.receiver.position
        )

        # Generate the targe's absolute range from the receiver
        if isinstance(self.__target_range, (tuple, list)):
            target_range = float(
                self.__target_range[0]
                + (self.__target_range[1] - self.__target_range[0])
                * self.__target_range_variable.sample(consistent_sample)
            )
        else:
            target_range = self.__target_range

        # Generate the target's azimuth angle of arrival
        if isinstance(self.__target_azimuth, (tuple, list)):
            target_azimuth = float(
                self.__target_azimuth[0]
                + (self.__target_azimuth[1] - self.__target_azimuth[0])
                * self.__target_azimuth_variable.sample(consistent_sample)
            )
        else:
            target_azimuth = self.__target_azimuth

        # Generate the target's zenith angle of arrival
        if isinstance(self.__target_zenith, (tuple, list)):
            target_zenith = float(
                self.__target_zenith[0]
                + (self.__target_zenith[1] - self.__target_zenith[0])
                * self.__target_zenith_variable.sample(consistent_sample)
            )
        else:
            target_zenith = self.__target_zenith

        # Generate the target's direction
        unit_direction = Direction.From_Spherical(target_azimuth, target_zenith).view(np.ndarray)

        # Generate the target's velocity
        if isinstance(self.__target_velocity, (tuple, list)):
            absolute_target_velocity = self.__target_velocity[0] + (
                self.__target_velocity[1] - self.__target_velocity[0]
            ) * self.__target_velocity_variable.sample(consistent_sample)
            target_velocity = unit_direction * absolute_target_velocity
        elif isinstance(self.__target_velocity, np.ndarray):  # pragma: no cover
            target_velocity = self.__target_velocity
        else:
            target_velocity = unit_direction * self.__target_velocity

        target_path = RadarTargetPath(
            unit_direction * target_range,
            target_velocity,
            self.__target_cross_section,
            float(2 * np.pi * self.__target_phase_variable.sample(consistent_sample)),
            self.__attenuate,
            False,
        )

        return [target_path]

    def to_HDF(self, group: Group) -> None:
        self.__consistent_realization.to_HDF(
            HDFSerializable._create_group(group, "consistent_realization")
        )
        if self.__target_range is not None:
            HDFSerializable._range_to_HDF(group, "target_range", self.__target_range)
        HDFSerializable._range_to_HDF(group, "target_azimuth", self.__target_azimuth)
        HDFSerializable._range_to_HDF(group, "target_zenith", self.__target_zenith)
        group.attrs["target_cross_section"] = self.__target_cross_section
        if isinstance(self.__target_velocity, np.ndarray):  # pragma: no cover
            HDFSerializable._write_dataset(group, "target_velocity", self.__target_velocity)
        else:
            HDFSerializable._range_to_HDF(group, "target_velocity", self.__target_velocity)
        group.attrs["attenuate"] = self.__attenuate
        group.attrs["gain"] = self.gain

    @staticmethod
    def From_HDF(
        group: Group,
        target_range_variable: ConsistentUniform,
        target_azimuth_variable: ConsistentUniform,
        target_zenith_variable: ConsistentUniform,
        target_velocity_variable: ConsistentUniform,
        target_phase_variable: ConsistentUniform,
        sample_hooks: Set[ChannelSampleHook[RadarChannelSample]],
    ) -> SingleTargetRadarChannelRealization:
        target_velocity: np.ndarray | Tuple[float, float] | float
        if "target_velocity" in group:  # pragma: no cover
            target_velocity = np.asarray(group["target_velocity"], dtype=np.float64)
        else:
            target_velocity = HDFSerializable._range_from_HDF(group, "target_velocity")

        target_range = None
        if "target_range" in group:  # pragma: no cover
            target_range = HDFSerializable._range_from_HDF(group, "target_range")

        return SingleTargetRadarChannelRealization(
            ConsistentRealization.from_HDF(group["consistent_realization"]),
            target_range_variable,
            target_azimuth_variable,
            target_zenith_variable,
            target_velocity_variable,
            target_phase_variable,
            target_range,
            HDFSerializable._range_from_HDF(group, "target_azimuth"),
            HDFSerializable._range_from_HDF(group, "target_zenith"),
            group.attrs["target_cross_section"],
            target_velocity,
            group.attrs["attenuate"],
            sample_hooks,
            group.attrs["gain"],
        )


class SingleTargetRadarChannel(RadarChannelBase[SingleTargetRadarChannelRealization], Serializable):
    """Model of a radar channel featuring a single reflecting target."""

    yaml_tag = "RadarChannel"

    __target_range: float | Tuple[float, float]
    __radar_cross_section: float
    __target_azimuth: float | Tuple[float, float]
    __target_zenith: float | Tuple[float, float]
    __target_exists: bool
    __target_velocity: float | Tuple[float, float] | np.ndarray

    def __init__(
        self,
        target_range: float | Tuple[float, float],
        radar_cross_section: float,
        target_azimuth: float | Tuple[float, float] = 0.0,
        target_zenith: float | Tuple[float, float] = 0.0,
        target_exists: bool = True,
        velocity: float | Tuple[float, float] | np.ndarray = 0,
        attenuate: bool = True,
        decorrelation_distance: float = float("inf"),
        **kwargs,
    ) -> None:
        """
        Args:

            target_range (float | Tuple[float, float]):
                Absolute distance of target and radar sensor in meters.
                Either a specific distance or a range of minimal and maximal target distance.

            radar_cross_section (float):
                Radar cross section (RCS) of the assumed single-point reflector in m**2

            target_azimuth (float | Tuple[float, float]), optional):
                Target location azimuth angle in radians, considering spherical coordinates.
                Zero by default.

            target_zenith (float | Tuple[float, float]), optional):
                Target location zenith angle in radians, considering spherical coordinates.
                Zero by default.

            target_exists (bool, optional):
                True if a target exists, False if there is only noise/clutter (default 0 True)

            velocity (float | Tuple[float, float] | np.ndarray , optional):
                Velocity as a 3D vector (or as a float), in m/s (default = 0)

            attenuate (bool, optional):
                If True, then signal will be attenuated depending on the range, RCS and losses.
                If False, then received power is equal to transmit power.

        Raises:
            ValueError:
                If radar_cross_section < 0.
                If carrier_frequency <= 0.
                If more than one antenna is considered.
        """

        # Initialize base class
        RadarChannelBase.__init__(self, attenuate=attenuate, **kwargs)

        # Initialize class properties
        self.__consistent_generator = ConsistentGenerator(self)
        self.__target_range_variable = self.__consistent_generator.uniform()
        self.__target_azimuth_variable = self.__consistent_generator.uniform()
        self.__target_zenith_variable = self.__consistent_generator.uniform()
        self.__target_velocity_variable = self.__consistent_generator.uniform()
        self.__target_phase_variable = self.__consistent_generator.uniform()

        self.target_range = target_range
        self.radar_cross_section = radar_cross_section
        self.target_azimuth = target_azimuth
        self.target_zenith = target_zenith
        self.target_exists = target_exists
        self.target_velocity = velocity
        self.decorrelation_distance = decorrelation_distance

    @property
    def target_range(self) -> float | Tuple[float, float]:
        """Absolute distance of target and radar sensor.

        Returns: Target range in meters.

        Raises:

            ValueError: If the range is smaller than zero.
        """

        return self.__target_range

    @target_range.setter
    def target_range(self, value: float | Tuple[float, float]) -> None:
        if isinstance(value, (float, int)):
            if value < 0.0:
                raise ValueError("Target range must be greater or equal to zero")

        elif isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError("Target range span must be a tuple of two")

            if value[1] < value[0]:
                raise ValueError("Target range span second value must be greater than first value")

            if value[0] < 0.0:
                raise ValueError("Target range span minimum must be greater or equal to zero")

        else:
            raise ValueError("Unknown targer range format")

        self.__target_range = value

    @property
    def target_velocity(self) -> float | Tuple[float, float] | np.ndarray:
        """Perceived target velocity.

        Returns: Velocity in m/s.
        """

        return self.__target_velocity

    @target_velocity.setter
    def target_velocity(self, value: float | Tuple[float, float] | np.ndarray) -> None:

        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError("Target velocity span must be a tuple of two")

            if value[1] < value[0]:
                raise ValueError(
                    "Target velocity span second value must be greater than first value"
                )

        self.__target_velocity = value

    @property
    def radar_cross_section(self) -> float:
        """Access configured radar cross section.

        Returns:
            float: radar cross section [m**2]
        """
        return self.__radar_cross_section

    @radar_cross_section.setter
    def radar_cross_section(self, value: float) -> None:
        """Modify the configured number of the radar cross section

        Args:
            value (float): The new RCS.

        Raises:
            ValueError: If `value` is less than zero.
        """

        if value < 0:
            raise ValueError("Radar cross section be greater than or equal to zero")

        self.__radar_cross_section = value

    @property
    def target_azimuth(self) -> float | Tuple[float, float]:
        """Target position azimuth in spherical coordiantes.

        Returns:

            Azimuth angle in radians.
        """

        return self.__target_azimuth

    @target_azimuth.setter
    def target_azimuth(self, value: float | Tuple[float, float]) -> None:
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError("Target azimuth span must be a tuple of two")

            if value[1] < value[0]:
                raise ValueError(
                    "Target azimuth span second value must be greater than first value"
                )

        self.__target_azimuth = value

    @property
    def target_zenith(self) -> float | Tuple[float, float]:
        """Target position zenith in spherical coordiantes.

        Returns:

            Zenith angle in radians.
        """

        return self.__target_zenith

    @target_zenith.setter
    def target_zenith(self, value: float | Tuple[float, float]) -> None:
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError("Target zenith span must be a tuple of two")

            if value[1] < value[0]:
                raise ValueError("Target zenith span second value must be greater than first value")

        self.__target_zenith = value

    @property
    def target_exists(self) -> bool:
        """Does an illuminated target exist?"""

        return self.__target_exists

    @target_exists.setter
    def target_exists(self, value: bool) -> None:
        self.__target_exists = value

    @property
    def decorrelation_distance(self) -> float:
        """Decorrelation distance of the channel.

        Raises:

            ValueError: If the decorrelation distance is smaller than zero.
        """

        return self.__decorrelation_distance

    @decorrelation_distance.setter
    def decorrelation_distance(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Decorrelation distance must be greater or equal to zero")

        self.__decorrelation_distance = value

    def _realize(self) -> SingleTargetRadarChannelRealization:
        return SingleTargetRadarChannelRealization(
            self.__consistent_generator.realize(self.decorrelation_distance),
            self.__target_range_variable,
            self.__target_azimuth_variable,
            self.__target_zenith_variable,
            self.__target_velocity_variable,
            self.__target_phase_variable,
            self.target_range if self.target_exists else None,
            self.target_azimuth,
            self.target_zenith,
            self.radar_cross_section,
            self.target_velocity,
            self.attenuate,
            self.sample_hooks,
            self.gain,
        )

    def recall_realization(self, group: Group) -> SingleTargetRadarChannelRealization:
        return SingleTargetRadarChannelRealization.From_HDF(
            group,
            self.__target_range_variable,
            self.__target_azimuth_variable,
            self.__target_zenith_variable,
            self.__target_velocity_variable,
            self.__target_phase_variable,
            self.sample_hooks,
        )
