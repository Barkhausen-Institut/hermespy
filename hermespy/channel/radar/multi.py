# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Set, Sequence

import numpy as np
from h5py import Group

from hermespy.core import Direction, HDFSerializable, Serializable
from hermespy.simulation.animation import Moveable, Trajectory, TrajectorySample
from ..channel import ChannelSampleHook, LinkState
from ..consistent import ConsistentGenerator, ConsistentRealization, ConsistentUniform
from .radar import (
    RadarChannelBase,
    RadarChannelRealization,
    RadarChannelSample,
    RadarInterferencePath,
    RadarTargetPath,
    RadarPath,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RadarTarget(ABC):
    """Abstract base class of radar targets.

    Radar targets represent reflectors of electromagnetic waves within :class:`RadarChannelBase<hermespy.channel.radar.radar.RadarChannelBase>` instances.
    """

    __static: bool

    def __init__(self, static: bool = False) -> None:
        """
        Args:

            static (bool, optional):
                Is the target visible during null hypothesis testing?
                Disabled by default.
        """

        self.__static = static

    @abstractmethod
    def sample_cross_section(
        self, impinging_direction: Direction, emerging_direction: Direction
    ) -> float:
        """Query the target's radar cross section.


        The target radr cross section is denoted by the vector :math:`\\sigma_{\\ell}`
        within the respective equations.

        Args:

            impinging_direction (Direction):
                Direction from which a far-field source impinges onto the target model.

            emerging_direction (Direction):
                Direction in which the scatter wave leaves the target model.

        Returns: The assumed radar cross section in :math:`m^2`.
        """
        ...  # pragma: no cover

    @abstractmethod
    def sample_trajectory(self, timestamp: float) -> TrajectorySample:
        """Sample the target's trajectory at a given time.

        Args:

            timestamp (float): Time at which to sample the trajectory in seconds.

        Returns: A sample of the trajectory.
        """
        ...  # pragma: no cover

    @property
    def static(self) -> bool:
        """Is the target visible in the null hypothesis?"""

        return self.__static


class RadarCrossSectionModel(ABC):
    """Base class for spatial radar cross section models."""

    @abstractmethod
    def get_cross_section(
        self, impinging_direction: Direction, emerging_direction: Direction
    ) -> float:
        """Query the model's cross section.

        Args:

            impinging_direction (Direction):
                Direction from which a far-field source impinges onto the cross section model.

            emerging_direction (Direction):
                Direction in which the scatter wave leaves the cross section model.

        Returns: The assumed cross section in :math:`m^2`.
        """
        ...  # pragma: no cover


class FixedCrossSection(RadarCrossSectionModel):
    """Model of a fixed cross section.

    Can be interpreted as a spherical target floating in space.
    """

    __cross_section: float

    def __init__(self, cross_section: float) -> None:
        """
        Args:

            cross_section (float):
                The cross section in :math:`\\mathrm{m}^2`.
        """

        self.cross_section = cross_section

    @property
    def cross_section(self) -> float:
        """The assumed cross section.

        Returns: The cross section in :math:`\\mathrm{m}^2`.

        Raises:

            ValueError: For cross sections smaller than zero.
        """

        return self.__cross_section

    @cross_section.setter
    def cross_section(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Radar cross section must be greater or equal to zero")

        self.__cross_section = value

    def get_cross_section(self, _: Direction, __: Direction) -> float:
        return self.__cross_section


class VirtualRadarTarget(Moveable, RadarTarget, Serializable):
    """Model of a spatial radar target only existing within a channe link."""

    yaml_tag = "VirtualTarget"

    __cross_section: RadarCrossSectionModel

    def __init__(
        self,
        cross_section: RadarCrossSectionModel,
        trajectory: Trajectory | None = None,
        static: bool = False,
    ) -> None:
        """
        Args:

            cross_section (RadarCrossSectionModel):
                The assumed cross section model.

            trajectory (Trajectory, optional):
                The assumed trajectory of the target.
                By default, the target is assumed to be static.

            static (bool, optional):
                See :meth:`RadarTarget.static`.
                Disabled by default.
        """

        # Initialize base classes
        Moveable.__init__(self, trajectory)
        RadarTarget.__init__(self, static=static)
        Serializable.__init__(self)

        # Initialize class attributes
        self.cross_section = cross_section

    @property
    def cross_section(self) -> RadarCrossSectionModel:
        """The represented radar cross section model."""

        return self.__cross_section

    @cross_section.setter
    def cross_section(self, value: RadarCrossSectionModel) -> None:
        self.__cross_section = value

    def sample_cross_section(
        self, impinging_direction: Direction, emerging_direction: Direction
    ) -> float:
        return self.cross_section.get_cross_section(impinging_direction, emerging_direction)

    def sample_trajectory(self, timestamp: float) -> TrajectorySample:
        return self.trajectory.sample(timestamp)


class PhysicalRadarTarget(RadarTarget, Serializable):
    """Model of a spatial radar target representing a moveable object.

    The radar target will always be modeled at its moveable global position.
    """

    yaml_tag = "PhysicalTarget"

    __cross_section: RadarCrossSectionModel
    __moveable: Moveable

    def __init__(
        self, cross_section: RadarCrossSectionModel, moveable: Moveable, static: bool = False
    ) -> None:
        """
        Args:

            cross_section (RadarCrossSectionModel):
                The assumed cross section model.

            moveable (Moveable):
                The moveable object this radar target represents.

            static (bool, optional):
                See :meth:`RadarTarget.static`.
                Disabled by default.
        """

        # Initialize base classes
        RadarTarget.__init__(self, static=static)
        Serializable.__init__(self)

        # Initialize properties
        self.cross_section = cross_section
        self.__moveable = moveable

    @property
    def cross_section(self) -> RadarCrossSectionModel:
        """The represented radar cross section model."""

        return self.__cross_section

    @cross_section.setter
    def cross_section(self, value: RadarCrossSectionModel) -> None:
        self.__cross_section = value

    @property
    def moveable(self) -> Moveable:
        """Moveble this radar model is attached to.

        Returns: Handle to the moveable object.
        """

        return self.__moveable

    def sample_cross_section(
        self, impinging_direction: Direction, emerging_direction: Direction
    ) -> float:
        return self.cross_section.get_cross_section(impinging_direction, emerging_direction)

    def sample_trajectory(self, timestamp: float) -> TrajectorySample:
        return self.moveable.trajectory.sample(timestamp)


class MultiTargetRadarChannelRealization(RadarChannelRealization):
    """Realization of a spatial multi target radar channel.

    Generated by the :meth:`realize<MultiTargetRadarChannel.realize>` method of :class:`MultiTargetRadarChannel`.
    """

    __target_realizations: Sequence[RadarTargetPath]

    def __init__(
        self,
        consistent_realization: ConsistentRealization,
        phase_variable: ConsistentUniform,
        targets: Set[RadarTarget],
        interference: bool,
        attenuate: bool,
        sample_hooks: Set[ChannelSampleHook[RadarChannelSample]],
        gain: float,
    ) -> None:

        # Initialize base classes
        RadarChannelRealization.__init__(self, sample_hooks, gain)

        # Initialize class attributes
        self.__consistent_realization = consistent_realization
        self.__phase_variable = phase_variable
        self.__targets = targets
        self.__interference = interference
        self.__attenuate = attenuate

    def __sample_target(self, target: RadarTarget, state: LinkState) -> RadarTargetPath:
        """Realize a single radar target's channel propagation path.

        Args:

            target (RadarTarget):
                The radar target to be realized.

            state (ChannelState):
                The current channel state.

        Returns: The realized propagation path.

        Raises:

            ValueError: If `carrier_frequency` is smaller or equal to zero.
            FloatingError: If transmitter or receiver are not specified.
            RuntimeError: If `target` and the channel's linked devices are located at identical global positions
        """

        # Query target global coordiante system transformations
        trajectory_sample = target.sample_trajectory(state.time)
        target_backwards_transform = trajectory_sample.pose
        target_forwards_transform = trajectory_sample.pose.invert()

        # Make sure the transmitter / receiver positions don't collide with target locations
        # This implicitly violates the far-field assumption and leads to numeric instabilities
        if np.array_equal(target_forwards_transform.translation, state.transmitter.position):
            raise RuntimeError(
                "Radar channel transmitter position colliding with an assumed target location"
            )

        if np.array_equal(target_forwards_transform.translation, state.receiver.position):
            raise RuntimeError(
                "Radar channel receiver position colliding with an assumed target location"
            )

        # Compute the impinging and emerging far-field wave direction from the target in local target coordinates
        target_impinging_direction = target_backwards_transform.transform_direction(
            target_forwards_transform.translation - state.transmitter.position, normalize=True
        )
        target_emerging_direction = target_backwards_transform.transform_direction(
            state.receiver.position - target_forwards_transform.translation, normalize=True
        )

        # Query the radar cross section from the target's model given impinging and emerging directions
        cross_section = target.sample_cross_section(
            target_impinging_direction, target_emerging_direction
        )

        # Query reflection phase shift
        consistent_sample = self.__consistent_realization.sample(
            target_forwards_transform.translation, state.receiver.position
        )
        reflection_phase = float(2 * np.pi * self.__phase_variable.sample(consistent_sample))

        # Return realized information wrapped in a target realization dataclass
        return RadarTargetPath(
            target_forwards_transform.translation,
            trajectory_sample.velocity,
            cross_section,
            reflection_phase,
            self.__attenuate,
            target.static,
        )

    def _generate_paths(self, state: LinkState) -> Sequence[RadarPath]:

        paths: List[RadarPath] = [self.__sample_target(target, state) for target in self.__targets]

        if self.__interference and np.any(state.transmitter.position != state.receiver.position):
            paths.append(RadarInterferencePath(self.__attenuate, True))

        return paths

    def to_HDF(self, group: Group) -> None:
        self.__consistent_realization.to_HDF(
            HDFSerializable._create_group(group, "consistent_realization")
        )
        group.attrs["gain"] = self.gain
        group.attrs["attenuate"] = self.__attenuate
        group.attrs["interference"] = self.__interference

    @staticmethod
    def From_HDF(
        group: Group,
        phase_variable: ConsistentUniform,
        targets: Set[RadarTarget],
        sample_hooks: Set[ChannelSampleHook[RadarChannelSample]],
    ) -> MultiTargetRadarChannelRealization:
        consistent_realization = ConsistentRealization.from_HDF(group["consistent_realization"])
        gain = group.attrs["gain"]
        attenuate = group.attrs["attenuate"]
        interference = group.attrs["interference"]

        return MultiTargetRadarChannelRealization(
            consistent_realization,
            phase_variable,
            targets,
            interference,
            attenuate,
            sample_hooks,
            gain,
        )


class MultiTargetRadarChannel(RadarChannelBase[MultiTargetRadarChannelRealization], Serializable):
    """Model of a spatial radar channel featuring multiple reflecting targets."""

    yaml_tag = "SpatialRadarChannel"

    interfernce: bool
    """Consider interference between linked devices.

    Only applies in the bistatic case, where transmitter and receiver are two dedicated device instances.
    """

    __targets: Set[RadarTarget]

    def __init__(
        self,
        attenuate: bool = True,
        interference: bool = True,
        decorrelation_distance: float = float("inf"),
        *args,
        **kwargs,
    ) -> None:
        """
        Args:

            attenuate (bool, optional):
                Should the propagated signal be attenuated during propagation modeling?
                Enabled by default.

            interference (bool, optional):
                Should the channel model consider interference between the linked devices?
                Enabled by default.

            decorrelation_distance (float, optional):
                Distance at which the channel's random variable realizations are considered uncorrelated.
                :math:`\\infty` by default, meaning the channel is static in space.
        """

        # Initialize base classes
        RadarChannelBase.__init__(self, attenuate, *args, **kwargs)
        Serializable.__init__(self)

        # Initialize attributes
        self.interference = interference
        self.decorrelation_distance = decorrelation_distance
        self.__targets = set()

        self.__consistent_generator = ConsistentGenerator(self)
        self.__phase_variable = self.__consistent_generator.uniform()

    @property
    def decorrelation_distance(self) -> float:
        """Decorrelation distance of the radar channel.

        Raises:

            ValueError: For decorrelation distances smaller than zero.
        """

        return self.__decorrelation_distance

    @decorrelation_distance.setter
    def decorrelation_distance(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Decorrelation distance must be greater or equal to zero")

        self.__decorrelation_distance = value

    @property
    def targets(self) -> Set[RadarTarget]:
        """Set of targets considered within the radar channel."""

        return self.__targets

    def add_target(self, target: RadarTarget) -> None:
        """Add a new target to the radar channel.

        Args:

            target (RadarTarget):
                Target to be added.
        """

        if target not in self.targets:
            self.__targets.add(target)

    def make_target(
        self, moveable: Moveable, cross_section: RadarCrossSectionModel, *args, **kwargs
    ) -> PhysicalRadarTarget:
        """Declare a moveable to be a target within the radar channel.

        Args:

            moveable (Moveable):
                Moveable to be declared as a target.

            cross_section (RadarCrossSectionModel):
                Radar cross section model of the target.

            *args:
                Additional positional arguments passed to the target's constructor.

            **kwargs:
                Additional keyword arguments passed to the target's constructor.

        Returns:

            PhysicalRadarTarget: The newly created target.
        """

        target = PhysicalRadarTarget(cross_section, moveable, *args, **kwargs)
        self.add_target(target)

        return target

    def _realize(self) -> MultiTargetRadarChannelRealization:
        return MultiTargetRadarChannelRealization(
            self.__consistent_generator.realize(self.decorrelation_distance),
            self.__phase_variable,
            self.targets,
            self.interference,
            self.attenuate,
            self.sample_hooks,
            self.gain,
        )

    def recall_realization(self, group: Group) -> MultiTargetRadarChannelRealization:
        return MultiTargetRadarChannelRealization.From_HDF(
            group, self.__phase_variable, self.targets, self.sample_hooks
        )
