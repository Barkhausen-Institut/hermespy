# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import cos, sin, exp, sqrt
from typing import Generic, Literal, overload, Type, TypeVar
from typing_extensions import override

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # type: ignore
from scipy.constants import pi, speed_of_light

from .executable import Executable
from .factory import Serializable, SerializableEnum, SerializationProcess, DeserializationProcess
from .transformation import Direction, Transformable, Transformation
from .visualize import VAT

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class AntennaMode(SerializableEnum):
    """Mode of operation of the antenna."""

    TX = 0
    """Transmit-only antenna"""

    RX = 1
    """Receive-only antenna"""

    DUPLEX = 2
    """Transmit-receive antenna"""


AAT = TypeVar("AAT", bound="AntennaArray")
"""Type of antenna array."""


class Antenna(ABC, Transformable):
    """Base class for the model of a single antenna element within an antenna array."""

    __mode: AntennaMode  # The mode this antenna is operating in, i.e. DUPLEX, TX or RX

    def __init__(
        self, mode: AntennaMode = AntennaMode.DUPLEX, pose: Transformation | None = None
    ) -> None:
        """
        Args:

            mode:
                Antenna's mode of operation.
                By default, a full duplex antenna is assumed.

            pose:
                The antenna's position and orientation with respect to its array.
        """

        # Init base class
        Serializable.__init__(self)
        Transformable.__init__(self, pose=pose)

        # Initialize attributes
        self.__mode = mode

    @property
    def mode(self) -> AntennaMode:
        """Antenna's mode of operation."""

        return self.__mode

    @mode.setter
    def mode(self, value: AntennaMode) -> None:
        # Do nothing if the mode does not change
        if self.__mode == value:
            return

        # Update the antenna's mode
        self.__mode = value

    @abstractmethod
    def copy(self: AT) -> AT:
        """Create a deep copy of the antenna.

        Returns:

            A deep copy of the antenna.
        """
        ...  # pragma: no cover

    @abstractmethod
    def local_characteristics(self, azimuth: float, elevation) -> np.ndarray:
        """Generate a single sample of the antenna's characteristics.

        The polarization is characterized by the angle-dependant field vector

        .. math::

            \\mathbf{F}(\\phi, \\theta) =
            \\begin{pmatrix}
                F_{\\mathrm{H}}(\\phi, \\theta) \\\\
                F_{\\mathrm{V}}(\\phi, \\theta) \\\\
            \\end{pmatrix}

        denoting the horizontal and vertical field components.
        The directional antenna gain can be computed from the polarization vector magnitude

        .. math::

            A(\\phi, \\theta) &= \\lVert \\mathbf{F}(\\phi, \\theta) \\rVert \\\\
                              &= \\sqrt{ F_{\\mathrm{H}}(\\phi, \\theta)^2 + F_{\\mathrm{V}}(\\phi, \\theta)^2 }

        Args:

            azimuth:
                Considered horizontal wave angle in radians :math:`\\phi`.

            elevation:
                Considered vertical wave angle in radians :math:`\\theta`.

        Returns:

            Two dimensional numpy array denoting the horizontal and vertical ploarization components
            of the antenna response vector.
        """
        ...  # pragma: no cover

    def global_characteristics(self, global_direction: Direction) -> np.ndarray:
        """Query the antenna's polarization characteristics towards a certain direction of interest.

        Args:

            global_direction:
                Cartesian direction unit vector of interest.

        Returns:
            Two-dimensional numpy vector representing the antenna's polarization components.
        """

        # Compute the local angle of interest for each antenna element
        local_direction = self.backwards_transformation.transform_direction(
            global_direction, normalize=False
        )

        # Query polarization vector for a-th antenna given local azimuth and zenith angles of interest
        local_antenna_character = self.local_characteristics(*local_direction.to_spherical())

        # Azimuth is denoted by phi in the standard
        # Zenith is denoted by theta in the standard
        azimuth_global, zenith_global = global_direction.to_spherical()
        azimuth_local, zenith_local = local_direction.to_spherical()

        # Phi unit vector implemention of equation (7.1-14) of ETSI TR 138901 version 17.0
        azimuth_global_unit = np.array(
            [-sin(azimuth_global), cos(azimuth_global), 0], dtype=np.float64
        )
        azimuth_local_unit = np.array(
            [-sin(azimuth_local), cos(azimuth_local), 0], dtype=np.float64
        )

        # Theta unit vector implemention of equation (7.1-13) of ETSI TR 138901 version 17.0
        zenith_global_unit = np.array(
            [
                cos(zenith_global) * cos(azimuth_global),
                cos(zenith_global) * sin(azimuth_global),
                -sin(zenith_global),
            ],
            dtype=np.float64,
        )
        zenith_local_unit = np.array(
            [
                cos(zenith_local) * cos(azimuth_local),
                cos(zenith_local) * sin(azimuth_local),
                -sin(zenith_local),
            ],
            dtype=np.float64,
        )

        # Implemention of equation (7.1-12) of ETSI TR 138901 version 17.0
        rotation = self.forwards_transformation[:3, :3]
        local_zenith_transformed = rotation @ zenith_local_unit
        local_azimuth_transformed = rotation @ azimuth_local_unit
        polarization_transformation = np.array(
            [
                [
                    np.inner(zenith_global_unit, local_zenith_transformed),
                    np.inner(zenith_global_unit, local_azimuth_transformed),
                ],
                [
                    np.inner(azimuth_global_unit, local_zenith_transformed),
                    np.inner(azimuth_global_unit, local_azimuth_transformed),
                ],
            ]
        )

        # Implemention of equation (7.1-11) of ETSI TR 138901 version 17.0
        global_antenna_character = polarization_transformation @ local_antenna_character

        # We're finally done
        return global_antenna_character

    def plot_polarization(self, angle_resolution: int = 180) -> Figure:
        """Visualize the antenna polarization depending on the angles of interest.

        Args:

            angle_resolution:
                Resolution of the polarization visualization.


        Returns:

            The created matplotlib figure.

        Raises:

            ValueError:
                If `angle_resolution` is smaller than one.
        """

        with Executable.style_context():
            axes: np.ndarray
            figure, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"})  # type: ignore[assignment]
            figure.suptitle("Antenna Polarization")

            azimuth_angles = 2 * pi * np.arange(angle_resolution) / angle_resolution - pi
            elevation_angles = (
                pi * np.arange(int(0.5 * angle_resolution)) / int(0.5 * angle_resolution) - 0.5 * pi
            )

            azimuth_samples, elevation_samples = np.meshgrid(azimuth_angles, elevation_angles)
            e_surface = np.empty((len(azimuth_angles) * len(elevation_angles), 3), dtype=float)
            e_magnitudes = np.empty(len(azimuth_angles) * len(elevation_angles), dtype=float)
            h_surface = np.empty((len(azimuth_angles) * len(elevation_angles), 3), dtype=float)
            h_magnitudes = np.empty(len(azimuth_angles) * len(elevation_angles), dtype=float)

            for i, (azimuth, elevation) in enumerate(
                zip(azimuth_samples.flat, elevation_samples.flat)
            ):
                e_magnitude, h_magnitude = self.local_characteristics(azimuth, elevation)

                e_magnitudes[i] = e_magnitude
                h_magnitudes[i] = h_magnitude

                e_surface[i, :] = (
                    e_magnitude * cos(azimuth) * cos(elevation),
                    e_magnitude * sin(azimuth) * cos(elevation),
                    e_magnitude * sin(elevation),
                )
                h_surface[i, :] = (
                    h_magnitude * cos(azimuth) * cos(elevation),
                    h_magnitude * sin(azimuth) * cos(elevation),
                    h_magnitude * sin(elevation),
                )

            triangles = tri.Triangulation(azimuth_samples.flatten(), elevation_samples.flatten())

            e_cmap = plt.cm.ScalarMappable(
                norm=colors.Normalize(e_magnitudes.min(), e_magnitudes.max()), cmap="jet"
            )
            e_cmap.set_array(e_magnitudes)
            h_cmap = plt.cm.ScalarMappable(
                norm=colors.Normalize(h_magnitudes.min(), h_magnitudes.max()), cmap="jet"
            )
            h_cmap.set_array(h_magnitudes)

            axes[0].set_title("E-Field")
            axes[0].plot_trisurf(
                e_surface[:, 0],
                e_surface[:, 1],
                e_surface[:, 2],
                triangles=triangles.triangles,
                cmap=e_cmap.cmap,
                norm=e_cmap.norm,
                linewidth=0.0,
            )
            axes[0].set_xlabel("X")
            axes[0].set_ylabel("Y")
            axes[0].set_zlabel("Z")

            axes[1].set_title("H-Field")
            axes[1].plot_trisurf(
                h_surface[:, 0],
                h_surface[:, 1],
                h_surface[:, 2],
                triangles=triangles.triangles,
                cmap=h_cmap.cmap,
                norm=h_cmap.norm,
                linewidth=0.0,
            )
            axes[1].set_xlabel("X")
            axes[1].set_ylabel("Y")
            axes[1].set_zlabel("Z")

            return figure

    def plot_gain(self, angle_resolution: int = 180) -> Figure:
        """Visualize the antenna gain depending on the angles of interest.

        Args:

            angle_resolution:
                Resolution of the polarization visualization.


        Returns:

            The created matplotlib figure.

        Raises:

            ValueError:
                If `angle_resolution` is smaller than one.
        """

        with Executable.style_context():
            axes: Axes3D
            figure, axes = plt.subplots(subplot_kw={"projection": "3d"})
            figure.suptitle("Antenna Gain")

            azimuth_angles = 2 * pi * np.arange(angle_resolution) / angle_resolution - pi
            elevation_angles = (
                pi * np.arange(int(0.5 * angle_resolution)) / int(0.5 * angle_resolution) - 0.5 * pi
            )

            azimuth_samples, elevation_samples = np.meshgrid(azimuth_angles, elevation_angles)
            surface = np.empty((len(azimuth_angles) * len(elevation_angles), 3), dtype=float)
            magnitudes = np.empty(len(azimuth_angles) * len(elevation_angles), dtype=float)

            for i, (azimuth, elevation) in enumerate(
                zip(azimuth_samples.flat, elevation_samples.flat)
            ):
                e_magnitude, h_magnitude = self.local_characteristics(azimuth, elevation)
                magnitude = sqrt(e_magnitude**2 + h_magnitude**2)
                magnitudes[i] = magnitude

                surface[i, :] = (
                    magnitude * cos(azimuth) * cos(elevation),
                    magnitude * sin(azimuth) * cos(elevation),
                    magnitude * sin(elevation),
                )

            triangles = tri.Triangulation(azimuth_samples.flatten(), elevation_samples.flatten())

            cmap = plt.cm.ScalarMappable(
                norm=colors.Normalize(magnitudes.min(), magnitudes.max()), cmap="jet"
            )
            cmap.set_array(magnitudes)

            axes.plot_trisurf(
                surface[:, 0],
                surface[:, 1],
                surface[:, 2],
                triangles=triangles.triangles,
                cmap=cmap.cmap,
                norm=cmap.norm,
                linewidth=0.0,
            )
            axes.set_xlabel("X")
            axes.set_ylabel("Y")
            axes.set_zlabel("Z")

            return figure

    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_integer(self.mode.value, "mode")
        process.serialize_array(self.pose, "pose")

    @classmethod
    def Deserialize_Antenna(
        cls, process: DeserializationProcess
    ) -> tuple[AntennaMode, Transformation]:
        mode = AntennaMode(process.deserialize_integer("mode"))
        pose = process.deserialize_array("pose", np.float64).view(Transformation)
        return mode, pose


AT = TypeVar("AT", bound=Antenna)
"""Type of antenna."""


class IdealAntenna(Antenna, Serializable):
    """Theoretic model of an ideal antenna.

    .. figure:: /images/api_antenna_idealantenna_gain.png
       :alt: Ideal Antenna Gain
       :scale: 70%
       :align: center

       Ideal Antenna Characteristics

    The assumed characteristic is

    .. math::

       \\mathbf{F}(\\phi, \\theta) =
          \\begin{pmatrix}
             \\sqrt{2} \\\\
             \\sqrt{2} \\\\
          \\end{pmatrix}

    resulting in unit gain in every direction.
    """

    def __init__(
        self, mode: AntennaMode = AntennaMode.DUPLEX, pose: Transformation | None = None
    ) -> None:
        """
        Args:

            mode:
                Antenna's mode of operation.
                By default, a full duplex antenna is assumed.

            pose:
                The antenna's position and orientation with respect to its array.
        """

        # Initialize base class
        Antenna.__init__(self, mode, pose)

    def copy(self) -> IdealAntenna:
        return IdealAntenna(self.mode, self.pose.copy())

    def local_characteristics(self, azimuth: float, elevation: float) -> np.ndarray:
        return np.array([2**-0.5, 2**-0.5], dtype=float)

    def serialize(self, process: SerializationProcess) -> None:
        Antenna.serialize(self, process)

    @classmethod
    def Deserialize(cls: Type[IdealAntenna], process: DeserializationProcess) -> IdealAntenna:
        mode, pose = Antenna.Deserialize_Antenna(process)
        return cls(mode, pose)


class LinearAntenna(Antenna, Serializable):
    """Model of a linearly polarized ideal antenna.

    The assumed characteristic is

    .. math::

       \\mathbf{F}(\\theta, \\phi, \\zeta) =
          \\begin{pmatrix}
             \\cos (\\zeta) \\\\
             \\sin (\\zeta) \\\\
          \\end{pmatrix}

    with :math:`zeta = 0` resulting in vertical polarization and :math:`zeta = \\pi / 2` resulting in horizontal polarization.
    """

    __slant: float

    def __init__(
        self,
        mode: AntennaMode = AntennaMode.DUPLEX,
        slant: float = 0.0,
        pose: Transformation | None = None,
    ):
        """Initialize a new linear antenna.

        Args:

            mode:
                Antenna's mode of operation.
                By default, a full duplex antenna is assumed.

            mode:
                Antenna's mode of operation.
                By default, a full duplex antenna is assumed.

            slant:
                Slant of the antenna in radians.

            pose:
                Pose of the antenna.
        """

        # Initialize base class
        Antenna.__init__(self, mode, pose)

        # Initialize class attributes
        self.__slant = slant

    @property
    def slant(self) -> float:
        """Slant of the antenna in radians."""

        return self.__slant

    @slant.setter
    def slant(self, value: float):
        self.__slant = value

    def copy(self) -> LinearAntenna:
        return LinearAntenna(self.mode, self.slant, self.pose.copy())

    def local_characteristics(self, azimuth: float, zenith: float) -> np.ndarray:
        return np.array([cos(self.slant), sin(self.slant)], dtype=np.float64)

    def serialize(self, process: SerializationProcess) -> None:
        Antenna.serialize(self, process)
        process.serialize_floating(self.slant, "slant")

    @classmethod
    def Deserialize(cls: Type[LinearAntenna], process: DeserializationProcess) -> LinearAntenna:
        mode, pose = Antenna.Deserialize_Antenna(process)
        slant = process.deserialize_floating("slant")
        return cls(mode, slant, pose)


class PatchAntenna(Antenna, Serializable):
    """Realistic model of a vertically polarized patch antenna.

    .. figure:: /images/api_antenna_patchantenna_gain.png
       :alt: Patch Antenna Gain
       :scale: 70%
       :align: center

       Patch Antenna Characteristics

    Refer to :footcite:t:`2012:jaeckel` for further information.
    """

    def __init__(
        self, mode: AntennaMode = AntennaMode.DUPLEX, pose: Transformation | None = None
    ) -> None:
        """
        Args:

            mode:
                Antenna's mode of operation.
                By default, a full duplex antenna is assumed.

            pose:
                The antenna's position and orientation with respect to its array.
        """

        # Initialize base class
        Antenna.__init__(self, mode, pose)

    def copy(self) -> PatchAntenna:
        return PatchAntenna(self.mode, self.pose.copy())

    def local_characteristics(self, azimuth: float, elevation: float) -> np.ndarray:
        vertical_azimuth = 0.1 + 0.9 * exp(-1.315 * azimuth**2)
        vertical_elevation = cos(elevation) ** 2

        return np.array([max(0.1, vertical_azimuth * vertical_elevation), 0.0], dtype=float)

    def serialize(self, process: SerializationProcess) -> None:
        Antenna.serialize(self, process)

    @classmethod
    def Deserialize(cls: Type[PatchAntenna], process: DeserializationProcess) -> PatchAntenna:
        mode, pose = Antenna.Deserialize_Antenna(process)
        return cls(mode, pose)


class Dipole(Antenna, Serializable):
    """Model of vertically polarized half-wavelength dipole antenna.

    .. figure:: /images/api_antenna_dipole_gain.png
       :alt: Dipole Antenna Gain
       :scale: 70%
       :align: center

       Dipole Antenna Characteristics

    The assumed characteristic is

    .. math::

       F_\\mathrm{V}(\\phi, \\theta) &= \\frac{ \\cos( \\frac{\\pi}{2} \\cos(\\theta)) }{ \\sin(\\theta) } \\\\
       F_\\mathrm{H}(\\phi, \\theta) &= 0

    """

    def __init__(
        self, mode: AntennaMode = AntennaMode.DUPLEX, pose: Transformation | None = None
    ) -> None:
        """
        Args:

            mode:
                Antenna's mode of operation.
                By default, a full duplex antenna is assumed.

            pose:
                The antenna's position and orientation with respect to its array.
        """

        # Initialize base class
        Antenna.__init__(self, mode, pose)

    def copy(self) -> Dipole:
        return Dipole(self.mode, self.pose.copy())

    def local_characteristics(self, azimuth: float, elevation: float) -> np.ndarray:
        vertical_polarization = (
            0.0 if elevation == 0.0 else cos(0.5 * pi * cos(elevation)) / sin(elevation)
        )
        return np.array([vertical_polarization, 0.0], dtype=float)

    def serialize(self, process: SerializationProcess) -> None:
        Antenna.serialize(self, process)

    @classmethod
    def Deserialize(cls: Type[Dipole], process: DeserializationProcess) -> Dipole:
        mode, pose = Antenna.Deserialize_Antenna(process)
        return cls(mode, pose)


class AntennaArrayBase(ABC, Transformable):
    """Base class for all antenna array models."""

    @property
    @abstractmethod
    def antennas(self) -> Sequence[Antenna]:
        """All individual antenna elements within this array."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def transmit_antennas(self) -> Sequence[Antenna]:
        """All transmitting antenna elements within this array."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def receive_antennas(self) -> Sequence[Antenna]:
        """All receiving antenna elements within this array."""
        ...  # pragma: no cover

    @property
    def num_antennas(self) -> int:
        """Number of antenna elements within this array."""

        return len(self.antennas)

    @property
    def num_transmit_antennas(self) -> int:
        """Number of transmitting antenna elements within this array."""

        return len(self.transmit_antennas)

    @property
    def num_receive_antennas(self) -> int:
        """Number of receiving antenna elements within this array."""

        return len(self.receive_antennas)

    @property
    def topology(self) -> np.ndarray:
        """Sensor array topology.

        Access the array topology as a :math:`M \\times 3` matrix indicating the cartesian locations
        of each antenna element within the local coordinate system.

        Returns:

            :math:`M \\times 3` topology matrix,
            where :math:`M` is the number of antenna elements.
        """

        if self.num_antennas == 0:
            return np.empty((0, 3), dtype=np.float64)

        global_toplogy = np.array(
            [a.forwards_transformation[:4, 3] for a in self.antennas], dtype=np.float64
        )
        local_topology = global_toplogy @ self.backwards_transformation.T

        return local_topology[:, :3].view(np.ndarray)

    @property
    def transmit_topology(self) -> np.ndarray:
        """Topology of transmitting antenna elements.

        Access the array topology as a :math:`M_{\\mathrm{Tx}} \\times 3` matrix indicating the cartesian locations
        of each transmitting antenna element within the local coordinate system.

        Returns:
            :math:`M_{\\mathrm{Tx}} \\times 3`
            topology matrix, where :math:`M_{\\mathrm{Tx}}` is the number of antenna elements.
        """

        if self.num_transmit_antennas == 0:
            return np.empty((0, 3), dtype=np.float64)

        global_toplogy = np.array(
            [a.forwards_transformation[:4, 3] for a in self.transmit_antennas], dtype=np.float64
        )
        local_topology = global_toplogy @ self.backwards_transformation.T

        return local_topology[:, :3].view(np.ndarray)

    @property
    def receive_topology(self) -> np.ndarray:
        """Topology of receiving antenna elements.

        Access the array topology as a :math:`M_{\\mathrm{Rx}} \\times 3` matrix indicating the cartesian locations
        of each receiving antenna element within the local coordinate system.

        Returns:
            :math:`M_{\\mathrm{Rx}} \\times 3` topology matrix,
            where :math:`M_{\\mathrm{Rx}}` is the number of antenna elements.
        """

        if self.num_receive_antennas == 0:
            return np.empty((0, 3), dtype=np.float64)

        global_toplogy = np.array(
            [a.forwards_transformation[:4, 3] for a in self.receive_antennas], dtype=np.float64
        )
        local_topology = global_toplogy @ self.backwards_transformation.T

        return local_topology[:, :3].view(np.ndarray)

    def _topology(self, mode: AntennaMode) -> np.ndarray:
        """Topology of antenna elements of a certain mode.

        Args:
            mode:  Antenna mode of interest.

        Returns:
            :math:`M \\times 3` topology matrix,
            where :math:`M` is the number of antenna elements.

        Raises:
            ValueError: If an unknown antenna mode is encountered.
        """

        if mode == AntennaMode.DUPLEX:
            return self.topology

        elif mode == AntennaMode.TX:
            return self.transmit_topology

        elif mode == AntennaMode.RX:
            return self.receive_topology

        else:
            raise ValueError("Unknown antenna mode encountered")

    @overload
    def characteristics(
        self, location: np.ndarray, mode: AntennaMode, frame: Literal["global", "local"] = "local"
    ) -> np.ndarray:
        """Sensor array characteristics towards a certain angle.

        Args:
            location: Cartesian position of the target of interest.
            mode: Antenna mode of interest.
            frame(Literal['local', 'global']):
                Coordinate system reference frame.
                `local` assumes `location` to be in the antenna array's native coordiante system.
                `global` assumes `location` and `azimuth` to be in the antenna array's root coordinate system.

        Returns:
            :math:`M \\times 2` topology matrix,
            where :math:`M` is the number of antenna elements.
        """
        ...  # pragma: no cover

    @overload
    def characteristics(
        self, direction: Direction, mode: AntennaMode, frame: Literal["global", "local"] = "local"
    ) -> np.ndarray:
        """Sensor array polarizations towards a certain angle.

        Args:
            direction: Direction of the angles of interest.
            mode: Antenna mode of interest.
            frame(Literal['local', 'global']):
                Coordinate system reference frame.
                `local` assumes `direction` to be in the antenna array's native coordiante system.
                `global` assumes `direction` to be in the antenna array's root coordinate system.

        Returns:
            :math:`M \\times 2` topology matrix,
            where :math:`M` is the number of antenna elements.
        """
        ...  # pragma: no cover

    def characteristics(self, arg_0: np.ndarray | Direction, mode: AntennaMode, frame: Literal["global", "local"] = "local") -> np.ndarray:  # type: ignore
        # Direction of interest with respect to the array's local coordinate system
        global_direction: Direction

        # Handle spherical parameters of function overload
        if not isinstance(arg_0, Direction):
            global_direction = Direction.From_Cartesian(
                arg_0 - self.global_position if frame == "global" else arg_0, True
            )

        # Handle cartesian vector parameters of function overload
        else:
            global_direction = (
                arg_0
                if frame == "global"
                else self.forwards_transformation.transform_direction(arg_0)
            )

        antennas: Sequence[Antenna]

        if mode == AntennaMode.DUPLEX:
            antenna_characteristics = np.empty((self.num_antennas, 2), dtype=float)
            antennas = self.antennas

        elif mode == AntennaMode.TX:
            antenna_characteristics = np.empty((self.num_transmit_antennas, 2), dtype=float)
            antennas = self.transmit_antennas

        elif mode == AntennaMode.RX:
            antenna_characteristics = np.empty((self.num_receive_antennas, 2), dtype=float)
            antennas = self.receive_antennas

        else:
            raise ValueError("Unknown antenna mode encountered")

        # Query the antenna characteristics for each antenna element
        for a, antenna in enumerate(antennas):
            antenna_characteristics[a] = antenna.global_characteristics(global_direction)

        return antenna_characteristics

    def plot_topology(self, mode: AntennaMode = AntennaMode.DUPLEX) -> tuple[Figure, VAT]:
        """Plot a scatter representation of the array topology.

        Args:
            mode:
                Antenna mode of interest.
                `DUPLEX` by default, meaning that all antenna elements are considered.

        Returns: The created figure.
        """

        with Executable.style_context():
            figure, axes = plt.subplots(1, 1, squeeze=False, subplot_kw={"projection": "3d"})
            figure.suptitle("Antenna Array Topology")

            ax: Axes3D = axes.flat[0]

            if mode == AntennaMode.TX or mode == AntennaMode.DUPLEX:
                topology = self._topology(AntennaMode.TX)
                ax.scatter(
                    topology[:, 0],
                    topology[:, 1],
                    topology[:, 2],
                    marker="^",
                    color="blue",
                    depthshade=False,
                )

            if mode == AntennaMode.RX or mode == AntennaMode.DUPLEX:
                topology = self._topology(AntennaMode.RX)
                ax.scatter(
                    topology[:, 0],
                    topology[:, 1],
                    topology[:, 2],
                    marker="o",
                    color="blue",
                    depthshade=False,
                )

            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_zlabel("Z [m]")

            return figure, axes

    def cartesian_phase_response(
        self,
        carrier_frequency: float,
        position: np.ndarray,
        frame: Literal["local", "global"] = "local",
        mode: AntennaMode = AntennaMode.DUPLEX,
    ) -> np.ndarray:
        """Phase response of the sensor array towards an impinging point source within its far-field.

        Assuming a point source at position :math:`\\mathbf{t} \\in \\mathbb{R}^{3}` within the sensor array's
        far field, so that :math:`\\lVert \\mathbf{t} \\rVert_2 \\gg 0`,
        the :math:`m`-th array element at position :math:`\\mathbf{q}_m \\in \\mathbb{R}^{3}` responds with a factor

        .. math::

            a_{m} = e^{ \\mathrm{j} \\frac{2 \\pi f_\\mathrm{c}}{\\mathrm{c}}
                        \\lVert \\mathbf{t} - \\mathbf{q}_{m} \\rVert_2 }

        to an electromagnetic waveform emitted with center frequency :math:`f_\\mathrm{c}`.
        The full array response vector is the,refore

        .. math::

           \\mathbf{a} = \\left[ a_1, a_2, \\dots, a_{M} \\right]^{\\intercal} \\in \\mathbb{C}^{M} \\mathrm{.}

        Args:

            carrier_frequency:
                Center frequency :math:`f_\\mathrm{c}` of the assumed transmitted signal in Hz.

            position:
                Cartesian location :math:`\\mathbf{t}` of the impinging target.

            frame:
                Coordinate system reference frame.
                `local` by default.
                `local` assumes `position` to be in the antenna array's native coordiante system.
                `global` assumes `position` to be in the antenna array's root coordinate system.

            mode:
                Antenna mode of interest.
                `DUPLEX` by default, meaning that all antenna elements are considered.

        Returns:
            The sensor array response vector :math:`\\mathbf{a}`.
            A one-dimensional, complex-valued numpy array modeling the phase responses of each antenna element.

        Raises:
            ValueError: If `position` is not a cartesian vector.
        """

        position = position.flatten()
        if len(position) != 3:
            raise ValueError("Target position must be a cartesian (three-dimensional) vector")

        # Transform from global to local coordinates if required
        if frame == "global":
            position = self.backwards_transformation.transform_position(position)

        # Expand the position by a new dimension
        position = position[:, np.newaxis]

        # Compute the distance between antenna elements and the point source
        distances = np.linalg.norm(self._topology(mode).T - position, axis=0, keepdims=False)

        # Transform the distances to complex phases, i.e. the array response
        phase_response = np.exp(-2j * pi * carrier_frequency * distances / speed_of_light)

        # That's it
        return phase_response

    def cartesian_array_response(
        self,
        carrier_frequency: float,
        position: np.ndarray,
        frame: Literal["local", "global"] = "local",
        mode: AntennaMode = AntennaMode.DUPLEX,
    ) -> np.ndarray:
        """Sensor array characteristics towards an impinging point source within its far-field.

        Args:

            carrier_frequency:
                Center frequency :math:`f_\\mathrm{c}` of the assumed transmitted signal in Hz.

            position:
                Cartesian location :math:`\\mathbf{t}` of the impinging target.

            frame(Literal['local', 'global']):
                Coordinate system reference frame.
                `global` by default.
                `local` assumes `position` to be in the antenna array's native coordiante system.
                `global` assumes `position` to be in the antenna array's root coordinate system.

            mode:
                Antenna mode of interest.
                `DUPLEX` by default, meaning that all antenna elements are considered.

        Returns:
            The sensor array response matrix :math:`\\mathbf{A} \\in \\mathbb{C}^{M \\times 2}`.
            A one-dimensional, complex-valued numpy matrix modeling the far-field charactersitics of each antenna element.

        Raises:
            ValueError: If `position` is not a cartesian vector.
        """

        position = position.flatten()
        if len(position) != 3:
            raise ValueError("Target position must be a cartesian (three-dimensional) vector")

        # Query far-field phase response and antenna element polarizations
        phase_response = self.cartesian_phase_response(carrier_frequency, position, frame, mode)
        polarization = self.characteristics(position, mode, frame)

        # The full array response is an element-wise multiplication of phase response and polarizations
        # Towards the assumed far-field source's position
        array_response = phase_response[:, None] * polarization
        return array_response

    def horizontal_phase_response(
        self,
        carrier_frequency: float,
        azimuth: float,
        elevation: float,
        mode: AntennaMode = AntennaMode.DUPLEX,
    ) -> np.ndarray:
        """Response of the sensor array towards an impinging point source within its far-field.

        Assuming a far-field point source impinges onto the sensor array from horizontal angles of arrival
        azimuth :math:`\\phi \\in [0,  2\\pi)` and elevation :math:`\\theta \\in [-\\pi,  \\pi]`,
        the wave vector

        .. math::

            \\mathbf{k}(\\phi, \\theta) = \\frac{2 \\pi f_\\mathrm{c}}{\\mathrm{c}}
            \\begin{pmatrix}
                \\cos( \\phi ) \\cos( \\theta ) \\\\
                \\sin( \\phi) \\cos( \\theta ) \\\\
                \\sin( \\theta )
            \\end{pmatrix}

        defines the phase of a planar wave in horizontal coordinates.
        The :math:`m`-th array element at position :math:`\\mathbf{q}_m \\in \\mathbb{R}^{3}` responds with a factor

        .. math::

            a_{m}(\\phi, \\theta) = e^{\\mathrm{j} \\mathbf{k}^\\intercal(\\phi, \\theta)\\mathbf{q}_{m} }

        to an electromagnetic waveform emitted with center frequency :math:`f_\\mathrm{c}`.
        The full array response vector is therefore

        .. math::

           \\mathbf{a}(\\phi, \\theta)  = \\left[ a_1(\\phi, \\theta) , a_2(\\phi, \\theta) , \\dots, a_{M}(\\phi, \\theta)  \\right]^{\\intercal} \\in \\mathbb{C}^{M} \\mathrm{.}

        Args:

            carrier_frequency:
                Center frequency :math:`f_\\mathrm{c}` of the assumed transmitted signal in Hz.

            azimuth:
                Azimuth angle :math:`\\phi` in radians.

            elevation:
                Elevation angle :math:`\\theta` in radians.

            mode:
                Antenna mode of interest.
                `DUPLEX` by default, meaning that all antenna elements are considered.

        Returns:
            The sensor array response vector :math:`\\mathbf{a}`.
            A one-dimensional, complex-valued numpy array modeling the phase responses of each antenna element.

        """

        # Compute the wave vector
        k = np.array(
            [cos(azimuth) * cos(elevation), sin(azimuth) * cos(elevation), sin(elevation)],
            dtype=float,
        )

        # Transform the distances to complex phases, i.e. the array response
        response = np.exp(
            2j * pi * carrier_frequency * np.inner(k, self._topology(mode)) / speed_of_light
        )

        # That's it
        return response

    def spherical_phase_response(
        self,
        carrier_frequency: float,
        azimuth: float,
        zenith: float,
        mode: AntennaMode = AntennaMode.DUPLEX,
    ) -> np.ndarray:
        """Response of the sensor array towards an impinging point source within its far-field.

        Assuming a far-field point source impinges onto the sensor array from spherical angles of arrival
        azimuth :math:`\\phi \\in [0,  2\\pi)` and zenith :math:`\\theta \\in [0,  \\pi]`,
        the wave vector

        .. math::

            \\mathbf{k}(\\phi, \\theta) = \\frac{2 \\pi f_\\mathrm{c}}{\\mathrm{c}}
            \\begin{pmatrix}
                \\cos( \\phi ) \\sin( \\theta ) \\\\
                \\sin( \\phi) \\sin( \\theta ) \\\\
                \\cos( \\theta )
            \\end{pmatrix}

        defines the phase of a planar wave in horizontal coordinates.
        The :math:`m`-th array element at position :math:`\\mathbf{q}_m \\in \\mathbb{R}^{3}` responds with a factor

        .. math::

            a_{m}(\\phi, \\theta) = e^{\\mathrm{j} \\mathbf{k}^\\intercal(\\phi, \\theta)\\mathbf{q}_{m} }

        to an electromagnetic waveform emitted with center frequency :math:`f_\\mathrm{c}`.
        The full array response vector is therefore

        .. math::

           \\mathbf{a}(\\phi, \\theta)  = \\left[ a_1(\\phi, \\theta) , a_2(\\phi, \\theta) , \\dots, a_{M}(\\phi, \\theta)  \\right]^{\\intercal} \\in \\mathbb{C}^{M} \\mathrm{.}

        Args:

            carrier_frequency:
                Center frequency :math:`f_\\mathrm{c}` of the assumed transmitted signal in Hz.

            azimuth:
                Azimuth angle :math:`\\phi` in radians.

            zenith:
                Zenith angle :math:`\\theta` in radians.

            mode:
                Antenna mode of interest.
                `DUPLEX` by default, meaning that all antenna elements are considered.

        Returns:
            The sensor array response vector :math:`\\mathbf{a}`.
            A one-dimensional, complex-valued numpy array modeling the phase responses of each antenna element.

        """

        # Compute the wave vector
        k = np.array(
            [cos(azimuth) * sin(zenith), sin(azimuth) * sin(zenith), cos(zenith)], dtype=float
        )

        # Transform the distances to complex phases, i.e. the array response
        response = np.exp(
            2j * pi * carrier_frequency * np.inner(k, self._topology(mode)) / speed_of_light
        )

        # That's it
        return response


class AntennaArrayState(Sequence, AntennaArrayBase):
    """Immutable state of an antenna array.

    Returned by the :meth:`state<AntennaArray.state>` of an antenna array.
    """

    __antennas: list[Antenna]

    def __init__(self, antennas: Sequence[Antenna], global_pose: Transformation) -> None:
        """
        Args:
            antennas: Phyiscal antenna elements within this array state.
            global_pose: Global pose of the represented antenna array.
        """

        # Initialize base class
        AntennaArrayBase.__init__(self, global_pose)

        # Initialize class attributes
        self.__antennas = list(antennas)
        for antenna in self.__antennas:
            antenna.set_base(self)

    @property
    def antennas(self) -> list[Antenna]:
        """All individual antenna elements within this array."""

        return self.__antennas

    @property
    def transmit_antennas(self) -> Sequence[Antenna]:
        """All transmitting antenna elements within this array."""

        return [a for a in self.antennas if a.mode != AntennaMode.RX]

    @property
    def receive_antennas(self) -> Sequence[Antenna]:
        """All receiving antenna elements within this array."""

        return [a for a in self.antennas if a.mode != AntennaMode.TX]

    def select_subarray(
        self, indices: int | slice | Sequence[int], mode: AntennaMode = AntennaMode.DUPLEX
    ) -> AntennaArrayState:
        """Select a subset of the antenna array state, given a sequence of port indices.

        Depending on the selected `AntennaMode`, the provided `indices` refer to ports only transmitting, receiving, or both.

        Args:
            indices: Index or slice of the antenna ports to be considered.
            mode: The modes of the ports to be selected.

        Returns: The subset of the antenna array state.
        """

        # Select the correct susbet of candidate ports depending on the AntennaMode
        considered_antennas: Sequence[Antenna]
        if mode == AntennaMode.DUPLEX:
            considered_antennas = self.antennas
        elif mode == AntennaMode.TX:
            considered_antennas = self.transmit_antennas
        elif mode == AntennaMode.RX:
            considered_antennas = self.receive_antennas
        else:
            raise ValueError("Invalid AntennaMode provided")

        # Select a subset of antenna ports depending on the provided indices
        subarray_antennas: list[Antenna]
        if isinstance(indices, int):
            subarray_antennas = [considered_antennas[indices]]
        elif isinstance(indices, slice):
            subarray_antennas = list(considered_antennas[indices])
        else:
            subarray_antennas = [considered_antennas[i] for i in indices]

        # Create a new state object
        return AntennaArrayState(subarray_antennas, self.pose)

    def __getitem__(self, indices: int | slice | Sequence[int]) -> AntennaArrayState:
        """Return a subset of the antenna array state.

        Shorthand to :meth:`.select_subarray`.
        """

        return self.select_subarray(indices)

    def __len__(self) -> int:
        """Number of antenna elements within this array."""

        return len(self.__antennas)


class AntennaArray(AntennaArrayBase, Generic[AT]):
    """Base class of a model of a set of antennas."""

    def __init__(self, pose: Transformation | None = None) -> None:
        """
        Args:

            pose:
                The antenna array's position and orientation with respect to its device.
                If not specified, the same orientation and position as the device is assumed.
        """

        # Initialize base class
        AntennaArrayBase.__init__(self, pose)

    @property
    def num_transmit_antennas(self) -> int:
        num_antennas = 0
        for a in self.antennas:
            if a.mode != AntennaMode.RX:
                num_antennas += 1
        return num_antennas

    @property
    def num_receive_antennas(self) -> int:
        num_antennas = 0
        for a in self.antennas:
            if a.mode != AntennaMode.TX:
                num_antennas += 1
        return num_antennas

    @property
    @abstractmethod
    def antennas(self) -> list[AT]:
        """All individual antenna elements within this array."""
        ...  # pragma: no cover

    @property
    def transmit_antennas(self) -> list[AT]:
        """Transmitting antennas within this array."""

        return [a for a in self.antennas if a.mode != AntennaMode.RX]

    @property
    def receive_antennas(self) -> list[AT]:
        """Receiving antennas within this array."""

        return [a for a in self.antennas if a.mode != AntennaMode.TX]

    def state(self, base_pose: Transformation) -> AntennaArrayState:
        """Return the current state of the antenna array.

        Args:
            base_pose: Assumed pose of the antenna array's base coordinate frame.

        Returns: The current immutable state of the antenna array.
        """

        # Create a copy of the antenna elements
        antenna_copies: list[AT] = [a.copy() for a in self.antennas]
        return AntennaArrayState(antenna_copies, base_pose)


class UniformArray(Generic[AT], AntennaArray[AT], Serializable):
    """Model of a Uniform Antenna Array."""

    __element: AT
    __antennas: list[AT]  # List of all individual antenna elements within this array
    __spacing: float  # Spacing betwene the antenna ports in m
    __dimensions: tuple[int, int, int]  # Number of ports in x-, y-, and z-direction

    def __init__(
        self,
        element: Type[AT] | AT,
        spacing: float,
        dimensions: Sequence[int],
        pose: Transformation | None = None,
    ) -> None:
        """
        Args:

            element:
                The antenna element uniformly repeated across the array.

            spacing:
                Spacing between the elements in m.

            dimensions:
                The number of elements in x-, y-, and z-dimension.

            pose:
                The anntena array's transformation with respect to its device.
        """

        # Initialize base class
        AntennaArray.__init__(self, pose=pose)

        element_instance = element if isinstance(element, Antenna) else element()
        self.__element = element_instance  # type: ignore[assignment]
        self.__spacing = 0.0
        self.__dimensions = (0, 0, 0)

        self.spacing = spacing
        self.dimensions = tuple(dimensions)

    def __update_elements(self) -> None:
        """Update antenna elements if the toplogy configuration has changed in any way."""

        grid = np.meshgrid(
            np.arange(self.__dimensions[0]),
            np.arange(self.__dimensions[1]),
            np.arange(self.__dimensions[2]),
        )

        self.__antennas: list[AT] = []
        for pos in self.__spacing * np.vstack((grid[0].flat, grid[1].flat, grid[2].flat)).T:
            antenna_instance = self.__element.copy()
            antenna_instance.position = pos
            antenna_instance.set_base(self)
            self.__antennas.append(antenna_instance)

    @property
    def spacing(self) -> float:
        """Spacing between the antenna elements.

        Returns: Spacing in m.

        Raises:
            ValueError: If `spacing` is less or equal to zero.
        """

        return self.__spacing

    @spacing.setter
    def spacing(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Spacing must be greater than zero")

        self.__spacing = value
        self.__update_elements()

    @property
    @override
    def antennas(self) -> list[AT]:
        return self.__antennas

    @property
    def num_antennas(self) -> int:
        return self.__dimensions[0] * self.__dimensions[1] * self.__dimensions[2]

    @property
    def dimensions(self) -> tuple[int, ...]:
        """Number of antennas in x-, y-, and z-dimension."""

        return self.__dimensions

    @dimensions.setter
    def dimensions(self, value: Sequence[int] | int) -> None:
        if isinstance(value, int):
            value = (value,)

        else:
            value = tuple(value)

        if len(value) == 1:
            value += 1, 1

        elif len(value) == 2:
            value += (1,)

        elif len(value) > 3:
            raise ValueError("Number of antennas must have three or less entries")

        self.__dimensions = value  # type: ignore
        self.__update_elements()

    def serialize(self, process: SerializationProcess) -> None:
        AntennaArray.serialize(self, process)
        process.serialize_array(np.asarray(self.__dimensions, np.float64), "dimensions")
        process.serialize_floating(self.__spacing, "spacing")
        process.serialize_object(self.__element, "element")

    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> UniformArray:
        pose = process.deserialize_array("pose", np.float64).view(Transformation)
        spacing = process.deserialize_floating("spacing")
        dimensions = process.deserialize_array("dimensions", np.float64).flatten().tolist()
        element = process.deserialize_object("element", Antenna)
        return cls(element, spacing, dimensions, pose)  # type: ignore[arg-type]


class CustomAntennaArray(Generic[AT], AntennaArray[AT], Serializable):
    """Model of a set of arbitrary antennas."""

    __antennas: list[AT]  # List of all individual antenna elements within this array

    def __init__(
        self, antennas: Sequence[AT] | None = None, pose: Transformation | None = None
    ) -> None:
        """
        Args:
            antennas:
                Sequence of antenna elements available within this array.
                If not specified, an empty array is assumed.

            pose:
                The anntena array's transformation with respect to its device.

        Raises:

            ValueError: If the argument lists contain an unequal amount of objects.
        """

        # Initialize base class
        AntennaArray.__init__(self, pose=pose)

        self.__antennas = []
        _antennas: Sequence[AT] = [] if antennas is None else antennas
        for antenna in _antennas:
            self.add_antenna(antenna)

    def add_antenna(self, antenna: AT) -> None:
        """Add a new antenna element to this array.

        Args:
            antenna: The antenna element to be added.

        Returns: The newly created port.
        """

        self.__antennas.append(antenna)
        antenna.set_base(self)

    @property
    @override
    def antennas(self) -> list[AT]:
        return self.__antennas

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_object_sequence(self.antennas, "antennas")
        process.serialize_array(self.pose, "pose")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> CustomAntennaArray:
        return CustomAntennaArray(
            process.deserialize_object_sequence("antennas", Antenna),
            process.deserialize_array("pose", np.float64).view(Transformation),
        )
