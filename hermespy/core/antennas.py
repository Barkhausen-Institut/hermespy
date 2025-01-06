# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from math import cos, sin, exp, sqrt
from typing import Generic, List, Literal, overload, Tuple, Type, TypeVar

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # type: ignore
from ruamel.yaml import Node, SafeRepresenter  # type: ignore
from scipy.constants import pi, speed_of_light

from .executable import Executable
from .factory import Serializable, SerializableEnum
from .transformation import Direction, Transformable, Transformation
from .visualize import VAT

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
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

APT = TypeVar("APT", bound="AntennaPort")
"""Type of antenna port."""


class Antenna(ABC, Generic[APT], Transformable):
    """Base class for the model of a single antenna element within an antenna array."""

    property_blacklist = {"port"}.union(Transformable.property_blacklist)
    __mode: AntennaMode  # The mode this antenna is operating in, i.e. DUPLEX, TX or RX
    __port: APT | None  # Antenna port this antenna belongs to

    def __init__(
        self, mode: AntennaMode = AntennaMode.DUPLEX, pose: Transformation | None = None
    ) -> None:
        """
        Args:

            mode (AntennaMode, optional):
                Antenna's mode of operation.
                By default, a full duplex antenna is assumed.

            pose (Transformation, optional):
                The antenna's position and orientation with respect to its array.
        """

        # Init base class
        Serializable.__init__(self)
        Transformable.__init__(self, pose=pose)

        # Initialize attributes
        self.__mode = mode
        self.__port = None

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

        # Notify the port that the antenna's mode has changed
        if self.port is not None:
            self.port.antennas_updated()

    @property
    def port(self) -> APT | None:
        """Antenna port this antenna is connected to."""

        return self.__port

    @port.setter
    def port(self, value: APT | None) -> None:
        # Do nothing if the port does not change
        if self.__port == value:
            return

        # Update this antenna's port reference and make it its reference coordinate frame
        self.__port = value

        if value is None:
            self.set_base(None)

        else:
            self.port.add_antenna(self)
            self.set_base(self.__port)

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

            azimuth (float):
                Considered horizontal wave angle in radians :math:`\\phi`.

            elevation (float):
                Considered vertical wave angle in radians :math:`\\theta`.

        Returns:

            Two dimensional numpy array denoting the horizontal and vertical ploarization components
            of the antenna response vector.
        """
        ...  # pragma: no cover

    def global_characteristics(self, global_direction: Direction) -> np.ndarray:
        """Query the antenna's polarization characteristics towards a certain direction of interest.

        Args:

            global_direction (Direction):
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

    def plot_polarization(self, angle_resolution: int = 180) -> plt.Figure:
        """Visualize the antenna polarization depending on the angles of interest.

        Args:

            angle_resolution (int, optional):
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

    def plot_gain(self, angle_resolution: int = 180) -> plt.Figure:
        """Visualize the antenna gain depending on the angles of interest.

        Args:

            angle_resolution (int, optional):
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


AT = TypeVar("AT", bound=Antenna)
"""Type of antenna."""


class IdealAntenna(Generic[APT], Antenna[APT], Serializable):
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

    yaml_tag = "IdealAntenna"
    """YAML serialization tag"""

    def __init__(
        self, mode: AntennaMode = AntennaMode.DUPLEX, pose: Transformation | None = None
    ) -> None:
        """
        Args:

            mode (AntennaMode, optional):
                Antenna's mode of operation.
                By default, a full duplex antenna is assumed.

            pose (Transformation, optional):
                The antenna's position and orientation with respect to its array.
        """

        # Initialize base class
        Antenna.__init__(self, mode, pose)

    def copy(self) -> IdealAntenna:
        return IdealAntenna(self.mode, self.pose.copy())

    def local_characteristics(self, azimuth: float, elevation: float) -> np.ndarray:
        return np.array([2**-0.5, 2**-0.5], dtype=float)


class LinearAntenna(Generic[APT], Antenna[APT], Serializable):
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

    yaml_tag = "LinearAntenna"

    __slant: float

    def __init__(
        self,
        mode: AntennaMode = AntennaMode.DUPLEX,
        slant: float = 0.0,
        pose: Transformation | None = None,
    ):
        """Initialize a new linear antenna.

        Args:

            mode (AntennaMode, optional):
                Antenna's mode of operation.
                By default, a full duplex antenna is assumed.

            mode (AntennaMode, optional):
                Antenna's mode of operation.
                By default, a full duplex antenna is assumed.

            slant (float):
                Slant of the antenna in radians.

            pose (Transformation, optional):
                Pose of the antenna.
        """

        # Initialize base class
        Antenna.__init__(self, mode, pose)
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


class PatchAntenna(Generic[APT], Antenna[APT], Serializable):
    """Realistic model of a vertically polarized patch antenna.

    .. figure:: /images/api_antenna_patchantenna_gain.png
       :alt: Patch Antenna Gain
       :scale: 70%
       :align: center

       Patch Antenna Characteristics

    Refer to :footcite:t:`2012:jaeckel` for further information.
    """

    yaml_tag = "PatchAntenna"
    """YAML serialization tag"""

    def __init__(
        self, mode: AntennaMode = AntennaMode.DUPLEX, pose: Transformation | None = None
    ) -> None:
        """
        Args:

            mode (AntennaMode, optional):
                Antenna's mode of operation.
                By default, a full duplex antenna is assumed.

            pose (Transformation, optional):
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


class Dipole(Generic[APT], Antenna[APT], Serializable):
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

    yaml_tag = "DipoleAntenna"
    """YAML serialization tag"""

    def __init__(
        self, mode: AntennaMode = AntennaMode.DUPLEX, pose: Transformation | None = None
    ) -> None:
        """
        Args:

            mode (AntennaMode, optional):
                Antenna's mode of operation.
                By default, a full duplex antenna is assumed.

            pose (Transformation, optional):
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


class AntennaPort(Generic[AT, AAT], Transformable, Serializable):
    """A single antenna port linking a set of antennas to an antenna array."""

    yaml_tag = "AntennaPort"
    __antennas: List[AT]  # Antennas connected to this port
    __transmit_antennas: List[AT]  # Transmitting antennas connected to this port
    __receive_antennas: List[AT]  # Receiving antennas connected to this port
    __array: AAT | None  # Array this antenna port belongs to

    def __init__(
        self,
        antennas: Sequence[AT] | None = None,
        pose: Transformation | None = None,
        array: AAT | None = None,
    ) -> None:
        """
        Args:

            antennas (Sequence[AT], optional):
                Sequence of antennas to be connected to this port.
                If not specified, no antennas are connected by default.

            pose (Transformation, optional):
                The antenna port's position and orientation with respect to its array.

            array (AAT, optional):
                Antenna array this port belongs to.
        """

        # Initialize base class
        Transformable.__init__(self, pose)

        # Initialize class attributes
        self.__antennas = []
        self.__transmit_antennas = []
        self.__receive_antennas = []
        self.__array = None
        self.array = array

        _antennas = [] if antennas is None else antennas
        for antenna in _antennas:
            self.add_antenna(antenna)

    @property
    def antennas(self) -> Sequence[AT]:
        """Antennas connected to this port."""

        return self.__antennas.copy()

    @property
    def num_antennas(self) -> int:
        """Number of antenna elements connected to this port."""

        return len(self.antennas)

    def antennas_updated(self) -> None:
        """Callback that is called whenever the list of connected antennas is updated.

        Should also be called after a connected antenna's mode has changed.
        """

        self.__transmit_antennas = []
        self.__receive_antennas = []

        for antenna in self.antennas:
            if antenna.mode == AntennaMode.DUPLEX:
                self.__transmit_antennas.append(antenna)
                self.__receive_antennas.append(antenna)
            elif antenna.mode == AntennaMode.TX:
                self.__transmit_antennas.append(antenna)
            elif antenna.mode == AntennaMode.RX:
                self.__receive_antennas.append(antenna)
            else:
                # This exception should never be raised
                raise RuntimeError("Unknow antenna mode encountered")

    def add_antenna(self, antenna: AT) -> None:
        """Add a new antenna to this port.

        Args:

            antenna (AT):
                The antenna to be added.

        Raises:

            ValueError: If the antenna is already connected to a different port.
        """

        if antenna.port is not None and antenna.port != self:
            raise ValueError("Antenna is already connected to a different port")

        if antenna not in self.antennas:
            self.__antennas.append(antenna)

            # Update the internal antenna lists
            self.antennas_updated()

        # Update the antenna's port reference
        antenna.port = self

    def remove_antenna(self, antenna: AT) -> None:
        """Remove an antenna from this port.

        Args:

            antenna (AT):
                The antenna to be removed.
        """

        if antenna in self.antennas:
            self.__antennas.remove(antenna)

            # Update the internal antenna lists
            self.antennas_updated()

        # Update the antenna's port reference
        if antenna.port is not None:
            antenna.port = None

    @property
    def num_transmit_antennas(self) -> int:
        """Number of transmitting antenna elements connected to this port."""

        return len(self.__transmit_antennas)

    @property
    def num_receive_antennas(self) -> int:
        """Number of receiving antenna elements connected to this port."""

        return len(self.__receive_antennas)

    @property
    def transmitting(self) -> bool:
        """Is this port connected to a transmitting antenna?"""

        return self.num_transmit_antennas > 0

    @property
    def receiving(self) -> bool:
        """Is this port connected to a receiving antenna?"""

        return self.num_receive_antennas > 0

    @property
    def transmit_antennas(self) -> Sequence[AT]:
        """Transmitting antennas connected to this port."""

        return self.__transmit_antennas.copy()

    @property
    def receive_antennas(self) -> Sequence[AT]:
        """Receiving antennas connected to this port."""

        return self.__receive_antennas.copy()

    @property
    def array(self) -> AAT | None:
        """Antenna array this antenna port belongs to."""

        return self.__array

    @array.setter
    def array(self, value: AAT | None) -> None:
        # Do nothing if the state does not change
        if self.__array == value:
            return

        self.__array = value
        self.set_base(self.__array)

    def copy(self) -> AntennaPort:
        """Create a deep copy of the antenna port.

        The copy will not be connected to any array.

        Returns: The copy.
        """

        return AntennaPort([antenna.copy() for antenna in self.antennas], deepcopy(self.pose), None)


class AntennaArrayBase(ABC, Generic[APT], Transformable):
    """Base class for all antenna array models."""

    @property
    @abstractmethod
    def ports(self) -> Sequence[APT]:
        """Sequence of all antenna ports within this array."""
        ...  # pragma: no cover

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
    def num_ports(self) -> int:
        """Number of antenna ports within this array."""

        return len(self.ports)

    @property
    def transmit_ports(self) -> Sequence[APT]:
        """Sequence of all transmitting ports within this array."""

        return [port for port in self.ports if port.transmitting]

    @property
    def receive_ports(self) -> Sequence[APT]:
        """Sequence of all receiving ports within this array."""

        return [port for port in self.ports if port.receiving]

    @property
    def num_transmit_ports(self) -> int:
        """Number of transmitting antenna ports within this array."""

        return len(self.transmit_ports)

    @property
    def num_receive_ports(self) -> int:
        """Number of receiving antenna ports within this array."""

        return len(self.receive_ports)

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

            mode (AntennaMode):
                Antenna mode of interest.

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

            location (numpy.ndarray):
                Cartesian position of the target of interest.

            mode (AntennaMode):
                Antenna mode of interest.

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

            direction (Direction):
                Direction of the angles of interest.

            mode (AntennaMode):
                Antenna mode of interest.

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

    def plot_topology(self, mode: AntennaMode = AntennaMode.DUPLEX) -> Tuple[plt.Figure, VAT]:
        """Plot a scatter representation of the array topology.

        Args:

            mode (AntennaMode, optional):
                Antenna mode of interest.
                `DUPLEX` by default, meaning that all antenna elements are considered.

        Returns:
            plt.Figure:
                The created figure.
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

            carrier_frequency (float):
                Center frequency :math:`f_\\mathrm{c}` of the assumed transmitted signal in Hz.

            position (numpy.ndarray):
                Cartesian location :math:`\\mathbf{t}` of the impinging target.

            frame (Literal['local', 'global']):
                Coordinate system reference frame.
                `local` by default.
                `local` assumes `position` to be in the antenna array's native coordiante system.
                `global` assumes `position` to be in the antenna array's root coordinate system.

            mode (AntennaMode, optional):
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

            carrier_frequency (float):
                Center frequency :math:`f_\\mathrm{c}` of the assumed transmitted signal in Hz.

            position (numpy.ndarray):
                Cartesian location :math:`\\mathbf{t}` of the impinging target.

            frame(Literal['local', 'global']):
                Coordinate system reference frame.
                `global` by default.
                `local` assumes `position` to be in the antenna array's native coordiante system.
                `global` assumes `position` to be in the antenna array's root coordinate system.

            mode (AntennaMode, optional):
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

            carrier_frequency (float):
                Center frequency :math:`f_\\mathrm{c}` of the assumed transmitted signal in Hz.

            azimuth (float):
                Azimuth angle :math:`\\phi` in radians.

            elevation (float):
                Elevation angle :math:`\\theta` in radians.

            mode (AntennaMode, optional):
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

            carrier_frequency (float):
                Center frequency :math:`f_\\mathrm{c}` of the assumed transmitted signal in Hz.

            azimuth (float):
                Azimuth angle :math:`\\phi` in radians.

            zenith (float):
                Zenith angle :math:`\\theta` in radians.

            mode (AntennaMode, optional):
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

    __ports: Sequence[AntennaPort]
    __elements: list[Antenna]

    def __init__(self, ports: Sequence[AntennaPort], global_pose: Transformation) -> None:
        """
        Args:

            ports (Sequence[AntennaPort]):
                Physical ports of the represented antenna array.

            global_pose (Transformation):
                Global pose of the represented antenna array.
        """

        # Initialize base class
        AntennaArrayBase.__init__(self, global_pose)

        # Initialize class attributes
        self.__ports = ports
        self.__elements = []
        for port in self.__ports:
            port.set_base(self)
            self.__elements.extend(port.antennas)

    @property
    def ports(self) -> Sequence[AntennaPort]:
        """All physical ports of the represented antenna array."""

        return self.__ports

    @property
    def antennas(self) -> Sequence[Antenna]:
        """All individual antenna elements within this array."""

        return self.__elements

    @property
    def transmit_antennas(self) -> Sequence[Antenna]:
        """All transmitting antenna elements within this array."""

        return [antenna for antenna in self.antennas if antenna.mode != AntennaMode.RX]

    @property
    def receive_antennas(self) -> Sequence[Antenna]:
        """All receiving antenna elements within this array."""

        return [antenna for antenna in self.antennas if antenna.mode != AntennaMode.TX]

    def select_subarray(
        self, indices: int | slice | Sequence[int], mode: AntennaMode = AntennaMode.DUPLEX
    ) -> AntennaArrayState:
        """Select a subset of the antenna array state, given a sequence of port indices.

        Depending on the selected `AntennaMode`, the provided `indices` refer to ports only transmitting, receiving, or both.

        Args:

            indices (int | slice | Sequence[int]):
                Index or slice of the antenna ports to be considered.

            mode (AntennaMode, optional):
                The modes of the ports to be selected.

        Returns: The subset of the antenna array state.
        """

        # Select the correct susbet of candidate ports depending on the AntennaMode
        if mode == AntennaMode.DUPLEX:
            considered_ports = self.ports
        elif mode == AntennaMode.TX:
            considered_ports = self.transmit_ports
        elif mode == AntennaMode.RX:
            considered_ports = self.receive_ports
        else:
            raise ValueError("Invalid AntennaMode provided")

        # Select a subset of antenna ports depending on the provided indices
        subarray_ports: list[AntennaPort]
        if isinstance(indices, int):
            subarray_ports = [considered_ports[indices]]
        elif isinstance(indices, slice):
            subarray_ports = list(considered_ports[indices])
        else:
            subarray_ports = [considered_ports[i] for i in indices]

        # Collect all antenna elements within the selected ports
        subarray_antennas: list[Antenna] = []
        for port in subarray_ports:
            subarray_antennas.extend(port.antennas)

        # Create a new state object
        # A little hacky, but does the trick
        state = AntennaArrayState([], self.pose)
        state.__elements = subarray_antennas
        state.__ports = subarray_ports
        return state

    def __getitem__(self, indices: int | slice | Sequence[int]) -> AntennaArrayState:
        """Return a subset of the antenna array state.

        Shorthand to :meth:`.select_subarray`.
        """

        return self.select_subarray(indices)

    def __len__(self) -> int:
        """Number of antenna elements within this array."""

        return self.num_ports


class AntennaArray(AntennaArrayBase[APT], Generic[APT, AT]):
    """Base class of a model of a set of antennas."""

    def __init__(self, pose: Transformation | None = None) -> None:
        """
        Args:

            pose (Transformation, optional):
                The antenna array's position and orientation with respect to its device.
                If not specified, the same orientation and position as the device is assumed.
        """

        # Initialize base class
        AntennaArrayBase.__init__(self, pose)

    @abstractmethod
    def _new_port(self) -> APT:
        """Create a new antenna port.

        Returns: The newly connected port.
        """
        ...  # pragma: no cover

    @property
    def transmit_ports(self) -> Sequence[APT]:
        return [port for port in self.ports if port.transmitting]

    @property
    def receive_ports(self) -> Sequence[APT]:
        return [port for port in self.ports if port.receiving]

    @property
    def num_transmit_antennas(self) -> int:
        num_antennas = sum(port.num_transmit_antennas for port in self.transmit_ports)
        return num_antennas

    @property
    def num_receive_antennas(self) -> int:
        num_antennas = sum(port.num_receive_antennas for port in self.receive_ports)
        return num_antennas

    def count_antennas(self, ports: Sequence[int]) -> int:
        """Count the number of antenna elements within a subset of ports.

        Args:

            ports (Sequence[int]):
                Indices of the ports to be considered.

        Returns: Number of antenna elements within the specified ports.

        Raises:

            IndexError: If an invalid port index is encountered.
        """

        num_antennas = 0
        for port_index in ports:
            num_antennas += self.ports[port_index].num_antennas

        return num_antennas

    def count_transmit_antennas(self, ports: Sequence[int]) -> int:
        """Count the number of transmitting antenna elements within a subset of ports.

        Args:

            ports (Sequence[int]):
                Indices of the ports to be considered.

        Returns: Number of transmitting antenna elements within the specified ports.

        Raises:

            IndexError: If an invalid port index is encountered.
        """

        num_antennas = 0
        for port_index in ports:
            num_antennas += self.transmit_ports[port_index].num_transmit_antennas

        return num_antennas

    def count_receive_antennas(self, ports: Sequence[int]) -> int:
        """Count the number of receiving antenna elements within a subset of ports.

        Args:

            ports (Sequence[int]):
                Indices of the ports to be considered.

        Returns: Number of receiving antenna elements within the specified ports.

        Raises:

            IndexError: If an invalid port index is encountered.
        """

        num_antennas = 0
        for port_index in ports:
            num_antennas += self.receive_ports[port_index].num_receive_antennas

        return num_antennas

    @property
    def antennas(self) -> List[AT]:
        """All individual antenna elements within this array."""

        antennas: List[AT] = []
        for port in self.ports:
            antennas.extend(port.antennas)

        return antennas

    @property
    def transmit_antennas(self) -> Sequence[AT]:
        """Transmitting antennas within this array."""

        antennas: List[AT] = []
        for port in self.transmit_ports:
            antennas.extend(port.transmit_antennas)

        return antennas

    @property
    def receive_antennas(self) -> Sequence[AT]:
        """Receiving antennas within this array."""

        antennas: List[AT] = []
        for port in self.receive_ports:
            antennas.extend(port.receive_antennas)

        return antennas

    def state(self, base_pose: Transformation) -> AntennaArrayState:
        """Return the current state of the antenna array.

        Args:

            base_pose (Transformation):
                Assumed pose of the antenna array's base coordinate frame.

        Returns: The current immutable state of the antenna array.
        """

        # Create a copy of the antenna elements
        port_copies = [port.copy() for port in self.ports]
        return AntennaArrayState(port_copies, base_pose)


class UniformArray(Generic[APT, AT], AntennaArray[APT, AT], Serializable):
    """Model of a Uniform Antenna Array."""

    yaml_tag = "UniformArray"
    property_blacklist = {"topology"}
    __base_port: APT
    __ports: List[APT]  # List of individual antenna ports within this array
    __spacing: float  # Spacing betwene the antenna ports in m
    __dimensions: Tuple[int, int, int]  # Number of ports in x-, y-, and z-direction

    def __init__(
        self,
        element: Type[AT] | AT | APT,
        spacing: float,
        dimensions: Sequence[int],
        pose: Transformation | None = None,
    ) -> None:
        """
        Args:

            element (Type[AT] | AT | APT):
                The element uniformly repeated across the array.
                If an antenna is passed instead of a port, a new port is automatically created.

            spacing (float):
                Spacing between the elements in m.

            dimensions (Sequence[int]):
                The number of elements in x-, y-, and z-dimension.

            pose (Tranformation, optional):
                The anntena array's transformation with respect to its device.
        """

        # Initialize base class
        AntennaArray.__init__(self, pose=pose)

        _base_port: APT
        if isinstance(element, AntennaPort):
            _base_port = element  # type: ignore
        else:
            _base_port = self._new_port()
            if isinstance(element, Antenna):
                _base_port.add_antenna(element)
            else:
                _base_port.add_antenna(element())

        self.__base_port = _base_port
        self.__spacing = 0.0
        self.__dimensions = (0, 0, 0)

        self.spacing = spacing
        self.dimensions = tuple(dimensions)

        self.__update_ports()

    def _new_port(self) -> APT:
        return AntennaPort()  # type: ignore

    @property
    def ports(self) -> Sequence[APT]:
        return self.__ports.copy()

    def __update_ports(self) -> None:
        """Update ports if the toplogy configuration has changed in any way."""

        grid = np.meshgrid(
            np.arange(self.__dimensions[0]),
            np.arange(self.__dimensions[1]),
            np.arange(self.__dimensions[2]),
        )
        positions = self.__spacing * np.vstack((grid[0].flat, grid[1].flat, grid[2].flat)).T

        self.__ports = [deepcopy(self.__base_port) for _ in range(self.num_antennas)]
        self.__antennas: List[AT] = []

        # ToDo:
        # Currently the forward kinematic chain, i.e. self.linked_frames will still hold references
        # to the old ports. This might cause a memory leak.

        for port, pos in zip(self.__ports, positions):
            # Update the port transformation
            port.position = pos
            port.set_base(self)

            # Update the internal antenna lists
            self.__antennas.extend(port.antennas)

    @property
    def spacing(self) -> float:
        """Spacing between the antenna elements.

        Returns:
            float: Spacing in m.

        Raises:
            ValueError:
                If `spacing` is less or equal to zero.
        """

        return self.__spacing

    @spacing.setter
    def spacing(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Spacing must be greater than zero")

        self.__spacing = value
        self.__update_ports()

    @property
    def num_antennas(self) -> int:
        return self.__dimensions[0] * self.__dimensions[1] * self.__dimensions[2]

    @property
    def dimensions(self) -> Tuple[int, ...]:
        """Number of antennas in x-, y-, and z-dimension.

        Returns: Number of antennas in each direction.
        """

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
        self.__update_ports()

    @property
    def antennas(self) -> List[AT]:
        return self.__antennas

    @property
    def base_port(self) -> APT:
        """Base port repeated across the array topology."""

        return self.__base_port

    @classmethod
    def to_yaml(cls: type[UniformArray], representer: SafeRepresenter, node: UniformArray) -> Node:
        # Ensure the antenna property is an instance and not a type
        additional_fields = {"element": node.base_port}
        return node._mapping_serialization_wrapper(representer, {"base_port"}, additional_fields)


class CustomAntennaArray(Generic[APT, AT], AntennaArray[APT, AT], Serializable):
    """Model of a set of arbitrary antennas."""

    yaml_tag = "CustomAntennaArray"

    __ports: List[APT]  # List of antenna ports within this array

    def __init__(
        self, ports: Sequence[APT | AT] = None, pose: Transformation | None = None
    ) -> None:
        """
        Args:

            ports (Sequence[APT | AT], optional):
                Sequence of antenna ports available within this array.
                If antennas are passed instead of ports, the ports are automatically created.
                If not specified, an empty array is assumed.

            pose (Transformation, optional):
                The anntena array's transformation with respect to its device.

        Raises:

            ValueError: If the argument lists contain an unequal amount of objects.
        """

        # Initialize base class
        AntennaArray.__init__(self, pose=pose)

        self.__ports = []

        _ports = [] if ports is None else ports
        for port in _ports:
            if isinstance(port, AntennaPort):
                self.add_port(port)  # type: ignore
            else:
                self.add_antenna(port)

    def _new_port(self) -> APT:
        return AntennaPort()  # type: ignore

    @property
    def ports(self) -> Sequence[APT]:
        return self.__ports.copy()

    def add_port(self, port: APT) -> None:
        """Add a new port to this array.

        Args:

            port (APT):
                The antenna port to be added.
        """

        # Do nothing if the antenna is already registered within this array
        if port.array is self and port in self.ports:
            return

        # Add information to the internal lists
        self.__ports.append(port)
        port.array = self

    def remove_port(self, port: APT) -> None:
        """Remove a port from this array.

        Args:

            port (APT):
                The antenna port to be removed.

        Raises:

            ValueError: If the port is not connected to this array.
        """

        # Do nothing if the antenna is not within this array
        if port not in self.__ports:
            raise ValueError("Port is not connected to this array")

        self.__ports.remove(port)
        port.array = None

    def add_antenna(self, antenna: AT) -> APT:
        """Add a new antenna element to this array.

        Convenience wrapper around :meth:`.add_port`,
        meaning a new port is automatically created and the antenna is added to it.

        Args:

            antenna (AT):
                The antenna element to be added.

        Raises:

            ValueError: If the antenna is already attached to another array or port.

        Returns: The newly created port.
        """

        # Raise an error if the antenna is already attached to another array or port
        if antenna.port is not None:
            raise ValueError("Antenna is already attached to another port")

        port = self._new_port()
        port.add_antenna(antenna)
        self.add_port(port)

        return port
