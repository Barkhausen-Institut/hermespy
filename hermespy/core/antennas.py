# -*- coding: utf-8 -*-
"""
=====================
Antenna Configuration
=====================
"""

from __future__ import annotations
from abc import abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from math import cos, sin, exp, sqrt
from typing import List, Literal, Optional, overload, Tuple, Type

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from scipy.constants import pi, speed_of_light

from .executable import Executable
from .factory import Serializable
from .signal_model import Signal
from .transformation import Direction, Transformable, Transformation

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Antenna(Transformable, Serializable):
    """Model of a single antenna.

    A set of antenna models defines an antenna array model.
    """

    yaml_tag = "Antenna"
    __array: Optional[AntennaArray]  # Array this antenna belongs to

    def __init__(self, pose: Transformation | None = None) -> None:
        """
        Args:

            pose (Transformation, optional):
                The antenna's position and orientation with respect to its array.
        """

        # Init base class
        Serializable.__init__(self)
        Transformable.__init__(self, pose=pose)

        self.__array = None

    @property
    def array(self) -> Optional[AntennaArray]:
        """Array this antenna belongs to.

        Returns:
            Optional[AntennaArray]:
                The array this antenna belong to.
                `None` if this antenna is considered floating.
        """

        return self.__array

    @array.setter
    def array(self, value: Optional[AntennaArray]) -> None:
        # Do nothing if the state does not change
        if self.__array == value:
            return

        if self.__array is not None:
            self.__array.remove_antenna(self)

        self.__array = value
        self.set_base(self.__array)

    def transmit(self, signal: Signal) -> Signal:
        """Transmit a signal over this antenna.

        The transmission may be distorted by the antennas impulse response / frequency characteristics.

        Args:

            signal (Signal):received
                The signal model to be transmitted.

        Returns:

            Signal:
                The actually transmitted (distorted) signal model.
        """

        # The default implementation is ideal, i.e. the signal is not distorted
        return signal

    def receive(self, signal: Signal) -> Signal:
        """Receive a signal over this antenna.

        The reception may be distorted by the antennas impulse response / frequency characteristics.

        Args:

            signal (Signal):
                The signal model to be received.

        Returns:

            Signal:
                The actually received (distorted) signal model.
        """

        # The default implementation is ideal, i.e. the signal is not distored
        return signal

    @abstractmethod
    def characteristics(self, azimuth: float, elevation) -> np.ndarray:
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
            of the antenna response.
        """
        ...  # pragma: no cover

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
            figure, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
            figure.suptitle("Antenna Polarization")

            azimuth_angles = 2 * pi * np.arange(angle_resolution) / angle_resolution - pi
            elevation_angles = pi * np.arange(int(0.5 * angle_resolution)) / int(0.5 * angle_resolution) - 0.5 * pi

            azimuth_samples, elevation_samples = np.meshgrid(azimuth_angles, elevation_angles)
            e_surface = np.empty((len(azimuth_angles) * len(elevation_angles), 3), dtype=float)
            e_magnitudes = np.empty(len(azimuth_angles) * len(elevation_angles), dtype=float)
            h_surface = np.empty((len(azimuth_angles) * len(elevation_angles), 3), dtype=float)
            h_magnitudes = np.empty(len(azimuth_angles) * len(elevation_angles), dtype=float)

            for i, (azimuth, elevation) in enumerate(zip(azimuth_samples.flat, elevation_samples.flat)):
                e_magnitude, h_magnitude = self.characteristics(azimuth, elevation)

                e_magnitudes[i] = e_magnitude
                h_magnitudes[i] = h_magnitude

                e_surface[i, :] = (e_magnitude * cos(azimuth) * cos(elevation), e_magnitude * sin(azimuth) * cos(elevation), e_magnitude * sin(elevation))
                h_surface[i, :] = (h_magnitude * cos(azimuth) * cos(elevation), h_magnitude * sin(azimuth) * cos(elevation), h_magnitude * sin(elevation))

            triangles = tri.Triangulation(azimuth_samples.flatten(), elevation_samples.flatten())

            e_cmap = plt.cm.ScalarMappable(norm=colors.Normalize(e_magnitudes.min(), e_magnitudes.max()), cmap="jet")
            e_cmap.set_array(e_magnitudes)
            h_cmap = plt.cm.ScalarMappable(norm=colors.Normalize(h_magnitudes.min(), h_magnitudes.max()), cmap="jet")
            h_cmap.set_array(h_magnitudes)

            axes[0].set_title("E-Field")
            axes[0].plot_trisurf(e_surface[:, 0], e_surface[:, 1], e_surface[:, 2], triangles=triangles.triangles, cmap=e_cmap.cmap, norm=e_cmap.norm, linewidth=0.0)
            axes[0].set_xlabel("X")
            axes[0].set_ylabel("Y")
            axes[0].set_zlabel("Z")

            axes[1].set_title("H-Field")
            axes[1].plot_trisurf(h_surface[:, 0], h_surface[:, 1], h_surface[:, 2], triangles=triangles.triangles, cmap=h_cmap.cmap, norm=h_cmap.norm, linewidth=0.0)
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
            figure, axes = plt.subplots(subplot_kw={"projection": "3d"})
            figure.suptitle("Antenna Gain")

            azimuth_angles = 2 * pi * np.arange(angle_resolution) / angle_resolution - pi
            elevation_angles = pi * np.arange(int(0.5 * angle_resolution)) / int(0.5 * angle_resolution) - 0.5 * pi

            azimuth_samples, elevation_samples = np.meshgrid(azimuth_angles, elevation_angles)
            surface = np.empty((len(azimuth_angles) * len(elevation_angles), 3), dtype=float)
            magnitudes = np.empty(len(azimuth_angles) * len(elevation_angles), dtype=float)

            for i, (azimuth, elevation) in enumerate(zip(azimuth_samples.flat, elevation_samples.flat)):
                e_magnitude, h_magnitude = self.characteristics(azimuth, elevation)
                magnitude = sqrt(e_magnitude**2 + h_magnitude**2)
                magnitudes[i] = magnitude

                surface[i, :] = (magnitude * cos(azimuth) * cos(elevation), magnitude * sin(azimuth) * cos(elevation), magnitude * sin(elevation))

            triangles = tri.Triangulation(azimuth_samples.flatten(), elevation_samples.flatten())

            cmap = plt.cm.ScalarMappable(norm=colors.Normalize(magnitudes.min(), magnitudes.max()), cmap="jet")
            cmap.set_array(magnitudes)

            axes.plot_trisurf(surface[:, 0], surface[:, 1], surface[:, 2], triangles=triangles.triangles, cmap=cmap.cmap, norm=cmap.norm, linewidth=0.0)
            axes.set_xlabel("X")
            axes.set_ylabel("Y")
            axes.set_zlabel("Z")

            return figure


class IdealAntenna(Antenna):
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

    def characteristics(self, azimuth: float, elevation: float) -> np.ndarray:
        return np.array([2**-0.5, 2**-0.5], dtype=float)


class PatchAntenna(Antenna):
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

    def characteristics(self, azimuth: float, elevation: float) -> np.ndarray:
        vertical_azimuth = 0.1 + 0.9 * exp(-1.315 * azimuth**2)
        vertical_elevation = cos(elevation) ** 2

        return np.array([max(0.1, vertical_azimuth * vertical_elevation), 0.0], dtype=float)


class Dipole(Antenna):
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

    def characteristics(self, azimuth: float, elevation: float) -> np.ndarray:
        vertical_polarization = 0.0 if elevation == 0.0 else cos(0.5 * pi * cos(elevation)) / sin(elevation)
        return np.array([vertical_polarization, 0.0], dtype=float)


class AntennaArrayBase(Transformable):
    """Base class of a model of a set of antennas."""

    @property
    @abstractmethod
    def num_antennas(self) -> int:
        """Number of antenna elements within this array.

        Returns:
            int: Number of antenna elements.
        """
        ...  # pragma: no cover

    @property
    def num_transmit_antennas(self) -> int:
        """Number of transmitting antenna elements within this array.

        Returns: Number of transmitting elements.
        """

        return self.num_antennas

    @property
    def num_receive_antennas(self) -> int:
        """Number of receiving antenna elements within this array.

        Returns: Number of receiving elements.
        """

        return self.num_antennas

    @property
    @abstractmethod
    def antennas(self) -> List[Antenna]:
        """All individual antenna elements within this array.

        Returns: List of antennas.
        """
        ...  # pragma: no cover

    @property
    def topology(self) -> np.ndarray:
        """Sensor array topology.

        Access the array topology as a :math:`M \\times 3` matrix indicating the cartesian locations
        of each antenna element within the local coordinate system.

        Returns:

            np.ndarray:
                :math:`M \\times 3` topology matrix, where :math:`M` is the number of antenna elements.
        """
        return np.array([antenna.forwards_transformation.translation for antenna in self.antennas], dtype=float)

    @overload
    def characteristics(self, location: np.ndarray, frame: Literal["global", "local"] = "local") -> np.ndarray:
        """Sensor array characteristics towards a certain angle.

        Args:

            location (np.ndarray):
                Cartesian position of the target of interest.

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
    def characteristics(self, direction: Direction, frame: Literal["global", "local"] = "local") -> np.ndarray:
        """Sensor array polarizations towards a certain angle.

        Args:

            direction (Direction):
                Direction of the angles of interest.

            frame(Literal['local', 'global']):
                Coordinate system reference frame.
                `local` assumes `direction` to be in the antenna array's native coordiante system.
                `global` assumes `direction` to be in the antenna array's root coordinate system.

        Returns:

            :math:`M \\times 2` topology matrix,
            where :math:`M` is the number of antenna elements.
        """
        ...  # pragma: no cover

    def characteristics(self, arg_0: np.ndarray | Direction, frame: Literal["global", "local"] = "local") -> np.ndarray:  # type: ignore
        # Direction of interest with respect to the array's local coordinate system
        global_direction: Direction

        # Handle spherical parameters of function overload
        if not isinstance(arg_0, Direction):
            global_direction = Direction.From_Cartesian(arg_0 - self.global_position if frame == "global" else arg_0, True)

        # Handle cartesian vector parameters of function overload
        else:
            global_direction = arg_0 if frame == "global" else self.forwards_transformation.transform_direction(arg_0)

        antenna_characteristics = np.empty((self.num_antennas, 2), dtype=float)
        for a, antenna in enumerate(self.antennas):
            # Compute the local angle of interest for each antenna element
            local_antenna_direction = antenna.backwards_transformation.transform_direction(global_direction, normalize=True)

            # Query polarization vector for a-th antenna given local azimuth and zenith angles of interest
            local_antenna_character = antenna.characteristics(*local_antenna_direction.to_spherical())

            # The global polarization is the forwards transformation of the local
            # H-V (X-Y) polarization components to the global (X-Y) system
            global_antenna_character = antenna.forwards_transformation.rotate_direction(local_antenna_character)[:2]

            # We're finally done
            antenna_characteristics[a, :] = global_antenna_character

        return antenna_characteristics

    def plot_topology(self) -> plt.Figure:
        """Plot a scatter representation of the array topology.

        Returns:
            plt.Figure:
                The created figure.
        """

        topology = self.topology

        with Executable.style_context():
            figure = plt.figure()
            figure.suptitle("Antenna Array Topology")

            axes = figure.add_subplot(projection="3d")
            axes.scatter(topology[:, 0], topology[:, 1], topology[:, 2])
            axes.set_xlabel("X [m]")
            axes.set_ylabel("Y [m]")
            axes.set_zlabel("Z [m]")

            return figure

    def cartesian_phase_response(self, carrier_frequency: float, position: np.ndarray, frame: Literal["local", "global"] = "local") -> np.ndarray:
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

            position (np.ndarray):
                Cartesian location :math:`\\mathbf{t}` of the impinging target.

            frame(Literal['local', 'global']):
                Coordinate system reference frame.
                `local` by default.
                `local` assumes `position` to be in the antenna array's native coordiante system.
                `global` assumes `position` to be in the antenna array's root coordinate system.

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
        distances = np.linalg.norm(self.topology.T - position, axis=0, keepdims=False)

        # Transform the distances to complex phases, i.e. the array response
        phase_response = np.exp(2j * pi * carrier_frequency * distances / speed_of_light)

        # That's it
        return phase_response

    def cartesian_array_response(self, carrier_frequency: float, position: np.ndarray, frame: Literal["local", "global"] = "local") -> np.ndarray:
        """Sensor array charactersitcis towards an impinging point source within its far-field.

        Args:

            carrier_frequency (float):
                Center frequency :math:`f_\\mathrm{c}` of the assumed transmitted signal in Hz.

            position (np.ndarray):
                Cartesian location :math:`\\mathbf{t}` of the impinging target.

            frame(Literal['local', 'global']):
                Coordinate system reference frame.
                `global` by default.
                `local` assumes `position` to be in the antenna array's native coordiante system.
                `global` assumes `position` to be in the antenna array's root coordinate system.

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
        phase_response = self.cartesian_phase_response(carrier_frequency, position, frame)
        polarization = self.characteristics(position, frame)

        # The full array response is an element-wise multiplication of phase response and polarizations
        # Towards the assumed far-field source's position
        array_response = phase_response[:, None] * polarization
        return array_response

    def horizontal_phase_response(self, carrier_frequency: float, azimuth: float, elevation: float) -> np.ndarray:
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

        Returns:

            np.ndarray:
                The sensor array response vector :math:`\\mathbf{a}`.
                A one-dimensional, complex-valued numpy array modeling the phase responses of each antenna element.

        """

        # Compute the wave vector
        k = np.array([cos(azimuth) * cos(elevation), sin(azimuth) * cos(elevation), sin(elevation)], dtype=float)

        # Transform the distances to complex phases, i.e. the array response
        response = np.exp(2j * pi * carrier_frequency * np.inner(k, self.topology) / speed_of_light)

        # That's it
        return response

    def spherical_phase_response(self, carrier_frequency: float, azimuth: float, zenith: float) -> np.ndarray:
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

        Returns:

            np.ndarray:
                The sensor array response vector :math:`\\mathbf{a}`.
                A one-dimensional, complex-valued numpy array modeling the phase responses of each antenna element.

        """

        # Compute the wave vector
        k = np.array([cos(azimuth) * sin(zenith), sin(azimuth) * sin(zenith), cos(zenith)], dtype=float)

        # Transform the distances to complex phases, i.e. the array response
        response = np.exp(-2j * pi * carrier_frequency * np.inner(k, self.topology) / speed_of_light)

        # That's it
        return response


class UniformArray(AntennaArrayBase, Serializable):
    """Model of a Uniform Antenna Array."""

    yaml_tag = "UniformArray"
    property_blacklist = {"topology"}
    __antenna: Type[Antenna] | Antenna
    __antennas: List[Antenna]  # List of individual antenna elements
    __spacing: float  # Spacing betwene the antenna elements
    __dimensions: Tuple[int, int, int]  # Number of antennas in x-, y-, and z-direction

    def __init__(self, antenna: Type[Antenna] | Antenna, spacing: float, dimensions: Sequence[int], pose: Transformation | None = None) -> None:
        """
        Args:

            antenna (Type[Antenna] | Antenna):
                The anntenna model this uniform array assumes.

            spacing (float):
                Spacing between the antenna elements in m.

            dimensions (Sequence[int]):
                The number of antennas in x-, y-, and z-dimension.

            pose (Tranformation, optional):
                The anntena array's transformation with respect to its device.
        """

        # Initialize base class
        AntennaArrayBase.__init__(self, pose=pose)

        self.__antenna = antenna
        self.__spacing = 0.0
        self.__dimensions = (0, 0, 0)

        self.spacing = spacing
        self.dimensions = tuple(dimensions)

        self.__update_antennas()

    def __update_antennas(self) -> None:
        """Update antenna elements if the toplogy has changed in any way."""

        grid = np.meshgrid(np.arange(self.__dimensions[0]), np.arange(self.__dimensions[1]), np.arange(self.__dimensions[2]))
        positions = self.__spacing * np.vstack((grid[0].flat, grid[1].flat, grid[2].flat)).T

        if isinstance(self.__antenna, type):
            self.__antennas = [self.__antenna(pose=Transformation.From_Translation(pos)) for pos in positions]

        else:
            self.__antennas = [deepcopy(self.__antenna) for _ in range(self.num_antennas)]
            for ant, pos in zip(self.__antennas, positions):
                ant.position = pos

        # Make sure the antennas recognize the antenna array as its cordinate system reference frame
        for ant in self.__antennas:
            ant.set_base(self)

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
        self.__update_antennas()

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

        if any([num < 1 for num in value]):
            raise ValueError("Number of antenna elements must be greater than zero in every dimension")

        self.__dimensions = value  # type: ignore
        self.__update_antennas()

    @property
    def antennas(self) -> List[Antenna]:
        return self.__antennas

    @property
    def antenna(self) -> Type[Antenna] | Antenna:
        """The assumed antenna model.

        Returns: The antenna model.
        """

        return self.__antenna


class AntennaArray(AntennaArrayBase, Serializable):
    """Model of a set of arbitrary antennas."""

    yaml_tag = "CustomArray"

    __antennas: List[Antenna]

    def __init__(self, antennas: List[Antenna] = None, pose: Transformation | None = None) -> None:
        """
        Args:

            antennas (List[Antenna], optional):
                Antenna models of each array element.

            pose (Transformation, optional):
                The anntena array's transformation with respect to its device.

        Raises:

            ValueError: If the argument lists contain an unequal amount of objects.
        """

        # Initialize base class
        AntennaArrayBase.__init__(self, pose=pose)

        self.__antennas = []

        antennas = [] if antennas is None else antennas
        for antenna in antennas:
            self.add_antenna(antenna)

    @property
    def antennas(self) -> List[Antenna]:
        """Antennas within this array.

        Returns:
            List[Antenna]:
                List of antenna elements.
        """

        return self.__antennas.copy()

    @property
    def num_antennas(self) -> int:
        """Number of antenna elements within this array.

        Returns:
            int:
                Number of antenna elements.
        """

        return len(self.__antennas)

    def add_antenna(self, antenna: Antenna) -> None:
        """Add a new antenna element to this array.

        Args:

            antenna (Antenna):
                The new antenna to be added.
        """

        # Do nothing if the antenna is already registered within this array
        if antenna.array is self and antenna in self.antennas:
            return

        # Add information to the internal lists
        self.__antennas.append(antenna)

        # Register the antenna array at the antenna element
        antenna.array = self

    def remove_antenna(self, antenna: Antenna) -> None:
        """Remove an antenna element from this array.

        Args:

            antenna (Antenna):
                The antenna element to be removed.
        """

        # Do nothing if the antenna is not within this array
        if antenna not in self.__antennas:
            return

        self.__antennas.remove(antenna)
        antenna.array = None
