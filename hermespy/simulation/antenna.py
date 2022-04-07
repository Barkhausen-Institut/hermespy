# -*- coding: utf-8 -*-
"""
================
Antenna Modeling
================
"""

from __future__ import annotations
from abc import abstractmethod
from math import cos, sin, exp, sqrt
from typing import List, Optional, Tuple

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from scipy.constants import pi, speed_of_light

from hermespy.core import Executable, FloatingError, Serializable, Signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Antenna(Serializable):
    """Model of a single antenna.

    A set of antenna models defines an antenna array model.
    """

    yaml_tag = u'Antenna'
    __array: Optional[AntennaArray]     # Array this antenna belongs to

    def __init__(self) -> None:

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

    @property
    def pos(self) -> np.ndarray:
        """Local position of the antenna within its local coordinate system.

        Returns:
            np.ndarray:
                Three-dimensional cartesian position vector.

        Raises:

            ValueError:
                Floating error if the antenna is not attached to an array.
        """

        if self.__array is None:
            raise FloatingError("Error trying to access the position of a floating antenna")

        return self.__array.antenna_position(self)

    @pos.setter
    def pos(self, value: np.ndarray) -> None:

        if self.__array is None:
            raise FloatingError("Error trying to access the position of a floating antenna")

        self.__array.set_antenna_position(self)

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
    def polarization(self,
                     azimuth: float,
                     elevation) -> Tuple[float, float]:
        """Generate a single sample of the antenna's polarization characteristics.


        .. math::
            \\mathbf{F}(\\phi, \\theta) =
            \\begin{pmatrix}
                F_{\\mathrm{V}}(\\phi, \\theta) \\\\
                F_{\\mathrm{H}}(\\phi, \\theta) \\\\
            \\end{pmatrix}

        Args:

            azimuth (float):
                Considered horizontal wave angle in radians :math:`\\phi`.

            elevation (float):
                Considered vertical wave angle in radians :math:`\\theta`.

        Returns:
            Tuple[float, float]:
                Vertical and horizontal polarization components of the antenna response.
        """

        ...

    def plot_polarization(self, angle_resolution: int = 180) -> plt.Figure:
        """Visualize the antenna polarization depending on the angles of interest.
        
        Args:

            angle_resolution (int, optional):
                Resolution of the polarization visualization.


        Returns:

            np.Figure:
                The created matplotlib figure.

        Raises:

            ValueError:
                If `angle_resolution` is smaller than one.
        """

        with Executable.style_context():

            figure, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
            figure.suptitle('Antenna Polarization')

            azimuth_angles = 2 * pi * np.arange(angle_resolution) / angle_resolution - pi
            elevation_angles = pi * np.arange(int(.5 * angle_resolution)) / int(.5 * angle_resolution) - .5 * pi

            azimuth_samples, elevation_samples = np.meshgrid(azimuth_angles, elevation_angles)
            e_surface = np.empty((len(azimuth_angles) * len(elevation_angles), 3), dtype=float)
            e_magnitudes = np.empty(len(azimuth_angles) * len(elevation_angles), dtype=float)
            h_surface = np.empty((len(azimuth_angles) * len(elevation_angles), 3), dtype=float)
            h_magnitudes = np.empty(len(azimuth_angles) * len(elevation_angles), dtype=float)
            
            for i, (azimuth, elevation) in enumerate(zip(azimuth_samples.flat, elevation_samples.flat)):
                
                e_magnitude, h_magnitude = self.polarization(azimuth, elevation)
                
                e_magnitudes[i] = e_magnitude
                h_magnitudes[i] = h_magnitude
                
                e_surface[i, :] = (e_magnitude * cos(azimuth) * cos(elevation),
                                   e_magnitude * sin(azimuth) * cos(elevation),
                                   e_magnitude * sin(elevation))
                h_surface[i, :] = (h_magnitude * cos(azimuth) * cos(elevation),
                                   h_magnitude * sin(azimuth) * cos(elevation),
                                   h_magnitude * sin(elevation))

            triangles = tri.Triangulation(azimuth_samples.flatten(), elevation_samples.flatten())

            e_cmap = plt.cm.ScalarMappable(norm=colors.Normalize(e_magnitudes.min(), e_magnitudes.max()), cmap='jet')
            e_cmap.set_array(e_magnitudes)
            h_cmap = plt.cm.ScalarMappable(norm=colors.Normalize(h_magnitudes.min(), h_magnitudes.max()), cmap='jet')
            h_cmap.set_array(h_magnitudes)

            axes[0].set_title('E-Field')
            axes[0].plot_trisurf(e_surface[:, 0], e_surface[:, 1], e_surface[:, 2], triangles=triangles.triangles, cmap=e_cmap.cmap, norm=e_cmap.norm, linewidth=0.)
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
            axes[0].set_zlabel('Z')

            axes[1].set_title('H-Field')
            axes[1].plot_trisurf(h_surface[:, 0], h_surface[:, 1], h_surface[:, 2], triangles=triangles.triangles, cmap=h_cmap.cmap, norm=h_cmap.norm, linewidth=0.)
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')
            axes[1].set_zlabel('Z')

            return figure

    def plot_gain(self, angle_resolution: int = 180) -> plt.Figure:
        """Visualize the antenna gain depending on the angles of interest.
        
        Args:

            angle_resolution (int, optional):
                Resolution of the polarization visualization.


        Returns:

            np.Figure:
                The created matplotlib figure.

        Raises:

            ValueError:
                If `angle_resolution` is smaller than one.
        """

        with Executable.style_context():

            figure, axes = plt.subplots(subplot_kw={"projection": "3d"})
            figure.suptitle('Antenna Gain')

            azimuth_angles = 2 * pi * np.arange(angle_resolution) / angle_resolution - pi
            elevation_angles = pi * np.arange(int(.5 * angle_resolution)) / int(.5 * angle_resolution) - .5 * pi

            azimuth_samples, elevation_samples = np.meshgrid(azimuth_angles, elevation_angles)
            surface = np.empty((len(azimuth_angles) * len(elevation_angles), 3), dtype=float)
            magnitudes = np.empty(len(azimuth_angles) * len(elevation_angles), dtype=float)

            for i, (azimuth, elevation) in enumerate(zip(azimuth_samples.flat, elevation_samples.flat)):
                
                e_magnitude, h_magnitude = self.polarization(azimuth, elevation)
                magnitude = sqrt(e_magnitude ** 2 + h_magnitude **2)    
                magnitudes[i] = magnitude
                
                surface[i, :] = (magnitude * cos(azimuth) * cos(elevation),
                                 magnitude * sin(azimuth) * cos(elevation),
                                 magnitude * sin(elevation))

            triangles = tri.Triangulation(azimuth_samples.flatten(), elevation_samples.flatten())

            cmap = plt.cm.ScalarMappable(norm=colors.Normalize(magnitudes.min(), magnitudes.max()), cmap='jet')
            cmap.set_array(magnitudes)

            axes.plot_trisurf(surface[:, 0], surface[:, 1], surface[:, 2], triangles=triangles.triangles, cmap=cmap.cmap, norm=cmap.norm, linewidth=0.)
            axes.set_xlabel('X')
            axes.set_ylabel('Y')
            axes.set_zlabel('Z')

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

    yaml_tag: u'IdealAntenna'

    def polarization(self, azimuth: float, elevation: float) -> Tuple[float, float]:

        return 2 ** -.5, 2 ** -.5


class PatchAntenna(Antenna):
    """Realistic model of a vertically polarized patch antenna.

    .. figure:: /images/api_antenna_patchantenna_gain.png
       :alt: Patch Antenna Gain
       :scale: 70%
       :align: center

       Patch Antenna Characteristics

    Refer to :footcite:t:`2012:jaeckel` for further information.
    """

    def polarization(self, azimuth: float, elevation: float) -> Tuple[float, float]:

        vertical_azimuth = .1 + .9 * exp(-1.315 * azimuth ** 2)
        vertical_elevation = cos(elevation) ** 2

        return max(.1, vertical_azimuth * vertical_elevation), 0.


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

    def polarization(self,
                     azimuth: float,
                     elevation) -> Tuple[float, float]:

        vertical_polarization = 0. if elevation == 0. else cos(.5 * pi * cos(elevation)) / sin(elevation)
        return vertical_polarization, 0.


class AntennaArrayBase(object):
    """Base class of a model of a set of antennas."""

    @property
    @abstractmethod
    def num_antennas(self) -> int:
        """Number of antenna elements within this array.

        Returns:
            int: Number of antenna elements.
        """

        ...

    @property
    @abstractmethod
    def topology(self) -> np.ndarray:
        """Sensor array topology.

        Access the array topology as a :math:`M \\times 3` matrix indicating the cartesian locations
        of each antenna element within the local coordinate system.

        Returns:

            np.ndarray:
                math:`M \\times 3` topology matrix, where :math:`M` is the number of antenna elements.
        """

        ...

    @abstractmethod
    def polarization(self,
                     azimuth: float,
                     elevation: float) -> np.ndarray:
        """Sensor array polarizations towards a certain angle.
           
        Args:
        
            azimuth (float):
                Azimuth angle of interest in radians.


            elevation (float):
                Elevation angle of interest in radians.

        Returns:

            np.ndarray:
                math:`M \\times 2` topology matrix, where :math:`M` is the number of antenna elements.
        """

        ...

    def plot_topology(self) -> plt.Figure:
        """Plot a scatter representation of the array topology.
        
        Returns:
            plt.Figure:
                The created figure.
        """

        topology = self.topology

        with Executable.style_context():

            figure = plt.figure()
            figure.suptitle('Antenna Array Topology')

            axes = figure.add_subplot(projection='3d')
            axes.scatter(topology[:, 0], topology[:, 1], topology[:, 2])
            axes.set_xlabel('X [m]')
            axes.set_ylabel('Y [m]')
            axes.set_zlabel('Z [m]')

            return figure

    def cartesian_response(self,
                           carrier_frequency: float,
                           position: np.ndarray) -> np.ndarray:
        """Response of the sensor array towards an impinging point source within its far-field.

        Assuming a point source at position :math:`\\mathbf{t} \\in \\mathbb{R}^{3}` within the sensor array's
        far field, so that :math:`\\lVert \\mathbf{t} \\rVert_2 \\gg 0`,
        the :math:`m`-th array element at position :math:`\\mathbf{q}_m \\in \\mathbb{R}^{3}` responds with a factor

        .. math::

            a_{m} = e^{ \\mathrm{j} \\frac{2 \\pi f_\\mathrm{c}}{\\mathrm{c}}
                        \\lVert \\mathbf{t} - \\mathbf{q}_{m} \\rVert_2 }

        to an electromagnetic waveform emitted with center frequency :math:`f_\\mathrm{c}`.
        The full array response vector is therefore

        .. math::

           \\mathbf{a} = \\left[ a_1, a_2, \\dots, a_{M} \\right]^{\\intercal} \\in \\mathbb{C}^{M} \\mathrm{.}

        Args:

            carrier_frequency (float):
                Center frequency :math:`f_\\mathrm{c}` of the assumed transmitted signal in Hz.

            position (np.ndarray):
                Cartesian location :math:`\\mathbf{t}` of the impinging target within the array's local coordinate system.

        Returns:

            np.ndarray:
                The sensor array response vector :math:`\\mathbf{a}`.
                A one-dimensional, complex-valued numpy array modeling the phase responses of each antenna element.

        Raises:

            ValueError: If `position` is not a cartesian vector.
        """

        position = position.flatten()
        if len(position) != 3:
            raise ValueError("Target position must be a cartesian (three-dimensional) vector")

        # Expand the position by a new dimension
        position = position[:, np.newaxis]

        # Compute the distance between antenna elements and the point source
        distances = np.linalg.norm(self.topology.T - position, axis=0, keepdims=False)

        # Transform the distances to complex phases, i.e. the array response
        response = np.exp(2j * pi * carrier_frequency * distances / speed_of_light)

        # That's it
        return response

    def horizontal_response(self,
                            carrier_frequency: float,
                            azimuth: float,
                            elevation: float) -> np.ndarray:
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
        k = np.array([cos(azimuth) * cos(elevation),
                      sin(azimuth) * cos(elevation),
                      sin(elevation)], dtype=float)

        # Transform the distances to complex phases, i.e. the array response
        response = np.exp(2j * pi * carrier_frequency * np.inner(k, self.topology) / speed_of_light)

        # That's it
        return response

    def spherical_response(self,
                           carrier_frequency: float,
                           azimuth: float,
                           zenith: float) -> np.ndarray:
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
        k = np.array([cos(azimuth) * sin(zenith),
                      sin(azimuth) * sin(zenith),
                      cos(zenith)], dtype=float)

        # Transform the distances to complex phases, i.e. the array response
        response = np.exp(-2j * pi * carrier_frequency * np.inner(k, self.topology) / speed_of_light)

        # That's it
        return response


class UniformArray(AntennaArrayBase, Serializable):
    """Model of a Uniform Antenna Array."""

    yaml_tag = u'UniformArray'
    __antenna: Antenna                      # The assumed antenna model
    __spacing: float                        # Spacing betwene the antenna elements
    __dimensions: Tuple[int, int, int]    # Number of antennas in x-, y-, and z-direction

    def __init__(self,
                 antenna: Antenna,
                 spacing: float,
                 dimensions: Tuple[int, ...]) -> None:
        """
        Args:

            antenna (Antenna):
                The anntenna model this uniform array assumes.

            spacing (float):
                Spacing between the antenna elements in m.

            dimensions (Tuple[int, ...]):
                The number of antennas in x-, y-, and z-dimension.
        """

        self.__antenna = antenna
        self.spacing = spacing
        self.dimensions = dimensions

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

        if value <= 0.:
            raise ValueError("Spacing must be greater than zero")

        self.__spacing = value

    @property
    def num_antennas(self) -> int:

        return self.__dimensions[0] * self.__dimensions[1] * self.__dimensions[2]

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        """Number of antennas in x-, y-, and z-dimension.

        Returns:
            Tuple[int, int, int]:
                Number of antennas in each direction.
        """

        return self.__dimensions

    @dimensions.setter
    def dimensions(self, value: Tuple[int, int, int]) -> None:

        if isinstance(value, int):
            value = value,

        if len(value) == 1:
            value += 1, 1

        elif len(value) == 2:
            value += 1,

        elif len(value) > 3:
            raise ValueError("Number of antennas must have three or less entries")

        if any([num < 1 for num in value]):
            raise ValueError("Number of antenna elements must be greater than zero in every dimension")

        self.__dimensions = value

    @property
    def topology(self) -> np.ndarray:

        grid = np.meshgrid(np.arange(self.__dimensions[0]), np.arange(self.__dimensions[1]), np.arange(self.__dimensions[2]))
        positions = self.__spacing * np.vstack((grid[0].flat, grid[1].flat, grid[2].flat)).T
        return positions

    def polarization(self,
                     azimuth: float,
                     elevation: float) -> np.ndarray:

        # Query polarization of the base antenna model
        polarization = np.array(self.__antenna.polarization(azimuth, elevation), dtype=float)[np.newaxis, :]
        polarization = np.tile(polarization, (self.num_antennas, 1))

        return polarization


class AntennaArray(Serializable):
    """Model of a set of arbitrary antennas."""

    __antennas: List[Antenna]           
    __positions: List[np.ndarray]
    __orientations: List[np.ndarray]

    def __init__(self) -> None:

        self.__antennas = []
        self.__positions = []
        self.__orientations = []

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

    def add_antenna(self,
                    antenna: Antenna,
                    position: np.ndarray,
                    orientation: Optional[np.ndarray]) -> None:
        """Add a new antenna element to this array.

        Args:

            antenna (Antenna):
                The new antenna to be added.

            position (np.ndarray):
                Position of the antenna within the local coordinate system.

            orientation (np.ndarray, optional):
                Orientation of the antenna within the local coordinate system.

        Raises:

            ValueError:
                If the `position` is not a three-dimensional vector.
                If the specified `orientation` is not a tuple of azimuth and elevation angles.
        """

        # Do nothing if the antenna is already registered within this array
        if antenna.array is self:
            return

        # Raise exception if the position is not a cartesian vector
        position = position.flatten()
        if len(position) != 3:
            raise ValueError("Antenna position must be a cartesian vector")

        # Raise exception if the orientation is not 2D for azimuth and elevation
        orientation = np.array([0., 0.], dtype=float) if orientation is None else orientation.flatten()
        if len(orientation) != 2:
            raise ValueError("Antenna orientation must contain azimuth and elevation angles information")

        # Add information to the internal lists
        self.__antennas.append(antenna)
        self.__positions.append(position)
        self.__orientations.append(orientation)

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

    @property
    def topology(self) -> np.ndarray:

        return np.array(self.__positions, dtype=float)

    def polarization(self,
                     azimuth: float,
                     elevation: float) -> np.ndarray:
        
        return np.array([ant.polarization(azimuth, elevation) for ant in self.__antennas], dtype=float)
