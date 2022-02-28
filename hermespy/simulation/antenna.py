# -*- coding: utf-8 -*-
"""
===========================
Antenna Simulation Modeling
===========================
"""

from __future__ import annotations
from abc import abstractmethod
from math import cos, sin, exp, sqrt
from typing import Optional, Tuple

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from scipy.constants import pi

from hermespy.core import Executable, FloatingError, Serializable

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

    def plot_ploarization(self, angle_resolution: int = 180) -> plt.Figure:
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
    """An ideal antenna model."""

    yaml_tag: u'IdealAntenna'

    def polarization(self, azimuth: float, elevation: float) -> Tuple[float, float]:

        return 2 ** -.5, 2 ** -.5


class PatchAntenna(Antenna):
    """Model of a Patch Antenna."""

    def polarization(self, azimuth: float, elevation: float) -> Tuple[float, float]:

        vertical_azimuth = .1 + .9 * exp(-1.315 * azimuth ** 2)
        vertical_elevation = cos(elevation) ** 2

        return max(.1, vertical_azimuth * vertical_elevation), 0.


class AntennaArrayBase(object):
    """Base class of a model of a set of antennas."""

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
        """Sensor array polaroizations towards a certain angle.
           
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


class UniformArray(AntennaArrayBase, Serializable):
    """Model of a Uniform Antenna Array."""

    yaml_tag = u'UniformArray'
    __antenna: Antenna                      # The assumed antenna model
    __spacing: float                        # Spacing betwene the antenna elements
    __num_antennas: Tuple[int, int, int]    # Number of antennas in x-, y-, and z-direction

    def __init__(self,
                 antenna: Antenna,
                 spacing: float,
                 num_antennas: Tuple[int, ...]) -> None:
        """
        Args:

            antenna (Antenna):
                The anntenna model this uniform array assumes.

            spacing (float):
                Spacing between the antenna elements in m.

            num_antennas (Tuple[int, ...]):
                The number of antennas in x-, y-, and z-dimension.
        """

        self.__antenna = antenna
        self.spacing = spacing
        self.num_antennas = num_antennas

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
    def num_antennas(self) -> Tuple[int, int, int]:
        """Number of antennas in x-, y-, and z-dimension.

        Returns:
            Tuple[int, int, int]:
                Number of antennas in each direction.
        """

        return self.__num_antennas

    @num_antennas.setter
    def num_antennas(self, value: Tuple[int, int, int]) -> None:

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

        self.__num_antennas = value

    @property
    def topology(self) -> np.ndarray:

        grid = np.meshgrid(np.arange(self.__num_antennas[0]), np.arange(self.__num_antennas[1]), np.arange(self.__num_antennas[2]))
        positions = self.__spacing * np.vstack((grid[0].flat, grid[1].flat, grid[2].flat)).T
        return positions

    def polarization(self,
                     azimuth: float,
                     elevation: float) -> np.ndarray:

        # Query polarization of the base antenna model
        polarization = np.array(self.__antenna.polarization(azimuth, elevation), dtype=float)[np.newaxis, :]

        # The polarization pattern is the repetition of the nuclear polarization
        num_antennas = self.num_antennas
        num_antennas = num_antennas[0] * num_antennas[1] * num_antennas[2]
        
        polarization = np.tile(polarization, (num_antennas, 1))
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
        orientation = np.array([0., 0.], dtyep=float) if orientation is None else orientation.flatten()
        if len(orientation) != 2:
            raise ValueError("Antenna orientation must contain azimuath and elevation angles information")

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


    @property
    def polarization(self,
                      azimuth: float,
                      elevation: float) -> np.ndarray:
        
        return np.narray([ant.polarization(azimuth, elevation) for ant in self.__antennas], dtype=float)
        


class Dipole(Antenna):
    """Model of a dipole antenna."""

    __length: float             # Length of the antenna in m

    def __init__(self, length: float) -> None:
        """
        Args:

            length (float):
                Length of the antenna in m.
        """

        self.length = length
        Antenna.__init__(self)


    @property
    def length(self) -> float:
        """Length of the antenna.

        Returns:
            float:
                Length in m.

        Raises:
            ValueError:
                For a `length` smaller or equal to zero.
        """

        return self.__length

    @length.setter
    def length(self, value: float) -> None:

        if value <= 0.:
            raise ValueError("Antenna length must be greater than zero")

        self.__length = value

class PatchAntenna(Antenna):
    """Model of a Patch Antenna."""

    def polarization(self, azimuth: float, elevation: float) -> Tuple[float, float]:

        vertical_azimuth = .1 + .9 * exp(-1.315 * azimuth ** 2)
        vertical_elevation = cos(elevation ** 2)

        return max(.1, vertical_azimuth * vertical_elevation), 0.


class AntennaArrayBase(object):
    """Base class of a model of a set of antennas."""

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
        """Sensor array polaroizations towards a certain angle.
           
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


class UniformArray(AntennaArrayBase, Serializable):
    """Model of a Uniform Antenna Array."""

    yaml_tag = u'UniformArray'
    __antenna: Antenna                      # The assumed antenna model
    __spacing: float                        # Spacing betwene the antenna elements
    __num_antennas: Tuple[int, int, int]    # Number of antennas in x-, y-, and z-direction

    def __init__(self,
                 antenna: Antenna,
                 spacing: float,
                 num_antennas: Tuple[int, ...]) -> None:
        """
        Args:

            antenna (Antenna):
                The anntenna model this uniform array assumes.

            spacing (float):
                Spacing between the antenna elements in m.

            num_antennas (Tuple[int, ...]):
                The number of antennas in x-, y-, and z-dimension.
        """

        self.__antenna = antenna
        self.spacing = spacing
        self.num_antennas = num_antennas

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
    def num_antennas(self) -> Tuple[int, int, int]:
        """Number of antennas in x-, y-, and z-dimension.

        Returns:
            Tuple[int, int, int]:
                Number of antennas in each direction.
        """

        return self.__num_antennas

    @num_antennas.setter
    def num_antennas(self, value: Tuple[int, int, int]) -> None:

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

        self.__num_antennas = value

    @property
    def topology(self) -> np.ndarray:

        grid = np.meshgrid(np.arange(self.__num_antennas[0]), np.arange(self.__num_antennas[1]), np.arange(self.__num_antennas[2]))
        positions = self.__spacing * np.vstack((grid[0].flat, grid[1].flat, grid[2].flat)).T
        return positions

    def polarization(self,
                     azimuth: float,
                     elevation: float) -> np.ndarray:

        # Query polarization of the base antenna model
        polarization = np.array(self.__antenna.polarization(azimuth, elevation), dtype=float)[np.newaxis, :]

        # The polarization pattern is the repetition of the nuclear polarization
        num_antennas = self.num_antennas
        num_antennas = num_antennas[0] * num_antennas[1] * num_antennas[2]
        
        polarization = np.tile(polarization, (num_antennas, 1))
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
        orientation = np.array([0., 0.], dtyep=float) if orientation is None else orientation.flatten()
        if len(orientation) != 2:
            raise ValueError("Antenna orientation must contain azimuath and elevation angles information")

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


    @property
    def polarization(self,
                      azimuth: float,
                      elevation: float) -> np.ndarray:
        
        return np.narray([ant.polarization(azimuth, elevation) for ant in self.__antennas], dtype=float)
        


class Dipole(Antenna):
    """Model of a dipole antenna."""

    __length: float             # Length of the antenna in m

    def __init__(self, length: float) -> None:
        """
        Args:

            length (float):
                Length of the antenna in m.
        """

        self.length = length
        Antenna.__init__(self)


    @property
    def length(self) -> float:
        """Length of the antenna.

        Returns:
            float:
                Length in m.

        Raises:
            ValueError:
                For a `length` smaller or equal to zero.
        """

        return self.__length

    @length.setter
    def length(self, value: float) -> None:

        if value <= 0.:
            raise ValueError("Antenna length must be greater than zero")

        self.__length = value
