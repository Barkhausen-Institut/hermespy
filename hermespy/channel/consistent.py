# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import numpy as np
from h5py import Group
from scipy.optimize import bisect
from scipy.stats import norm

from hermespy.core import HDFSerializable, RandomNode

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ConsistentVariable(ABC):
    """Base class for spatially consistent random variables."""

    __shape: Tuple[int, ...]
    __offset: int
    __size: int

    def __init__(
        self, generator: ConsistentGenerator, shape: Tuple[int, ...] | None = None
    ) -> None:
        """
        Args:
            generator (ConsistentGenerator): Generator to which this variable belongs.
            shape (Tuple[int, ...] | None, optional): Shape of the output array. Scalar by default.
        """

        # Initialize attributes
        self.__shape = (1,) if shape is None else shape
        self.__size = int(np.prod(self.shape))
        self.__offset = generator.add_variable(self)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the output array."""

        return self.__shape

    @property
    def size(self) -> int:
        """Number of scalar samples to be generated."""

        return self.__size

    @property
    def offset(self) -> int:
        """Offset of the variable in the generated samples."""

        return self.__offset

    def sample(self, sample: ConsistentSample) -> np.ndarray:
        """Sample the variable given a sample of the realization.

        Args:

            sample (ConsistentSample): Sample of the realization.

        Returns: Numpy array of samples of dimensions matching the variable size.
        """

        # Fetch the scalar samples
        scalar_samples = sample.fetch_scalars(self.offset, self.size)

        # Reshape to match the variable size
        reshaped_samples = scalar_samples.reshape(self.shape)
        return reshaped_samples


class ConsistentSample(object):
    """Sample of a consistent realization.

    Generated by calling :meth:`ConsistentRealization.sample`.
    """

    @abstractmethod
    def fetch_scalars(self, offset: int, num_scalars: int) -> np.ndarray:
        """Fetch the scalar samples.

        Args:

            offset (int): Offset of the scalar samples within this realization.
            num_scalars (int): Number of scalar samples to fetch.

        Returns: Numpy vector of scalar samples.
        """
        ...  # pragma: no cover


class ConsistentRealization(ABC, HDFSerializable):
    """Base class for a realization of a consistent random generator."""

    @abstractmethod
    def sample(self, position_a: np.ndarray, position_b: np.ndarray) -> ConsistentSample:
        """Sample the realization given two locations in euclidean space.

        Args:

            position_a (numpy.ndarray): First position in euclidean space.
            position_b (numpy.ndarray): Second position in euclidean space.

        Returns: Sample of the realization.
        """
        ...  # pragma: no cover

    @classmethod
    def from_HDF(cls, group: Group) -> ConsistentRealization:
        # Check for the group tag to decide which realization to load
        realization_type = group.attrs["type"]
        if realization_type == "DualConsistentRealization":
            return DualConsistentRealization.from_HDF(group)
        elif realization_type == "StaticConsistentRealization":
            return StaticConsistentRealization.from_HDF(group)
        else:
            raise ValueError(f"Unknown realization type: {realization_type}")


class DualConsistentSample(ConsistentSample):
    """Sample of a dual consistent realization.

    Generated by calling :meth:`DualConsistentRealization.sample`.
    """

    def __init__(self, scalar_samples: np.ndarray) -> None:
        """Initialize a dual consistent sample."""

        self.__scalar_samples = scalar_samples

    def fetch_scalars(self, offset: int, num_scalars: int) -> np.ndarray:
        return self.__scalar_samples[offset : offset + num_scalars]


class DualConsistentRealization(ConsistentRealization):
    """Realization of a set of dual consistent random variables."""

    __frequencies: np.ndarray
    __phases: np.ndarray

    def __init__(self, frequencies: np.ndarray, phases: np.ndarray) -> None:
        """
        Args:
            frequencies (numpy.ndarray): Frequencies of the spatially consistent process.
            phases (numpy.ndarray): Phases of the spatially consistent process.
        """

        self.__frequencies = frequencies
        self.__phases = phases

    @property
    def frequencies(self) -> np.ndarray:
        """Frequencies of the spatially consistent process."""

        return self.__frequencies

    @property
    def phases(self) -> np.ndarray:
        """Phases of the spatially consistent process."""

        return self.__phases

    def sample(self, position_a: np.ndarray, position_b: np.ndarray) -> DualConsistentSample:
        # Eq. 3
        scalar_samples: np.ndarray = (2 / self.__frequencies.shape[2]) ** 0.5 * np.sum(
            np.cos(
                np.tensordot(position_a, self.__frequencies[..., 0], (0, 0))
                + np.tensordot(position_b, self.__frequencies[..., 1], (0, 0))
                + self.__phases
            ),
            axis=-1,
            keepdims=False,
        )

        return DualConsistentSample(scalar_samples)

    def to_HDF(self, group: Group) -> None:
        group.attrs["type"] = "DualConsistentRealization"
        self._write_dataset(group, "frequencies", self.frequencies)
        self._write_dataset(group, "phases", self.phases)

    @classmethod
    def from_HDF(cls, group: Group) -> DualConsistentRealization:
        frequencies = np.array(group["frequencies"], dtype=np.float64)
        phases = np.array(group["phases"], dtype=np.float64)
        return DualConsistentRealization(frequencies, phases)


class StaticConsistentSample(ConsistentSample):
    """Consistent sample that is invariant in space."""

    __scalar_samples: np.ndarray

    def __init__(self, scalar_samples: np.ndarray) -> None:
        """
        Args:

            scalar_samples (numpy.ndarray):
                Scalar samples of the realization.
        """

        self.__scalar_samples = scalar_samples

    def fetch_scalars(self, offset: int, num_scalars: int) -> np.ndarray:
        return self.__scalar_samples[offset : offset + num_scalars]


class StaticConsistentRealization(ConsistentRealization):
    """Consistent realization that is immutable in space."""

    __scalar_samples: np.ndarray

    def __init__(self, scalar_samples: np.ndarray) -> None:
        """
        Args:
            scalar_samples (numpy.ndarray): Scalar samples of the realization.
        """

        self.__scalar_samples = scalar_samples.flatten()

    def sample(self, position_a: np.ndarray, position_b: np.ndarray) -> DualConsistentSample:
        return DualConsistentSample(self.__scalar_samples)

    def to_HDF(self, group: Group) -> None:
        group.attrs["type"] = "StaticConsistentRealization"
        self._write_dataset(group, "scalar_samples", self.__scalar_samples)

    @classmethod
    def from_HDF(cls, group: Group) -> StaticConsistentRealization:
        return StaticConsistentRealization(np.array(group["scalar_samples"], dtype=np.float64))


class ConsistentGenerator(object):
    """Generator of consistent random variables."""

    __rng: np.random.Generator | RandomNode
    __offset: int
    __variables: List[ConsistentVariable]
    __cdf_cache: Dict[Tuple[float, int], np.ndarray] = dict()

    @staticmethod
    def __radial_velocity_cdf(fr: float, a: float, U: float) -> float:
        """Implementation of the probability density function in Eq. 10

        Args:
            fr (float): Radial velocity.
            a (float): Correlation distance.
            U (float): Expected CDF value.
        """

        return (
            2 / np.pi * np.arctan(2 * np.pi * fr / a)
            - 4 * a * fr / (4 * np.pi**2 * fr**2 + a**2)
            - U
        )

    def __init__(self, rng: np.random.Generator | RandomNode) -> None:
        """
        Args:

            rng (np.random.Generator | RandomNode): Random number generator used to initialize this random variable.
        """

        # Initialize attributes
        self.__rng = rng
        self.__offset = 0
        self.__variables = []

    def gaussian(self, shape: Tuple[int, ...] | None = None) -> ConsistentGaussian:
        """Create a dual consistent Gaussian random variable."""

        variable = ConsistentGaussian(self, shape)
        self.add_variable(variable)

        return variable

    def uniform(self, shape: Tuple[int, ...] | None = None) -> ConsistentUniform:
        """Create a dual consistent uniform random variable."""

        variable = ConsistentUniform(self, shape)
        self.add_variable(variable)

        return variable

    def boolean(self, shape: Tuple[int, ...] | None = None) -> ConsistentBoolean:
        """Create a dual consistent boolean random variable."""

        variable = ConsistentBoolean(self, shape)
        self.add_variable(variable)

        return variable

    def add_variable(self, variable: ConsistentVariable) -> int:
        """Add a dual consistent random variable to the generator.

        Return the variable's offset in the generated samples.
        """

        if variable in self.__variables:
            return variable.offset

        variable_offset = self.__offset
        self.__offset += variable.size

        self.__variables.append(variable)
        return variable_offset

    def __sample_cdf(self, decorrelation_distance: float, num_samples: int = 1000) -> np.ndarray:
        """Sample the CDF of the radial velocity distribution.

        This operation is rather computationally expensive,
        therefore results are cached in both memory and on disk during runtime.

        Args:

            decorrelation_distance (float): Euclidean distance at which a sample of this Gaussian process is considered to be uncorrelated with another sample.
            num_samples (int, optional): Number of samples to generate. 1000 by default.

        Returns: Numpy array of radial velocities.
        """

        # Check the in-memory cache
        cached_cdf = ConsistentGenerator.__cdf_cache.get((decorrelation_distance, num_samples))
        if cached_cdf is not None:
            return cached_cdf

        # Eq. 12
        u_candidates = np.linspace(0, 1, 1 + num_samples, endpoint=True, dtype=np.float64)[:-1]
        radial_velocities = np.empty(num_samples, dtype=np.float64)
        a = 1 / decorrelation_distance

        fr_max = 1
        for indices, u in np.ndenumerate(u_candidates):

            # Find an appropriate upper bound to start the bisect function
            # Not sure if this is the best way to do it
            # Note that the CDF is monotonically increasing and uper bounded by 1
            while self.__radial_velocity_cdf(fr_max, a, u) < 0:  # type: ignore[arg-type]
                fr_max *= 2

            # Solve the equation
            radial_velocities[indices] = bisect(self.__radial_velocity_cdf, 0, fr_max, args=(a, u))

        # Cache the result
        ConsistentGenerator.__cdf_cache[(decorrelation_distance, num_samples)] = radial_velocities

        return radial_velocities

    def realize(
        self, decorrelation_distance: float, num_sinusoids: int = 30
    ) -> ConsistentRealization:
        """
        Args:

            decorrelation_distance (float): Euclidean distance at which a sample of this Gaussian process is considered to be uncorrelated with another sample.
            num_sinusoids (int, optional): Number of sinusoids used to approximate the Gaussian process. 30 by default.

        Returns: Realization of the consistent generator.
        """

        # Collect the required number of scalar random variables
        num_scalars = 0
        for variable in self.__variables:
            num_scalars += variable.size

        dimensions = (num_scalars, num_sinusoids, 2)
        rng = self.__rng if isinstance(self.__rng, np.random.Generator) else self.__rng._rng

        # If the decorrelation distance is infinite, the process is static
        if decorrelation_distance == float("inf"):
            return StaticConsistentRealization(rng.standard_normal(num_scalars))

        # Sample the radial velocities
        radial_velocity_candidates = self.__sample_cdf(decorrelation_distance)
        radial_velocities = rng.choice(radial_velocity_candidates, size=dimensions)

        # Eq. 11
        azimuth_angles = rng.uniform(
            0, 2 * np.pi, size=dimensions
        )  # Phi in the respective equations
        zenith_angles = np.arccos(
            1 - rng.uniform(0, 2, size=dimensions)
        )  # Theta in the respective equations

        # Eq. 5
        sin_zenith_angles = np.sin(zenith_angles)
        radial_directions = np.array(
            [
                np.cos(azimuth_angles) * sin_zenith_angles,
                np.sin(azimuth_angles) * sin_zenith_angles,
                np.cos(zenith_angles),
            ]
        )

        # Consolidate frequencies and phases to a realization
        frequencies = 2 * np.pi * radial_velocities * radial_directions
        phases = rng.uniform(0, 2 * np.pi, size=(num_scalars, num_sinusoids))
        return DualConsistentRealization(frequencies, phases)


class ConsistentGaussian(ConsistentVariable):
    """Spatially consistent normally distributed Gaussian variable."""

    def sample(self, sample: ConsistentSample, mean: float = 0.0, std: float = 1.0) -> np.ndarray:

        # Fetch the scalar samples
        samples = ConsistentVariable.sample(self, sample)

        # Transform to the desired distribution
        return mean + std * samples


class ConsistentUniform(ConsistentVariable):
    """Spatially consistent uniformly distributed random variable."""

    def sample(self, sample: ConsistentSample) -> np.ndarray:

        # Fetch the scalar samples
        samples = ConsistentVariable.sample(self, sample)

        # Transform to the uniform distribution over the gaussian CDF
        return norm.cdf(samples)


class ConsistentBoolean(ConsistentVariable):
    """Spatially consistent boolean random variable."""

    def sample(self, sample: ConsistentSample) -> np.ndarray:

        # Fetch the scalar samples
        samples = ConsistentVariable.sample(self, sample)

        # Transform to the boolean distribution
        return samples > 0.0
