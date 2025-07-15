# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from functools import reduce
from typing import Callable, Sequence

import numpy as np

from ..logarithmic import LogarithmicSequence, ValueType

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RegisteredDimension(property):
    """Register a class property getter as a PyMonte simulation dimension.

    Registered properties may specify their simulation stage impacts and therefore significantly
    increase simulation runtime in cases where computationally demanding section re-calculations
    can be reduced.
    """

    __first_impact: str | None
    __last_impact: str | None
    __title: str | None

    def __init__(
        self,
        _property: property,
        first_impact: str | None = None,
        last_impact: str | None = None,
        title: str | None = None,
    ) -> None:
        """
        Args:

            first_impact:
                Name of the first simulation stage within the simulation pipeline
                which is impacted by manipulating this property.
                If not specified, the initial stage is assumed.

            last_impact:
                Name of the last simulation stage within the simulation pipeline
                which is impacted by manipulating this property.
                If not specified, the final stage is assumed.

            title:
                Displayed title of the dimension.
                If not specified, the dimension's name will be assumed.
        """

        self.__first_impact = first_impact
        self.__last_impact = last_impact
        self.__title = title

        property.__init__(
            self,
            getattr(_property, "fget", None),
            getattr(_property, "fset", None),
            getattr(_property, "fdel", None),
            getattr(_property, "doc", None),
        )

    @classmethod
    def is_registered(cls, object: object) -> bool:
        """Check if any object is a registered PyMonte simulation dimension.

        Args:

            object:
                The object in question.

        Returns:

            A boolean indicator.
        """

        return isinstance(object, cls)

    @property
    def first_impact(self) -> str | None:
        """First impact of the dimension within the simulation pipeline."""

        return self.__first_impact

    @property
    def last_impact(self) -> str | None:
        """Last impact of the dimension within the simulation pipeline."""
        return self.__last_impact

    @property
    def title(self) -> str | None:
        """Displayed title of the dimension."""

        return self.__title

    def getter(self, fget: Callable[[object], object]) -> RegisteredDimension:
        updated_property = property.getter(self, fget)
        return RegisteredDimension(
            updated_property, self.first_impact, self.last_impact, self.title
        )

    def setter(self, fset: Callable[[object, object], None]) -> RegisteredDimension:
        updated_property = property(self.fget, fset, self.fdel, self.__doc__)
        return RegisteredDimension(
            updated_property, self.first_impact, self.last_impact, self.title
        )

    def deleter(self, fdel: Callable[[object], None]) -> RegisteredDimension:
        updated_property = property.deleter(self, fdel)
        return RegisteredDimension(
            updated_property, self.first_impact, self.last_impact, self.title
        )


def register(*args, **kwargs) -> Callable[[object], RegisteredDimension]:
    """Shorthand to register a property as a MonteCarlo dimension.

    Args:

        _property: The property to be registered.
    """

    return lambda _property: RegisteredDimension(_property, *args, **kwargs)


class SamplePoint(object):
    """Sample point of a single grid dimension.

    A single :class:`.GridDimension` holds a sequence of sample points
    accesible by the :attr:`sample_points<.GridDimension.sample_points>` property.
    During simulation runtime, the simulation will dynamically reconfigure
    the scenario selecting a single sample point out of each :class:`.GridDimension`
    per generated simulation sample.
    """

    __value: object
    __title: str | None

    def __init__(self, value: object, title: str | None = None) -> None:
        """
        Args:

            value:
                Sample point value with which to configure the grid dimension.

            title:
                String representation of this sample point.
                If not specified, the simulation will attempt to infer an adequate representation.
        """

        self.__value = value
        self.__title = title

    @property
    def value(self) -> object:
        """Sample point value with which to configure the grid dimension"""

        return self.__value

    @property
    def title(self) -> str:
        """String representation of this sample point"""

        if self.__title is not None:
            return self.__title

        if type(self.__value).__str__ is not object.__str__:
            return str(self.__value)

        if isinstance(self.__value, float):
            return f"{self.__value:.2g}"

        if isinstance(self.__value, int):
            return f"{self.__value}"

        # Return the values class name if its not representable otherwise
        return self.__value.__class__.__name__


class GridDimensionInfo(object):
    """Basic information about a grid dimension."""

    __sample_points:list[SamplePoint]
    __title: str | None
    __plot_scale: str
    __tick_format: ValueType

    def __init__(
        self,
        sample_points: list[SamplePoint],
        title: str,
        plot_scale: str,
        tick_format: ValueType,
    ) -> None:
        
        self.__sample_points = sample_points
        self.__title = title
        self.__plot_scale = plot_scale
        self.__tick_format = tick_format

    @property
    def sample_points(self) -> list[SamplePoint]:
        """Sample points of this grid dimension."""

        return self.__sample_points

    @property
    def num_sample_points(self) -> int:
        """Number of dimension sample points.

        Returns: Number of sample points.
        """

        return len(self.__sample_points)

    @property
    def title(self) -> str:
        """Title of this grid dimension."""

        return self.__title

    @property
    def plot_scale(self) -> str:
        """Scale of the scalar evaluation plot.

        Refer to the `Matplotlib`_ documentation for a list of a accepted values.

        .. _Matplotlib: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_yscale.html
        """

        return self.__plot_scale

    @property
    def tick_format(self) -> ValueType:
        """Axis tick format of the scalar evaluation plot."""

        return self.__tick_format


class GridDimension(GridDimensionInfo):
    """Single axis within the simulation grid.

    A grid dimension represents a single simulation parameter that is to
    be varied during simulation runtime to observe its effects on the evaluated
    performance indicators.
    The values the represented parameter is configured to are
    :class:`SamplePoints<.SamplePoint>`.
    """

    __considered_objects: tuple[object, ...]
    __dimension: str
    __setter_lambdas: tuple[Callable, ...]
    __first_impact: str | None
    __last_impact: str | None

    def __init__(
        self,
        considered_objects: object | tuple[object, ...],
        dimension: str,
        sample_points: Sequence[SamplePoint | object],
        title: str | None = None,
        plot_scale: str | None = None,
        tick_format: ValueType | None = None,
    ) -> None:
        """
        Args:

            considered_objects:
                The considered objects of this grid section.

            dimension:
                Path to the attribute.

            sample_points:
                Sections the grid is sampled at.

            title:
                Title of this dimension.
                If not specified, the attribute string is assumed.

            plot_scale:
                Scale of the axis within plots.

            tick_format:
                Format of the tick labels.
                Linear by default.

        Raises:
            ValueError: If the selected `dimension` does not exist within the `considered_object`.
        """

        _considered_objects = (
            considered_objects if isinstance(considered_objects, tuple) else (considered_objects,)
        )
        self.__considered_objects = tuple()

        property_path = dimension.split(".")
        object_path = property_path[:-1]
        property_name = property_path[-1]

        # Infer plot scale of the x-axis
        _plot_scale: str
        if plot_scale is None:
            if isinstance(sample_points, LogarithmicSequence):
                _plot_scale = "log"

            else:
                _plot_scale = "linear"

        else:
            _plot_scale = plot_scale

        # Infer tick format of the x-axis
        _tick_format: ValueType
        if tick_format is None:
            if isinstance(sample_points, LogarithmicSequence):
                _tick_format = ValueType.DB

            else:
                _tick_format = ValueType.LIN

        else:
            _tick_format = tick_format

        self.__setter_lambdas = tuple()
        self.__dimension = dimension
        self.__first_impact = None
        self.__last_impact = None

        for considered_object in _considered_objects:
            # Make sure the dimension exists
            try:
                dimension_mother_object = reduce(
                    lambda obj, attr: getattr(obj, attr), object_path, considered_object
                )
                dimension_registration: RegisteredDimension = getattr(
                    type(dimension_mother_object), property_name
                )
                dimension_value = getattr(dimension_mother_object, property_name)

            except AttributeError:
                raise ValueError(
                    "Dimension '" + dimension + "' does not exist within the investigated object"
                )

            if len(sample_points) < 1:
                raise ValueError("A simulation grid dimension must have at least one sample point")

            # Update impacts if the dimension is registered as a PyMonte simulation dimension
            if RegisteredDimension.is_registered(dimension_registration):
                first_impact = dimension_registration.first_impact
                last_impact = dimension_registration.last_impact

                if self.__first_impact and first_impact != self.__first_impact:
                    raise ValueError(
                        "Diverging impacts on multi-object grid dimension initialization"
                    )

                if self.__last_impact and last_impact != self.__last_impact:
                    raise ValueError(
                        "Diverging impacts on multi-object grid dimension initialization"
                    )

                self.__first_impact = first_impact
                self.__last_impact = last_impact

                # Updated the depicted title if the dimension offers an option and it wasn't exactly specified
                if title is None and dimension_registration.title is not None:
                    title = dimension_registration.title

            self.__considered_objects += (considered_object,)

            # If the dimension value is a scalar dimension, we can directly use the lshift operator to configure
            # the object with the sample point values, given that the sample points are scalars as well
            if isinstance(dimension_value, ScalarDimension) and np.all(
                np.vectorize(np.isscalar)(sample_points)
            ):
                self.__setter_lambdas += (dimension_value.__lshift__,)
                title = dimension_value.title

            # Otherwise, the dimension value is a regular attribute and we need to create a setter lambda
            else:
                self.__setter_lambdas += (
                    self.__create_setter_lambda(considered_object, dimension),
                )

        # Initialize base class
        GridDimensionInfo.__init__(
            self,
            [s if isinstance(s, SamplePoint) else SamplePoint(s) for s in sample_points],
            dimension if title is None else title,
            _plot_scale,
            _tick_format,
        )

    @property
    def considered_objects(self) -> tuple[object, ...]:
        """Considered objects of this grid section."""

        return self.__considered_objects

    @property
    def dimension(self) -> str:
        """Dimension property name."""

        return self.__dimension

    def configure_point(self, point_index: int) -> None:
        """Configure a specific sample point.

        Args:
            point_index: Index of the sample point to configure.

        Raises:
            ValueError: For invalid indexes.
        """

        if point_index < 0 or point_index >= len(self.sample_points):
            raise ValueError(
                f"Index {point_index} is out of the range for grid dimension '{self.title}'"
            )

        for setter_lambda in self.__setter_lambdas:
            setter_lambda(self.sample_points[point_index].value)

    @property
    def first_impact(self) -> str | None:
        """Index of the first impacted simulation pipeline stage.

        :py:obj:`None`, if the stage is unknown.
        """

        return self.__first_impact

    @property
    def last_impact(self) -> str | None:
        """Index of the last impacted simulation pipeline stage.

        :py:obj:`None`, if the stage is unknown.
        """

        return self.__last_impact

    @staticmethod
    def __create_setter_lambda(considered_object: object, dimension: str) -> Callable:
        """Generate a setter lambda callback for a selected grid dimension.

        Args:

            considered_object:
                The considered object root.

            dimension:
                String representation of dimension location relative to the investigated object.

        Returns: The setter lambda.
        """

        stages = dimension.split(".")
        object_reference = reduce(
            lambda obj, attr: getattr(obj, attr), stages[:-1], considered_object
        )

        # Return a lambda to the function if the reference is callable
        function_reference = getattr(object_reference, stages[-1])
        if callable(function_reference):
            return lambda args: function_reference(args)

        # Return a setting lambda if the reference is not callable
        # Implicitly we assume that every non-callable reference is an attribute
        return lambda args: setattr(object_reference, stages[-1], args)
    

class ScalarDimension(ABC):
    """Base class for objects that can be configured by scalar values.

    When a property of type :class:`ScalarDimension` is defined as a simulation parameter :class:`GridDimension`,
    the simulation will automatically configure the object with the scalar value of the sample point
    during simulation runtime.

    The configuration operation is represented by the lshift operator `<<`.
    """

    @abstractmethod
    def __lshift__(self, scalar: float) -> None:
        """Configure the object with a scalar value.

        Args:
            scalar: Scalar value to configure the object with.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def title(self) -> str:
        """Title of the scalar dimension.

        Displayed in plots and tables during simulation runtime.
        """
        ...  # pragma: no cover
