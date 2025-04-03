# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, Sequence, Tuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import StemContainer
from matplotlib.image import AxesImage
from matplotlib.collections import QuadMesh

from .executable import Executable

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


VAT = np.ndarray[Tuple[int, int], np.dtype[Any]]
"""Type alias for a numpy array of matplotlib axes."""

VLT = np.ndarray[Tuple[int, int], np.dtype[Any]]
"""Type alias for a numpy array of matplotlib lines."""


class Visualization(ABC):
    """Information generated by plotting a Visualizable."""

    __figure: plt.Figure
    __axes: VAT

    def __init__(self, figure: plt.Figure | None, axes: VAT) -> None:
        """
        Args:

            figure:
                The figure containing the plot.
                May be :py:obj:`None` if the figure is unknown or unavailable.

            axes:
                The individual axes contained within the figure.
                A numpy object array of shape (nrows, ncols) containing matplotlib axes.
        """

        self.__figure = figure
        self.__axes = axes

    @property
    def figure(self) -> plt.Figure | None:
        """The figure containing the plot."""

        return self.__figure

    @property
    def axes(self) -> VAT:
        """The individual axes contained within the figure."""

        return self.__axes

    def show(self) -> None:
        """Show this visualization only.

        Note that, depending on the visualizuation, this may be a blocking command.
        """

        self.__figure.show(warn=False)


VT = TypeVar("VT", bound=Visualization)
"""Type variable for a visualization."""


class PlotVisualization(Visualization):
    """Information generated by plotting a Visualizable."""

    __lines: VLT

    def __init__(self, figure: plt.Figure, axes: VAT, lines: VLT) -> None:
        """
        Args:

            figure:
                The figure containing the plot.

            axes:
                The individual axes contained within the figure.
                A numpy object array of shape (nrows, ncols) containing matplotlib axes.

            lines:
                The lines contained within the axes.
                A numpy object array of shape (nrows, ncols) containing matplotlib lines for each axis.
        """

        # Assert that the axes and lines are compatible
        if axes.shape != lines.shape:
            raise ValueError(
                f"Shape of axes and lines do not match ({axes.shape} != {lines.shape})"
            )

        # Initialize base class
        Visualization.__init__(self, figure, axes)

        # Initialize class attributes
        self.__lines = lines

    @property
    def lines(self) -> VLT:
        """The lines contained within the axes."""

        return self.__lines


class StemVisualization(Visualization):
    """Information generated by plotting a Visualizable."""

    __container: StemContainer

    def __init__(self, figure: plt.Figure | None, axes: VAT, container: StemContainer) -> None:
        """
        Args:

            figure:
                The figure containing the plot.
                May be :py:obj:`None` if the figure is unknown or unavailable.

            axes:
                The individual axes contained within the figure.
                A numpy object array of shape (nrows, ncols) containing matplotlib axes.

            container:
                The container containing the stem plot.
        """

        # Initialize base class
        Visualization.__init__(self, figure, axes)

        # Initialize class attributes
        self.__container = container

    @property
    def container(self) -> StemContainer:
        """The container containing the stem plot."""

        return self.__container


class ScatterVisualization(Visualization):
    """Information generated by plotting a Visualizable."""

    __paths: VLT  # Path collection representing the scatter plot

    def __init__(self, figure: plt.Figure | None, axes: VAT, paths: VLT) -> None:
        """
        Args:

            figure:
                The figure containing the plot.
                May be :py:obj:`None` if the figure is unknown or unavailable.

            axes:
                The individual axes contained within the figure.
                A numpy object array of shape (nrows, ncols) containing matplotlib axes.

            paths:
                The path collection representing the scatter plot.
        """

        # Initialize base class
        Visualization.__init__(self, figure, axes)

        # Initialize class attributes
        self.__paths = paths

    @property
    def paths(self) -> VLT:
        """The path collection representing the scatter plot."""

        return self.__paths


class ImageVisualization(Visualization):
    """Information generated by plotting a Visualizable."""

    __image: AxesImage  # Axes image representing the image plot

    def __init__(self, figure: plt.Figure, axes: VAT, image: AxesImage) -> None:
        """
        Args:

            figure:
                The figure containing the plot.

            axes:
                The individual axes contained within the figure.
                A numpy object array of shape (nrows, ncols) containing matplotlib axes.

            image:
                The axes image representing the image plot.
        """

        # Initialize base class
        Visualization.__init__(self, figure, axes)

        # Initialize class attributes
        self.__image = image

    @property
    def image(self) -> AxesImage:
        """The axes image representing the image plot."""

        return self.__image


class QuadMeshVisualization(Visualization):
    """Information generated by plotting a Visualizable."""

    __mesh: QuadMesh

    def __init__(self, figure: plt.Figure, axes: VAT, mesh: QuadMesh) -> None:
        """
        Args:

            figure:
                The figure containing the plot.

            axes:
                The individual axes contained within the figure.
                A numpy object array of shape (nrows, ncols) containing matplotlib axes.

            mesh:
                The quad mesh representing the image plot.
        """

        # Initialize base class
        Visualization.__init__(self, figure, axes)

        # Initialize class attributes
        self.__mesh = mesh

    @property
    def mesh(self) -> QuadMesh:
        """The mesh representing the image plot."""

        return self.__mesh


class Visualizable(Generic[VT], ABC):
    """Base class for visualizable results."""

    __visualization: VT | None  # The most recent visualization

    def __init__(self) -> None:
        # Initialize class attributes
        self.__visualization = None

    @property
    def title(self) -> str:
        """Title of the visualizable.

        Returns: Title string.
        """

        return self.__class__.__name__

    @property
    def visualization(self) -> VT | None:
        """The most recent visualization."""

        return self.__visualization

    def _get_color_cycle(self) -> Sequence[str]:
        """Style color rotation."""

        with Executable.style_context():
            return plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def _axes_dimensions(self, **kwargs) -> Tuple[int, int]:
        """Determine the number of matplotlib axes to be created.

        Returns: Number of rows and columns of axes.
        """

        return 1, 1

    def create_figure(self, **kwargs) -> Tuple[plt.FigureBase, VAT]:
        """Create a new figure for plotting.

        Returns: Newly generated figure and axes to plot into.
        """

        return plt.subplots(*self._axes_dimensions(**kwargs), squeeze=False)

    @abstractmethod
    def _prepare_visualization(self, figure: plt.Figure | None, axes: VAT, **kwargs) -> VT:
        """Prepare axes and respective lines for plotting.

        Args:

            figure:
                Figure to which the `axes` belong.
                If unknown or unavailable, :py:obj:`None` is passed.

            axes:
                Axes to plot into.
                The dimensions must match the result of :meth:`Visualizable._axes_dimensions`.

            \**kwargs:
                Additional arguments to be passed to :meth:`Visualizable._new_axes`.

        Returns: Newly generated visualization.
        """
        ...  # pragma: no cover

    def visualize(
        self, axes: VAT | plt.Axes | None = None, *, title: str | None = None, **kwargs
    ) -> VT:
        """Generate a visual representation of this object using Matplotlib.

        Args:

            axes:
                The Matplotlib axes object into which the information should be plotted.
                If not specified, the routine will generate and return a new figure.

            title:
                Title of the generated plot.
                If not specified, :attr:`Visualizable.title` will be applied.

        Returns: Plotted information including axes and lines.
        """

        # Prepare the figure and axes for plotting
        with Executable.style_context():
            if axes is not None:
                _axes: VAT = axes if isinstance(axes, np.ndarray) else np.array([[axes]])
                figure = _axes.flat[0].get_figure()

            else:
                figure, _axes = self.create_figure(**kwargs) if axes is None else (None, axes)
                figure.suptitle(title or self.title)

            self.__visualization = self._prepare_visualization(figure, _axes, **kwargs)

        # Visualize the content into the supplied axes
        self._update_visualization(self.__visualization, **kwargs)

        # Return visualization handle
        return self.__visualization

    def update_visualization(self, visualization: VT | None = None, **kwargs) -> None:
        """Update an existing visualization with new data.

        Args:

            visualization:
                The visualization to update.
                If not specified, the most recent visualization will be updated.

        Raises:

            RuntimeError: If no visualization is provided and no visualization is cached.
        """

        if visualization:
            self._update_visualization(visualization, **kwargs)

        else:
            if self.__visualization:
                self._update_visualization(self.__visualization, **kwargs)
            else:
                raise RuntimeError("No visualization cached to update")

    @abstractmethod
    def _update_visualization(self, visualization: VT, **kwargs) -> None:
        """Update the visualization."""
        ...  # pragma: no cover


class VisualizableAttribute(Generic[VT], Visualizable[VT]):
    """Base class for attributes mocking plot functions."""

    def __call__(self, axes: VAT | None = None, *, title: str | None = None, **kwargs) -> VT:
        """Plot a visualizable.

        Args:

            axes:
                The Matplotlib axes object into which the information should be plotted.
                If not specified, the routine will generate and return a new figure.

            title:
                Title of the generated plot.
                If not specified, :attr:`Visualizable.title` will be applied.

            \**kwargs: Additional arguments for the plot routine.

        Returns: Plotted information including axes and lines.
        """

        return self.visualize(axes, title=title, **kwargs)
