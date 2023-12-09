# -*- coding: utf-8 -*-
"""
=============
Visualization
=============
"""

from __future__ import annotations
from typing import Any, overload, Sequence, Tuple

import matplotlib.pyplot as plt
from nptyping import NDArray, Shape

from .executable import Executable

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


VAT = NDArray[Shape["*, *"], Any]
"""Type alias for a numpy array of matplotlib axes."""


class Visualizable(object):
    """Base class for visualizable results."""

    @property
    def title(self) -> str:
        """Title of the visualizable.

        Returns: Title string.
        """

        return self.__class__.__name__

    def _get_color_cycle(self) -> Sequence[str]:
        """Style color rotation.

        Returns: Sequence of color codes.
        """

        with Executable.style_context():
            return plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def _new_axes(self, **kwargs) -> Tuple[plt.Figure, VAT]:
        """Generate a new figure and axes to plot into.

        Can be overriden by subclasses to configure custom axes flags.

        Args:

            **kwargs: Additional arguments which cane be used by subclasses.

        Returns: Tuple of matplotlib figure and axes.
        """

        figure, axes = plt.subplots(1, 1, squeeze=False)
        return figure, axes

    def _prepare_axes(
        self, axes: VAT | None = None, title: str | None = None, **kwargs
    ) -> Tuple[plt.FigureBase, VAT]:
        """Prepare axes to plot into.

        Args:

            axes (VAT, optional):
                Axes to plot into.
                If not provided, a new figure and axes will be generated.
                See :meth:`Visualizable._new_axes`

            title (str, optional):
                Title of the generated figure.
                Only applied if `axes` are not provided and a new figure is generated.
                If not specified, :meth:`Visualizable.title` will be applied.

        Returns:
            Tuple of the newly generated figure and configured axis.
        """

        figure: plt.FigureBase

        if axes is None:
            with Executable.style_context():
                figure, axes = self._new_axes(**kwargs)
                figure.suptitle(self.title if title is None else title)

        else:
            if axes.size < 1:
                raise ValueError("Provided axes array is empty")

            figure: plt.Figure = axes.flat[0].get_figure()  # type: ignore

            # Make sure we can infer the containing figure from the provided axes
            if figure is None:
                raise RuntimeError("Axes are not associated with a figure")

        return figure, axes

    @overload
    def plot(self, axes: VAT) -> plt.FigureBase:
        ...  # pragma: no cover

    @overload
    def plot(self) -> plt.FigureBase:
        ...  # pragma: no cover

    def plot(self, axes: VAT | None = None, *, title: str | None = None) -> plt.FigureBase:
        """Plot a visualizable.

        Args:

            axes (VAT, optional):
                The axis object into which the information should be plotted.
                If not specified, the routine will generate and return a new figure.

            title (str, optional):
                Title of the generated plot.

        Returns:

            The newly generated matplotlib figure.
        """

        figure, axes = self._prepare_axes(axes, title)

        # Visualize the content into the supplied _axes
        self._plot(axes)

        # Return a figure if a new one was created
        return figure

    def _plot(self, axes: VAT) -> None:
        """Subroutine to plot the visualizable into a matplotlib axis.

        Args:

            axes (VAT):
                The axis object into which the information should be plotted.

            **kwargs:
                Additional plotting arguments.
        """

        axis: plt.Axes
        for axis in axes.flat:  # type: ignore
            axis.text(0.5, 0.5, "NO PLOT AVAILABLE", horizontalalignment="center")
