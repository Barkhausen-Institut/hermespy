# -*- coding: utf-8 -*-
"""
=============
Visualization
=============
"""

from __future__ import annotations
from typing import overload

import matplotlib.pyplot as plt

from .executable import Executable

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Visualizable(object):
    """Base class for visualizable results."""

    @property
    def title(self) -> str:
        """Title of the visualizable.

        Returns: Title string.
        """

        return self.__class__.__name__

    @overload
    def plot(self, axes: plt.Axes) -> None:
        ...  # pragma: no cover

    @overload
    def plot(self) -> plt.Figure:
        ...  # pragma: no cover

    def plot(self, axes: plt.Axes | None = None) -> None | plt.Figure:

        figure: plt.Figure | None

        if axes is None:

            with Executable.style_context():
                figure, axes = plt.subplots()

            figure.suptitle(self.title)

        else:
            figure, axes = None, axes

        # Visualize the content into the supplied _axes
        self._plot(axes)

        # Return a figure if a new one was created
        return figure

    def _plot(self, axes: plt.Axes) -> None:
        """Subroutine to plot the visualizable into a matplotlib axis.

        Args:

            axes (plt.Axes):
                The axis object into which the information should be plotted.

            **kwargs:
                Additional plotting arguments.
        """

        axes.text(0.5, 0.5, "NO PLOT AVAILABLE", horizontalalignment="center")
