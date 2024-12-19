# -*- coding: utf-8 -*-
"""
============
Tile Figures
============

Curtesy of
https://stackoverflow.com/questions/61503168/how-to-tile-matplotlib-figures-evenly-on-screen
"""

from typing import Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:  # pragma: no cover
    from screeninfo import get_monitors  # type: ignore

except ModuleNotFoundError:  # pragma: no cover

    def get_monitors():  # type: ignore
        return []


__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


def screen_geometry(monitor: int = 0) -> Tuple[int, ...]:
    """Query the screen geometry using the screeninfo module

    Args:

        monitor:
            The monitor index to query.
            Defaults to 0.

    Returns:
        A 4-tuple (x, y, w, h) specifying the geometry of the screen area.
    """

    monitors = get_monitors()
    if len(monitors) == 0:
        return (0, 0, 900, 600)

    else:
        selected_monitor = monitors[monitor]
        return (
            selected_monitor.x,
            selected_monitor.y,
            selected_monitor.width,
            selected_monitor.height,
        )


def set_figure_geometry(fig, backend, x, y, w, h) -> None:
    if backend in ("Qt5Agg", "Qt4Agg"):
        fig.canvas.manager.window.setGeometry(x, y, w, h)
        # fig.canvas.manager.window.statusBar().setVisible(False)
        # fig.canvas.toolbar.setVisible(True)
    elif backend in ("TkAgg",):
        fig.canvas.manager.window.wm_geometry("%dx%d+%d+%d" % (w, h, x, y))
    else:
        # print("This backend is not supported yet.")
        # print("Set the backend with matplotlib.use(<name>).")
        return


def tile_figures(cols=3, rows=2, screen_rect=None, tile_offsets=None) -> None:
    """
    Tile figures. If more than cols*rows figures are present, cols and
    rows are adjusted. For now, a Qt- or Tk-backend is required.

        import matplotlib
        matplotlib.use('Qt5Agg')
        matplotlib.use('TkAgg')

    Arguments:
        cols, rows:     Number of cols, rows shown. Will be adjusted if the
                        number of figures is larger than cols*rows.
        screen_rect:    A 4-tuple specifying the geometry (x,y,w,h) of the
                        screen area used for tiling (in pixels). If None, the
                        system's screen is queried using the screeninfo module.
        tile_offsets:   A 2-tuple specifying the offsets in x- and y- direction.
                        Can be used to compensate the title bar height.
    """
    assert isinstance(cols, int) and cols > 0
    assert isinstance(rows, int) and rows > 0
    assert screen_rect is None or len(screen_rect) == 4
    backend = mpl.get_backend()
    if screen_rect is None:
        screen_rect = screen_geometry()
    if tile_offsets is None:
        tile_offsets = (0, 0)
    sx, sy, sw, sh = screen_rect
    sx += tile_offsets[0]
    sy += tile_offsets[1]
    fig_ids = plt.get_fignums()
    # Adjust tiles if necessary.
    tile_aspect = cols / rows
    while len(fig_ids) > cols * rows:
        cols += 1
        rows = max(np.round(cols / tile_aspect), rows)
    # Apply geometry per figure.
    w = int(sw / cols)
    h = int(sh / rows)
    for i, num in enumerate(fig_ids):
        fig = plt.figure(num)
        x = (i % cols) * (w + tile_offsets[0]) + sx
        y = (i // cols) * (h + tile_offsets[1]) + sy
        set_figure_geometry(fig, backend, x, y, w, h)
