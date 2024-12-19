# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, patch

from hermespy.tools.tile import screen_geometry, set_figure_geometry, tile_figures

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestTile(TestCase):
    """Test tiling herlper functions"""

    def test_screen_geometry(self) -> None:
        """Test the screen geometry query function"""

        with patch("hermespy.tools.tile.get_monitors") as get_monitors_mock:
            screen_mock = Mock()
            screen_mock.x = 0
            screen_mock.y = 0
            screen_mock.width = 1920
            screen_mock.height = 1080

            get_monitors_mock.return_value = [screen_mock]
            self.assertSequenceEqual((0, 0, 1920, 1080), screen_geometry())

            get_monitors_mock.return_value = []
            self.assertSequenceEqual((0, 0, 900, 600), screen_geometry())

    def test_set_figure_geometry(self) -> None:
        """Test the figure geometry setting function"""

        qt_figure = Mock()
        set_figure_geometry(qt_figure, "Qt5Agg", 0, 0, 1920, 1080)
        qt_figure.canvas.manager.window.setGeometry.assert_called_once_with(0, 0, 1920, 1080)

        tk_figure = Mock()
        set_figure_geometry(tk_figure, "TkAgg", 0, 0, 1920, 1080)
        tk_figure.canvas.manager.window.wm_geometry.assert_called_once_with("1920x1080+0+0")

        unsupported_figure = Mock()
        set_figure_geometry(unsupported_figure, "Unsupported", 0, 0, 1920, 1080)
        unsupported_figure.assert_not_called()

    def test_tile_figures(self) -> None:
        """Test the figure tiling function over the full screen"""

        num_figures = 3
        figures = [Mock() for _ in range(num_figures)]

        with patch("matplotlib.get_backend") as get_backend_mock, patch("matplotlib.pyplot.get_fignums") as get_fignums_mock, patch("matplotlib.pyplot.figure") as figure_mock, patch("hermespy.tools.tile.get_monitors", return_value=[]):  # No monitors, so the screen geometry is (0, 0, 900, 600')
            get_backend_mock.return_value = "Qt5Agg"
            get_fignums_mock.return_value = [n for n in range(num_figures)]
            figure_mock.side_effect = lambda n: figures[n]

            tile_figures(cols=2, rows=1)

        for figure in figures:
            figure.canvas.manager.window.setGeometry.assert_called_once()
