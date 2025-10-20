# -*- coding: utf-8 -*-

from __future__ import annotations
from contextlib import redirect_stdout
from logging import getLogger, Logger
from os import devnull
from typing import Any


import numpy as np

try:  # pragma: no cover
    with redirect_stdout(open(devnull, "w")):
        from oct2py import Oct2Py, Oct2PyError, Struct

except Exception:  # pragma: no cover
    Oct2Py = None
    Oct2PyError = RuntimeError
    Struct = Any

from .interface import QuadrigaInterface

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class QuadrigaOctaveInterface(QuadrigaInterface):  # pragma: no cover
    """Quadriga Octave Interface."""

    __octave: Oct2Py
    __logger: Logger

    def __init__(self, octave_bin: str | None = None, **kwargs: Any) -> None:
        """Quadriga Octave interface object initialization.

        Args:

            octave_bin:
                Path to the octave cli executable.

            kwargs:
                Interface arguments.
        """

        # Init base class
        QuadrigaInterface.__init__(self, **kwargs)

        # Init octave session
        self.__logger = getLogger("octave_logger")
        self.__octave = Oct2Py(logger=self.__logger)  # executable=octave_bin)

        # Add quadriga source folder to octave lookup paths
        self.__octave.addpath(self.path_quadriga_src)

        # Add launch script folder to octave loopkup paths
        self.__octave.addpath(self.path_launch_script)

    def _run_quadriga(self, **parameters) -> np.ndarray:
        # Push parameters to quadriga
        for key, value in parameters.items():
            # Convert numpy arrays to lists
            if isinstance(value, np.ndarray):
                value = value.tolist()

            self.__octave.push(key, value)

        # Launch octave
        try:
            self.__octave.eval("launch_quadriga")

        except Oct2PyError as error:  # pragma: no cover
            raise RuntimeError(error)

        # Pull & return results
        cirs = self.__octave.pull("cirs")
        if isinstance(cirs, Struct):  # pragma: no cover
            cirs = np.array([[cirs]])

        return cirs
