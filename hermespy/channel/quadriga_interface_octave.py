# -*- coding: utf-8 -*-
"""Octave interface to the Quadriga channel model."""

from __future__ import annotations
from logging import getLogger, Logger
from typing import Optional, List, Any

import numpy as np
from oct2py import Oct2Py, Oct2PyError, Struct

from .quadriga_interface import QuadrigaInterface

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronauer@barkhauseninstitut.org"
__status__ = "Prototype"


class QuadrigaOctaveInterface(QuadrigaInterface):
    """Quadriga Octave Interface."""

    __octave: Oct2Py
    __logger: Logger

    def __init__(self,
                 octave_bin: Optional[str] = None,
                 **kwargs: Any) -> None:
        """Quadriga Octave interface object initialization.

        Args:
        
            octave_bin (str, optional):
                Path to the octave cli executable.
                
            kwargs (Any):
                Interface arguments.
        """

        # Init base class
        QuadrigaInterface.__init__(self, **kwargs)

        # Init octave session
        self.__logger = getLogger('octave_logger')
        self.__octave = Oct2Py(logger=self.__logger)  # executable=octave_bin)

        # Add quadriga source folder to octave lookup paths
        self.__octave.addpath(self.path_quadriga_src)
        
        # Add launch script folder to octave loopkup paths
        self.__octave.addpath(self.path_launch_script)

    def _run_quadriga(self, **parameters) -> List[Any]:

        # Push parameters to quadriga
        for key, value in parameters.items():

            # Convert numpy arrays to lists
            if isinstance(value, np.ndarray):
                value = value.tolist()

            self.__octave.push(key, value)

        # Launch octave
        try:
            self.__octave.eval('launch_quadriga')

        except Oct2PyError as error:
            raise RuntimeError(error)

        # Pull & return results
        cirs = self.__octave.pull("cirs")

        if isinstance(cirs, Struct):
            cirs = np.array([[cirs]])

        return cirs
