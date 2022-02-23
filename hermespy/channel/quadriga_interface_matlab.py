# -*- coding: utf-8 -*-
"""Matlab interface to the Quadriga channel model."""

from __future__ import annotations
from typing import Optional, List, Any
from matlab.engine import MatlabEngine, start_matlab
import matlab
import numpy as np

from .quadriga_interface import QuadrigaInterface

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.6"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronaue@barkhauseninstitut.org"
__status__ = "Prototype"


class QuadrigaMatlabInterface(QuadrigaInterface):
    """Quadriga Matlab interface."""

    __engine: MatlabEngine

    def __init__(self,
                 path_quadriga_src: Optional[str] = None,
                 antenna_kind: Optional[str] = None,
                 scenario_label: Optional[str] = None) -> None:
        """Quadriga Matlab interface object initialization.

        Args:
            path_quadriga_src (str, optional): Path to the Quadriga Matlab source files.
            antenna_kind (str, optional): Type of antenna considered.
            scenario_label (str, optional): Scenario label.
        """

        # Init base class
        QuadrigaInterface.__init__(self, path_quadriga_src, antenna_kind, scenario_label)

        # Start the matlab engine
        self.__engine = start_matlab()

    def _run_quadriga(self, **parameters) -> List[Any]:

        # Create the Matlab workspace from the given parameters
        for key, value in parameters:

            if isinstance(value, np.ndarray):

                if value.dtype is float:
                    value = matlab.double(value.tolist())

                elif value.dtype is int:
                    value = matlab.int32(value.tolist())

                else:
                    value = matlab.object(value.tolist())

            elif isinstance(value, float):
                value = matlab.double(value)

            elif isinstance(value, int):
                value = matlab.int32(value)

            else:
                value = matlab.object(value)

            self.__engine.workspace[key] = value

        # Launch Matlab
        self.__engine.launch_quadriga_script(nargout=0)

        # Fetch & return results
        return self.__engine.workspace["cirs"]
