# -*- coding: utf-8 -*-
# pragma: no cover

from __future__ import annotations

try:  # pragma: no cover
    from matlab.engine import MatlabEngine, start_matlab
    import matlab

except ImportError:  # pragma: no cover
    MatlabEngine = None
    start_matlab = None
    matlab = None

import numpy as np

from .quadriga_interface import QuadrigaInterface

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class QuadrigaMatlabInterface(QuadrigaInterface):
    """Quadriga Matlab interface."""

    __engine: MatlabEngine

    def __init__(self, *args, **kwargs) -> None:
        # Init base class
        QuadrigaInterface.__init__(self, *args, **kwargs)

        # Start the matlab engine
        self.__engine = start_matlab()

    def _run_quadriga(self, **parameters) -> np.ndarray:
        # Create the Matlab workspace from the given parameters
        for key, value in parameters.items():
            if isinstance(value, np.ndarray):
                if value.dtype is float:  # pragma: no cover
                    value = matlab.double(value.tolist())

                elif value.dtype is int:  # pragma: no cover
                    value = matlab.int32(value.tolist())

                else:  # pragma: no cover
                    value = matlab.object(value.tolist())

            elif isinstance(value, float):  # pragma: no cover
                value = matlab.double(value)

            elif isinstance(value, int):  # pragma: no cover
                value = matlab.int32(value)

            else:  # pragma: no cover
                value = matlab.object(value)

            self.__engine.workspace[key] = value

        # Launch Matlab
        self.__engine.launch_quadriga_script(nargout=0)

        # Fetch & return results
        return self.__engine.workspace["cirs"]
