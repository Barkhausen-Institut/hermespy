import os

from parameters_parser.parameters_general import ParametersGeneral
from parameters_parser.parameters_scenario import ParametersScenario


class Parameters:
    """This class implements a parameters parser to be used within HermesPy."""

    def __init__(self, path: str) -> None:
        """creates a parsing object, that will read the parameter files contained in directory 'path'.

        Parameter files have predetermined names. The following are currently necessary:
        - 'settings_general.ini' (mandatory) : file containing parameters that control the simulation flow
        - 'settings_scenario.ini' (mandatory) : file describing the simulation scenario (modems, channels)"""
        self.general = ParametersGeneral()
        self.scenario = ParametersScenario()

        self._path = path

    def read_params(self) -> None:

        filename = "settings_general.ini"
        self.general.read_params(os.path.join(self._path, filename))

        filename = "settings_scenario.ini"
        self.scenario.read_params(os.path.join(self._path, filename))
