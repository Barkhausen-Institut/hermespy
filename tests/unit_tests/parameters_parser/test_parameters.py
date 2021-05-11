import unittest
from unittest.mock import patch, Mock

from parameters_parser.parameters_general import ParametersGeneral
from parameters_parser.parameters_scenario import ParametersScenario
from parameters_parser.parameters import Parameters


class TestParameters(unittest.TestCase):
    def setUp(self) -> None:
        path = 'myPath'
        self.params = Parameters(path)

    def test_proper_initialization(self) -> None:
        self.assertTrue(isinstance(self.params.general, ParametersGeneral))
        self.assertTrue(isinstance(self.params.scenario, ParametersScenario))

    @patch.object(ParametersGeneral, 'read_params')
    @patch.object(ParametersScenario, 'read_params')
    def test_reading_parameters(self, mock_params_scenario_read_params: Mock,
                                mock_params_general_read_params: Mock) -> None:
        self.params.read_params()
        mock_params_general_read_params.assert_called_once()
        mock_params_scenario_read_params.assert_called_once()
