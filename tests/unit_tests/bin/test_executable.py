# -*- coding: utf-8 -*-

from importlib import import_module
from unittest import TestCase
from unittest.mock import Mock, patch

from ruamel.yaml.constructor import ConstructorError

from hermespy.core import Executable

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestHermes(TestCase):
    """Test the HermesPy Command Line Interface"""

    def setUp(self) -> None:
        self.executable_mock = Mock(spec=Executable)

        self.console_patch = patch("hermespy.bin.hermes.Console")
        self.console_mock = self.console_patch.start()

        self.factory_patch = patch("hermespy.bin.hermes.Factory.from_path")
        self.from_path_mock = self.factory_patch.start()
        self.from_path_mock.return_value = [self.executable_mock]

        self.copy_patch = patch("shutil.copy")
        self.copy_mock = self.copy_patch.start()

        from hermespy.bin.hermes import hermes_simulation

        self.hermes_simulation = lambda *args: hermes_simulation(*args)

    def tearDown(self) -> None:
        self.console_patch.stop()
        self.factory_patch.stop()
        self.copy_patch.stop()

    def test_hermes_validation(self) -> None:
        """Test the HermesPy Command Line Interface argument validation"""

        self.from_path_mock.return_value = []
        with self.assertRaises(SystemExit):
            self.hermes_simulation(["no_directory"])

        def raiseConstructError(*args):
            mark = Mock()
            mark.line = 1
            mark.name = "test"
            raise ConstructorError(problem_mark=mark)

        self.from_path_mock.side_effect = raiseConstructError
        with self.assertRaises(SystemExit):
            self.hermes_simulation(["no_directory"])

    def test_hermes(self) -> None:
        """Test Hermespy Command Line Interface execution"""

        args = ["no_directory", "--log", "-s", "black", "-o", "no_directory"]
        self.hermes_simulation(args)
