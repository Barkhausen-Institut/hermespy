# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import patch, MagicMock

from hermespy.tools.shermes import sHermes

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


@patch('hermespy.tools.shermes.run')
@patch('hermespy.tools.shermes.which')
class TestSlurmEntry(TestCase):
    """Test the shermes entry command."""

    def test_sanity_check(self, which_mock: MagicMock, run_mock: MagicMock) -> None:
        """Test exit on sanit checks"""

        # sbash command not detected
        which_mock.return_value = None

        with self.assertRaises(SystemExit):
            sHermes(['test_script.py'])

        which_mock.assert_called_once_with("sbatch")

    def test_run_execution(self, which_mock: MagicMock, run_mock: MagicMock) -> None:
        """Test run execution"""

        # sbash comand detected
        which_mock.return_value = "/usr/bin/sbash"

        sHermes(['test_script.py'])

        # Check if the run command was called with the expected arguments
        run_mock.assert_called_once()
