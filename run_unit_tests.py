
import unittest
import sys
import os

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronaue@barkhauseninstitut.org"
__status__ = "Prototype"

os.environ["MPLBACKEND"] = "agg"
loader = unittest.TestLoader()
unit_tests = loader.discover('./tests/unit_tests', top_level_dir='.')
unit_test_runner = unittest.runner.TextTestRunner()
unit_test_results = unit_test_runner.run(unit_tests)

if unit_test_results.errors != [] or unit_test_results.failures != []:
    sys.exit(1)

os.environ["MPLBACKEND"] = ""
