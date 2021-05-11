
import unittest
import sys
import os

os.environ["MPLBACKEND"] = "agg"
loader = unittest.TestLoader()
unit_tests = loader.discover('./tests/unit_tests', top_level_dir='.')
unit_test_runner = unittest.runner.TextTestRunner()
unit_test_results = unit_test_runner.run(unit_tests)

if unit_test_results.errors != [] or unit_test_results.failures != []:
    sys.exit(1)

os.environ["MPLBACKEND"] = ""
