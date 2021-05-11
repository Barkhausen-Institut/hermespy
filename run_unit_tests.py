
import unittest
import sys
import os
print("a")
os.environ["MPLBACKEND"] = "agg"
print("a")
loader = unittest.TestLoader()
unit_tests = loader.discover('./tests/unit_tests', top_level_dir='.')
print("a")
unit_test_runner = unittest.runner.TextTestRunner()
unit_test_results = unit_test_runner.run(unit_tests)

if unit_test_results.errors != [] or unit_test_results.failures != []:
    sys.exit(1)

os.environ["MPLBACKEND"] = ""
