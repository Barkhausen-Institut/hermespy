import unittest
import sys
import os

os.environ["MPLBACKEND"] = "agg"

loader = unittest.TestLoader()
integrity_tests = loader.discover('./tests/integrity', top_level_dir='.')
integrity_tests_runner = unittest.runner.TextTestRunner()
integrity_tests_results = integrity_tests_runner.run(integrity_tests)

if integrity_tests_results.errors != [] or integrity_tests_results.failures != []:
    sys.exit(1)

os.environ["MPLBACKEND"] = ""
