# -*- coding: utf-8 -*-

import os
from sys import path, argv
import warnings
from binascii import hexlify
from typing import Union
from unittest.loader import TestLoader
from unittest.runner import TextTestRunner
from unittest.signals import registerResult
from unittest import TestCase, TestSuite
import memray
import json
from datetime import datetime

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TextTestRunnerWrapper(TextTestRunner):
    """Overloads TextTestRunner.run to record time and memory allocation per test class.
    Uses memray (github.com/bloomberg/memray)."""

    def testSuitContainsTestCase(self, tests: Union[TestSuite, TestCase]) -> bool:
        """Check if given TestSuite contains at least one TestCase.

        Args:
            tests (TestSuite | TestCase)

        Return:
            contains (bool): True if tests contains at least one TestCase, False otherwise."""

        if isinstance(tests, TestCase):
            return True
        for i, t in enumerate(tests):
            if self.testSuitContainsTestCase(t):
                return True
        return False

    def run(self, tests: Union[TestSuite, TestCase]):
        """Runs each test in enumerate(tests) and records memory allocations and time.
        Returns a list of test results for each test suite run."""
        # init results
        self.perf_results = []
        self.results = []
        # init temp file
        temp_filename = hexlify(os.urandom(8)).decode("ascii")
        # make tests arg enumeratable if it a TestCase
        tests_ = [tests] if isinstance(tests, TestCase) else tests
        # for each test class
        for index, test in enumerate(tests_):
            if not self.testSuitContainsTestCase(test):
                continue
            # init results
            testName = test.__repr__().split(' ')[2][8:]
            # init memray report file
            memray_temp_file = memray.FileDestination(f'{temp_filename}.bin', overwrite=True)
            # further code is mostly copy-pasted from TextTestRunner
            result = self._makeResult()
            registerResult(result)
            result.failfast = self.failfast
            result.buffer = self.buffer
            result.tb_locals = self.tb_locals
            with warnings.catch_warnings():
                if self.warnings:
                    warnings.simplefilter(self.warnings)
                    if self.warnings in ['default', 'always']:
                        warnings.filterwarnings('module',
                                                category=DeprecationWarning,
                                                message=r'Please use assert\w+ instead.')
                startTestRun = getattr(result, 'startTestRun', None)
                if startTestRun is not None:
                    startTestRun()
                try:
                    with memray.Tracker(destination=memray_temp_file):
                        test(result)
                finally:
                    stopTestRun = getattr(result, 'stopTestRun', None)
                    if stopTestRun is not None:
                        stopTestRun()
            result.printErrors()
            # parse the results
            os.system(f"memray stats --json -f -o {temp_filename}.json {temp_filename}.bin 2>&1 > /dev/null")
            f = open(f'{temp_filename}.json')
            j = json.load(f)
            f.close()
            bytesAllocated = j['total_bytes_allocated']
            timeTaken = datetime.fromisoformat(j['metadata']['end_time'])
            timeTaken = timeTaken - datetime.fromisoformat(j['metadata']['start_time'])
            timeTaken = timeTaken.microseconds // 1000
            self.perf_results.append((testName, timeTaken, bytesAllocated))

            expectedFails = unexpectedSuccesses = skipped = 0
            try:
                results = map(len, (result.expectedFailures,
                                    result.unexpectedSuccesses,
                                    result.skipped))
            except AttributeError:
                pass
            else:
                expectedFails, unexpectedSuccesses, skipped = results

            infos = []
            if not result.wasSuccessful():
                self.stream.write("FAILED")
                failed, errored = len(result.failures), len(result.errors)
                if failed:
                    infos.append("failures=%d" % failed)
                if errored:
                    infos.append("errors=%d" % errored)
            else:
                self.stream.write("OK")
            if skipped:
                infos.append("skipped=%d" % skipped)
            if expectedFails:
                infos.append("expected failures=%d" % expectedFails)
            if unexpectedSuccesses:
                infos.append("unexpected successes=%d" % unexpectedSuccesses)
            if infos:
                self.stream.writeln(" (%s)" % (", ".join(infos),))
            else:
                self.stream.write("\n")
            self.stream.flush()
            self.results.append(result)
        # cleanup
        os.remove(f'{temp_filename}.json')
        os.remove(f'{temp_filename}.bin')
        # return
        return self.results


if __name__ == "__main__":
    # Remove the source directory from path lookup to prevent aliasing
    repository = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    for dir in path:
        if dir.lower() == repository:
            path.remove(dir)

    # Warn the user if we cannot detect hermes
    try:
        import hermespy

    except ModuleNotFoundError:
        print("Hermes could not be detected. Are you sure you installed it without the editable flag?")
        exit(-1)

    # Run all tests as usual
    test_loader = TestLoader()
    test_runner = TextTestRunnerWrapper(verbosity=2, failfast=False)
    tests_dir = os.path.join(repository, "tests")

    if len(argv) < 2:
        start_dir = "."

    else:
        start_dir = argv[1]

    if os.path.isfile(start_dir):
        module_name = start_dir.replace(".py", "").replace(".\\", "").replace("./", "").replace("/", ".").replace("\\", ".")
        tests = test_loader.loadTestsFromName(module_name)

    else:
        tests = test_loader.discover(start_dir, top_level_dir=tests_dir)

    test_results = test_runner.run(tests)

    # Write the performance results into a file
    results_dir = os.path.join(repository, "scripts/performance/results" + os.path.abspath(start_dir)[len(tests_dir):])
    os.makedirs(results_dir, exist_ok=True)
    file = open(results_dir + "/results.json", "wt")
    json.dump(test_runner.perf_results, file, indent=2)
    file.close()

    # Return with a proper exit code indicating test success / failure
    for test_result in test_results:
        if not test_result.wasSuccessful():
            exit(1)
    exit(0)
