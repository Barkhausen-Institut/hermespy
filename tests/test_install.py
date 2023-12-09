import os
from sys import path, argv
from unittest.loader import TestLoader
from unittest.runner import TextTestRunner

if __name__ == "__main__":
    # Remove the source directory from path lookup to prevent aliasing
    repository = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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
    test_runner = TextTestRunner(verbosity=2, failfast=False)

    if len(argv) < 2:
        start_dir = "."

    else:
        start_dir = argv[1]

    if os.path.isfile(start_dir):
        module_name = start_dir.replace(".py", "").replace(".\\", "").replace("./", "").replace("/", ".").replace("\\", ".")
        tests = test_loader.loadTestsFromName(module_name)

    else:
        tests = test_loader.discover(start_dir, top_level_dir=os.path.join(repository, "tests"))

    test_result = test_runner.run(tests)

    # Return with a proper exit code indicating test success / failure
    exit(int(not test_result.wasSuccessful()))
