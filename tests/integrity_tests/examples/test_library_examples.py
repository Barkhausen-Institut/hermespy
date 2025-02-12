# -*- coding: utf-8 -*-

import logging
from contextlib import ExitStack
from os import path as os_path
from sys import path as sys_path
from unittest import TestCase
from unittest.mock import patch, PropertyMock
from warnings import filterwarnings

import ray as ray
from matplotlib import use as matplotlib_use

from unit_tests.utils import SimulationTestContext

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestLibraryExamples(TestCase):
    """Test library example execution without exceptions"""

    def setUp(self) -> None:
        # Switch matplotlib to a non-interactive backend to prevent
        # window popups during testing
        matplotlib_use("Agg")

        # Ignore matplotlib warnings stemming from the non-interactive backend
        filterwarnings("ignore", category=UserWarning)

        library_dir = os_path.abspath(os_path.join(os_path.dirname(__file__), "..", "..", "..", "_examples", "library"))
        sys_path.append(library_dir)

        # Patch the executable debug flag to True
        self.debug_patch = patch("hermespy.core.executable.Executable.debug", new_callable=PropertyMock)
        mock_debug = self.debug_patch.start()
        mock_debug.return_value = True

    def tearDown(self) -> None:
        # Remove the executable debug flag patch
        self.debug_patch.stop()

    @classmethod
    def setUpClass(cls) -> None:
        ray.init(local_mode=True, num_cpus=1, ignore_reinit_error=True, logging_level=logging.ERROR)

    @classmethod
    def tearDownClass(cls) -> None:
        ray.shutdown()

    def test_getting_started_link(self) -> None:
        """Test getting started library link example execution"""

        try:
            import getting_started_link  # type: ignore  # noqa: F401
        except Exception as e:
            self.fail(f"Exception raised: {e}")

    @patch("sys.stdout")
    def test_getting_started_loop(self, mock_stdout) -> None:
        """Test getting started library loop example"""

        try:
            import getting_started_loop  # type: ignore  # noqa: F401
        except Exception as e:
            self.fail(f"Exception raised: {e}")

    def test_getting_started_mobility(self) -> None:
        """Test getting started library mobility example execution"""

        with SimulationTestContext(patch_plot=True):
            try:
                import getting_started_mobility  # type: ignore  # noqa: F401
            except Exception as e:
                self.fail(f"Exception raised: {e}")

    def test_getting_started_ofdm_link(self) -> None:
        """Test getting started library OFDM link example execution"""

        try:
            import getting_started_ofdm_link  # type: ignore  # noqa: F401
        except Exception as e:
            self.fail(f"Exception raised: {e}")

    def test_getting_started_simulation_multidim(self) -> None:
        """Test getting started library multidimensional simulation example execution"""

        with SimulationTestContext(patch_plot=True):
            try:
                import getting_started_simulation_multidim  # type: ignore  # noqa: F401
            except Exception as e:
                self.fail(f"Exception raised: {e}")

    # Test deactivated because patching the matplotlib backend
    # Causes Ray's cloudpickle to throw an exception
    def __test_getting_started_simulation(self) -> None:
        """Test getting started library simulation example execution"""

        with SimulationTestContext(patch_plot=True):
            try:
                import getting_started_simulation  # type: ignore  # noqa: F401
            except Exception as e:
                self.fail(f"Exception raised: {e}")

    def test_usrp_loop(self) -> None:
        """Test USRP loop example execution"""

        with ExitStack() as stack:
            stack.enter_context(SimulationTestContext(patch_plot=True))

            from hermespy.hardware_loop import PhysicalScenarioDummy, PhysicalDeviceDummy

            new_device = PhysicalDeviceDummy()

            def new_device_callback(self, *args, **kwargs):
                if new_device not in self.devices:
                    self.add_device(new_device)
                return new_device

            new_device_patch = stack.enter_context(patch.object(PhysicalScenarioDummy, "new_device", autospec=True))
            new_device_patch.side_effect = new_device_callback

            stack.enter_context(patch("hermespy.hardware_loop.UsrpSystem", PhysicalScenarioDummy))
            stack.enter_context(patch("hermespy.hardware_loop.HardwareLoop.new_dimension"))

            import usrp_loop  # type: ignore  # noqa: F401
