# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Mapping
from unittest import TestCase
from tempfile import TemporaryDirectory
import os.path as path

import nbformat as nbform
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestNotebooks(TestCase):
    def setUp(self) -> None:
        # Create a new tempo
        self.dir = TemporaryDirectory()

    def tearDown(self) -> None:
        self.dir.cleanup()

    def __load_notebook(self, notebook: str) -> nbform.NotebookNode:
        """Load a notebook as a notebooknode.

        Args:

            notebook (str):
                Name of the notebook to be loaded.

        Returns:
            The generated notebook node.
        """

        filename = path.dirname(path.abspath(__file__))
        notebook_path = path.join(filename, "..", "..", "..", "docssource", "notebooks", notebook)

        notebook = nbform.read(notebook_path, as_version=4)

        return notebook

    @staticmethod
    def __patch_notebook(notebook: nbform.NotebookNode, cell: int, patches: Mapping[int, str] | None = None, inserts: Mapping[int, str] | None = None) -> None:
        """Internal subroutine to patch notebook source code.

        Args:

            notebook(nbform.NotebookNode):
                The notebook to be patched.

            cell (int):
                The cell to be patched.

            patches (Mapping[int, str], optional):
                Map of lines to be patched to the new line.

            inserts (Mapping[int, str], optional):
                Map of lines to be inserted to the new line.
        """

        # Fetch original cell lines
        new_lines: list = notebook["cells"][cell]["source"].split("\n")

        # Patch lines
        if patches is not None:
            for k, p in patches.items():
                new_lines[k] = p

        # Insert additional lines
        if inserts is not None:
            for k, i in sorted(inserts.items(), reverse=True):
                new_lines.insert(k, i)

        notebook["cells"][cell]["source"] = "\n".join(new_lines)

    def __test_notebook(self, notebook: nbform.NotebookNode) -> None:
        """Internal subroutine to test a full notebook.

        Fails the test if the notebook execution results in an exception.

        Args:

            notebook (nbform.NotebookNode):
                The notebook to be tested.
        """

        # Create new executor instance
        executor = ExecutePreprocessor(kernel_name="python3")

        # Execute the notebook
        try:
            executor.preprocess(notebook, {"metadata": {"path": self.dir.name}})

        except CellExecutionError as error:
            self.fail(error)

    def test_audiodevice(self) -> None:
        """Test the audio device loop example notebook"""

        notebook = self.__load_notebook("audio.ipynb")
        self.__patch_notebook(notebook, 2, patches={2: "from hermespy.hardware_loop import PhysicalDeviceDummy"})
        self.__patch_notebook(notebook, 4, patches={0: "device = PhysicalDeviceDummy()"})
        self.__test_notebook(notebook)

    def test_beamforming_implementation(self) -> None:
        """Test the beamforming implemntation example notebook"""

        notebook = self.__load_notebook("beamforming_implementation.ipynb")
        self.__patch_notebook(notebook, 8, inserts={2: "import ray as ray\n", 9: "ray.init(local_mode=True)"}, patches={9: "simulation = Simulation(console_mode=ConsoleMode.SILENT, num_actors=1)", 24: "simulation.num_samples = 1"})
        self.__test_notebook(notebook)

    def test_beamforming_usage(self) -> None:
        """Test beamforming usage notebook"""

        notebook = self.__load_notebook("beamforming_usage.ipynb")
        self.__patch_notebook(notebook, 8, inserts={0: "import ray as ray\n", 7: "ray.init(local_mode=True)"}, patches={8: "simulation = Simulation(console_mode=ConsoleMode.SILENT, num_actors=1, num_samples=1)"})
        self.__patch_notebook(notebook, 25, patches={21: "simulation.num_drops = 1"})
        self.__patch_notebook(notebook, 31, patches={14: "simulation.num_drops = 1"})
        self.__test_notebook(notebook)

    def test_channel(self) -> None:
        """Test the channel implementation example notebook"""

        notebook = self.__load_notebook("channel.ipynb")
        self.__patch_notebook(notebook, 4, 
            patches= {
                8: "simulation = Simulation(console_mode=ConsoleMode.SILENT, num_actors=1)",
                34: "simulation.num_samples = 1",
            },
            inserts={1: "import ray as ray", 8: "ray.init(local_mode=True)"},
        )
        self.__test_notebook(notebook)

    def test_datasets(self) -> None:
        """Test the datasets recording example notebook"""

        notebook = self.__load_notebook("datasets.ipynb")
        self.__test_notebook(notebook)

    def test_evaluator(self) -> None:
        """Test the evaluator implementation example notebook"""

        notebook = self.__load_notebook("evaluator.ipynb")
        self.__patch_notebook(
            notebook, 4,
            inserts={0: "import ray as ray\n", 6: "ray.init(local_mode=True)"},
            patches={
                6: "simulation = Simulation(console_mode=ConsoleMode.SILENT, num_actors=1)",
            },
        )
        self.__test_notebook(notebook)

    def test_fec(self) -> None:
        """Test the forward error correction implementation example notebook"""

        notebook = self.__load_notebook("fec_coding.ipynb")
        self.__patch_notebook(notebook, 8, inserts={0: "import ray as ray\n", 5: "ray.init(local_mode=True)"}, patches={5: "simulation = Simulation(console_mode=ConsoleMode.SILENT, num_actors=1)", 17: "simulation.num_samples = 1"})
        self.__test_notebook(notebook)

    def test_precoding(self) -> None:
        """Test the MIMO precoding implementation example notebook"""

        notebook = self.__load_notebook("precoding.ipynb")
        self.__patch_notebook(notebook, 4, inserts={0: "import ray as ray\n", 5: "ray.init(local_mode=True)"}, patches={5: "simulation = Simulation(console_mode=ConsoleMode.SILENT, num_actors=1, num_samples=1)"})
        self.__test_notebook(notebook)

    def test_roc(self) -> None:
        """Test the receiver operation characteristics example notebook"""

        notebook = self.__load_notebook("roc.ipynb")
        self.__patch_notebook(notebook, 2, inserts={0: "import ray as ray\n", 16: "ray.init(local_mode=True)"})
        self.__patch_notebook(notebook, 6, patches={22: "simulation.num_samples = 1"})
        self.__patch_notebook(notebook, 8, patches={10: "hardware_loop.num_samples = 1"})
        self.__test_notebook(notebook)

    def test_waveform(self) -> None:
        """Test the communication waveform implementation example notebook"""

        notebook = self.__load_notebook("waveform.ipynb")
        self.__patch_notebook(notebook, 10, patches={3: "simulation = Simulation(console_mode=ConsoleMode.SILENT, num_actors=1)", 11: "simulation.num_samples = 1"}, inserts={0: "import ray as ray", 3: "ray.init(local_mode=True)"})
        self.__test_notebook(notebook)
