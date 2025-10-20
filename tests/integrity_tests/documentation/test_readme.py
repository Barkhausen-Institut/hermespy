# -*- coding: utf-8 -*-

from os import path
from unittest import TestCase
import re

from unit_tests.utils import SimulationTestContext

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestReadme(TestCase):
    def test_readme(self) -> None:

        # Load the readme as a string
        readme_location = path.join(path.dirname(path.abspath(__file__)), "..", "..", "..", "README.md")
        with open(readme_location, "r") as f:
            readme_text = f.read()

        # Extract python code blocks
        code_blocks = re.findall(r"```python(.*?)```", readme_text, re.MULTILINE | re.DOTALL | re.S)

        # Replace the num_samples with one to avoid long simulation times
        code_blocks[1] = re.sub(r"num_samples = \d+", "num_samples = 1", code_blocks[1])

        # Replace the num_drops with one to avoid long simulation times
        code_blocks[2] = re.sub(r"num_drops = \d+", "num_drops = 1", code_blocks[2])

        # Execute the code blocks
        with SimulationTestContext():
            for c, code_block in enumerate(code_blocks):
                with self.subTest(msg=f"Code block {c}"):
                    try:
                        exec(code_block)
                    except Exception as e:
                        self.fail(f"Code block {c} failed: {e}")
