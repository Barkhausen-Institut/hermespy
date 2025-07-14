# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

from hermespy.core.pymonte.artifact import ArtifactTemplate, MonteCarloSample

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestArtifactTemplate(TestCase):
    """Test the template for scalar artifacts"""

    def setUp(self) -> None:
        self.artifact_value = 1.2345
        self.artifact = ArtifactTemplate[float](self.artifact_value)

    def test_init(self) -> None:
        """Initialization parameter should be properly stored as class attributes"""

        self.assertEqual(self.artifact_value, self.artifact.artifact)

    def test_artifact(self) -> None:
        """Artifact property should return the represented scalar artifact"""

        self.assertEqual(self.artifact_value, self.artifact.artifact)

    def test_str(self) -> None:
        """String representation should return a string"""

        self.assertIsInstance(self.artifact.__str__(), str)

    def test_to_scalar(self) -> None:
        """Scalar conversion routine should return the represented artifact"""

        self.assertEqual(self.artifact_value, self.artifact.to_scalar())


class TestMonteCarloSample(TestCase):
    """Test the Monte Carlo sample class"""

    def setUp(self) -> None:
        self.grid_section = (0, 1, 2, 3)
        self.sample_index = 5

        self.evaluation_artifacts = []
        for _ in range(5):
            artifact = Mock()
            artifact.to_scalar.return_value = 1.0
            self.evaluation_artifacts.append(artifact)

        self.sample = MonteCarloSample(self.grid_section, self.sample_index, self.evaluation_artifacts)

    def test_init(self) -> None:
        """Initialization arguments should be properly stored as object attributes"""

        self.assertCountEqual(self.grid_section, self.sample.grid_section)
        self.assertEqual(self.sample_index, self.sample.sample_index)
        self.assertCountEqual(self.evaluation_artifacts, self.sample.artifacts)
        self.assertEqual(len(self.evaluation_artifacts), self.sample.num_artifacts)

    def test_artifact_scalars(self) -> None:
        """Artifact scalars property should call the artifact conversion routine for each scalar"""

        scalars = self.sample.artifact_scalars

        self.assertEqual(len(self.evaluation_artifacts), len(scalars))

        for artifact in self.evaluation_artifacts:
            artifact.to_scalar.assert_called()
