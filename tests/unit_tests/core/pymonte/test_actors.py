# -*- coding: utf-8 -*-

from typing import Callable
from unittest import TestCase
from unittest.mock import patch

import ray as ray

from hermespy.core.pymonte.actors import MonteCarloActor
from hermespy.core.pymonte.grid import GridDimension
from .object import TestObjectMock
from .test_evaluation import SumEvaluator, ProductEvaluator

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


@ray.remote(num_cpus=1)
class MonteCarloActorMock(MonteCarloActor[TestObjectMock]):
    """Mock of a Monte Carlo Actor"""

    def init_stage(self) -> None:
        return

    def exit_stage(self) -> None:
        return

    @staticmethod
    def stage_identifiers() -> list[str]:
        return ["init_stage", "exit_stage"]

    def stage_executors(self) -> list[Callable]:
        return [self.init_stage, self.exit_stage]


class TestMonteCarloActor(TestCase):
    """Test the Monte Carlo actor"""

    @classmethod
    def setUpClass(cls) -> None:
        ray.init(local_mode=True, num_cpus=1, ignore_reinit_error=True, logging_level=logging.ERROR)

    @classmethod
    def tearDownClass(cls) -> None:
        # Shut down ray
        ray.shutdown()

    def setUp(self) -> None:
        self.investigated_object = TestObjectMock()
        self.investigated_object.property = 1
        self.dimensions = [GridDimension(self.investigated_object, "property_a", [1, 2, 6, 7, 8])]
        self.evaluators = [SumEvaluator(self.investigated_object), ProductEvaluator(self.investigated_object)]

    def test_run(self) -> None:
        """Running the actor should produce the expected result"""

        actor = MonteCarloActorMock.remote((self.investigated_object, self.dimensions, self.evaluators), 0)

        for sample_idx in range(self.dimensions[0].num_sample_points - 1):
            program = [(sample_idx,), (1 + sample_idx,)]

            result: ActorRunResult = ray.get(actor.run.remote(program))
            self.assertEqual(2, len(result.samples))

    def test_run_exepction_handling(self) -> None:
        """Running the actor should produce the expected result when throwing exepctions"""

        def throw_exepction(*args):
            raise RuntimeError("test")

        with patch.object(self.dimensions[0], "configure_point") as configure_point_mock:
            configure_point_mock.side_effect = throw_exepction
            patched_actor = MonteCarloActorMock.remote((self.investigated_object, self.dimensions, self.evaluators), 0)

            result: ActorRunResult = ray.get(patched_actor.run.remote([(0,)]))
            self.assertEqual("test", result.message)
