# -*- coding: utf-8 -*-

from logging import ERROR
from typing import Callable
from typing_extensions import override
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from hermespy.core.pymonte.actors import MonteCarloActor, MonteCarloCollector, MonteCarloQueueManager
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


class MonteCarloActorMock(MonteCarloActor[TestObjectMock]):
    """Mock of a Monte Carlo Actor"""

    def init_stage(self) -> None:
        return

    def exit_stage(self) -> None:
        return

    @staticmethod
    @override
    def stage_identifiers() -> list[str]:
        return ["init_stage", "exit_stage"]

    @override
    def stage_executors(self) -> list[Callable]:
        return [self.init_stage, self.exit_stage]


class TestQueueManager(TestCase):
    """Test the Monte Carlo queue manager"""
    
    def setUp(self):
        self.test_object = TestObjectMock()
        self.grid = [
            GridDimension(self.test_object, "property_a", [1, 2, 3, 4, 5]),
            GridDimension(self.test_object, "property_b", [1, 2, 3, 4, 5]),
        ]
        self.num_samples = 7
        self.batch_size = 6
        
        self.queue_manager = MonteCarloQueueManager(
            self.grid,
            self.num_samples,
            self.batch_size,
        )
        
    def test_next_batch(self) -> None:
        """Test the next batch generation"""
        
        # Generate the first batch
        batch = self.queue_manager.next_batch()
        
        # The number of samples in the batch should be equal to the batch size
        self.assertEqual(len(batch), self.batch_size)
        
        num_sections = 5 * 5 * self.num_samples
        num_batches = int(num_sections / self.batch_size - 1)
        
        for i in range(num_batches + 5):
            batch = self.queue_manager.next_batch()
            if len(batch) == 0:
                break
        
        self.assertEqual(i - 1, num_batches)
    
    def test_nonexisting_section(self) -> None:
        """Disabling a non-existing section should not raise an error"""

        self.queue_manager.disable_section((100, 100))

    def test_query_progress(self) -> None:
        """Test the progress query"""

        # The progress should be zero at the beginning
        progress, _ = self.queue_manager.query_progress()
        self.assertEqual(progress, 0.0)

        # Generate a batch and check the progress
        self.queue_manager.next_batch()
        progress, _ = self.queue_manager.query_progress()
        self.assertGreater(progress, 0.0)
        
        # Disable all sections
        for section_index in np.ndindex((5, 5)):
            self.queue_manager.disable_section(section_index)
            
        # The progress should be 1.0 now
        progress, _ = self.queue_manager.query_progress()
        self.assertEqual(progress, 1.0)


class TestMonteCarloCollector(TestCase):
    """Test the Monte Carlo collector"""

    def setUp(self) -> None:
        self.test_object = TestObjectMock()
        self.grid = [
            GridDimension(self.test_object, "property_a", [1, 2, 3, 4, 5]),
            GridDimension(self.test_object, "property_b", [1, 2, 3, 4, 5]),
        ]
        self.evaluators = [SumEvaluator(self.test_object), ProductEvaluator(self.test_object)]
        
        self.queue_manager = MonteCarloQueueManager(self.grid, 10, 6)
        self.actors = []
        self.collector = MonteCarloCollector(
            self.queue_manager,
            self.actors,
            self.grid,
            self.evaluators,
        )

    def test_run(self) -> None:
        """Test the collector's run method"""
        pass

    def test_runtime_estimates(self) -> None:
        """Test querying runtime estimates from the collector"""

        runtime_esimates = self.collector.query_estimates()
        self.assertEqual(len(runtime_esimates), len(self.evaluators))

    @patch("hermespy.core.pymonte.actors.wait", autospec=True)
    def test_fetch_results(self, wait_mock: Mock) -> None:
        """Test fetching results from the collector"""

        # Mock the wait function to return a single section
        wait_mock.return_value = [[] for _ in range(len(self.evaluators))]

        # Fetch the results
        results = self.collector.fetch_results()

        # Check that the results are correct
        self.assertEqual(len(results), len(self.evaluators))


class TestMonteCarloActor(TestCase):
    """Test the Monte Carlo actor"""

    def setUp(self) -> None:
        self.investigated_object = TestObjectMock()
        self.investigated_object.property = 1
        self.dimensions = [GridDimension(self.investigated_object, "property_a", [1, 2, 6, 7, 8])]

    def test_init_validation(self) -> None:
        """Test the actor's initialization argument validation"""

        invalid_stage_arguments = {'invalid_stage': None}
        with self.assertRaises(ValueError):
            MonteCarloActorMock(Mock(), (self.investigated_object, self.dimensions, []), 0, invalid_stage_arguments)

    def test_investigated_object(self) -> None:
        """Investigated object property should return the correct object"""
        
        actor = MonteCarloActorMock(Mock(), (self.investigated_object, self.dimensions, []), 0)
        
        self.assertIs(actor._investigated_object, self.investigated_object)

    @patch("hermespy.core.pymonte.actors.get", autospec=True)
    def test_run(self, get_mock: Mock) -> None:
        """Test the actor's run method"""

        # Manipulate the get function to return a single section followed by an empty program to break the run method's loop
        get_counter = 0
        def get_mock_side_effect(*args, **kwargs):
            nonlocal get_counter
            if get_counter < 1:
                get_counter += 1
                return [(0,),]
            else:
                return []
        get_mock.side_effect = get_mock_side_effect

        actor = MonteCarloActorMock(Mock(), (self.investigated_object, self.dimensions, []), 0)
        
        # Run a single progpram section
        actor.run()
        
        # Fetch the results
        results = actor.fetch_results()
        
        # Check that the results are correct
        self.assertEqual(len(results), 1)
        
        # Check that the result list has been emptied
        self.assertEqual(len(actor.fetch_results()), 0)
