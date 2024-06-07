# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from hermespy.core import Transformation
from hermespy.simulation.animation import BITrajectoryB, LinearTrajectory, Moveable, StaticTrajectory, TrajectorySample

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestTrajectorySample(TestCase):
    """Test trajectory sample data class."""
    
    def setUp(self) -> None:
        
        self.timestamp = 12345
        self.pose = Transformation.From_RPY(np.array([1, 2, 3]), np.array([4, 5, 6]))
        self.velocity = np.array([7, 8, 9])
        
        self.sample = TrajectorySample(self.timestamp, self.pose, self.velocity)
        
    def test_init(self) -> None:
        """Initialization parameters are stored correctly."""
        
        trajectory_sample = TrajectorySample(self.timestamp, self.pose, self.velocity)
        
        self.assertEqual(trajectory_sample.timestamp, self.timestamp)
        assert_array_equal(trajectory_sample.velocity, self.velocity)
        assert_array_equal(trajectory_sample.pose, self.pose)


class TestLinearTrajectory(TestCase):
    """Test the linear trajectory class."""
    
    def setUp(self) -> None:
        
        self.initial_pose = Transformation.From_RPY(np.array([1, 2, 3]), np.array([4, 5, 6]))
        self.final_pose = Transformation.From_RPY(np.array([7, 8, 9]), np.array([10, 11, 12]))
        self.duration = 9.876
        self.start = 1.234
        
        self.linear_trajectory = LinearTrajectory(self.initial_pose, self.final_pose, self.duration, self.start)

    def test_init_validation(self) -> None:
        """Initialization should raise an error if the duration is negative."""
        
        with self.assertRaises(ValueError):
            LinearTrajectory(self.initial_pose, self.final_pose, -1, self.start)
            
        with self.assertRaises(ValueError):
            LinearTrajectory(self.initial_pose, self.final_pose, 1, -1)

    def test_max_timestamp(self) -> None:
        """Max timestamp should be the sum of start and duration."""
        
        self.assertAlmostEqual(self.linear_trajectory.max_timestamp, self.start + self.duration)

    def test_sample_before_start(self) -> None:
        """Sampling before start should return the initial pose."""
        
        sample = self.linear_trajectory.sample(self.start - 1)
        assert_array_equal(self.initial_pose, sample.pose)
        
    def test_sample_after_end(self) -> None:
        """Sampling after end should return the final pose."""
        
        sample = self.linear_trajectory.sample(self.start + self.duration + 1)
        assert_array_equal(self.final_pose, sample.pose)
        
    def test_sample(self) -> None:
        """Sampling within the trajectory should return the correct pose."""
        
        sample = self.linear_trajectory.sample(self.start + self.duration / 2)
        assert_array_equal(sample.pose.translation, (self.initial_pose.translation + self.final_pose.translation) / 2)


class TestStaticTrajectory(TestCase):
    """Test static trajectory class."""
    
    def setUp(self) -> None:
        self.pose = Transformation.From_RPY(np.array([1, 2, 3]), np.array([4, 5, 6]))
        self.velocity = np.array([7, 8, 9])
        
        self.trajectory = StaticTrajectory(self.pose, self.velocity)

    def test_init(self) -> None:
        """Initialization parameters are stored correctly."""
        
        assert_array_equal(self.trajectory.pose, self.pose)
        assert_array_equal(self.trajectory.velocity, self.velocity)
        
    def test_sample(self) -> None:
        """Sampling should return the correct pose."""
        
        sample = self.trajectory.sample(12345)
        assert_array_equal(sample.pose, self.pose)
        assert_array_equal(sample.velocity, self.velocity)


class TestMoveable(TestCase):
    """Test moveable base class"""

    def setUp(self) -> None:
        
        self.trajectory = LinearTrajectory(
            Transformation.From_Translation(np.array([1, 2, 3])),
            Transformation.From_Translation(np.array([8, 2, 3])),
            10, 2)
        self.moveable = Moveable(self.trajectory)

    def test_init(self) -> None:
        """Initialization parameters are stored correctly"""

        self.assertIs(self.moveable.trajectory, self.trajectory)

    def test_trajectory_setget(self) -> None:
        """Trajectory property getter should return setter argument"""
        
        expected_trajectory = LinearTrajectory(
            Transformation.From_Translation(np.array([1, 2, 5])),
            Transformation.From_Translation(np.array([8, 2, 5])),
            10, 2)
        
        self.moveable.trajectory = expected_trajectory
        self.assertIs(self.moveable.trajectory, expected_trajectory)
        
    def test_max_trajectory_timestamp(self) -> None:
        """Max timestamp should be the trajectory's max timestamp"""
        
        self.assertAlmostEqual(self.moveable.max_trajectory_timestamp, self.trajectory.max_timestamp)


class TestBITrajectoryB(TestCase):
    """Test the BI trajectory class."""
    
    def setUp(self) -> None:
        
        self.height = 10
        self.duration = 11.2345
        
        self.trajectory = BITrajectoryB(self.height, self.duration)

    def test_max_timestamp(self) -> None:
        """Max timestamp should be the duration"""

        self.assertEqual(self.duration, self.trajectory.max_timestamp)
        
    def test_sample(self) -> None:
        """Sample should return valid poses for each leg"""
        
        leg_timestamps = [1.5 * self.duration, 4.5 * self.duration / 5, 3.5 * self.duration / 5, 2.5 * self.duration / 5, 1.0 * self.duration / 5]
        for leg_timetamp in leg_timestamps:
            sample = self.trajectory.sample(leg_timetamp)
            self.assertEqual(sample.timestamp, leg_timetamp)
