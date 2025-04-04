# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.constants import speed_of_light
from scipy.ndimage import gaussian_filter

from hermespy.radar import PointDetection, RadarCube, RadarPointCloud, ThresholdDetector, MaxDetector, CFARDetector
from unit_tests.utils import SimulationTestContext
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Egor Achkasov"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestPointDetection(TestCase):
    """Test the base class for radar point detections"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        self.position = self.rng.normal(size=3)
        self.velocity = self.rng.normal(size=3)
        self.power = 1.2345

        self.point = PointDetection(self.position, self.velocity, self.power)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        assert_array_equal(self.position, self.point.position)
        assert_array_equal(self.velocity, self.point.velocity)
        self.assertEqual(self.power, self.point.power)

    def test_init_validation(self) -> None:
        """Initialization should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            _ = PointDetection(np.zeros(2), self.velocity, self.power)

        with self.assertRaises(ValueError):
            _ = PointDetection(self.position, np.zeros(2), self.power)

        with self.assertRaises(ValueError):
            _ = PointDetection(self.position, self.velocity, 0.0)

    def test_from_spherical(self) -> None:
        """Point detections should be constructable from spherical coordinates"""

        zenith = np.pi / 2
        azimuth = np.pi / 2
        range = 1.234
        velocity = 3.45
        power = 6.789

        point = PointDetection.FromSpherical(zenith, azimuth, range, velocity, power)

        self.assertAlmostEqual(0, point.position[0])
        self.assertAlmostEqual(range, point.position[1])
        self.assertAlmostEqual(0, point.position[2])
        assert_array_almost_equal(np.array([0, velocity, 0]), point.velocity)
        self.assertEqual(power, point.power)

    def test_eq(self) -> None:
        """Check if NotImplemented is returned if comparing with something other then PointDetection"""

        self.assertTrue(PointDetection.__eq__(self.point, "something") == NotImplemented)

    def test_serialization(self) -> None:
        """Test max detector serialization"""
        
        test_roundtrip_serialization(self, self.point)


class TestRadarPointCloud(TestCase):
    """Test the radar point cloud class"""

    def setUp(self) -> None:
        self.cloud = RadarPointCloud(max_range=10.0)

    def test_init_validation(self) -> None:
        """Initialization should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            _ = RadarPointCloud(max_range=0.0)

        with self.assertRaises(ValueError):
            _ = RadarPointCloud(max_range=-1.0)

    def test_properties(self) -> None:
        """Properties should be properly stored as class attributes"""

        self.assertEqual(10.0, self.cloud.max_range)

    def test_add_point(self) -> None:
        """Points should be properly added to the point cloud"""

        point = PointDetection(np.zeros(3), np.zeros(3), 1.0)

        self.cloud.add_point(point)

        self.assertEqual(1, self.cloud.num_points)
        self.assertEqual(point, self.cloud.points[0])

    def test_plot(self) -> None:
        """Point clouds should be properly plotted"""

        with SimulationTestContext():
            self.cloud.add_point(PointDetection(np.zeros(3), np.zeros(3), 1.0))
            _ = self.cloud.visualize()

    def test_serialization(self) -> None:
        """Test radar point cloud serialization"""

        test_roundtrip_serialization(self, self.cloud)

class TestThresholdDetector(TestCase):
    """Test the threshold detector class"""

    def setUp(self) -> None:
        self.min_power = 0.1

        self.detector = ThresholdDetector(self.min_power)

    def test_properties(self) -> None:
        """Properties should be properly stored as class attributes"""

        self.assertEqual(self.min_power, self.detector.min_power)
        self.assertTrue(self.detector.normalize)
        self.assertTrue(self.detector.peak_detection)

    def test_min_power_validation(self) -> None:
        """Min power property setter should raise ValueErrors on invalid arguments"""

        with self.assertRaises(ValueError):
            self.detector.min_power = -1.0

        with self.assertRaises(ValueError):
            self.detector.min_power = 0.0

    def test_min_power_setget(self) -> None:
        """Min power property getter should return setter argument"""

        expected_min_power = 1.234
        self.detector.min_power = expected_min_power

        self.assertEqual(expected_min_power, self.detector.min_power)

    def test_detect_filtered(self) -> None:
        """Threshold detector should properly detect points with filtering enabled"""

        self.detector.peak_detection = True

        carrier_frequency = 72e9
        range_bins = np.arange(101)
        doppler_bins = np.arange(15) - 7
        angle_bins = np.array([[-1, 0.0], [0.0, 0.0], [1, 0.0]])

        data = np.zeros((3, 15, 101))
        data[1, 4, :51] = np.arange(51) / 50
        data[1, 4, 51:] = np.arange(50, 0, -1) / 51

        cube = RadarCube(data, angle_bins, doppler_bins, range_bins, carrier_frequency)
        cloud = self.detector.detect(cube)

        self.assertEqual(1, cloud.num_points)
        self.assertAlmostEqual(1.0, cloud.points[0].power)
        assert_array_almost_equal(np.array([0.0, 0.0, range_bins[50]]), cloud.points[0].position)
        assert_array_almost_equal(np.array([0, 0, doppler_bins[4] * speed_of_light / carrier_frequency]), cloud.points[0].velocity)

    def test_detect_unfiltered(self) -> None:
        """Threshold detector should properly detect points with filtering disabled"""

        self.detector.peak_detection = False

        carrier_frequency = 72e9
        range_bins = np.arange(101)
        doppler_bins = np.arange(15) - 7
        angle_bins = np.array([[-1, 0.0], [0.0, 0.0], [1, 0.0]])

        data = np.zeros((3, 15, 101))
        data[1, 2, 3] = 0.5

        cube = RadarCube(data, angle_bins, doppler_bins, range_bins, carrier_frequency)
        cloud = self.detector.detect(cube)

        self.assertEqual(1, cloud.num_points)
        self.assertAlmostEqual(0.5, cloud.points[0].power)
        assert_array_almost_equal(np.array([0.0, 0.0, range_bins[3]]), cloud.points[0].position)
        assert_array_almost_equal(np.array([0, 0, doppler_bins[2] * speed_of_light / carrier_frequency]), cloud.points[0].velocity)

    def test_serialization(self) -> None:
        """Test threshold detector serialization"""
        
        test_roundtrip_serialization(self, self.detector)


class TestMaxDetector(TestCase):
    "Test the max detector class"

    def setUp(self) -> None:
        self.detector = MaxDetector()

    def test_detect(self) -> None:
        """Max detector should properly detect points"""

        carrier_frequency = 72e9
        range_bins = np.arange(101)
        doppler_bins = np.arange(15) - 7
        angle_bins = np.array([[-1, 0.0], [0.0, 0.0], [1, 0.0]])

        data = np.zeros((3, 15, 101))

        data[1, 2, 3] = 0.5
        data[2, 3, 4] = 0.1

        cube = RadarCube(data, angle_bins, doppler_bins, range_bins, carrier_frequency)
        cloud = self.detector.detect(cube)

        self.assertEqual(1, cloud.num_points)
        self.assertAlmostEqual(0.5, cloud.points[0].power)
        assert_array_almost_equal(np.array([0.0, 0.0, range_bins[3]]), cloud.points[0].position)
        assert_array_almost_equal(np.array([0, 0, doppler_bins[2] * speed_of_light / carrier_frequency]), cloud.points[0].velocity)

    def test_empty_detect(self) -> None:
        """Max detector should return an empty point cloud if no points are detected"""

        range_bins = np.arange(101)
        velocity_bins = np.arange(15) - 7
        angle_bins = np.array([[-1, 0.0], [0.0, 0.0], [1, 0.0]])

        data = np.zeros((3, 15, 101))

        cube = RadarCube(data, angle_bins, velocity_bins, range_bins)
        cloud = self.detector.detect(cube)

        self.assertEqual(0, cloud.num_points)

    def test_serialization(self) -> None:
        """Test max detector serialization"""
        
        test_roundtrip_serialization(self, self.detector)


class TestCFARDetector(TestCase):
    """Test the CFAR detector class"""

    def setUp(self) -> None:
        self.num_training_cells = (4, 10)
        self.num_guard_cells = (2, 4)
        self.pfa = 0.5

        self.detector = CFARDetector(self.num_training_cells, self.num_guard_cells, self.pfa)

    def test_properties(self) -> None:
        """Properties should be properly stored as class attributes"""

        self.assertEqual(self.num_training_cells, self.detector.num_training_cells)
        self.assertEqual(self.num_guard_cells, self.detector.num_guard_cells)

    def test_num_training_cells_validation(self) -> None:
        """num_training_cells setter should raise ValueErrors on invalid arguments"""

        # check wrong tuple size
        with self.assertRaises(ValueError):
            self.detector.num_training_cells = 1
        with self.assertRaises(ValueError):
            self.detector.num_training_cells = (1, 2, 3)

        # check non-positive and odd values
        with self.assertRaises(ValueError):
            self.detector.num_training_cells = (-1, 2)
        with self.assertRaises(ValueError):
            self.detector.num_training_cells = (0, 2)

    def test_num_training_cells_setget(self) -> None:
        """num_training_cells getter should return setter argument"""

        expected_num_training_cells = self.num_training_cells
        self.detector.num_training_cells = expected_num_training_cells

        self.assertEqual(expected_num_training_cells, self.detector.num_training_cells)

    def test_num_guard_cells_validation(self) -> None:
        """num_guard_cells setter should raise ValueErrors on invalid arguments"""

        # check wrong tuple size
        with self.assertRaises(ValueError):
            self.detector.num_guard_cells = 1
        with self.assertRaises(ValueError):
            self.detector.num_guard_cells = (1, 2, 3)

        # check non-positive and odd values
        with self.assertRaises(ValueError):
            self.detector.num_guard_cells = (-1, 2)

    def test_num_guard_cells_setget(self) -> None:
        """num_guard_cells getter should return setter argument"""

        expected_num_guard_cells = self.num_guard_cells
        self.detector.num_guard_cells = expected_num_guard_cells

        self.assertEqual(expected_num_guard_cells, self.detector.num_guard_cells)

    def test_pfa_validation(self) -> None:
        """pfa setter should raise ValueError on any value outside the [0, 1] interval"""

        with self.assertRaises(ValueError):
            self.detector.pfa = -1
        with self.assertRaises(ValueError):
            self.detector.pfa = 1.2

    def test_pfa_setget(self) -> None:
        """pfa getter should return the setter argument"""

        expected_pfa = self.pfa
        self.detector.pfa = expected_pfa

        self.assertEqual(expected_pfa, self.detector.pfa)

    def test_detect_validation(self) -> None:
        """Check if an exception is rosen on a too large window"""

        num_velocity_bins = self.num_guard_cells[0] + self.num_training_cells[0] - 1
        num_range_bins = self.num_guard_cells[1] + self.num_training_cells[1] - 1
        data = np.zeros((3, num_velocity_bins, num_range_bins))
        range_bins = np.arange(num_range_bins)
        velocity_bins = np.arange(num_velocity_bins) - num_velocity_bins // 2
        angle_bins = np.array([[-1, 0.0], [0.0, 0.0], [1, 0.0]])
        cube = RadarCube(data, angle_bins, velocity_bins, range_bins, carrier_frequency=72e9)
        with self.assertRaises(ValueError):
            self.detector.detect(cube)

    def test_detect(self) -> None:
        """CFAR detector should properly detect"""

        # set up data dimensions
        num_range_bins = 505
        num_velocity_bins = 100
        range_bins = np.arange(num_range_bins)
        velocity_bins = np.arange(num_velocity_bins) - num_velocity_bins // 2
        angle_bins = np.array([[-1, 0.0], [0.0, 0.0], [1, 0.0]])

        # generate data
        data = np.zeros((3, num_velocity_bins, num_range_bins))
        # generate 10 random points
        np.random.seed(42)
        points = np.random.randint(0, num_velocity_bins * num_range_bins, 10)
        for p in points:
            data[1, p // num_range_bins, p % num_range_bins] = 1
        # blur the points with a gaussian filter
        data[1] = gaussian_filter(data[1], 1.1)
        # add white noise
        data[1] += np.random.random(data[1].shape) / 80.0
        data[1] = np.clip(data[1], 0, 1)

        cube = RadarCube(data, angle_bins, velocity_bins, range_bins, carrier_frequency=72e9)
        cloud = self.detector.detect(cube)

        # all the generated points must be detected
        azimuth, zenith = cube.angle_bins[1]
        for p in points:
            px = p // num_range_bins
            py = p % num_range_bins
            velocity = cube.velocity_bins[px]
            range = cube.range_bins[py]
            power_indicator = cube.data[1, px, py]
            point_detection_expected = PointDetection.FromSpherical(zenith, azimuth, range, velocity, power_indicator)
            assert point_detection_expected in cloud.points

    def test_empty_detect(self) -> None:
        """CFAR detector should return an empty point cloud if no points are detected"""

        range_bins = np.arange(101)
        velocity_bins = np.arange(15) - 7
        angle_bins = np.array([[-1, 0.0], [0.0, 0.0], [1, 0.0]])

        data = np.zeros((3, 15, 101))

        cube = RadarCube(data, angle_bins, velocity_bins, range_bins, carrier_frequency=72e9)
        cloud = self.detector.detect(cube)

        self.assertEqual(0, cloud.num_points)

    def test_serialization(self) -> None:
        """Test CFAR detector serialization"""
        
        test_roundtrip_serialization(self, self.detector)
