# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

from hermespy.core.pymonte.grid import GridDimension, register, RegisteredDimension, SamplePoint, ScalarDimension
from hermespy.core import LogarithmicSequence, ValueType

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSamplePoint(TestCase):
    """Test the grid sample point class"""

    def test_init(self) -> None:
        """Initialization should properly store the arguments as object attributes"""

        value = 1.0
        title = "xyz"
        point = SamplePoint(value, title)

        self.assertEqual(value, point.value)
        self.assertEqual(title, point.title)

    def test_title(self) -> None:
        """Title property should return the correct string representation"""

        str_value = "abc"
        str_point = SamplePoint(str_value)
        self.assertEqual(str_value, str_point.title)

        float_value = 1.2
        float_point = SamplePoint(float_value)
        self.assertEqual("1.2", float_point.title)

        int_value = 1
        int_point = SamplePoint(int_value)
        self.assertEqual("1", int_point.title)

        object_value = object()
        object_point = SamplePoint(object_value)
        self.assertEqual(object_value.__class__.__name__, object_point.title)


class ScalarDimensionMock(ScalarDimension):
    """Mock class for testing scalar dimension base class"""

    value: float

    def __init__(self) -> None:

        self.value = 0.0
        ScalarDimension.__init__(self)

    def __lshift__(self, scalar: float) -> None:
        self.value = scalar

    @property
    def title(self) -> str:
        return "Mock"


class TestGridDimension(TestCase):
    """Test the simulation grid dimension class"""

    def setUp(self) -> None:
        class MockObject(object):
            def __init__(self) -> None:
                self.__dimension = 1234
                self.__scalar_dimension = ScalarDimensionMock()

            @register(first_impact="a", last_impact="b", title="testtitle")
            @property
            def dimension(self) -> int:
                return self.__dimension

            @dimension.setter
            def dimension(self, value: float) -> None:
                """Setter should raise a ValueError on invalid arguments"""

                self.__dimension = value

            @property
            def scalar_dimension(self) -> ScalarDimensionMock:
                return self.__scalar_dimension

            def set_dimension(self, value: float) -> None:
                self.dimension = value

        self.considered_object = MockObject()
        self.sample_points = [1, 2, 3, 4]

        self.dimension = GridDimension(self.considered_object, "dimension", self.sample_points)
        self.scalar_dimension = GridDimension(self.considered_object, "scalar_dimension", self.sample_points)

    def test_logarithmic_init(self) -> None:
        """Initialization with logarithmic sample points should configure the plot scale to logarithmic"""

        sample_points = LogarithmicSequence(self.sample_points)
        dimension = GridDimension(self.considered_object, "dimension", sample_points)

        self.assertEqual("log", dimension.plot_scale)

    def test_plot_scale_init(self) -> None:
        """Initialization with a plot scale should set the plot scale"""

        dimension = GridDimension(self.considered_object, "dimension", self.sample_points, plot_scale="log", tick_format=ValueType.DB)

        self.assertEqual("log", dimension.plot_scale)
        self.assertEqual(ValueType.DB, dimension.tick_format)

    def test_init_validation(self) -> None:
        """Initialization should ra ise ValueErrors on invalid arguments"""

        with self.assertRaises(ValueError):
            GridDimension(self.considered_object, "nonexistentdimension", self.sample_points)

        with self.assertRaises(ValueError):
            GridDimension(self.considered_object, "dimension", [])

    def test_registered_dimension_validation(self) -> None:
        """Initialization should raise ValueError if multiple registered dimesions don't share impacts"""

        class MockObjetB(object):
            def __init__(self) -> None:
                self.__dimension = 1234

            @register(first_impact="c", last_impact="b")
            @property
            def dimension(self) -> int:
                return self.__dimension

            @dimension.setter
            def dimension(self, value: float) -> None:
                """Setter should raise a ValueError on invalid arguments"""

                self.__dimension = value

            def set_dimension(self, value: float) -> None:
                self.dimension = value

        class MockObjetC(object):
            def __init__(self) -> None:
                self.__dimension = 1234

            @register(first_impact="a", last_impact="d")
            @property
            def dimension(self) -> int:
                return self.__dimension

            @dimension.setter
            def dimension(self, value: float) -> None:
                """Setter should raise a ValueError on invalid arguments"""

                self.__dimension = value

            def set_dimension(self, value: float) -> None:
                self.dimension = value

        mock_b = MockObjetB()
        mock_c = MockObjetC()

        with self.assertRaises(ValueError):
            _ = GridDimension((self.considered_object, mock_b), "dimension", self.sample_points)

        with self.assertRaises(ValueError):
            _ = GridDimension((self.considered_object, mock_c), "dimension", self.sample_points)

    def test_function_dimension(self) -> None:
        """Test pointing to a function instead of a property"""

        self.dimension = GridDimension(self.considered_object, "set_dimension", self.sample_points)

        self.dimension.configure_point(0)
        self.assertEqual(self.sample_points[0], self.considered_object.dimension)

    def test_considered_object(self) -> None:
        """Considered object property should return considered object"""

        self.assertIs(self.considered_object, self.dimension.considered_objects[0])

    def test_dimension(self) -> None:
        """Dimension propertsy should return the correct dimension"""

        self.assertEqual("dimension", self.dimension.dimension)

    def test_sample_points(self) -> None:
        """Sample points property should return sample points"""

        sample_point_values = [p.value for p in self.dimension.sample_points]
        self.assertSequenceEqual(self.sample_points, sample_point_values)

    def test_num_sample_points(self) -> None:
        """Number of sample points property should return the correct amount of sample points"""

        self.assertEqual(4, self.dimension.num_sample_points)

    def test_configure_point_validation(self) -> None:
        """Configuring a point with an invalid index should raise a ValueError"""

        with self.assertRaises(ValueError):
            self.dimension.configure_point(4)

    def test_configure_point(self) -> None:
        """Configuring a point should set the property correctly"""

        expected_value = self.sample_points[3]
        self.dimension.configure_point(3)

        self.assertEqual(expected_value, self.considered_object.dimension)

    def test_title(self) -> None:
        """Title property should infer the correct title"""

        self.dimension.title = None
        self.assertEqual("dimension", self.dimension.title)

        self.dimension.title = "xyz"
        self.assertEqual("xyz", self.dimension.title)

    def test_plot_scale_setget(self) -> None:
        """Plot scale property getter should return setter argument"""

        scale = "loglin"
        self.dimension.plot_scale = scale

        self.assertEqual(scale, self.dimension.plot_scale)


class TestRegisteredDimension(TestCase):
    """Test the registered dimension"""

    def setUp(self) -> None:
        self.property = property(lambda: 10)
        self.first_impact = "a"
        self.last_impact = "b"
        self.title = "c"

        self.dimension = RegisteredDimension(self.property, self.first_impact, self.last_impact, self.title)

    def test_is_registered(self) -> None:
        """Is registered should return the correct result"""

        self.assertTrue(RegisteredDimension.is_registered(self.dimension))

    def test_properties(self) -> None:
        """Properties should return the correct values"""

        self.assertEqual(self.first_impact, self.dimension.first_impact)
        self.assertEqual(self.last_impact, self.dimension.last_impact)
        self.assertEqual(self.title, self.dimension.title)

    def test_getter(self) -> None:
        """Getter should return the correct value"""

        self.assertIsInstance(self.dimension.getter(Mock()), RegisteredDimension)

    def test_setter(self) -> None:
        """Setter should return the correct value"""

        self.assertIsInstance(self.dimension.setter(Mock()), RegisteredDimension)

    def test_deleter(self) -> None:
        """Deleter should return the correct value"""

        self.assertIsInstance(self.dimension.deleter(Mock()), RegisteredDimension)

    def test_decoration(self) -> None:
        """The decorator should return a property registered within the simulation registry"""

        expected_first_impact_a = "123124"
        expected_last_impact_a = "21341312"

        expected_first_impact_b = "1231223234"
        expected_last_impact_b = "213413123232"

        class TestClassA:
            def __init__(self) -> None:
                self.__value_a = 1.2345

            @register(first_impact=expected_first_impact_a, last_impact=expected_last_impact_a)
            @property
            def test_dimension(self) -> float:
                return self.__value_a

            @test_dimension.setter
            def test_dimension(self, value: float) -> None:
                self.__value_a = value

        class TestClassB:
            def __init__(self) -> None:
                self.test_dimension = 6.789

            @register(first_impact=expected_first_impact_b, last_impact=expected_last_impact_b)
            @property
            def test_dimension(self) -> float:
                return self.__value_b

            @test_dimension.setter
            def test_dimension(self, value: float) -> None:
                self.__value_b = value

        self.assertTrue(RegisteredDimension.is_registered(TestClassA.test_dimension))
        self.assertEqual(expected_first_impact_a, TestClassA.test_dimension.first_impact)
        self.assertEqual(expected_last_impact_a, TestClassA.test_dimension.last_impact)

        self.assertTrue(RegisteredDimension.is_registered(TestClassB.test_dimension))
        self.assertEqual(expected_first_impact_b, TestClassB.test_dimension.first_impact)
        self.assertEqual(expected_last_impact_b, TestClassB.test_dimension.last_impact)

        expected_value_a = 1.2345
        expected_value_b = 6.7890

        class_a = TestClassA()
        class_b = TestClassB()
        class_a.test_dimension = expected_value_a
        class_b.test_dimension = expected_value_b

        self.assertEqual(expected_value_a, class_a.test_dimension)
        self.assertEqual(expected_value_b, class_b.test_dimension)
