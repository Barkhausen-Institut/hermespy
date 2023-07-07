# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from hermespy.core.transformation import Direction, Transformation, TransformableBase, TransformableLink, Transformable
from .test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestDirection(TestCase):
    
    def test_spherical(self) -> None:
        """Test initialization and transformation from spherical coordinates."""
        
        angles = np.pi * np.array([[0., 0.],
                                   [0., 1.],
                                   [0., .5],
                                   [.5, .5],
                                   [1., .5],
                                   [-.5, .5]])
        
        expected_directions = np.array([[0, 0, 1],
                                        [0, 0, -1],
                                        [1, 0, 0],
                                        [0, 1, 0],
                                        [-1, 0, 0],
                                        [0, -1, 0]])
        
        transformed_angles = np.empty(angles.shape, dtype=float)
        transformed_directions = np.empty(expected_directions.shape, dtype=float)
        
        for a, angle in enumerate(angles):
            
            direction = Direction.From_Spherical(*angle)

            transformed_directions[a, :] = direction
            transformed_angles[a, :] = direction.to_spherical()
            
        assert_array_almost_equal(expected_directions, transformed_directions)
        assert_array_almost_equal(angles, transformed_angles)
        
    def test_cartesian(self) -> None:
        """Test initialization from cartesian vectors"""
        
        vector = np.array([1, 2, 3])
        expected_unit_vector = vector / np.linalg.norm(vector)
        
        unnormalized_direction = Direction.From_Cartesian(vector, normalize=False)
        normalized_direction = Direction.From_Cartesian(vector, normalize=True)
        
        assert_array_almost_equal(vector, unnormalized_direction)
        assert_array_almost_equal(expected_unit_vector, normalized_direction)

    def test_cartesian_validation(self) -> None:
        """Initializaion from cartesian vectors should raise a ValueError for invalid arguments"""

        with self.assertRaises(ValueError):
            _ = Direction.From_Cartesian(np.arange(5))


class TestTransformation(TestCase):

    def test_no_init(self) -> None:
        """No transformation initialization should yield a correct transformation matrix"""

        transformation = Transformation.No()

        assert_array_equal(np.zeros(3), transformation.rotation_rpy)
        assert_array_equal(np.zeros(3), transformation.translation)

    def test_from_rpy_init(self) -> None:
        """Rpy transformation initialization should yield a correct transformation matrix"""

        expected_rpy = np.array([.123, .5555, .345])
        expected_translation = np.array([4., 5., 6.])

        transformation = Transformation.From_RPY(expected_rpy, expected_translation)

        assert_array_almost_equal(expected_rpy, transformation.rotation_rpy)
        assert_array_almost_equal(expected_translation, transformation.translation)
        
    def test_from_translation_init(self) -> None:
        """Translation transformation initialization should yield a correct transformation matrix"""

        expected_rpy = np.zeros(3)
        expected_translation = np.arange(3)
        
        transformation = Transformation.From_Translation(expected_translation)
        
        assert_array_almost_equal(expected_rpy, transformation.rotation_rpy)
        assert_array_almost_equal(expected_translation, transformation.translation)
        
    def test_from_translation_validation(self) -> None:
        """Translation transformation initialization should raise ValueErrors on invalid arguments"""
        
        with self.assertRaises(ValueError):
            _ = Transformation.From_Translation(np.arange(5))
        
    def test_translation_setget(self) -> None:
        """Translation property getter should return setter argument"""
        
        transformation = Transformation.No()
        expected_translation = np.arange(3)
        transformation.translation = expected_translation
        
        assert_array_almost_equal(expected_translation, transformation.translation)
        
    def test_translation_validation(self) -> None:
        """Translation property setter should raise ValueErrors on invalid arguments"""
        
        transformation = Transformation.No()
        
        with self.assertRaises(ValueError):
            transformation.translation = np.arange(5)
        
    def test_rotation_setget(self) -> None:
        """Rotation property getter should return setter argument"""
        
        transformation = Transformation.No()
        expected_rotation = np.arange(3)
        transformation.rotation_rpy = expected_rotation
        
        assert_array_almost_equal(expected_rotation, transformation.rotation_rpy)

    def test_rotation_singularity_handling(self) -> None:
        """Rotation property getter should return setter argument for the singularity"""

        transformation = Transformation.No()
        expected_rotation = np.array([0, np.pi / 2, 0])
        transformation.rotation_rpy = expected_rotation
        transformation[0, 0] = 0.

        assert_array_almost_equal(expected_rotation, transformation.rotation_rpy)

    def test_transform_position(self) -> None:
        """Test transformation of a cartesian position vector"""
        
        transformation = Transformation.No()
        position = np.array([1, 2])
        
        transformed_position = transformation.transform_position(position)
        assert_array_almost_equal(position, transformed_position)
        
        scale = 1.23456
        position = scale * np.array([1, 0, 0], dtype=float)
        offset = np.array([4, 5, 6], dtype=float)
        rotation_candidates = np.pi * np.array([[0, 0, 1],
                                                [0, 1, 0],
                                                [1, 0, 0]])
        expected_positions = offset + scale * np.array([[-1, 0, 0],
                                                        [-1, 0, 0],
                                                        [1, 0, 0]], dtype=float)
        transformed_positions = np.empty(expected_positions.shape, dtype=float)
        for p, rotation in enumerate(rotation_candidates):
        
            transformation = Transformation.From_RPY(rotation, offset)
            transformed_positions[p, :] = transformation.transform_position(position)
            
        assert_array_almost_equal(expected_positions, transformed_positions)

    def test_transform_position_valdiation(self) -> None:
        """Transforming a position should raise a ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            _ = Transformation.No().transform_position(np.arange(5))

    def test_rotate_direction(self) -> None:
        """Test rotation of a directional unit vector"""
        
        direction = np.array([1, 0, 0], dtype=float)
        
        rotation_candidates = np.pi * np.array([[0, 0, 1],
                                                [0, 1, 0],
                                                [1, 0, 0]])
        expected_directions = np.array([[-1, 0, 0],
                                        [-1, 0, 0],
                                        [1, 0, 0]], dtype=float)
        
        rotated_directions = np.empty(expected_directions.shape, dtype=float)
        for d, rotation in enumerate(rotation_candidates):
            
            transformation = Transformation.From_RPY(rotation, np.zeros(3))
            rotated_directions[d, :] = transformation.rotate_direction(direction)

        assert_array_almost_equal(expected_directions, rotated_directions)
                   
    def test_rotat_direction_validation(self) -> None:
        """Rotating a directional unit vector should raise a ValuError on invalid arguments""" 
        
        with self.assertRaises(ValueError):
            _ = Transformation.No().rotate_direction(np.arange(5))
    
    def test_transform_direction(self) -> None:
        """Test transformation of directions"""
        
        direction = np.array([1, 0, 0], dtype=float)
        
        rotation_candidates = np.pi * np.array([[0, 0, 1],
                                                [0, 1, 0],
                                                [1, 0, 0]])
        expected_directions = np.array([[-1, 0, 0],
                                        [-1, 0, 0],
                                        [1, 0, 0]], dtype=float)
        
        rotated_directions = np.empty(expected_directions.shape, dtype=float)
        for d, rotation in enumerate(rotation_candidates):
            
            transformation = Transformation.From_RPY(rotation, np.zeros(3))
            rotated_directions[d, :] = transformation.transform_direction(direction)

        assert_array_almost_equal(expected_directions, rotated_directions)
    
    def test_serialization(self) -> None:
        """Test YAML serialization"""
        
        test_yaml_roundtrip_serialization(self, Transformation.No())
    

class TestTransformableLink(TestCase):
    
    def setUp(self) -> None:
        
        self.link = TransformableBase()
        
    def test_remove_link_validation(self) -> None:
        """Removing a non-existing link should raise a Runtimer error"""
        
        with self.assertRaises(RuntimeError):
            self.link.remove_link(Transformable())


class TestTransformableBase(TestCase):
    
    def setUp(self) -> None:
        
        self.transformable = TransformableBase()
        
    def test_forwards_transformation(self) -> None:
        """Forwards transformation property should return the correct transormation"""
        
        expected_transformation = Transformation.No()
        assert_array_equal(expected_transformation, self.transformable.forwards_transformation)
        
    def test_base_updated(self) -> None:
        """Calling the base update routine should raise a RuntimeError"""
        
        with self.assertRaises(RuntimeError):
            self.transformable._kinematics_updated()
            
    def test_set_base(self) -> None:
        """Calling the set base routine should raise RuntimeError on any argument not None"""
        
        with self.assertRaises(RuntimeError):
            self.transformable.set_base(Transformable())


class TestTransformable(TestCase):
    
    def setUp(self) -> None:

        self.transformable = Transformable()
        
        self.base = TransformableBase()
        self.base.add_link(self.transformable)
        
    def test_position_setget(self) -> None:
        """Position property getter should return setter argument"""

        position = np.arange(3)
        self.transformable.position = position

        assert_array_equal(position, self.transformable.position)
        assert_array_almost_equal(position, self.transformable.forwards_transformation.translation)

    def test_position_validation(self) -> None:
        """Position property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.transformable.position = np.arange(4)

        with self.assertRaises(ValueError):
            self.transformable.position = np.array([[1, 2, 3]])

        try:
            self.transformable.position = np.arange(1)

        except ValueError:
            self.fail()
            
    def test_position_expansion(self) -> None:
        """Position property setter should expand vector dimensions if required."""

        position = np.array([1.0])
        expected_position = np.array([1.0, 0.0, 0.0])
        self.transformable.position = position

        assert_array_almost_equal(expected_position, self.transformable.position)
        assert_array_almost_equal(expected_position, self.transformable.global_position)
            
    def test_orientation_setget(self) -> None:
        """Orientation property getter should return setter argument"""
        
        expected_orientation = np.array([.1, .2, .3], dtype=float)
        
        self.transformable.orientation = expected_orientation
        
        assert_array_almost_equal(expected_orientation, self.transformable.orientation)
        assert_array_almost_equal(expected_orientation, self.transformable.global_orientation)
        
    def test_orientation_validation(self) -> None:
        """Orientation property setter should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.transformable.orientation = np.zeros((2, 3))

    def test_set_base(self) -> None:
        """Updating the base reference frame should correctly update the kinematics"""
        
        self.transformable.set_base(self.base)
        self.assertFalse(self.transformable.is_base)
        
        expected_position_offset = np.array([1, 1, 1], dtype=float)
        expected_base = Transformable(Transformation.From_Translation(expected_position_offset))
        
        self.base.add_link(expected_base)
        self.transformable.set_base(expected_base)
        
        self.assertFalse(self.transformable.is_base)
        self.assertCountEqual([expected_base], self.base.linked_frames)
        self.assertCountEqual([self.transformable], expected_base.linked_frames)
        
        assert_array_almost_equal(expected_position_offset, self.transformable.forwards_transformation.translation)
        
    def test_transformable_to_local_coordinates(self) -> None:
        """Test transformation of transformable to local coordinates"""

        transformable = Transformable(Transformation.From_RPY(pos=np.array([3, 4, 5]),
                                                              rpy=np.array([.2, .3, .4])))
        
        local_transformation = self.transformable.to_local_coordinates(transformable)
        assert_array_almost_equal(transformable.position, local_transformation.translation)

        self.transformable.position = np.array([3, 4, 5])
        self.transformable.orientation = np.array([.2, .3, .4])
        
        local_transformation = self.transformable.to_local_coordinates(transformable)
        assert_array_almost_equal(np.zeros(3), local_transformation.translation)
        assert_array_almost_equal(np.zeros(3), local_transformation.rotation_rpy)

    def test_transformation_to_local_coordiantes(self) -> None:
        """Test transformation of transformation to local coordinates"""
        
        transformable = Transformation.From_RPY(pos=np.array([3, 4, 5]),
                                                rpy=np.array([.2, .3, .4]))
        
        local_transformation = self.transformable.to_local_coordinates(transformable)
        assert_array_almost_equal(transformable.translation, local_transformation.translation)

        self.transformable.position = np.array([3, 4, 5])
        self.transformable.orientation = np.array([.2, .3, .4])
        
        local_transformation = self.transformable.to_local_coordinates(transformable)
        assert_array_almost_equal(np.zeros(3), local_transformation.translation)
        assert_array_almost_equal(np.zeros(3), local_transformation.rotation_rpy)
        
    def test_parameters_to_local_coordiantes(self) -> None:
        """Test transformation of parametric transformation to local coordinates"""
        
        pos = np.array([3, 4, 5])
        rpy = np.array([.2, .3, .4])
        
        local_transformation = self.transformable.to_local_coordinates(pos, rpy)
        assert_array_almost_equal(pos, local_transformation.translation)

        self.transformable.position = pos
        self.transformable.orientation = rpy
        
        local_transformation = self.transformable.to_local_coordinates(pos, rpy)
        assert_array_almost_equal(np.zeros(3), local_transformation.translation)
        assert_array_almost_equal(np.zeros(3), local_transformation.rotation_rpy)

    def test_to_local_coordiantes_validation(self) -> None:
        """Local coordinate transformation should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.transformable.to_local_coordinates(12345)

    def test_base_updates(self) -> None:
        """Updating frames further up the chain should update the kinematics"""
        
        second_link = Transformable(Transformation.From_Translation(np.array([-1, -2, -3])))
        self.base.add_link(second_link)
        self.transformable.set_base(second_link)
        
        assert_array_almost_equal(second_link.position, self.transformable.forwards_transformation.translation)
    
        second_link.position = np.array([-2, -3, -4])
        assert_array_almost_equal(second_link.position, self.transformable.forwards_transformation.translation)
