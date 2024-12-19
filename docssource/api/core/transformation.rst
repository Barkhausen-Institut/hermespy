=================
Coordinate System
=================

.. inheritance-diagram:: hermespy.core.transformation.Transformation hermespy.core.transformation.Transformable hermespy.core.transformation.TransformableLink hermespy.core.transformation.TransformableBase hermespy.core.transformation.Direction
   :parts: 1

HermesPy's coordinate sytem implementation defines classes for forward and backwards kinematics along a kinematic chain of multiple linked coordinate systems.
They can be used to describe the position and orientation of objects in a 3D space depending on the position and orientation of other objects.
More precisely, antenna elements within antenna arrays can be described in relation to the antenna array itself, which in turn can be described in relation to the device it is attached to.
Devices can be described in relation to the global coordinate system, which is defined by the simulation environment and attached to moving geometric objects in the case of spatial simulations.

.. autoclass:: hermespy.core.transformation.Transformation

.. autoclass:: hermespy.core.transformation.Transformable

.. autoclass:: hermespy.core.transformation.TransformableLink

.. autoclass:: hermespy.core.transformation.TransformableBase

.. autoclass:: hermespy.core.transformation.Direction

.. footbibliography::
