==========
Animation
==========

.. inheritance-diagram:: hermespy.simulation.animation.Trajectory hermespy.simulation.animation.StaticTrajectory hermespy.simulation.animation.LinearTrajectory
   :parts: 1

Movements of devices and other objects within simulations may be described by :class:`Trajectories<hermespy.simulation.animation.Trajectory>`,
which model the veloctiy, position and orientation of the object as a function of time.
Generally, objects that may be animated within a simulation should inherit from the :class:`Moveable<hermespy.simulation.animation.Moveable>` class,
which exposes the :meth:`trajectory<hermespy.simulation.animation.Moveable.trajectory>` property.
The currently available trajectory types are:

.. list-table:: Trajectory Types
    :header-rows: 1
    
    * - Type
      - Description
    * - :class:`StaticTrajectory<hermespy.simulation.animation.StaticTrajectory>`
      - A trajectory that does not change over time.
    * - :class:`LinearTrajectory<hermespy.simulation.animation.LinearTrajectory>`
      - A trajectory that moves in a straight line at a constant velocity.

The following example demonstrates how to assign a linear trajectory to a device
within a simulation and plot the scenario:

.. literalinclude:: ../scripts/examples/simulation_animation.py
    :language: python
    :linenos:
    :lines: 10-29

.. autoclass:: hermespy.simulation.animation.Trajectory

.. autoclass:: hermespy.simulation.animation.StaticTrajectory

.. autoclass:: hermespy.simulation.animation.LinearTrajectory

.. autoclass:: hermespy.simulation.animation.Moveable

.. autoclass:: hermespy.simulation.animation.TrajectorySample

.. footbibliography::
