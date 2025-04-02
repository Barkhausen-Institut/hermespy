====================
Simulation Scenario
====================

.. inheritance-diagram:: hermespy.simulation.simulation.SimulationScenario
   :parts: 1

Simulation scenarios contain the complete description of a physical layer scenario to be simulated.
This includes primarily the set of physical devices and their linking channel models.
In the case of Monte Carlo simulations, they are managed by the :class:`Simulation<hermespy.simulation.simulation.Simulation>` class
and acessible through the :attr:`scenario<hermespy.core.pipeline.Pipeline.scenario>` property.

.. autoclass:: hermespy.simulation.simulation.SimulationScenario

.. footbibliography::