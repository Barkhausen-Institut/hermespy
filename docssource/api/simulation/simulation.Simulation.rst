=============
Simulation
=============

.. inheritance-diagram:: hermespy.simulation.simulation.Simulation
   :parts: 1

The :class:`Simulation<hermespy.simulation.simulation.Simulation>` class extends
HermesPy's core distributed :class:`MonteCarlo<hermespy.core.pymonte.monte_carlo.MonteCarlo>`
simulation framework to support parameterized physical layer simulations of wireless
communication and sensing scenarios.
It manages the configuration of the physical layer description represented by
:class:`SimulationScenarios<hermespy.simulation.simulation.SimulationScenario>`.
Once the user has configured the physical layer description,
a call to :meth:`run<hermespy.simulation.simulation.Simulation.run>` will launch a full
Monte Carlo simulation iterating over the configured parameter grid.

During simulation runtime, the follwing sequence of simulation stages is repeatedly executed:

#. Generate digital base-band signal models to be transmitted by the simulated devices
#. Simulate the hardware effects of devices during transmission
#. Realize all channel models
#. Propagate the device transmissions over the channel realizations
#. Simulate the hardware effects of devices during reception
#. Digitally process base-band signal models received by the simulated devices
#. Evaluate the configured performance metrics

.. autoclass:: hermespy.simulation.simulation.Simulation

.. footbibliography::
