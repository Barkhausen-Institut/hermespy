==============
Antenna Arrays
==============

.. inheritance-diagram:: hermespy.simulation.antennas.SimulatedAntennaArray
   :top-classes: hermespy.core.antennas.AntennaArray
   :parts: 1

Simulated antenna arrays are the simulation module's extension of
the core :class:`AnteannaArray<hermespy.core.antennas.AntennaArray>`
decsription by :class:`RfChain<hermespy.simulation.rf.chain.RFChain>` models
and polarimetric antenna descriptions. 
They are used to configure the frontend description of
:class:`SimulatedDevices<hermespy.simulation.simulated_device.SimulatedDevice>`.
The current implementation supports two types of antenna arrays:

.. list-table::
   :header-rows: 1

   * - Array Model
     - Description

   * - :doc:`Custom<antennas.SimulatedCustomArray>`
     - Custom antenna array with aribtray antenna types, positions and orientations

   * - :doc:`Uniform<antennas.SimulatedUniformArray>`
     - Uniformly antenna array with equidistantly spaced antennas of the same type

They can be configured by assigning them to the :attr:`antennas<hermespy.simulation.simulated_device.SimulatedDevice.antennas>` attribute
of a :class:`SimulatedDevice<hermespy.simulation.simulated_device.SimulatedDevice>`:

.. literalinclude:: ../../scripts/examples/simulation_antennas_SimulatedAntennaArray.py
   :language: python
   :linenos:
   :lines: 12-14

The *SimulatedAntennaArray* in this example snippet has to be replaced by one of the
available implementations.
During simulation runtime, the aforementioned :attr:`antennas<hermespy.simulation.simulated_device.SimulatedDevice.antennas>`
property can be exploited to compare the performance of different antenna array models:

.. literalinclude:: ../../scripts/examples/simulation_antennas_SimulatedAntennaArray.py
   :language: python
   :linenos:
   :lines: 17-25
  
In this case, every configured performance evaluator wille be executed for both antenna array models,
collecting evaluation samples for each of them.

.. autoclass:: hermespy.simulation.antennas.SimulatedAntennaArray

.. toctree::
   :hidden:
   :maxdepth: 1

   antennas.SimulatedCustomArray
   antennas.SimulatedUniformArray

.. footbibliography::
