========
Antenna
========

.. inheritance-diagram:: hermespy.simulation.antennas.SimulatedAntenna
   :top-classes: hermespy.core.antennas.Antenna
   :parts: 1

Simulated antennas are the base class for all antenna models within the context
of simulations.
They model the polarimetric radiation pattern of antenna elements in addition to their
relative position and orientation within their respective array.
The currently implemented models are:

.. list-table::
   :header-rows: 1

   * - Antenna Model
     - Description

   * - :doc:`Dipole<simulation.antennas.SimulatedDipole>`
     - Dipole antenna as commonly used in radio, television and broadcasting

   * - :doc:`Ideal Isotropic<simulation.antennas.SimulatedIdealAntenna>`
     - Ideal isotropic antenna with a uniform radiation pattern

   * - :doc:`Linear<simulation.antennas.SimulatedLinearAntenna>`
     - Linear antenna with a cosine radiation pattern
     
   * - :doc:`Patch<simulation.antennas.SimulatedPatchAntenna>`
     - Patch antenna as commonly used in communication antenna arrays

.. autoclass:: hermespy.simulation.antennas.SimulatedAntenna

.. toctree::
   :hidden:
   :maxdepth: 1

   simulation.antennas.SimulatedDipole
   simulation.antennas.SimulatedIdealAntenna
   simulation.antennas.SimulatedLinearAntenna
   simulation.antennas.SimulatedPatchAntenna

.. footbibliography::
