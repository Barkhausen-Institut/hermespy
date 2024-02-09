============
Phase Noise
============

.. inheritance-diagram:: hermespy.simulation.rf_chain.phase_noise.PhaseNoise
   :parts: 1

Phase noise refers to the effect random fluctuations in a non-ideal osciallators phase
have on a signal modulated by that osicallator.
The effect of phase noise is to spread the signal out in the frequency domain.
The currently implemented phase noise models are

.. list-table::
   :header-rows: 1

   * - Phase Noise Model
     - Description

   * - :doc:`simulation.rf_chain.phase_noise.NoPhaseNoise`
     - No phase noise is added to the signal.

   * - :doc:`simulation.rf_chain.phase_noise.OscillatorPhaseNoise`
     - Phase noise is added to the signal based on its freuency domain characteristics.

Configuring a :class:`SimulatedDevice's<hermespy.simulation.simulated_device.SimulatedDevice>` phase noise model
requires setting the :attr:`phase_noise<hermespy.simulation.rf_chain.rf_chain.RfChain.phase_noise>` property
of the device's :attr:`rf_chain<hermespy.simulation.simulated_device.SimulatedDevice.rf_chain>`:

.. literalinclude:: ../scripts/examples/simulation_phase_noise.py
   :language: python
   :linenos:
   :lines: 5-10

Of course, the abstract *PhaseNoise* model in the above snippet has to be replaced by one the implementations
listed above.

.. autoclass:: hermespy.simulation.rf_chain.phase_noise.PhaseNoise

.. toctree::
   :hidden:

   simulation.rf_chain.phase_noise.NoPhaseNoise
   simulation.rf_chain.phase_noise.OscillatorPhaseNoise

.. footbibliography::
