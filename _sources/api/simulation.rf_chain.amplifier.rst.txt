=============
Amplification
=============

.. inheritance-diagram:: hermespy.simulation.rf_chain.power_amplifier.PowerAmplifier
   :parts: 1

Amplifcation modeling represents a single processing step within the radio-frequency hardware model of
simulated devices. Amplifier models implement an equation

.. math::

   s'(t) = f\lbrace s(t), t \rbrace \in \mathbb{C}

distorting complex-valued signal samples :math:`s(t) \in \mathbb{C}` feeding into the power amplifier.
For time-invariant (i.e. memoryless) models :math:`f\lbrace s(t), t \rbrace = f\lbrace s(t) \rbrace`.

The currently available amplifier models are

.. list-table::
   :header-rows: 1

   * - Amplification Model
     - Description

   * - :doc:`simulation.rf_chain.amplifier.ClippingPowerAmplifier`
     - Model of a power amplifier driven into saturation.

   * - :doc:`simulation.rf_chain.amplifier.CustomPowerAmplifier`
     - Power amplification model with fully customizable gain characteristics.

   * - :doc:`simulation.rf_chain.amplifier.PowerAmplifier`
     - Ideal power amplifier model. HermesPy's default configuration.

   * - :doc:`simulation.rf_chain.amplifier.RappPowerAmplifier`
     - Power amplification model following :footcite:t:`1991:rapp`.
    
   * - :doc:`simulation.rf_chain.amplifier.SalehPowerAmplifier`
     - Power amplification model following :footcite:t:`1981:saleh`.

Configuring a :class:`SimulatedDevice's<hermespy.simulation.simulated_device.SimulatedDevice>` amplification
requires setting the :attr:`phase_noise<hermespy.simulation.rf_chain.rf_chain.RfChain.power_amplifier>` property
of the device's :attr:`rf_chain<hermespy.simulation.simulated_device.SimulatedDevice.rf_chain>`:

.. literalinclude:: ../scripts/examples/simulation_amplifier.py
   :language: python
   :linenos:
   :lines: 5-10

Of course, the abstract *PowerAmplifier* model in the above snippet has to be replaced by one the implementations
listed above.
The following figure visualizes the gain characteristics for the implemented amplification models for a saturation point :math:`s_\mathrm{sat} = 1`.

.. plot:: scripts/plot_pa_characteristics.py
   :align: center

.. toctree::
   :hidden:

   simulation.rf_chain.amplifier.ClippingPowerAmplifier
   simulation.rf_chain.amplifier.CustomPowerAmplifier
   simulation.rf_chain.amplifier.PowerAmplifier
   simulation.rf_chain.amplifier.RappPowerAmplifier
   simulation.rf_chain.amplifier.SalehPowerAmplifier

.. footbibliography::
