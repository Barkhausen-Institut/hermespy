==========================
A/D Conversion
==========================

Analog-to-digital conversion is the process of converting analog signals from a sensor such as
an antenna to a digitally sampled representation, and vice-versa.
The analog-to-digial-converter (ADC) is usually the last stage of a receive chain and the
digial-to-analog-converter (DAC) is the first stage of a transmit chain.

HermesPy's digital conversion modeling considers quantization noise stemming from the fact that digital
representations of analog signals are only an apprixmate representation in amplitude.
Additionally, ADC's require gain control in order to adjust the input signal to the ADC's dynamic range.

Configuring a :class:`SimulatedDevice's<hermespy.simulation.simulated_device.SimulatedDevice>` analog-digital conversion model
requires setting the :attr:`adc<hermespy.simulation.rf_chain.rf_chain.RfChain.adc>` property
of the device's :attr:`rf_chain<hermespy.simulation.simulated_device.SimulatedDevice.rf_chain>`:

.. literalinclude:: ../scripts/examples/simulation_adc.py
   :language: python
   :linenos:
   :lines: 6-12

The model can be further refined by specifying the quantization and gain control properties.

.. literalinclude:: ../scripts/examples/simulation_adc.py
   :language: python
   :linenos:
   :lines: 14-17

.. toctree::
   :glob:
   :hidden:

   simulation.rf_chain.adc.*

.. footbibliography::
