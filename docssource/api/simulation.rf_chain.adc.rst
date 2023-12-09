=============================
Analog-to-Digital Conversion
=============================

Models the hardware imperfections of analog-to-digital conversion (ADC)
and digital-to-analog conversion (DAC).
Currently considers quantization and automatic gain control (AGC).

The following figure visualizes the quantizer responses.

.. plot:: scripts/plot_quantizer.py
   :align: center

.. toctree::
   :glob:
   :maxdepth: 2

   simulation.rf_chain.adc.*

.. footbibliography::
