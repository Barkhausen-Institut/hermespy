=====================
Radio-Frequency Chain
=====================

.. inheritance-diagram:: hermespy.simulation.rf_chain.rf_chain.RfChain
   :parts: 1

The RF chain model is used to simulate the impairments of the RF front-end of a
wireless device.
During simulation runtime, the RF chain configuration is applied to the signal
before transmission over the air and after reception before further processing. 
More specifically, during transmission, the base-band samples to be transmitted
are initially converted to the analog domain by a digital-to-analog converter,
followed by modeling mixer impairments (I/Q imbalance and phase noise), and
finally by a power amplifier:

.. mermaid::
   :align: center

   graph LR
   dac[DAC] --> iq[I/Q] --> pn[PN] --> pa[PA]

During reception, the received signal is first amplified by a low-noise amplifier,
followed by mixer impairments (I/Q imbalance and phase noise), and finally converted
to the digital domain by an analog-to-digital converter:

.. mermaid::
   :align: center

   graph LR
   lna[LNA] --> pn[PN] --> iq[I/Q] --> adc[ADC]

.. autoclass:: hermespy.simulation.rf_chain.rf_chain.RfChain

Isolation model (to be implemented): :footcite:t:`2018:kiayni`

.. footbibliography::