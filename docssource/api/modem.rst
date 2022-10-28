====================
Communication Module
====================

This module provides functionalities to transmit information in form of bits over a wireless link.

.. autoclasstree:: hermespy.modem
   :strict:
   :namespace: hermespy

It consists of the base configuration classes for communication modems

.. toctree::

   modem.modem
   modem.bits_source
   modem.symbols
   modem.waveform_generator
   modem.evaluators

as well as multiple communication waveform implementations

.. toctree::

   modem.waveform_single_carrier
   modem.waveform_generator_chirp_fsk
   modem.waveform_generator_ofdm
   modem.waveform_correlation_synchronization
