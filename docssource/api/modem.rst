=============
Communication
=============

This module provides functionalities to transmit information in form of bits over a wireless link.

.. autoclasstree:: hermespy.modem
   :strict:
   :namespace: hermespy

It consists of the base configuration classes for communication modems

.. toctree::

   modem.modem
   modem.modem.SimplexLink
   modem.bits_source
   modem.symbols
   modem.waveform
   modem.evaluators
   modem.evaluators.BitErrorEvaluator

as well as multiple communication waveform implementations

.. toctree::

   modem.waveform_single_carrier
   modem.waveform_single_carrier.RootRaisedCosine
   modem.waveform_chirp_fsk
   modem.waveform_ofdm
   modem.waveform_correlation_synchronization

Its precoding subpackage includes MIMO precoding algorithms
for communication symbol streams

.. toctree::

   modem.precoding.symbol_precoding
   modem.precoding.dft
   modem.precoding.single_carrier
   modem.precoding.spatial_multiplexing
   modem.precoding.ratio_combining
   modem.precoding.space_time_block_coding
