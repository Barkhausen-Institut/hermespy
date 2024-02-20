=====
OFDM
=====

Orthogonal Frequency Devision Multiplexing (OFDM) is a method of encoding
digital data on multiple carrier frequencies.
Within HermesPy, OFDM can be modeled by a flexible propgramming interface allowing for the definition
of custom time-freqeuency grids.

Considering a simplex-link scenario of two modems communicating over a 3GPP 5G TDL channel

.. literalinclude:: ../scripts/examples/modem_waveforms_ofdm.py
   :language: python
   :linenos:
   :lines: 10-21

configuring an OFDM waveform requires the specification of the resource-time
grid onto which the transmitted data and pilot symbols are placed:

.. literalinclude:: ../scripts/examples/modem_waveforms_ofdm.py
   :language: python
   :linenos:
   :lines: 23-37

The grid considers :math:`128` orthogonal subcarriers each modulated with a unique symbol,
with `128` repetitions in time-domain, so that overall :math:`16384` symbols are transmitted per frame.
The grid alternates between two types of symbol sections, one carrying a reference element on every :math:`8`-th subcarrier
and one consisting only of data symbols.

Additionally, post-processing routines for channel estimation and channel equalization 
may be specified on the waveform level

.. literalinclude:: ../scripts/examples/modem_waveforms_ofdm.py
   :language: python
   :linenos:
   :lines: 39-45

.. toctree::
   :glob:
   :maxdepth: 2

   modem.waveforms.orthogonal.ofdm.*

.. footbibliography::
