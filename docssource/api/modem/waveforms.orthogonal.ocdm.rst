=====
OCDM
=====

.. inheritance-diagram:: hermespy.modem.waveforms.orthogonal.ocdm.OCDMWaveform
    :parts: 1
    :top-classes: hermespy.modem.waveform.CommunicationWaveform

Orthogonal Chirp Division Multiplexing (OCDM) is a method of encoding
digital data into multiple orthogonal chirps, i.e. waveforms with zero
cross-correlation.

Considering a simplex-link scenario of two modems communicating over a 3GPP 5G TDL channel

.. literalinclude:: ../../scripts/examples/modem_waveforms_ocdm.py
   :language: python
   :linenos:
   :lines: 10-21

configuring an OCDM waveform requires the specification of the resource-time
grid onto which the transmitted data and pilot symbols are placed:

.. literalinclude:: ../../scripts/examples/modem_waveforms_ocdm.py
   :language: python
   :linenos:
   :lines: 23-37

The grid considers :math:`128` orthogonal subcarriers each modulated with a unique symbol,
with :math:`128` repetitions in time-domain, so that overall :math:`16384` symbols are transmitted per frame.
The grid alternates between two types of symbol sections, one carrying a reference element on every :math:`8`-th subcarrier
and one consisting only of data symbols.

Additionally, post-processing routines for channel estimation and channel equalization 
may be specified on the waveform level

.. literalinclude:: ../../scripts/examples/modem_waveforms_ocdm.py
   :language: python
   :linenos:
   :lines: 39-45

.. autoclass:: hermespy.modem.waveforms.orthogonal.ocdm.OCDMWaveform

.. footbibliography::
