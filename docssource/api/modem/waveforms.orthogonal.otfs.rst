====
OTFS
====

.. inheritance-diagram:: hermespy.modem.waveforms.orthogonal.otfs.OTFSWaveform
   :parts: 1

Orthogonal Time Frequency Space (OTFS) modulation is a modulation scheme that is designed to provide high spectral efficiency and low latency.
It is particularly well suited for high mobility scenarios, such as satellite and mobile communications.
Within HermesPy, OTFS is implemented as type of :class:`OrthogonalWaveform<hermespy.modem.waveforms.orthogonal.waveform.OrthogonalWaveform>`
and a precoding of :class:`OFDM<hermespy.modem.waveforms.orthogonal.ofdm.OFDMWaveform>`.

Considering a simplex-link scenario of two modems communicating over a 3GPP 5G TDL channel

.. literalinclude:: ../../scripts/examples/modem_waveforms_otfs.py
   :language: python
   :linenos:
   :lines: 10-21

configuring an OTFS waveform requires the specification of the resource-time
grid onto which the transmitted data and pilot symbols are placed:

.. literalinclude:: ../../scripts/examples/modem_waveforms_otfs.py
   :language: python
   :linenos:
   :lines: 23-37

The grid considers :math:`128` orthogonal subcarriers each modulated with a unique symbol,
with :math:`128` repetitions in time-domain, so that overall :math:`16384` symbols are transmitted per frame.
The grid alternates between two types of symbol sections, one carrying a reference element on every :math:`8`-th subcarrier
and one consisting only of data symbols.

Additionally, post-processing routines for channel estimation and channel equalization 
may be specified on the waveform level

.. literalinclude:: ../../scripts/examples/modem_waveforms_otfs.py
   :language: python
   :linenos:
   :lines: 39-45

.. autoclass:: hermespy.modem.waveforms.orthogonal.otfs.OTFSWaveform

.. footbibliography::
