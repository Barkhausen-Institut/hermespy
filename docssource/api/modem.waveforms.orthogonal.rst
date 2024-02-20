===========
Orthogonal
===========

Orthogonal waveforms are HermesPy's base class for all multicarrier waveforms such as OFDM, OCDM and OTFS
that encode multiple data streams into a time-resource grid.

Considering a simplex-link scenario of two modems communicating over a 3GPP 5G TDL channel

.. literalinclude:: ../scripts/examples/modem_waveforms_orthogonal.py
   :language: python
   :linenos:
   :lines: 10-21

configuring an orthogonal waveform requires the specification of the resource-time
grid onto which the transmitted data and pilot symbols are placed:

.. literalinclude:: ../scripts/examples/modem_waveforms_orthogonal.py
   :language: python
   :linenos:
   :lines: 23-36

The grid considers :math:`128` orthogonal subcarriers each modulated with a unique symbol,
with `128` repetitions in time-domain, so that overall :math:`16384` symbols are transmitted per frame.
The grid alternates between two types of symbol sections, one carrying a reference element on every :math:`8`-th subcarrier
and one consisting only of data symbols.

Additionally, post-processing routines for channel estimation and channel equalization 
may be specified on the waveform level

.. literalinclude:: ../scripts/examples/modem_waveforms_orthogonal.py
   :language: python
   :linenos:
   :lines: 38-44

Naturally, the abstract base class *OrthogonalWaveform* is not meant to be used directly,
and has to be replaced by one of the available implementations such as OFDM and OCDM.

.. toctree::
   :hidden:
   :maxdepth: 1

   modem.waveforms.orthogonal.ofdm
   modem.waveforms.orthogonal.ocdm
   modem.waveforms.orthogonal.OrthogonalWaveform
   modem.waveforms.orthogonal.GridElement
   modem.waveforms.orthogonal.GridResource
   modem.waveforms.orthogonal.GridSection
   modem.waveforms.orthogonal.GuardSection
   modem.waveforms.orthogonal.PilotSection
   modem.waveforms.orthogonal.SymbolSection
