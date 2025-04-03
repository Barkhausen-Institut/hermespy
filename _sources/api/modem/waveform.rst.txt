==========
Waveforms
==========

Communication waveforms are the central configuration option of :class:`Modems<hermespy.modem.modem.BaseModem>`.
They describe the signal processing steps required to generate
base-band waveforms carrying information during transmission and, inversely,
the signal processing steps required to estimate information from base-band signals
during reception.

The following waveforms are currently supported:

.. include:: waveform._table.rst

.. autoclass:: hermespy.modem.waveform.CommunicationWaveform

.. autoclass:: hermespy.modem.waveform.PilotCommunicationWaveform

.. autoclass:: hermespy.modem.waveform.ConfigurablePilotWaveform

.. autoclass:: hermespy.modem.waveform.CWT

.. toctree::
   :maxdepth: 1
   :hidden:

   waveform.single_carrier
   waveforms.orthogonal
   waveform.chirp_fsk
   waveform.Synchronization
   waveform.ChannelEstimation
   waveform.ChannelEqualization
   waveform.PilotSymbolSequence
