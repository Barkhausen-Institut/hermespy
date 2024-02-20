==========
Waveforms
==========

Communication waveforms are the central configuration option of :class:`Modems<hermespy.modem.modem.BaseModem>`.
They describe the signal processing steps required to generate
base-band waveforms carrying information during transmission and, inversely,
the signal processing steps required to estimate information from base-band signals
during reception.

The following waveforms are currently supported:

.. include:: modem.waveform._table.rst

.. autoclass:: hermespy.modem.waveform.CommunicationWaveform

.. autoclass:: hermespy.modem.waveform.PilotCommunicationWaveform

.. autoclass:: hermespy.modem.waveform.ConfigurablePilotWaveform

.. autoclass:: hermespy.modem.waveform.WaveformType

.. toctree::
   :maxdepth: 1
   :hidden:

   modem.waveform.single_carrier
   modem.waveforms.orthogonal
   modem.waveform.chirp_fsk
   modem.waveform.Synchronization
   modem.waveform.ChannelEstimation
   modem.waveform.ChannelEqualization
