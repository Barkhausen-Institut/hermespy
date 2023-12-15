===============
Single Carrier
===============

.. inheritance-diagram:: hermespy.modem.waveform_single_carrier.FilteredSingleCarrierWaveform hermespy.modem.waveform_single_carrier.RolledOffSingleCarrierWaveform
   :parts: 1
   :top-classes: hermespy.modem.waveform.CommunicationWaveform

Single carrier waveforms modulate only a single discrete carrier frequency.
They are typically applying a shaping filter around the modulated frequency,
of which the following are currently available:

.. toctree::
   :maxdepth: 1
   :glob:

   modem.waveform.single_carrier.*

.. autoclass:: hermespy.modem.waveform_single_carrier.FilteredSingleCarrierWaveform

.. autoclass:: hermespy.modem.waveform_single_carrier.RolledOffSingleCarrierWaveform

.. footbibliography::
