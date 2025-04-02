======
C-FSK
======

.. inheritance-diagram:: hermespy.modem.waveform_chirp_fsk.ChirpFSKWaveform hermespy.modem.waveform_chirp_fsk.ChirpFSKSynchronization hermespy.modem.waveform_chirp_fsk.ChirpFSKCorrelationSynchronization
   :parts: 1
   :top-classes: hermespy.modem.waveform.CommunicationWaveform, hermespy.modem.waveform.Synchronization

Chirp Frequency Shift Keying (C-FSK) is a modulation scheme that encodes 
information in the start and stop frequency of an FMCW ramp.

The waveform can be configured by specifying the number of number of pilot- and data chirps
contained within each frame, as well as the duration and bandwidth each of the chirps:

.. literalinclude:: ../../scripts/examples/modem_waveforms_cfsk.py
   :language: python
   :linenos:
   :lines: 13-19

Afterwards, additional processing steps such as synchronization can be added to the waveform description:

.. literalinclude:: ../../scripts/examples/modem_waveforms_cfsk.py
   :language: python
   :linenos:
   :lines: 21-22

In order to generate and evaluate communication transmissions or receptions, waveforms should be added to :class:`modem<hermespy.modem.modem.BaseModem>` implementations.
Refer to :doc:`modem.TransmittingModem`, :doc:`modem.ReceivingModem` or :doc:`modem.SimplexLink` for more information.
For instructions how to implement custom waveforms, refer to :doc:`/notebooks/waveform`.

.. autoclass:: hermespy.modem.waveform_chirp_fsk.ChirpFSKWaveform

.. autoclass:: hermespy.modem.waveform_chirp_fsk.ChirpFSKSynchronization

.. autoclass:: hermespy.modem.waveform_chirp_fsk.ChirpFSKCorrelationSynchronization

.. footbibliography::
