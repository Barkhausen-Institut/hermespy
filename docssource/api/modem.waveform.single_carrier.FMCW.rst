======
FMCW
======

.. inheritance-diagram:: hermespy.modem.waveform_single_carrier.FMCWWaveform
   :parts: 1
   :top-classes: hermespy.modem.waveform.CommunicationWaveform

The Frequency-Modulated Continuous Waveform (FMCW) is a single-carrier modulation scheme filtering the communication
symbols with rectangle pulse shape:

.. list-table:: Pulse Properties

   * - .. plot::

            import matplotlib.pyplot as plt
            from hermespy.modem import FMCWWaveform

            waveform = FMCWWaveform(oversampling_factor=128, bandwidth=100e6, symbol_rate=1e6, num_preamble_symbols=1, num_data_symbols=0)
            waveform.plot_filter()
            plt.show()

     - .. plot::

            import matplotlib.pyplot as plt
            from hermespy.modem import FMCWWaveform
            
            waveform = FMCWWaveform(oversampling_factor=128, bandwidth=100e6, symbol_rate=1e6, num_preamble_symbols=1, num_data_symbols=0)
            waveform.plot_filter_correlation()
            
            plt.show()

The waveform can be configured by specifying the number of number of data- and preamble symbols
contained within each frame, as well as the considered symbol rate:

.. literalinclude:: ../scripts/examples/modem_waveforms_fmcw.py
   :language: python
   :linenos:
   :lines: 17-25

Afterwards, additional processing steps such as synchronization, channel estimation, equalization,
and the pilot symbol sequence can be added to the waveform:

.. literalinclude:: ../scripts/examples/modem_waveforms_fmcw.py
   :language: python
   :linenos:
   :lines: 27-39

In order to generate and evaluate communication transmissions or receptions, waveforms should be added to :class:`modem<hermespy.modem.modem.BaseModem>` implementations.
Refer to :doc:`modem.modem.TransmittingModem`, :doc:`modem.modem.ReceivingModem` or :doc:`modem.modem.SimplexLink` for more information.
For instructions how to implement custom waveforms, refer to :doc:`/notebooks/waveform`.

.. autoclass:: hermespy.modem.waveform_single_carrier.FMCWWaveform

.. footbibliography::
