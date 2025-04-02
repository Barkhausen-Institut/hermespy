=============
Simplex Link
=============

.. inheritance-diagram:: hermespy.modem.modem.SimplexLink
   :parts: 1
   :top-classes: hermespy.core.device.Receiver, hermespy.core.device.Transmitter, hermespy.modem.modem.BaseModem

Simplex links represent the signal processing chain of a unidirectional communication betweeen
a transmitting device and a receiving device,
implementing the digital signal processing before digital-to-analog conversion and after analog-to-digital conversion, respectively.
They are a combination of a :doc:`modem.TransmittingModem` and a :doc:`modem.ReceivingModem`.


After a :class:`SimplexLink<hermespy.modem.modem.SimplexLink>` is added as a
type of :class:`Transmitter<hermespy.core.device.Transmitter>` to a :class:`Device<hermespy.core.device.Device>`,
a call to :meth:`Device.transmit<hermespy.core.device.Device.transmit>` will be delegated
to :meth:`TransmittingModem._transmit()<hermespy.modem.modem.TransmittingModemBase._transmit>`.
Similarly, after a :class:`SimplexLink<hermespy.modem.modem.SimplexLink>` is added as a
type of :class:`Receiver<hermespy.core.device.Receiver>` to a :class:`Device<hermespy.core.device.Device>`,
a call to :meth:`Device.receive<hermespy.core.device.Device.receive>` will be delegated
to :meth:`ReceivingModem._receive()<hermespy.modem.modem.ReceivingModemBase._receive>`

.. literalinclude:: ../../scripts/examples/modem_SimplexLink.py
   :language: python
   :linenos:
   :lines: 19-35

For a detailed description of the transmit and receive routines,
refer to the :doc:`modem.TransmittingModem` and a :doc:`modem.ReceivingModem` of
the base classes.

The barebone configuration can be extend by additional components such as
:class:`Custom Bit Sources<hermespy.modem.bits_source.BitsSource>`,
:class:`Synchronization<hermespy.modem.waveform.Synchronization>`,
:class:`TransmitSignalCoding<hermespy.core.precoding.TransmitSignalCoding>`,
:class:`ReceiveSignalCoding<hermespy.core.precoding.ReceiveSignalCoding>`,
:class:`Channel Estimation<hermespy.modem.waveform.ChannelEstimation>`,
:class:`TransmitSymbolCoding<hermespy.modem.precoding.symbol_precoding.TransmitSymbolCoding>`,
:class:`ReceiveSymbolCoding<hermespy.modem.precoding.symbol_precoding.ReceiveSymbolCoding>`,
:class:`Channel Equalization<hermespy.modem.waveform.ChannelEqualization>` and
:class:`Bit Encoders<hermespy.fec.coding.Encoder>`:

.. literalinclude:: ../../scripts/examples/modem_SimplexLink.py
   :language: python
   :linenos:
   :lines: 24-58

.. autoclass:: hermespy.modem.modem.SimplexLink

.. footbibliography::
