=============
Bit Sources
=============

.. inheritance-diagram:: hermespy.modem.bits_source.RandomBitsSource hermespy.modem.bits_source.StreamBitsSource hermespy.modem.bits_source.BitsSource
   :parts: 1


Bit sources represent, as the title suggest, a source of (hard) communication bits
to be transmitted over a modem.
They are one of the default configuration parameters of a :class:`TransmittingModem<hermespy.modem.modem.TransmittingModem>`.

Every bit source implementation is expected to inherit from the :class:`BitsSource<hermespy.modem.bits_source.BitsSource>` base class,
which in turn represents a random node.
There are currently two basic types of bit sources available:

* :class:`RandomBitsSource<hermespy.modem.bits_source.RandomBitsSource>` instances implement a random stream of bits
* :class:`StreamBitsSource<hermespy.modem.bits_source.StreamBitsSource>` instances implement a deterministic stream of bits

.. autoclass:: hermespy.modem.bits_source.BitsSource

.. autoclass:: hermespy.modem.bits_source.RandomBitsSource

.. autoclass:: hermespy.modem.bits_source.StreamBitsSource

.. footbibliography::
