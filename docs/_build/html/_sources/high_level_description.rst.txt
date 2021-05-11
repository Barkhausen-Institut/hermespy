High Level Description
======================

In general, we strongly advise to watch our `introductory video <https://www.barkhauseninstitut.org/opensource/hermespy>`_ at first.
In the following, we will describe the software architecture based on the :ref:`block-diagram`

=====
Modem
=====

Each transmitter modem is associated with one **Bit Source**, which is implemented in :py:class:`source.bits_source.BitsSource`.
The purpose of the **Bit Source** is to create bits that fit into one drop (this term will be explained later on).

Each transmitter modem proceeds in the following way:

1. **Encoder**: Bits are encoded by the **Encoder**. All encoders can be found in the module :py:mod:`modem.coding`
2. **Modulator**: A signal is created which is then modulated. This happens in the module :py:mod:`modem.digital_modem`
3. **RfChain**: Rf Impairments are added to the signal afterwards.
4. The signal is sent.

Within HermesPy, the class :py:class:`modem.modem.Modem` has instances of:

* a Bit Source
* an encoder
* a :py:class:`modem.digital_modem.DigitalModem` which serves as a base class for the different modulation schemes
* the RfChain (which is not implemented yet).

Hence, the white and the blue box, i.e. *Bit Source 1/2* and *Tx Modem 1/2* are represented by the class
:py:class:`modem.modem.Modem` internally. The block *Modulator* in the block diagram is represented
by the classes deriving from :py:class:`modem.digital_modem.DigitalModem`.

=======
Channel
=======

After the signal was sent from the :py:class:`modem.modem.Modem`, it is perturbed by a channel which is can be found in the :py:mod:`channel` module.
Each :py:class:`modem.modem.Modem` is associated to a channel. Therefore, for each transmitter-receiver-modem-pair exists one channel.
The channels are treated independently (exception: quadriga). After propagation, :py:class:`channel.noise.Noise` is added given
the SNR values defined in the settings files prior to the simulation. 

The **Receiver Modems** are of type :py:class:`modem.modem.Modem` as well. The distinction between receiving and
transmitting modems is only made within the class itself. 

========
Quadriga
========

HermesPy supports the `Quadriga <https://quadriga-channel-model.de/>`_ channel model. Although publicly available,
the source code can be found in **3rdparty/** of our repository. Quadriga is run in either Matlab or Octave,
depending on the parameter setting in the **_settings** directory.

As Quadriga treats the channels between all transceiver pairs as a common channel, which is in contrast
to HermesPy treating all channels independently, the class :py:class:`channel.quadriga_interface.QuadrigaInterface` 
had to be implemented performing the mapping.

==========
Statistics
==========

After the bits are decoded, measurement metrics (e.g. BER, BLER, PSD...) are calculated, based on
e.g. the :py:class:`source.bits_source.BitsSource` that is part of each :py:class:`modem.modem.Modem`. Results are stored in an instance
of the :py:class:`simulator_core.statistics.Statistics` class. 

The simulation itself has stopping criteria defined by confidence intervals in the settings files.
Each "simulation run" is called a **drop**. One drop has a certain time length which can be a (non-integer)
multitude of one frame.


.. _block-diagram:
.. figure:: images/block_diagram_hermespy.svg
   :scale: 50 %

   Block Diagram of HermesPy.
   
   The figure depicts the system setup of HermesPy. Channels are treated independently (exception: Quadriga).

