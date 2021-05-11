Explanation of Platform Approach
================================

The December 2020 "Platform Release" is stated to be a platform release. But what does "platform" mean?

HermesPy is meant to be a joint development of the research community for an Open Source link-level simulator that can be used
with a licence-free software (here: python) to simulate flexible scenarios for the development of the PHY of future and current
mobile communication systems.

We understand the term **platform** in two ways:

* software architecture platform
* simulation platform

**Software architecture platform** means for us that you can easily implement new encoders, channel models, complete modems, etc.
Some flexibility is already provided by plenty of parameters that can be varied.

As an example, we have two abstract base classes, namely the 

- :py:class:`modem.digital_modem.DigitalModem` which serves as a waveform generator
- :py:class:`modem.coding.encoder.Encoder` as a base class for encoding

Including the BitsSource and RfChain (which has no implementation yet), instances of these four classes build a modem in the current release.

If one wants to, e.g. implement a new Encoder, one simply has to derive from the Encoder class.

**Simulation platform** means that you can change the simulation itself to your liking. That means you can define the
number of transmitter/receiver modems, channel characteristics, simulation length, etc. Each non-standard model
that is implemented can easily be changed in the **_settings/** directory.

=======
Roadmap
=======

Regarding the **software architecture platform** we plan to

* introduce a new class ReceiverModem as receiver modems can be very flexible and don't need to behave as receiver modems.
* Transmitter modems will be composed of the following:

   * Encoder
   * WaveFormGenerator
   * RfImpairments

* DigitalModem will be renamed to WaveFormGenerator
* ReceiverModem will not follow a general architecture due to its flexible nature. Instead, our emphasis will be on building blocks that will be well-documented based on a template provided by us.

Nonetheless, the software architecture itself will not change radically. As we are still in discussions on
how to change the architecture, UML diagrams will be provided as soon as decisions are taken.

