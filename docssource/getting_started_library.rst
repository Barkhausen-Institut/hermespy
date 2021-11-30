========
Library
========

.. |br| raw:: html

     <br>

This section outlines how to include HermesPy into your own Python projects and provides
basic reference examples to get new users accustomed with the API.

In its core, the HermesPy API aims to abstract the process of wireless communication signal processing
in an object-oriented class structure.
Each processing step is represented by a dedicated class and can be adapted and customized
by the library user.
Considering a single link between a receiving and transmitting modem,
the software architecture is displayed in the following flowchart.

.. mermaid::

   %%{init: {'theme': 'dark'}}%%
   flowchart TD

      subgraph TX[Transmitter]
            direction LR

            BSTX(BitSource) --> BCTX(Bit-Encoding) --> MAPTX

            subgraph WRX[WaveformGenerator]
                direction LR
                MAPTX(Mapping) --> PRETX(Precoding) --> MODTX(Modulation)
            end


            MODTX --> RFTX(RF-Chain)
      end

      TX --> F[Channel] --> RX

      subgraph RX[Receiver]
            direction RL

            RFRX(RF-Chain) --> SRX(Synchronization)

            subgraph WTX[WaveformGenerator]
                direction RL
                SRX --> MODRX(De-Modulation) --> PRERX(Precoding) --> MAPRX(Un-Mapping)
            end

            MAPRX --> BCRX(Bit-Decoding)

      end

At its core, each HermesPy :doc:`Scenario </api/hermespy.scenario.scenario>` consists of multiple links
between a :doc:`Transmitter </api/hermespy.modem.transmitter>`
and :doc:`Receiver </api/hermespy.modem.receiver>`, which are both :doc:`Modems </api/hermespy.modem.modem>`.
Transmitters feed :doc:`Signal</api/hermespy.signal.signal>` models of electromagnetic waves
into a wireless transmission :doc:`Channel </api/hermespy.channel.channel>`.
After propagation over said channel, receivers subsequently pick up the distorted signals.

Both transmitters and receivers perform a sequence of processing steps in order to
exchange information:

#. :doc:`BitsSource</api/hermespy.source.bits_source>` *(transmitters only)* |br|
   Generate a sequence of bits to be transmitted.

#. :doc:`Bit-Encoding</api/hermespy.coding.encoder_manager>` |br|
   Perform operations on the bit-sequence to add redundancy and correct errors.

#. :doc:`Waveform-Generation </api/hermespy.modem.waveform_generator>` |br|
   Map bits to communication symbols, modulate the symbols to electromagnetic baseband-signals.

#. :doc:`Radio-Frequency Chain </api/hermespy.modem.rf_chain>` |br|
   Mix and amplify the baseband-signals to radio-frequency-band signals.

Note that receivers perform the inverse processing steps in reverse order.

Getting Started
---------------
Assuming HermesPy is properly installed within the currently selected Python environment,
users may define custom scenarios to be investigated.

For instance, the following code generates the samples of a single communication frame
transmitted by a PSK/QAM modem:

.. code-block:: python

