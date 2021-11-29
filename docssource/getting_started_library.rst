========
Library
========

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

