"""
.. mermaid::

   %%{init: {'theme': 'dark'}}%%
   flowchart LR

       subgraph Radar

           direction LR

           subgraph Waveform
               Modulation
               TargetEstimation --- Demodulation
           end

           subgraph BeamForming

               TxBeamform[Tx Beamforming]
               RxBeamform[Rx Beamforming]
           end

           Modulation --> TxBeamform
           Demodulation --- RxBeamform

       end

       subgraph Device

           direction TB
           txslot>Tx Slot]
           rxslot>Rx Slot]
       end

   estimations{{Target Estimations}}
   txsignal{{Tx Signal Model}}
   rxsignal{{Rx Signal Model}}

   TxBeamform --> txsignal
   RxBeamform --- rxsignal
   txsignal --> txslot
   rxsignal --- rxslot

   TargetEstimation --- estimations
"""