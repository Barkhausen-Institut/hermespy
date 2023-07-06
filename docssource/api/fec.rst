================
Error Correction
================

The forward error correction coding module provides a pipeline description
of operations on the bit level of communication signal processing chains.

.. autoclasstree:: hermespy.fec
   :strict:
   :namespace: hermespy.fec

It consists of the basic pipeline description

.. toctree::

   fec.coding

as well as specific processing step implementations

.. toctree::

   fec.bch
   fec.block_interleaver
   fec.crc
   fec.ldpc
   fec.polar
   fec.repetition
   fec.rs
   fec.rsc
   fec.scrambler
   fec.turbo