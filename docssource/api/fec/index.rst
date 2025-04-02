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

   coding

as well as specific processing step implementations

.. toctree::

   bch
   block_interleaver
   crc
   ldpc
   polar
   repetition
   rs
   rsc
   scrambler
   turbo