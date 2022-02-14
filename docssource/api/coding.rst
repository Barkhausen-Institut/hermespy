=====================
Channel Coding Module
=====================

The channel coding module provides a pipeline description
of operations on the bit level of communication signal processing chains.

.. autoclasstree:: hermespy.coding
   :strict:
   :namespace: hermespy.coding

It consists of the basic pipeline description

.. toctree::

   coding.coding

as well as specific processing step implementations

.. toctree::

   coding.block_interleaver
   coding.cyclic_redundancy_check
   coding.ldpc
   coding.repetition
   coding.scrambler
   coding.turbo