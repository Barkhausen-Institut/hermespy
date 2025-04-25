===============
Coding Pipeline
===============

.. inheritance-diagram:: hermespy.fec.coding.Encoder hermespy.fec.coding.EncoderManager
   :parts: 1

This module introduces the concept of bit :class:`.Encoder` steps,
which form single chain link within a channel coding processing chain.

Considering an arbitrary coding scheme consisting of multiple steps,
the process of encoding bit streams during transmission and decoding them during
subsequent reception is modeled by a chain of :class:`.Encoder` instances:

.. mermaid::

   flowchart LR

      input([Input Bits]) --> n_i[...]
      n_i --> n_a[Encoder N-1] --> n_b[Encoder N] --> n_c[Encoder N+1]  --> n_o[...]
      n_o --> output([Coded Bits])

During transmission encoding the processing chain is sequentially executed from left to right,
during reception decoding in reverse order.

Within bit streams, :class:`.Encoder` instances sequentially encode block sections of :math:`K_n` bits into
code sections of :math:`L_n` bits.
Therefore, the rate of the :math:`n`-th :class:`.Encoder`

.. math::

   R_n = \frac{K_n}{L_n}

is defined as the relation between input and output block length.
The pipeline configuration as well as the encoding step execution is managed by the :class:`.EncoderManager`.
Provided with a frame of :math:`K` input bits, the manager will generate a coded frame of :math:`L` bits by
sequentially executing all :math:`N` configured encoders.
Considering a frame of :math:`K_{\mathrm{Frame}, n}` input bits to the :math:`n`-th encoder within the pipeline,
the manager will split the frame into

.. math::

   M_n(K_{\mathrm{Frame}, n}) = \left\lceil \frac{K_{\mathrm{Frame}, n}}{K_n} \right\rceil

blocks to be encoded independently.
The last block will be padded with zeros should it not contain sufficient bits.
While this may not be exactly standard-compliant behaviour, it is a necessary simplification to enable
arbitrary combinations of encoders.
Therefore, the coding rate of the whole pipeline

.. math::

   R = \frac{K}{L} = \frac{K}{M_N \cdot R_N}

can only be defined recursively considering the number of input blocks :math:`M_N` and rate :math:`R_N` of the last
encoder with in the pipeline, respectively.

.. autoclass:: hermespy.fec.coding.EncoderManager

.. autoclass:: hermespy.fec.coding.Encoder

.. footbibliography::
