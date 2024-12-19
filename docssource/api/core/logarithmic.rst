============
Logarithmics
============

.. inheritance-diagram:: hermespy.core.logarithmic.Logarithmic hermespy.core.logarithmic.LogarithmicSequence
   :parts: 1

Logarithmic numbers represent Decibel (dB) parameters within Hermes' API.
However, they will always act as their linear value when being interacted with,
in order to preserve compatibility with any internal equation,
since equations internally assume all parameters to be linear.

Note that therefore,

.. code-block::

   a = Logarithmic(10)
   b = Logarithmic(20)

   c = a + b
   print(c)
   >>> 20dB

will return in the output :math:`20.41392685158225` instead of :math:`30`,
since internally the linear representations will be summed.
Instead, use the multiplication operator to sum Logarithmics, i.e.

.. code-block::

   a = Logarithmic(10)
   b = Logarithmic(20)

   c = a * b
   print(c)
   >>> 30dB

.. autoclass:: hermespy.core.logarithmic.Logarithmic

.. autoclass:: hermespy.core.logarithmic.LogarithmicSequence

.. autoclass:: hermespy.core.logarithmic.dB

.. autoclass:: hermespy.core.ValueType

.. footbibliography::
