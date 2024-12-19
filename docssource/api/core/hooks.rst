======
Hooks
======

.. inheritance-diagram:: hermespy.core.hooks.Hook hermespy.core.hooks.Hookable
   :parts: 1

Hooks and hookables represent callback functions and their respective call sites within HermesPy's processing pipeline.
They are deployed to report the results of different signal processing stages to trigger additional caching and post-processing steps.

.. autoclass:: hermespy.core.hooks.Hook

.. autoclass:: hermespy.core.hooks.Hookable

.. autoclass:: hermespy.core.hooks._RT

.. footbibliography::
