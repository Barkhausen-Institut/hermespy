==================
Signal Extraction
==================

.. inheritance-diagram:: hermespy.core.evaluators.SignalExtractor hermespy.core.evaluators.SignalExtraction hermespy.core.evaluators.ExtractedSignals  
   :parts: 1

The signal extraction evaluator collects full sample streams at transmitting and receiving DSP layers
during simulation or hardware loop runtime.

.. literalinclude:: ../../scripts/examples/core_evaluators_extraction.py
   :language: python
   :linenos:
   :lines: 10-23

.. autoclass:: hermespy.core.evaluators.SignalExtractor

.. autoclass:: hermespy.core.evaluators.SignalExtraction

.. autoclass:: hermespy.core.evaluators.ExtractedSignals

.. footbibliography::
