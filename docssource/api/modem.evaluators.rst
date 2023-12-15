============
Evaluators
============

Communication evaluators are used to extract key performance indicators of
communication links between modems.

.. mermaid::

   classDiagram

   class CommunicationEvaluator {
    
      <<Abstract>>

      +TransmittingModem transmitting_modem
      +ReceivingModem receiving_modem
   }

   class BitErrorArtifact {
 
      +float artifact
      +to_scalar() float
   }

   class BitErrorEvaluation {

      +str title
      +artifact() BitErrorArtifact
   }

   class BitErrorEvaluator {

      +evaluate() BitErrorEvaluation
      +str abbreviation
      +str title
   }

   class BlockErrorArtifact {
 
      +float artifact
      +to_scalar() float
   }

   class BlockErrorEvaluation {

      +str title
      +artifact() BlockErrorArtifact
   }

   class BlockErrorEvaluator {

      +evaluate() BlockErrorEvaluation
      +str abbreviation
      +str title
   }

   class FrameErrorArtifact {
 
      +float artifact
      +to_scalar() float
   }

   class FrameErrorEvaluation {

      +str title
      +artifact() FrameErrorArtifact
   }

   class FrameErrorEvaluator {

      +evaluate() FrameErrorEvaluation
      +str abbreviation
      +str title
   }

   class ThroughputArtifact {
 
      +float artifact
      +to_scalar() float
   }

   class ThroughputEvaluation {

      +str title
      +artifact() ThroughputArtifact
   }

   class ThroughputEvaluator {

      +evaluate() ThroughputEvaluation
      +str abbreviation
      +str title
   }

   BitErrorEvaluator --|> CommunicationEvaluator
   BlockErrorEvaluator --|> CommunicationEvaluator
   FrameErrorEvaluator --|> CommunicationEvaluator
   ThroughputEvaluator --|> CommunicationEvaluator

   BitErrorEvaluator --> BitErrorArtifact : create
   BitErrorEvaluator --> BitErrorEvaluation : create
   BlockErrorEvaluator --> BlockErrorArtifact : create
   BlockErrorEvaluator --> BlockErrorEvaluation : create
   FrameErrorEvaluator --> FrameErrorArtifact : create
   FrameErrorEvaluator --> FrameErrorEvaluation : create
   ThroughputEvaluator --> ThroughputArtifact : create
   ThroughputEvaluator --> ThroughputEvaluation : create

   link CommunicationEvaluator "modem.evaluators.CommunicationEvaluator.html"
   link BitErrorArtifact "modem.evaluators.BitErrorArtifact.html"
   link BitErrorEvaluation "modem.evaluators.BitErrorEvaluation.html"
   link BitErrorEvaluator "modem.evaluators.BitErrorEvaluator.html"
   link BlockErrorArtifact "modem.evaluators.BlockErrorArtifact.html"
   link BlockErrorEvaluation "modem.evaluators.BlockErrorEvaluation.html"
   link BlockErrorEvaluator "modem.evaluators.BlockErrorEvaluator.html"
   link FrameErrorArtifact "modem.evaluators.FrameErrorArtifact.html"
   link FrameErrorEvaluation "modem.evaluators.FrameErrorEvaluation.html"
   link FrameErrorEvaluator "modem.evaluators.FrameErrorEvaluator.html"
   link ThroughputArtifact "modem.evaluators.ThroughputArtifact.html"
   link ThroughputEvaluation "modem.evaluators.ThroughputEvaluation.html"
   link ThroughputEvaluator "modem.evaluators.ThroughputEvaluator.html"


The implemented :doc:`Communication Evaluators<modem.evaluators.CommunicationEvaluator>` all inherit from the identically named common
base which gets initialized by selecting the two :doc:`Modem<modem.modem.BaseModem>` instances whose communication
should be evaluated.
The currently considered performance indicators are

.. include:: modem.evaluators._table.rst

Configuring :doc:`Communication Evaluators<modem.evaluators.CommunicationEvaluator>` to evaluate the communication process between two
:doc:`Modem<modem.modem.BaseModem>` instances is rather straightforward:

.. code-block:: python

   # Create two separate modem instances
   modem_alpha = Modem()
   modem_beta = Modem()

   # Create a bit error evaluation as a communication evaluation example
   communication_evaluator = BitErrorEvaluator(modem_alpha, modem_beta)

   # Extract evaluation artifact
   communication_artifact = communication_evaluator.evaluate()

   # Visualize artifact
   communication_artifact.plot()

.. toctree::
   :hidden:

   modem.evaluators.ber
   modem.evaluators.bler
   modem.evaluators.fer
   modem.evaluators.throughput
   modem.evaluators.CommunicationEvaluator

.. footbibliography::

