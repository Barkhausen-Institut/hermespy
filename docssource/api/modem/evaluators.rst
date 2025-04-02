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

   class EVMArtifact {
 
      +float artifact
      +to_scalar() float
   }

   class EVMEvaluation {

      +str title
      +artifact() EVMArtifact
   }

   class ConstellationEVM {

      +evaluate() EVMEvaluation
      +str abbreviation
      +str title
   }

   BitErrorEvaluator --|> CommunicationEvaluator
   BlockErrorEvaluator --|> CommunicationEvaluator
   FrameErrorEvaluator --|> CommunicationEvaluator
   ThroughputEvaluator --|> CommunicationEvaluator
   ConstellationEVM --|> CommunicationEvaluator

   BitErrorEvaluator --> BitErrorArtifact : create
   BitErrorEvaluator --> BitErrorEvaluation : create
   BlockErrorEvaluator --> BlockErrorArtifact : create
   BlockErrorEvaluator --> BlockErrorEvaluation : create
   FrameErrorEvaluator --> FrameErrorArtifact : create
   FrameErrorEvaluator --> FrameErrorEvaluation : create
   ThroughputEvaluator --> ThroughputArtifact : create
   ThroughputEvaluator --> ThroughputEvaluation : create
   ConstellationEVM --> EVMEvaluation : create
   EVMEvaluation --> EVMArtifact : create

   link CommunicationEvaluator "evaluators.CommunicationEvaluator.html"
   link BitErrorArtifact "evaluators.BitErrorArtifact.html"
   link BitErrorEvaluation "evaluators.BitErrorEvaluation.html"
   link BitErrorEvaluator "evaluators.BitErrorEvaluator.html"
   link BlockErrorArtifact "evaluators.BlockErrorArtifact.html"
   link BlockErrorEvaluation "evaluators.BlockErrorEvaluation.html"
   link BlockErrorEvaluator "evaluators.BlockErrorEvaluator.html"
   link FrameErrorArtifact "evaluators.FrameErrorArtifact.html"
   link FrameErrorEvaluation "evaluators.FrameErrorEvaluation.html"
   link FrameErrorEvaluator "evaluators.FrameErrorEvaluator.html"
   link ThroughputArtifact "evaluators.ThroughputArtifact.html"
   link ThroughputEvaluation "evaluators.ThroughputEvaluation.html"
   link ThroughputEvaluator "evaluators.ThroughputEvaluator.html"
   link EVMArtifact "evaluators.evm.html#hermespy.modem.evaluators.evm.EVMArtifact"
   link EVMEvaluation "evaluators.evm.html#hermespy.modem.evaluators.evm.EVMEvaluation"
   link ConstellationEVM "evaluators.evm.html#hermespy.modem.evaluators.evm.ConstellationEVM"

The implemented :doc:`Communication Evaluators<evaluators.CommunicationEvaluator>` all inherit from the identically named common
base which gets initialized by selecting the two :doc:`Modem<modem.BaseModem>` instances whose communication
should be evaluated.
The currently considered performance indicators are

.. include:: evaluators._table.rst

Configuring :doc:`Communication Evaluators<evaluators.CommunicationEvaluator>` to evaluate the communication process between two
:doc:`Modem<modem.BaseModem>` instances is rather straightforward:

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

   evaluators.ber
   evaluators.bler
   evaluators.fer
   evaluators.throughput
   evaluators.evm
   evaluators.CommunicationEvaluator

.. footbibliography::

