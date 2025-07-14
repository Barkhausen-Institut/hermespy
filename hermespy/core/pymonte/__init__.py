# -*- coding: utf-8 -*-

from .artifact import Artifact, ArtifactTemplate, FAT
from .actors import MonteCarloActor, MonteCarloSample
from .evaluation import Evaluation, EvaluationTemplate, EvaluationResult, Evaluator
from .grid import GridDimension, RegisteredDimension, register, ScalarDimension
from .monte_carlo import MonteCarlo, MonteCarloResult
from .scalar import ScalarEvaluationResult

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"

__all__ = [
    "Artifact",
    "ArtifactTemplate",
    "FAT",
    "MonteCarloActor",
    "MonteCarloSample",
    "Evaluation",
    "EvaluationTemplate",
    "EvaluationResult",
    "Evaluator",
    "GridDimension",
    "RegisteredDimension",
    "register",
    "ScalarDimension",
    "MonteCarlo",
    "MonteCarloResult",
    "ScalarEvaluationResult",
]