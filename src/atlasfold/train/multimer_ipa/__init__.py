"""Multimer IPA training pipeline."""

from .model_train import AtlasFoldMultimerIPAForTrain
from .train_module import TrainingModuleIPA

__all__ = ["AtlasFoldMultimerIPAForTrain", "TrainingModuleIPA"]
