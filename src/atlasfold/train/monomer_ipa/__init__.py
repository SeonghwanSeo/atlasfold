"""Monomer IPA training pipeline."""

from .model_train import AtlasFoldIPAForTrain
from .train_module import TrainingModuleIPA

__all__ = ["AtlasFoldIPAForTrain", "TrainingModuleIPA"]
