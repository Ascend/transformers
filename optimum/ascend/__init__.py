from .training_args import NPUTrainingArguments
from .trainer import NPUTrainer
from .trainer_utils import patch_set_seed


# Monkey patch
patch_set_seed()