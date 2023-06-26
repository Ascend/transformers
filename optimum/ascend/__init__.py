from .training_args import NPUTrainingArguments
from .trainer import NPUTrainer
from .trainer_utils import patch_set_seed
from .modeling_utils import patch_shard_checkpoint


# Monkey patch
patch_set_seed()
patch_shard_checkpoint()