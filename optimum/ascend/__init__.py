from .training_args import NPUTrainingArguments
from .trainer import NPUTrainer
from .trainer_utils import patch_set_seed
from .modeling_utils import adapt_transformers_to_npu, patch_shard_checkpoint


# Monkey patch
adapt_transformers_to_npu()
patch_set_seed()
patch_shard_checkpoint()
