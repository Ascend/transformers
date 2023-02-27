import sys
import transformers
import optimum.ascend

for k, v in sys.modules.items():
    if "transformers" in k:
        if getattr(v, "Trainer", None):
            setattr(v, "Trainer", optimum.ascend.NPUTrainer)
        if getattr(v, "TrainingArguments", None):
            setattr(v, "TrainingArguments", optimum.ascend.NPUTrainingArguments)
