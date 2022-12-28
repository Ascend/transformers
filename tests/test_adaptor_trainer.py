import sys
import transformers_npu
import transformers


def test_TrainingArguments():
    for k, v in sys.modules.items():
        if "transformers" in k:
            cls = getattr(v, "Trainer", None)
            if cls:
                assert cls == transformers_npu.adaptor_trainer.TrainerNPU
