import sys
import transformers_npu
import transformers


def test_OptimizerNames():
    assert hasattr(transformers.training_args.OptimizerNames, "ADAMW_APEX_FUSED_NPU")


def test_TrainingArguments():
    args = transformers.training_args.TrainingArguments("/dev/null")
    assert hasattr(args.optim, "ADAMW_APEX_FUSED_NPU")

    for k, v in sys.modules.items():
        if "transformers" in k:
            cls = getattr(v, "TrainingArguments", None)
            if cls:
                assert cls == transformers_npu.adaptor_training_args.TrainingArgumentsNPU

    attr_list = dir(transformers.TrainingArguments)
    for arr in ["loss_scale", "use_combine_ddp", "use_combine_grad", "distributed_process_group_timeout"]:
        assert arr in attr_list
