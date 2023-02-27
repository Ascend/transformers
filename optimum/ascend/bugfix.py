import importlib.util
import collections.abc
import numpy as np
import torch
from transformers.utils import ModelOutput


if importlib.util.find_spec("apex") is not None:
    import apex

    container_abcs = collections.abc
    def applier_npu(value, fn):
        if isinstance(value, torch.Tensor):
            return fn(value)
        elif isinstance(value, str):
            return value
        elif isinstance(value, np.ndarray):
            return value
        elif hasattr(value, "to"):  # Allow handling of custom batch classes
            return fn(value)
        elif isinstance(value, container_abcs.Mapping):
            if isinstance(value, ModelOutput):
                for k in value:
                    value[k] = applier_npu(value[k], fn)
                return value
            else:
                return {applier_npu(k, fn): applier_npu(v, fn) for k, v in value.items()}
        elif isinstance(value, container_abcs.Iterable):
            return type(value)(applier_npu(v, fn) for v in value)
        else:
            # Do I want this to fire off even if someone chooses to pass something ordinary like
            # an int or float?  May be more annoying than it's worth.
            # print("Warning:  unrecognized type in applier.  If your input data is a custom class, "
            #     "provide it with a .to(dtype) method which converts its floating-point Tensors to dtype. "
            #     "Amp will check for your custom to() and invoke it to cast the batch's "
            #     "floating-point Tensors to the appropriate type. "
            #     "Also, if your data is a custom class, it is your responsibility to ensure that "
            #     "any Tensors you want to be cuda are already cuda."
            return value

    apex.amp._initialize.applier = applier_npu