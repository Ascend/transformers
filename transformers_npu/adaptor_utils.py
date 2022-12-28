import importlib.util


def is_torch_npu_available():
    return importlib.util.find_spec("torch_npu") is not None