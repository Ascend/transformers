import transformers_npu


def test_is_torch_npu_available():
    assert transformers_npu.is_torch_npu_available()
