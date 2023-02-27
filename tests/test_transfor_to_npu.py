import sys
import unittest

import optimum.ascend
from optimum.ascend import transfor_to_npu

PATCH_CLS = {
    "Trainer": optimum.ascend.NPUTrainer,
    "TrainingArguments": optimum.ascend.NPUTrainingArguments,
}


class TestTransforToNPU(unittest.TestCase):
    """
    Unit tests for transfor_to_npu
    """

    def test_transfor_to_npu(self):
        for k, v in sys.modules.items():
            if "transformers" in k:
                for patch_cls_k, patch_cls_v in PATCH_CLS.items():
                    cls = getattr(v, patch_cls_k, None)
                    if cls:
                        self.assertEqual(cls, patch_cls_v)
