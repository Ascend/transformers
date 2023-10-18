# coding=utf-8
# Copyright 2023 HuggingFace Inc..
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import List

import safetensors
from accelerate.utils import write_basic_config

from diffusers import DiffusionPipeline, UNet2DConditionModel


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


# These utils relate to ensuring the right error message is received when running scripts
class SubprocessCallException(Exception):
    pass

def run_command(command: List[str], return_stdout=False):
    """
    Runs `command` with `subprocess.check_output` and will potentially return the `stdout`. Will also properly capture
    if an error occurred while running `command`
    """
    try:
        p = subprocess.Popen(' '.join(command),stdout=subprocess.PIPE, bufsize=1, shell=True)
        for line in iter(p.stdout.readline, b''):
            print(line)
    except subprocess.CalledProcessError as e:
        raise SubprocessCallException(
            f"Command `{' '.join(command)}` failed with the following error:\n\n{e.output.decode()}"
        ) from e


stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class ExamplesTestsAccelerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmpdir = tempfile.mkdtemp()
        cls.configPath = os.path.join(cls._tmpdir, "default_config.yml")

        write_basic_config(save_location=cls.configPath)
        cls._launch_args = ["accelerate", "launch", "--config_file", cls.configPath]

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(cls._tmpdir)

    def test_custom_diffusion(self):
        print("test_custom_diffusion start")
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                train_custom_diffusion.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --instance_data_dir diffusers/cat_toy_example
                --instance_prompt '<new1>'
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 1.0e-05
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --modifier_token '<new1>'
                --no_safe_serialization
                --output_dir {tmpdir}
                """.split()

            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_custom_diffusion_weights.bin")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "<new1>.bin")))

    def test_custom_diffusion_checkpointing_checkpoints_total_limit(self):
        print("test_custom_diffusion_checkpointing_checkpoints_total_limit start")
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            train_custom_diffusion.py
            --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
            --instance_data_dir diffusers/cat_toy_example
            --output_dir {tmpdir}
            --instance_prompt '<new1>'
            --resolution 64
            --train_batch_size 1
            --modifier_token '<new1>'
            --dataloader_num_workers 0
            --max_train_steps 6
            --checkpoints_total_limit 2
            --checkpointing_steps 2
            --no_safe_serialization
            """.split()

            run_command(self._launch_args + test_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_custom_diffusion_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(self):
        print("test_custom_diffusion_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints start")
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            train_custom_diffusion.py
            --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
            --instance_data_dir diffusers/cat_toy_example
            --output_dir {tmpdir}
            --instance_prompt '<new1>'
            --resolution 64
            --train_batch_size 1
            --modifier_token '<new1>'
            --dataloader_num_workers 0
            --max_train_steps 9
            --checkpointing_steps 2
            --no_safe_serialization
            """.split()

            run_command(self._launch_args + test_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4", "checkpoint-6", "checkpoint-8"},
            )

            resume_run_args = f"""
            train_custom_diffusion.py
            --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
            --instance_data_dir diffusers/cat_toy_example
            --output_dir {tmpdir}
            --instance_prompt '<new1>'
            --resolution 64
            --train_batch_size 1
            --modifier_token '<new1>'
            --dataloader_num_workers 0
            --max_train_steps 11
            --checkpointing_steps 2
            --resume_from_checkpoint checkpoint-8
            --checkpoints_total_limit 3
            --no_safe_serialization
            """.split()

            run_command(self._launch_args + resume_run_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-6", "checkpoint-8", "checkpoint-10"},
            )

if __name__ == '__main__':
    unittest.main(argv=[' ','ExamplesTestsAccelerate.test_custom_diffusion'])

