import os
import unittest
from unittest.mock import patch

import training_config


class TrainingConfigTests(unittest.TestCase):
    def test_base_model_path_defaults_to_project_gpt2(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(training_config.base_model_path().endswith("gpt2"))

    def test_base_model_path_can_be_overridden(self):
        with patch.dict(os.environ, {"GPT2_MODEL_PATH": "custom-model"}, clear=True):
            self.assertEqual(training_config.base_model_path(), "custom-model")

    def test_training_knobs_default_for_6gb_gpu(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                training_config.training_knobs(),
                {
                    "per_device_train_batch_size": 2,
                    "gradient_accumulation_steps": 4,
                    "max_steps": -1,
                    "gradient_checkpointing": True,
                },
            )

    def test_training_knobs_read_environment(self):
        env = {
            "TRAIN_BATCH_SIZE": "1",
            "GRAD_ACCUM_STEPS": "8",
            "TRAIN_MAX_STEPS": "20",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                training_config.training_knobs(default_batch_size=2, default_grad_accum=4),
                {
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 8,
                    "max_steps": 20,
                    "gradient_checkpointing": True,
                },
            )

    def test_gradient_checkpointing_can_be_disabled(self):
        with patch.dict(os.environ, {"DISABLE_GRADIENT_CHECKPOINTING": "1"}, clear=True):
            self.assertFalse(training_config.gradient_checkpointing_enabled())
            self.assertFalse(training_config.training_knobs()["gradient_checkpointing"])

    def test_model_dir_is_loadable_returns_true_when_loader_succeeds(self):
        class Loader:
            @classmethod
            def from_pretrained(cls, path):
                cls.path = path
                return object()

        self.assertTrue(training_config.model_dir_is_loadable("checkpoint/final", Loader))
        self.assertEqual(Loader.path, "checkpoint/final")

    def test_model_dir_is_loadable_returns_false_when_loader_fails(self):
        class Loader:
            @classmethod
            def from_pretrained(cls, path):
                raise OSError(f"cannot load {path}")

        self.assertFalse(training_config.model_dir_is_loadable("checkpoint/final", Loader))


if __name__ == "__main__":
    unittest.main()
