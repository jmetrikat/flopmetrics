"""Exercise dependency compatibility on CPU without downloading models."""

import importlib
import unittest

import torch
from peft import LoraConfig, get_peft_model
from torch.profiler import ProfilerActivity
from transformers import AutoModelForCausalLM, LlamaConfig

from flopmetrics.network import SimpleLinear
from flopmetrics.profiler import TorchProfiler


class DependencySmokeTests(unittest.TestCase):
    def test_library_modules_import(self):
        for module in (
            "evaluation",
            "model",
            "multiprocessing",
            "ncu",
            "network",
            "profiler",
        ):
            with self.subTest(module=module):
                importlib.import_module(f"flopmetrics.{module}")

    def test_linear_forward_and_backward_on_cpu(self):
        layer = SimpleLinear(4, 4, device=torch.device("cpu"))
        output = layer(torch.ones(4, 2))

        self.assertEqual(output.shape, (4, 2))
        self.assertTrue(torch.isfinite(output).all().item())
        output.sum().backward()
        self.assertIsNotNone(layer.W.grad)
        self.assertTrue(torch.isfinite(layer.W.grad).all().item())

    def test_torch_profiler_counts_cpu_matmul(self):
        left = torch.ones(4, 4)
        right = torch.ones(4, 2)
        with TorchProfiler(activities=[ProfilerActivity.CPU]) as profiler:
            with profiler.record_context("matmul"):
                torch.mm(left, right)

        self.assertEqual(profiler.get_total_flops(), 64)
        self.assertEqual(profiler.get_flops_by_step().loc["matmul", "flops"], 64)
        self.assertGreater(profiler.get_total_time(device="CPU"), 0)
        self.assertFalse(profiler.to_pandas().empty)

    def test_tiny_llama_lora_training_on_cpu(self):
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=16,
        )
        model = get_peft_model(
            AutoModelForCausalLM.from_config(config),
            LoraConfig(r=2, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
        )
        input_ids = torch.tensor([[1, 2, 3, 4]])
        output = model(input_ids=input_ids, labels=input_ids)

        self.assertTrue(torch.isfinite(output.loss).item())
        output.loss.backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(p.grad is not None for p in trainable))
        self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in trainable))


if __name__ == "__main__":
    unittest.main()
