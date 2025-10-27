"""Performance evaluation module (using lm_eval)."""
from abc import abstractmethod
from multiprocessing import Queue
import json
import os
import warnings
from typing import Callable, Dict, Iterable, List, Tuple, Union
from dataclasses import dataclass
from math import sqrt
from statistics import mean, stdev
import random
import torch
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM
from tqdm import tqdm
from datasets import Dataset, Split
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import DataCollatorForLanguageModeling

from flopmetrics.multiprocessing import start_separate_process
from flopmetrics.profiler import Profiler, NvidiaProfiler, TorchProfiler as TorchProfiler
from flopmetrics.model import load_model


class ModelEvaluator:
    """Abstract class for model evaluation."""

    def __init__(self, model, tokenizer, verbose, device=None):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate.
            tokenizer: Tokenizer for the model.
            verbose: Verbosity flag.
            device: Device to run on. If None, uses CUDA if available.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.verbose = verbose

    @abstractmethod
    def evaluate(self, execute: List[str] = None) -> Dict[str, float]:
        """Abstract evaluate method."""
        return {}


class ModelPerformanceEvaluator(ModelEvaluator):
    """Evaluates model performance on tasks using lm_eval_harness."""

    def __init__(
        self,
        model,
        tokenizer,
        few_shot: int = 0,
        verbose: bool = False,
        device=None,
    ):
        """
        Args:
            model: The model to evaluate (HuggingFace compatible).
            tokenizer: The tokenizer for the model.
            few_shot: Number of few-shot examples to use in prompting.
            verbose: Verbosity of the evaluation.
            device: Device to run the evaluation on.
        """
        super().__init__(model, tokenizer, verbose, device)
        self.eval_model = HFLM(pretrained=model, tokenizer=tokenizer, parralelize=True)
        self.few_shot = few_shot
        self.task_manager = TaskManager()
        self.task_groups = {
            key: el["yaml_path"].split("tasks/")[-1].split("/")[0]
            for key, el in self.task_manager.task_index.items()
            if isinstance(el["yaml_path"], str)
        }
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def evaluate(self, execute: List[str] = None) -> Dict[str, Union[float, str]]:
        """
        Evaluate model performance on specified tasks.

        Args:
            execute: List of task IDs. If None, defaults to a subset of GLUE tasks.

        Returns:
            Dictionary of metrics per task.
        """

        if execute is None:
            execute = [
                "sst2",
                "mrpc",
                "qqp",
                "mnli",
                "mnli_mismatch",
                "qnli",
                "rte",
                "wnli",
            ]
        assert all(
            task in self.task_manager.all_tasks for task in execute
        ), "Unknown tasks given in the execute list"

        eval_config = {
            "model": self.eval_model,
            "tasks": execute,
            "num_fewshot": self.few_shot,
            "batch_size": "auto",
            "device": str(self.device),
        }

        eval_results = evaluator.simple_evaluate(**eval_config)

        results = {}
        for task in execute:
            key = f"{self.task_groups[task]}_{task}"
            task_results = eval_results["results"][task]
            for task_key, task_value in task_results.items():
                if "alias" == task_key:
                    continue
                elif "_stderr" in task_key:
                    results[f"{key}_{task_key.split("_")[0]}_sem"] = float(task_value)
                elif ",none" in task_key:
                    results[f"{key}_{task_key.split(",")[0]}_score"] = float(task_value)
                else:
                    results[f"{key}_{task_key}"] = float(task_value)
            else:
                print(f"Task {task} not found in results.")

        if self.verbose:
            print("==== LMEval Harness Results ====")
            for k, v in results.items():
                print(k, ":", v)

        return results


@dataclass
class EnergyEvaluationArguments:
    """Arguments for energy evaluation."""

    num_samples: int = 110
    input_length: int = 100
    num_warmup_samples: int = 10
    batch_size: int = 16
    max_new_tokens: int = 20  # optional, only used for the generation execution
    nvidia_query_interval: int = 5

    def __post_init__(self):
        assert self.num_samples > 0, "num_samples must be greater than 0"
        assert self.input_length > 0, "input_length must be greater than 0"
        assert self.num_warmup_samples >= 0, "num_warmup_samples must be greater or equal to 0"
        assert self.nvidia_query_interval > 0, "nvidia_query_interval must be greater than 0"
        assert (
            self.num_samples > self.num_warmup_samples
        ), "num_samples must be greater than num_warmup_samples"


class ModelEnergyEvaluator(ModelEvaluator):
    """evaluates a model forward and backward passes

    Attributes:
        model (torch.nn.Module): the model to evaluate
        tokenizer (transformers.Tokenizer): the tokenizer for the model
        args (EnergyEvaluationArguments): the arguments for the evaluation (optional, default: EnergyEvaluationArguments())
        dataset (Dataset): the dataset to evaluate the model on (optional, if not provided a random dataset will be generated)
        verbose (bool): the verbosity of the evaluation (optional, default: False)

    Example:
        model_name = "google/gemma-2-2b-it"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        evaluator = ModelEnergyEvaluator(model, tokenizer, verbose=True)
        metrics = evaluator.evaluate()
    """

    def __init__(
        self,
        model,
        tokenizer,
        args: EnergyEvaluationArguments = EnergyEvaluationArguments(),
        dataset: Dataset = None,
        verbose: bool = False,
        device=None,
    ):
        """
        Initialize energy evaluator.

        Args:
            model: Model to evaluate.
            tokenizer: Tokenizer for the model.
            args: Evaluation arguments.
            dataset: Dataset to use. If None, generates random dataset.
            verbose: Verbosity flag.
            device: Device to run on. If None, uses CUDA if available.
        """
        super().__init__(model, tokenizer, verbose, device)
        self.tokenizer_vocab: List = list(self.tokenizer.vocab.keys())[:3]
        self.args: EnergyEvaluationArguments = args
        self.optimizer = AdamW(model.parameters(), lr=1e-5)

        self.is_dataset_random = dataset is None
        self.dataset: Dataset = dataset
        self.data_loader: DataLoader = None
        self.data_batches: Iterable = None

        self.execute_map = {
            "forward": self._model_forward,
            "backward": self._model_backward,
            "forward_backward": self._model_forward_backward,
            "generate": self._model_generate,
        }

    def _generate_random_dataset(self) -> Dataset:
        """
        Generate a random dataset.

        Returns:
            Generated dataset.
        """
        input_ids = []
        for _ in tqdm(
            range(self.args.num_samples * self.args.batch_size),
            desc="Generating random dataset",
            disable=not self.verbose,
        ):
            random_tokens = [
                random.choice(self.tokenizer_vocab) for _ in range(self.args.input_length)
            ]
            input_ids_sample = self.tokenizer.convert_tokens_to_ids(random_tokens)
            input_ids_sample = torch.Tensor(input_ids_sample).to(torch.int64)
            input_ids.append(input_ids_sample)

        attention_mask = torch.ones_like(input_ids[0]).to(torch.int64)
        dataset = Dataset.from_dict(
            {
                "input_ids": input_ids,
                "labels": input_ids,
                "attention_mask": [attention_mask] * self.args.num_samples * self.args.batch_size,
            },
            split=Split.TEST,
        )
        return dataset

    def _ensure_data(self):
        """Load dataset and create data loader."""
        if self.dataset is None:
            self.dataset = self._generate_random_dataset()
        else:
            assert (
                "input_ids" in self.dataset.column_names
                and "attention_mask" in self.dataset.column_names
                and "labels" in self.dataset.column_names
            ), "Dataset must have input_ids, attention_mask and labels columns"
        if self.data_loader is not None:
            return
        collate_fn = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        self.data_loader = DataLoader(
            self.dataset, batch_size=self.args.batch_size, collate_fn=collate_fn
        )
        self.data_batches = lambda desc: tqdm(self.data_loader, desc=desc, disable=not self.verbose)

    def _batch_to_device(self, batch) -> Dict[torch.Tensor, torch.Tensor]:
        """
        Move batch tensors to device.

        Args:
            batch: Batch to move.

        Returns:
            Batch with tensors moved to device.
        """
        return {k: v.to(self.device) for k, v in batch.items()}

    def _get_exec_fn_name(self, exec_fn: Callable[[dict, Profiler], None]) -> str:
        """
        Get human-readable name from function.

        Args:
            exec_fn: Function to get name from.

        Returns:
            Human-readable function name.
        """
        return exec_fn.__name__.replace("_", " ")

    def evaluate(
        self, execute: List[str] = None, metrics: List[str] = None
    ) -> Dict[str, Union[float, str]]:
        """
        Evaluate model forward and backward passes.

        Args:
            execute: List of passes to evaluate. Options: forward, backward, forward_backward, generate.
            metrics: List of metrics to evaluate. Options: joules, flops, cpu_time, gpu_time.

        Returns:
            Dictionary of evaluation results.
        """
        if execute is None:
            execute = self.execute_map.keys()
        if metrics is None:
            metrics = ["joules", "flops", "cpu_time", "gpu_time"]

        self._ensure_data()
        eval_results = {}

        if "generate" in execute:
            warnings.warn(
                "Generation on random data is not reliably measuring the exact model behavior which influences the measurements. Please use a proper dataset for generation to simulate exact behavior."
            )

        for exec_identifier in execute:
            assert exec_identifier in self.execute_map, f"Invalid execute value: {exec_identifier}"
            exec_fn = self.execute_map[exec_identifier]

            if "joules" in metrics:
                energy_measurements = self._interest_energy(exec_fn)
                energy_measurements = energy_measurements[self.args.num_warmup_samples :]
                eval_results[f"{exec_identifier}_energy_sum"] = sum(energy_measurements)
                eval_results[f"{exec_identifier}_energy_mean"] = mean(energy_measurements)
                eval_results[f"{exec_identifier}_energy_sem"] = stdev(energy_measurements) / sqrt(
                    len(energy_measurements)
                )

            if "flops" in metrics:
                eval_results[f"{exec_identifier}_flops_sum"] = int(self._interest_flops(exec_fn))

            if "cpu_time" in metrics or "gpu_time" in metrics:
                cpu_time_measurements, gpu_time_measurements = self._interest_time(exec_fn)
                if "cpu_time" in metrics:
                    cpu_time_measurements = cpu_time_measurements[self.args.num_warmup_samples :]
                    eval_results[f"{exec_identifier}_cpu_time_mean"] = mean(cpu_time_measurements)
                    eval_results[f"{exec_identifier}_cpu_time_sem"] = stdev(
                        cpu_time_measurements
                    ) / sqrt(len(cpu_time_measurements))

                if "gpu_time" in metrics:
                    gpu_time_measurements = gpu_time_measurements[self.args.num_warmup_samples :]
                    eval_results[f"{exec_identifier}_gpu_time_mean"] = mean(gpu_time_measurements)
                    eval_results[f"{exec_identifier}_gpu_time_sem"] = stdev(
                        gpu_time_measurements
                    ) / sqrt(len(gpu_time_measurements))

        return eval_results

    def _interest_flops(self, exec_fn: Callable[[dict, Profiler], None]) -> int:
        """
        Get FLOPs for interest step.

        Args:
            exec_fn: Function to execute model pass.

        Returns:
            FLOPs for interest step.
        """
        if self.verbose:
            print(f"Evaluating:{self._get_exec_fn_name(exec_fn)} flops")
        prof = self._torch_profile_single(exec_fn)
        return prof.get_flops_by_step().loc["interest", "flops"]

    def _interest_energy(self, exec_fn: Callable[[dict, Profiler], None]) -> List[float]:
        """
        Get energy consumption for interest step.

        Args:
            exec_fn: Function to execute model pass.

        Returns:
            Energy measurements for interest step.
        """
        prof = self._nvidia_profile_batched(
            exec_fn, f"Evaluating:{self._get_exec_fn_name(exec_fn)} energy"
        )
        return prof.get_total_energy(record_steps=["interest"], return_data=True)

    def _interest_time(
        self, exec_fn: Callable[[dict, Profiler], None]
    ) -> Tuple[List[float], List[float]]:
        """
        Get CPU and GPU time for interest step.

        Args:
            exec_fn: Function to execute model pass.

        Returns:
            Tuple of (CPU time measurements, GPU time measurements).
        """
        cpu_time_measurements = []
        gpu_time_measurements = []
        for batch in self.data_batches(f"Evaluating:{self._get_exec_fn_name(exec_fn)} time"):
            batch = self._batch_to_device(batch)
            with TorchProfiler() as prof:
                exec_fn(batch, prof)
            time_by_step = prof.get_time_by_step()
            cpu_time_measurements.append(time_by_step.loc["interest", "cpu_time"])
            gpu_time_measurements.append(time_by_step.loc["interest", "gpu_time"])
        return cpu_time_measurements, gpu_time_measurements

    def nvidia_plot(self, exec_idetifier: str):
        """
        Plot energy consumption time series for a pass.

        Args:
            exec_idetifier: Identifier of the pass to plot.

        Returns:
            Plotly figure with energy consumption time series.
        """
        self._ensure_data()
        assert exec_idetifier in self.execute_map, f"Invalid execute value: {exec_idetifier}"
        prof = self._nvidia_profile_single(self.execute_map[exec_idetifier])
        return prof.get_time_series_plot()

    def _model_forward(self, batch: dict, prof: Profiler):
        """
        Execute forward pass with profiling.

        Args:
            batch: Batch to execute forward pass on.
            prof: Profiler to record energy and FLOPs.
        """
        prof.record_step("other")
        batch = self._batch_to_device(batch)
        with torch.no_grad():
            prof.record_step("interest")
            _ = self.model(**batch)
        prof.record_step("other")

    def _model_backward(self, batch: dict, prof: Profiler):
        """
        Execute backward pass with profiling.

        Args:
            batch: Batch to execute backward pass on.
            prof: Profiler to record energy and FLOPs.
        """
        prof.record_step("other")
        batch = self._batch_to_device(batch)
        self.optimizer.zero_grad()
        prof.record_step("forward")
        outputs = self.model(**batch)
        prof.record_step("interest")
        outputs.loss.backward()
        self.optimizer.step()
        prof.record_step("other")

    def _model_forward_backward(self, batch: dict, prof: Profiler):
        """
        Execute forward and backward passes together with profiling.

        Args:
            batch: Batch to execute passes on.
            prof: Profiler to record energy and FLOPs.
        """
        prof.record_step("other")
        batch = self._batch_to_device(batch)
        self.optimizer.zero_grad()
        prof.record_step("interest")
        outputs = self.model(**batch)
        outputs.loss.backward()
        self.optimizer.step()
        prof.record_step("other")

    def _model_generate(self, batch: dict, prof: Profiler):
        """
        Execute generation pass with profiling.

        Args:
            batch: Batch to execute generation on.
            prof: Profiler to record energy and FLOPs.
        """
        prof.record_step("other")
        batch = self._batch_to_device(batch)
        prof.record_step("interest")
        _ = self.model.generate(
            **batch,
            max_new_tokens=self.args.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prof.record_step("other")

    def _nvidia_profile_batched(
        self, exec_fn: Callable[[dict, Profiler], None], verbose_desc: str
    ) -> NvidiaProfiler:
        """
        Profile model with NvidiaProfiler on batched data.

        Args:
            exec_fn: Function to execute model pass.
            verbose_desc: Description for progress bar.

        Returns:
            NvidiaProfiler with energy measurements.
        """
        with NvidiaProfiler(self.args.nvidia_query_interval) as prof:
            for batch in self.data_batches(verbose_desc):
                exec_fn(batch, prof)
        return prof

    def _torch_profile_single(self, exec_fn: Callable[[dict, Profiler], None]) -> TorchProfiler:
        """
        Profile model with TorchProfiler on a single batch.

        Args:
            exec_fn: Function to execute model pass.

        Returns:
            TorchProfiler with FLOPs measurements.
        """
        single_batch = next(iter(self.data_loader))
        single_batch = self._batch_to_device(single_batch)
        with TorchProfiler() as prof:
            exec_fn(single_batch, prof)
        return prof

    def _nvidia_profile_single(self, exec_fn: Callable[[dict, Profiler], None]) -> NvidiaProfiler:
        """
        Profile model with NvidiaProfiler on a single batch.

        Args:
            exec_fn: Function to execute model pass.

        Returns:
            NvidiaProfiler with energy measurements.
        """
        single_batch = next(iter(self.data_loader))
        single_batch = self._batch_to_device(single_batch)
        with NvidiaProfiler(self.args.nvidia_query_interval) as prof:
            exec_fn(single_batch, prof)
        return prof


def performance_eval(model, tokenizer):
    """
    Evaluate model performance on tasks.

    Args:
        model: Model to evaluate.
        tokenizer: Tokenizer for the model.

    Returns:
        Dictionary of performance metrics.
    """
    performance_evaluator = ModelPerformanceEvaluator(model, tokenizer, verbose=True)
    performance_metrics = performance_evaluator.evaluate()
    return performance_metrics


def energy_eval(model_name: str, tokenizer_name: str = None):
    """
    Evaluate model energy consumption.

    Args:
        model_name: Name of the model to load.
        tokenizer_name: Name of the tokenizer to load. If None, uses model_name.

    Returns:
        Dictionary of energy metrics.
    """
    print("Evaluating energy...")
    model, tokenizer = load_model(model_name, tokenizer_name)
    energy_evaluator = ModelEnergyEvaluator(
        model,
        tokenizer,
        EnergyEvaluationArguments(
            num_samples=110,
            input_length=100,
            num_warmup_samples=10,
            nvidia_query_interval=5,
        ),
        verbose=True,
    )
    energy_metrics = energy_evaluator.evaluate()
    return energy_metrics


def energy_eval_process_wrapper(result_queue: Queue, *args, **kwargs):
    """
    Wrapper function for energy evaluation in a separate process.

    Args:
        result_queue: Queue to put results in.
        *args: Positional arguments for energy_eval.
        **kwargs: Keyword arguments for energy_eval.
    """
    energy_metrics = energy_eval(*args, **kwargs)
    result_queue.put(energy_metrics)


def energy_eval_wrapped(output_dir: str, model_name: str, tokenizer_name: str = None):
    """
    Evaluate model energy consumption and save results to file.

    Args:
        output_dir: Directory to save results.
        model_name: Name of the model to load.
        tokenizer_name: Name of the tokenizer to load. If None, uses model_name.
    """
    os.makedirs(output_dir, exist_ok=True)

    energy_metrics = start_separate_process(
        energy_eval_process_wrapper,
        [model_name, tokenizer_name],
    )
    os.makedirs(os.path.dirname(f"{output_dir}/energy_metrics.json"), exist_ok=True)
    with open(f"{output_dir}/energy_metrics.json", "w", encoding="utf-8") as f:
        json.dump(energy_metrics, f)
