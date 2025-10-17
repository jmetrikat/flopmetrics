"""Performance evaluation module (using lm_eval_harness)."""

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

from metriwatt.multiprocessing import start_separate_process
from metriwatt.profiler import Profiler, NvidiaProfiler, TorchProfiler as TorchProfiler
from metriwatt.model import load_model


class ModelEvaluator:
    """Abstract class for the evaluation of a model"""

    def __init__(self, model, tokenizer, verbose, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.verbose = verbose

    @abstractmethod
    def evaluate(self, execute: List[str] = None) -> Dict[str, float]:
        """abstract evaluate method
        Args:
            execute (List[str], optional): list of execution types to evaluate. Defaults to None.
        Returns:
            Dict[str, Union[float, str]]: dictionary with the metrics
        """
        return {}


class ModelPerformanceEvaluator(ModelEvaluator):
    """
    Evaluates a model's performance on tasks supported by lm_eval_harness (e.g. GLUE)
    and returns a dictionary of metrics.
    """

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
            model (torch.nn.Module): The model to evaluate (HuggingFace compatible).
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
            few_shot (int, optional): Number of few-shot examples to use in prompting.
            verbose (bool, optional): Verbosity of the evaluation.
            device (torch.device, optional): Device to run the evaluation on.
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
        Runs the lm_eval_harness to compute performance metrics on the specified tasks.
        Returns a dictionary containing metrics per task.

        Args:
            execute (List[str], optional): List of GLUE tasks or any tasks recognized
                by lm_eval_harness. Must be the harness IDs, for example
                ["sst2", "mrpc", ...].
                If None, defaults to a subset of GLUE tasks.
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
    """holds evaluator arguments"""

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
        """generates a random dataset with input_length and num_samples from EnergyEvaluationArguments"""

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
        """loads the dataset and creates the data loader
        Raises:
            AssertionError: if the dataset does not have the required columns
        Sets:
            self.data_loader: the data loader for the dataset
            self.data_batches: the batches of the data loader
        """
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
        """moves the batch to the device
        Args:
            batch (dict): the batch to move to the device
        Returns:
            dict: the batch moved to the device
        """
        return {k: v.to(self.device) for k, v in batch.items()}

    def _get_exec_fn_name(self, exec_fn: Callable[[dict, Profiler], None]) -> str:
        """returns the name of the function
        Args:
            exec_fn (Callable[[dict, Profiler], None]): the function to get the name of
        Returns:
            str: the name of the function
        """
        return exec_fn.__name__.replace("_", " ")

    def evaluate(
        self, execute: List[str] = None, metrics: List[str] = None
    ) -> Dict[str, Union[float, str]]:
        """evaluates the model forward and backward passes
        Args:
            execute (List[str]): the list of passes to evaluate (default: all available], options: listed in self.execute_map.keys())
            metrics (List[str]): the list of metrics to evaluate (default: all available), options: ["joules", "flops", "cpu_time", "gpu_time"]
        Returns:
            dict: the evaluation results
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
        """returns the flops of the interest step
        Args:
            exec_fn (Callable[[dict, Profiler], None]): the function to execute the model pass
        Returns:
            int: the flops of the interest step
        """
        if self.verbose:
            print(f"Evaluating:{self._get_exec_fn_name(exec_fn)} flops")
        prof = self._torch_profile_single(exec_fn)
        return prof.get_flops_by_step().loc["interest", "flops"]

    def _interest_energy(self, exec_fn: Callable[[dict, Profiler], None]) -> List[float]:
        """returns the energy of the interest step
        Args:
            exec_fn (Callable[[dict, Profiler], None]): the function to execute the model pass
        Returns:
            List[float]: the energy measurements of the interest step
        """
        prof = self._nvidia_profile_batched(
            exec_fn, f"Evaluating:{self._get_exec_fn_name(exec_fn)} energy"
        )
        return prof.get_total_energy(record_steps=["interest"], return_data=True)

    def _interest_time(
        self, exec_fn: Callable[[dict, Profiler], None]
    ) -> Tuple[List[float], List[float]]:
        """returns the time of the interest step
        Args:
            exec_fn (Callable[[dict, Profiler], None]): the function to execute the model pass
        Returns:
            Tuple[List[float], List[float]]: the cpu and gpu time measurements of the interest step
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
        """plots the energy and flops of the model passes
        Args:
            exec_idetifier (str): the identifier of the pass to plot
        Returns:
            Plot: the plot of the energy and flops
        """
        self._ensure_data()
        assert exec_idetifier in self.execute_map, f"Invalid execute value: {exec_idetifier}"
        prof = self._nvidia_profile_single(self.execute_map[exec_idetifier])
        return prof.get_time_series_plot()

    def _model_forward(self, batch: dict, prof: Profiler):
        """executes the model forward pass
        Sets the prof record step to "interest" for the forward pass as this is the step we are interested in
        Args:
            batch (dict): the batch to execute the forward pass
            prof (Profiler): the profiler to record the energy and flops
        """
        prof.record_step("other")
        batch = self._batch_to_device(batch)
        with torch.no_grad():
            prof.record_step("interest")
            _ = self.model(**batch)
        prof.record_step("other")

    def _model_backward(self, batch: dict, prof: Profiler):
        """executes the model backward pass
        Sets the prof record step to "interest" for the backward pass as this is the step we are interested in
        Args:
            batch (dict): the batch to execute the backward pass
            prof (Profiler): the profiler to record the energy and flops
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
        """executes the model forward and backward pass
        Sets the prof record step to "interest" for the forward and backward pass as this is the step we are interested in
        Args:
            batch (dict): the batch to execute the backward pass
            prof (Profiler): the profiler to record the energy and flops
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
        """executes the model generate pass
        Sets the prof record step to "interest" for the generate pass as this is the step we are interested in
        Args:
            batch (dict): the batch to execute the generate pass
            prof (Profiler): the profiler to record the energy and flops
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
        """profiles the model with NvidiaProfiler
        Args:
            exec_fn (Callable[[dict, Profiler], None]): the function to execute the model pass
            verbose_desc (str): the description for the tqdm progress bar
        Returns:
            Profiler: NvidiaProfiler with energy measurements
        """
        with NvidiaProfiler(self.args.nvidia_query_interval) as prof:
            for batch in self.data_batches(verbose_desc):
                exec_fn(batch, prof)
        return prof

    def _torch_profile_single(self, exec_fn: Callable[[dict, Profiler], None]) -> TorchProfiler:
        """profiles the model with TorchProfiler
        Args:
            exec_fn (Callable[[dict, Profiler], None]): the function to execute the model pass
        Returns:
            Profiler: TorchProfiler flops measurements
        """
        single_batch = next(iter(self.data_loader))
        single_batch = self._batch_to_device(single_batch)
        with TorchProfiler() as prof:
            exec_fn(single_batch, prof)
        return prof

    def _nvidia_profile_single(self, exec_fn: Callable[[dict, Profiler], None]) -> NvidiaProfiler:
        """profiles the model with NvidiaProfiler
        Args:
            exec_fn (Callable[[dict, Profiler], None]): the function to execute the model pass
        Returns:
            Profiler: NvidiaProfiler with energy measurements
        """
        single_batch = next(iter(self.data_loader))
        single_batch = self._batch_to_device(single_batch)
        with NvidiaProfiler(self.args.nvidia_query_interval) as prof:
            exec_fn(single_batch, prof)
        return prof


def performance_eval(model, tokenizer):
    performance_evaluator = ModelPerformanceEvaluator(model, tokenizer, verbose=True)
    performance_metrics = performance_evaluator.evaluate()
    return performance_metrics


def energy_eval(model_name: str, tokenizer_name: str = None):
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
    energy_metrics = energy_eval(*args, **kwargs)
    result_queue.put(energy_metrics)


def energy_eval_wrapped(output_dir: str, model_name: str, tokenizer_name: str = None):
    os.makedirs(output_dir, exist_ok=True)

    energy_metrics = start_separate_process(
        energy_eval_process_wrapper,
        [model_name, tokenizer_name],
    )
    os.makedirs(os.path.dirname(f"{output_dir}/energy_metrics.json"), exist_ok=True)
    with open(f"{output_dir}/energy_metrics.json", "w", encoding="utf-8") as f:
        json.dump(energy_metrics, f)
