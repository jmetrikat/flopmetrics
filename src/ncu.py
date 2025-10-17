import argparse
import importlib
import subprocess
from typing import Callable, Iterable, List
import io
import pandas as pd
from tqdm import tqdm

from lora_bp.profiler import TorchProfiler


class NCUProfiler:

    def __init__(
        self,
        metrics: List[str] = [
            "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",
        ],
        ncu_executable: str = "ncu",
    ):
        self.metrics = metrics
        self.ncu_executable = ncu_executable
        assert (
            subprocess.getstatusoutput(ncu_executable)[0] == 0
        ), f"ncu executable not found at {ncu_executable}"
        self.result = None

    def profile(self, executable: str) -> str:
        """
        Profiles the executable with ncu
        Args:
            executable: str, executable to profile
        Returns:
            str: function stdout
        """
        command = f"{self.ncu_executable} --csv --metrics {','.join(self.metrics)} {executable}"
        result = subprocess.run(command, capture_output=True, shell=True)
        out = result.stdout.decode()
        assert all(
            "==ERROR==" not in ln for ln in out.split("\n")
        ), f"Error in ncu profiling: \n{out}"
        self._parse_result(out)
        lines = out.split("\n")
        connected_line = [i for i, line in enumerate(lines) if "== Connected" in line][
            0
        ]
        disconnected_line = [
            i for i, line in enumerate(lines) if "== Disconnected" in line
        ][0]
        return "\n".join(lines[connected_line + 1 : disconnected_line])

    def profile_function(self, function: Callable, arguments: dict) -> str:
        """
        Runs a python function and profiles it with ncu. The function is not run directly but through a python importing its module and calling the function then passing the arguments.
        Args:
            function: Callable, function to run
            arguments: dict, arguments to pass to the function
        Returns:
            str: function stdout

        Example:
            profiler.profile_function("lora_bp.ncu.atoms", "run_mm", {"size": 1000})
        """
        module = str(function.__module__)
        function = str(function.__name__)
        executable_command = f"import {module} as m; m.{function}({', '.join([f'{k}={v}' for k, v in arguments.items()])})"
        return self.profile(f"python3 -c '{executable_command}'")

    def _parse_result(self, result: str):
        lines = result.split("\n")
        first_csv_line = [i for i, line in enumerate(lines) if line.startswith('"ID"')][
            0
        ]
        csv = "\n".join(lines[first_csv_line:])
        df = pd.read_csv(io.StringIO(csv))
        if df["Metric Value"].dtype == str:
            df["Metric Value"] = df["Metric Value"].str.replace(",", "")
        df["Metric Value"] = df["Metric Value"].astype(float)
        self.result = df[["Kernel Name", "Metric Name", "Metric Value"]]

    def get_total_flops(self, precision="single"):
        """
        Calculates the flop count from instruction counts
        Args:
            precision: str, "single", "half" or "double"
        Returns:
            int: flop count calculated from instruction counts: mul + add + 2 * fma
        """
        assert (
            self.result is not None
        ), 'You need to run the profiler first, profile("executable")'
        assert precision in [
            "single",
            "half",
            "double",
        ], "precision must be 'single', 'half' or 'double'"

        precision_prefix = {"single": "f", "half": "h", "double": "d"}[precision]

        assert any(
            f"{precision_prefix}mul" in metric for metric in self.metrics
        ), f"need {precision_prefix}mul counter metric"
        assert any(
            f"{precision_prefix}add" in metric for metric in self.metrics
        ), f"need {precision_prefix}add counter metric"
        assert any(
            f"{precision_prefix}fma" in metric for metric in self.metrics
        ), f"need {precision_prefix}fma counter metric"

        no_mul = self.result.loc[
            self.result["Metric Name"].str.contains(f"{precision_prefix}mul"),
            "Metric Value",
        ].sum()
        no_add = self.result.loc[
            self.result["Metric Name"].str.contains(f"{precision_prefix}add"),
            "Metric Value",
        ].sum()
        no_fma = self.result.loc[
            self.result["Metric Name"].str.contains(f"{precision_prefix}fma"),
            "Metric Value",
        ].sum()
        return no_mul + no_add + 2 * no_fma


def flops_comparison(function: Callable, args: dict, verbose: bool = False):
    """
    Compares the actual FLOPs with the theoretical FLOPs for a given function and arguments
    Args:
        function: Callable, function to run
        args: dict, arguments for the function
        verbose: bool, print progress
    Returns:
        int, int: actual FLOPs, theoretical FLOPs
    """
    ncu = NCUProfiler()
    if verbose:
        print(f"Profiling {function.__name__} with arguments {args}")
    out = ncu.profile_function(function, args)
    actual = ncu.get_total_flops()
    with TorchProfiler() as prof:
        function(**args)
    theoretical = prof.get_total_flops()
    actual_metadata = {
        "kernels": ncu.result["Kernel Name"].unique().tolist(),
    }
    theoretical_metadata = {
        "functions": prof.to_pandas()["name"].unique().tolist(),
    }
    return actual, theoretical, actual_metadata, theoretical_metadata


def scaling_flops_comparison(
    range: Iterable, function: Callable, args_func: Callable, verbose: bool = False
):
    """
    Iterates over a range of values and compares the actual FLOPs with the theoretical FLOPs for a given function and arguments
    Args:
        range: Iterable, range of values to iterate over
        function: Callable, function to run
        args_func: Callable, function to generate arguments for the function, receives an element from the range and returns a dict of arguments for the function
        verbose: bool, print progress
    Returns:
        pd.DataFrame: DataFrame with columns "actual" and "theoretical" and index of the range.
    """
    actual, theoretical = [], []
    actual_metadata, theoretical_metadata = [], []
    for el in tqdm(
        range, disable=not verbose, desc=f"FLOPs comparison of {function.__name__}"
    ):
        args = args_func(el)
        actual_flops, theoretical_flops, a_metadata, t_metadata = flops_comparison(
            function, args
        )
        actual.append(actual_flops)
        theoretical.append(theoretical_flops)
        actual_metadata.append(a_metadata)
        theoretical_metadata.append(t_metadata)
    df = pd.DataFrame(
        {
            "actual": actual,
            "theoretical": theoretical,
            "actual_metadata": actual_metadata,
            "theoretical_metadata": theoretical_metadata,
        },
        index=range,
    )
    df = pd.concat(
        [df.drop(["actual_metadata"], axis=1), df["actual_metadata"].apply(pd.Series)],
        axis=1,
    )
    df = pd.concat(
        [
            df.drop(["theoretical_metadata"], axis=1),
            df["theoretical_metadata"].apply(pd.Series),
        ],
        axis=1,
    )
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "module",
        help="module to import, should be a module string like 'lora_bp.ncu.atoms'",
    )
    parser.add_argument(
        "function", help="function to run, should be a name of a function in the module"
    )
    parser.add_argument(
        "arguments",
        nargs=argparse.REMAINDER,
        help="arguments to pass to the function, remaining arguments with format key:type=value for example size:int=1000 for each argument of the function",
    )
    args = parser.parse_args()

    arguments = {
        el.split(":")[0]: eval(f"{el.split(":")[1].split("=")[0]}({el.split("=")[1]})")
        for el in args.arguments
    }

    module = importlib.import_module(args.module)
    func = getattr(module, args.function)

    actual_flops, theoretical_flops, actual_metadata, theoretical_metadata = (
        flops_comparison(func, arguments, verbose=True)
    )
    print("--- Results ---")
    print(f"Actual Metadata: {actual_metadata}")
    print(f"Theoretical Metadata: {theoretical_metadata}")
    print(
        f"Actual: {int(actual_flops):_}, Theoretical: {int(theoretical_flops):_}, Diff: {(int(actual_flops - theoretical_flops)):_}, FLOPs Utilization: {(theoretical_flops / actual_flops * 100):.2f}%"
    )
