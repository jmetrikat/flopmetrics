"""Classes for profiling runtime of training and inference."""
import subprocess
from contextlib import contextmanager
from datetime import datetime
from multiprocessing import Array, Event, Process, Value
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch.autograd.profiler_util import EventList, FunctionEvent
from torch.profiler import ProfilerActivity, profile

from flopmetrics.multiprocessing import (
    FileCacheResultHandler,
    MPQueueResultHandler,
    ResultHandler,
)


class Profiler:
    """Abstract Class for Profilers."""

    def __init__(self):
        self.record_steps: list[tuple[datetime, str]] = []
        self.record_step("__init__")

    def record_step(self, name: str) -> None:
        """Saves a record step with a name

        Args:
            name (str): the name of the step

        """
        self.record_steps.append((datetime.now(), name))

    @contextmanager
    def record_context(self, name: str):
        """Record step as a context manager.
        Starts a record_step with the given name,
        after execution of the context it records a step with the name "__other__"

        Args:
            name (str): the name of the context

        """
        try:
            self.record_step(name)
            yield None
        finally:
            self.record_step("__other__")


class TorchProfiler(profile, Profiler):
    # pylint: disable=W0212:protected-access
    """subclass of (torch.profiler) profile"""

    def __init__(self, *args, **kwargs):
        defaults = {
            "with_flops": True,
            "profile_memory": True,
            "activities": [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        }
        kwargs = {**defaults, **kwargs}
        profile.__init__(self, *args, **kwargs)
        Profiler.__init__(self)

        self.numeric_columns = [
            "flops",
            "count",
            "self_device_time_total",
            "self_cpu_time_total",
            "device_time_total",
            "cpu_time_total",
            "self_device_memory_usage",
            "self_cpu_memory_usage",
            "device_memory_usage",
            "cpu_memory_usage",
        ]

    def _get_profiler_events(self) -> EventList[FunctionEvent]:
        """Ensures the profiler has function events and returns them."""
        assert self.profiler, "Profiling not stopped correctly"
        self.profiler._ensure_function_events()
        return self.profiler._function_events

    def to_pandas(self) -> pd.DataFrame:
        """Converts profiler events to pandas dataframe

        Returns:
            pd.DataFrame() containing row-wise all events (calculations like aten::mm) with metrics

        """
        matched_events = self._get_profiler_events_by_record_step()

        rows = []
        for step, events in matched_events.items():
            for event in events:
                row = {col: getattr(event, col, None) for col in self.numeric_columns}
                row["name"] = getattr(event, "name", None)
                row["is_annotation"] = getattr(event, "is_user_annotation", None)
                row["device"] = getattr(event, "device_type", None).name
                row["record_step"] = step
                rows.append(row)

        df = pd.DataFrame(rows)
        df.loc[df["device"] == "CPU", "self_device_time_total"] = 0
        df.loc[df["device"] == "CPU", "device_time_total"] = 0
        df.loc[df["device"] == "CUDA", "self_cpu_time_total"] = 0
        df.loc[df["device"] == "CUDA", "cpu_time_total"] = 0
        df["self_cpu_time_total_percentage"] = (
            df["self_cpu_time_total"] / df["self_cpu_time_total"].sum() * 100
        )
        df["cpu_time_total_percentage"] = (
            df["cpu_time_total"] / df["cpu_time_total"].sum() * 100
        )
        return df

    def summary(self) -> pd.DataFrame:
        """Generates a summary as a pandas dataframe

        Returns:
            pd.DataFrame() containing summed metrics for all event types (e.g. aten::mul, aten::mm)

        """
        df: pd.DataFrame = self.to_pandas()
        df = df[~df["is_annotation"]]
        df = df[["name"] + self.numeric_columns].groupby("name").sum()
        df = df.sort_values(by=["flops", "count"])
        return df

    def totals(self) -> pd.Series:
        """Sums all metrics for the whole profiling

        Returns:
            pd.Series containing the total sums with keys

        """
        df: pd.DataFrame = self.to_pandas()
        df = df[~df["is_annotation"]]
        df = df[self.numeric_columns].sum(axis=0)
        return df

    def get_total_time(self, device: str = "CUDA") -> float:
        """Computes the total device time from the profiler's events without converting to pandas df
        Args:
            device (str): the device to calculate the total time for (default = CUDA), options: CPU, CUDA
        Returns:
            float: the total device time in us
        """
        assert device in ["CPU", "CUDA"], "device must be either CPU or CUDA"
        events = self._get_profiler_events()
        time_field = (
            "self_cpu_time_total" if device == "CPU" else "self_device_time_total"
        )
        total_time = sum(
            getattr(event, time_field, 0.0)
            for event in events
            if event.device_type.name == device and not event.is_user_annotation
        )
        return total_time

    def get_total_flops(self) -> float:
        """Computes the total FLOPs from the profiler's events without converting to pandas df

        Returns:
            float: The sum of FLOPs for all events.

        """
        events = self._get_profiler_events()
        total_flops = sum(getattr(event, "flops", 0.0) for event in events)
        return int(total_flops)

    def _get_profiler_events_by_record_step(self) -> dict[str, list[FunctionEvent]]:
        """Returns the profiler events grouped by record steps
        Returns:
            Dict[str, List[FunctionEvent]]: the profiler events grouped by record steps.
        """
        events = self._get_profiler_events()
        matched_events = {step: [] for _, step in self.record_steps}
        base_timestamp = self.profiler.kineto_results.trace_start_ns() * 1e-3

        for event in events:
            event_start = event.time_range.start
            event_ts = base_timestamp + event_start

            diffs = [
                (event_ts - ts.timestamp() * 1e6, name)
                for ts, name in self.record_steps
            ]
            pos_diffs = [(diff, name) for diff, name in diffs if diff >= 0]
            matched = min(pos_diffs, key=lambda x: x[0])
            matched_events[matched[1]].append(event)

        return matched_events

    def get_flops_by_step(self) -> pd.DataFrame:
        """Computes the FLOPs for each step in the record_steps list based on time_range

        Returns:
            pd.DataFrame: A DataFrame with the FLOPs for each step.

        """
        matched_events = self._get_profiler_events_by_record_step()
        flops_by_step = {
            name: sum(event.flops for event in events)
            for name, events in matched_events.items()
        }
        df = pd.DataFrame.from_dict(flops_by_step, orient="index", columns=["flops"])

        return df

    def get_time_by_step(self) -> pd.DataFrame:
        """Computes the time for each step in the record_steps list based on time_range
        Returns:
            pd.DataFrame: A DataFrame with the time is us for each step.
        """
        matched_events = self._get_profiler_events_by_record_step()
        time_by_step = {}
        for step, events in matched_events.items():
            gpu_time, cpu_time = 0.0, 0.0
            for event in events:
                if event.is_user_annotation:
                    continue
                if event.device_type.name == "CUDA":
                    gpu_time += getattr(event, "self_device_time_total", 0.0)
                elif event.device_type.name == "CPU":
                    cpu_time += getattr(event, "self_cpu_time_total", 0.0)
            time_by_step[step] = (cpu_time, gpu_time)
        df = pd.DataFrame.from_dict(
            time_by_step,
            orient="index",
            columns=["cpu_time", "gpu_time"],
        )
        return df


class NvidiaProfiler(Profiler):
    """ "
    Profiler for gpu energy consumption as context manager using nvidia-smi.
    The Profiler starts a seperate process for profiling using nvidia-smi.
    If one enters the context the parallel process gets started and it waits until the profiling got some data before continuing
    the execution inside the context. If the context closes all data is collected and the seperate process gets killed.

    Args:
        interval (int): milliseconds interval of profiler steps
        cache_file (str or None): file path to store the profiling data in a csv file. If None the data is stored in memory
            and collected when the profiler context exits otherwise a csv file is created and the data is written to it and read from it
            after the context exits
        force_cache (bool): if True the cache file will be overwritten if it already exists

    """

    def __init__(
        self,
        interval: int = 1,
        cache_file: str | None = None,
        force_cache: bool = False,
    ):
        self.current_record_step = Array("c", 1000)
        self.interval: float = interval
        self.data: list[tuple[int, datetime, float]] = []
        self.should_profiling_run = Value("i", 1)
        self.profiling_started: Event = Event()  # type: ignore
        self.profiling_stopped: Event = Event()  # type: ignore
        self.result_handler: ResultHandler = (
            MPQueueResultHandler()
            if not cache_file
            else FileCacheResultHandler(cache_file, force_cache)
        )
        self.result_handler.set_columns(
            ("gpu_id", "timestamp", "power", "memory", "record_step"),
            (int, str, float, float, str),
        )
        self.process: Process = Process(
            target=NvidiaProfiler._nvidiasmi_profiling_process,
            args=(
                self.should_profiling_run,
                self.profiling_started,
                self.profiling_stopped,
                self.result_handler,
                self.current_record_step,
                self.interval,
            ),
        )
        super().__init__()

    def record_step(self, name):
        super().record_step(name)
        self.current_record_step.value = name.encode("utf-8")

    @staticmethod
    def _nvidiasmi_profiling_process(
        should_run,
        started,
        stopped,
        result_handler: ResultHandler,
        current_record_step,
        interval: int,
    ):
        """Static method for the seperate profiling process. Use this method in a multiprocessing process.
        Opens a subprocess with nvidia-smi and saves gpu_id, timestamp and power for every interval seconds and puts these values in the queue.

        Args:
            should_run (multiprocessing.Value (int)): should be 1 initially to run the subprocess, change it to 0 to stop the profiling
            started (multiprocessing.Event): notifies the process starter that profiling runs
            stopped (multiprocessing.Event): notifies the process starter that profiling ended
            queue (multiprocessing.Queue): handles the shared memory, call queue.get() to the firt put in (gpu_id, timestamp, power and used memory)
            interval (int): the interval in milliseconds the profiler should check nvidia-smi

        """

        def read_data(ln):
            vals: list[Any] = ln.strip().split(", ")
            gid: int = int(vals[0])
            ts: datetime = datetime.strptime(vals[1], "%Y/%m/%d %H:%M:%S.%f")
            pwr: float = float(vals[2].split(" ")[0])
            mem: float = float(vals[3].split(" ")[0])
            return (gid, ts, pwr, mem, current_record_step.value.decode("utf-8"))

        with (
            subprocess.Popen(
                f"nvidia-smi --query-gpu=index,timestamp,power.draw,memory.used --format=csv -lms {interval}",
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
            ) as nvidiasmi_process,
            result_handler as result,
        ):
            with nvidiasmi_process.stdout as out:
                _ = out.readline()
                if should_run.value:
                    started.set()
                while should_run.value:
                    data = read_data(out.readline())
                    result.put(data)
        result.put(None)
        stopped.set()

    def __enter__(self):
        assert subprocess.getstatusoutput("nvidia-smi")[0] == 0, (
            "Could not find nvidia-smi tool"
        )
        self.process.start()
        self.profiling_started.wait()
        return self

    def __exit__(self, *args, **kwargs):
        self.should_profiling_run.value = 0
        self.profiling_stopped.wait()
        self.data = self.result_handler.get_all()
        self.process.join()
        self.process.terminate()

    @staticmethod
    def from_cache(cache_file: str) -> "NvidiaProfiler":
        """Creates a NvidiaProfiler object from a cache file.
        Note that only the data is loaded from the cache file, the profiler object is not
        started and the record_steps will be empty but
        available in the data directly.

        Args:
            cache_file (str): the path to the cache file

        Returns:
            NvidiaProfiler: the NvidiaProfiler object with the data from the cache file

        """
        prof = NvidiaProfiler(cache_file=cache_file)
        prof.data = prof.result_handler.get_all()
        return prof

    def to_pandas(self) -> pd.DataFrame:
        """Generates a pandas dataframe from the profiled data

        Returns:
            pandas DataFrame containing columns gpu_index, timestamp, power (in watts), memory (in MiB)

        """
        df: pd.DataFrame = pd.DataFrame(
            self.data,
            columns=["gpu_id", "timestamp", "power", "memory", "record_step"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S.%f")
        return df

    def get_profiled_gpus(self) -> list[int]:
        """Returns:
        a list of all gpu ids which got profiled

        """
        df: pd.DataFrame = self.to_pandas()
        return df["gpu_id"].unique().tolist()

    def get_total_energy(
        self,
        gpu_ids: list[int] | None = None,
        record_steps: list[str] | None = None,
        return_data: bool = False,
    ) -> float:
        """Summes the power to get the total energy

        Args:
            gpu_id (List[int]): the ids of the gpu to calculate the total energy for (default = None, the first gpu id is used)
            record_steps (List[str]): records_steps to include in the summation (default = None, all record_steps will be included)
            return_data (bool): whether to return the raw data of the selection (default = False)

        Returns:
            the total energy recorded in watt seconds
            if return_data is True the list of measurements for each record_step in watt seconds is returned

        """
        if not self.data:
            return 0.0
        df: pd.DataFrame = self.to_pandas()
        df["record_step_id"] = df["record_step"].ne(df["record_step"].shift()).cumsum()
        gpu_ids = gpu_ids or [df["gpu_id"].unique()[0]]
        df = df[df["gpu_id"].isin(gpu_ids)]
        df["time_interval"] = df["timestamp"].diff().dt.total_seconds().fillna(0)
        df["energy_interval"] = df["power"] * df["time_interval"]
        record_steps = record_steps or list(df["record_step"].unique())
        df = df[df["record_step"].isin(record_steps)]
        if return_data:
            return list(df.groupby("record_step_id")["energy_interval"].sum())
        return df["energy_interval"].sum()

    def get_total_time(self) -> float:
        """Returns the total profiling time in seconds

        Returns:
            time difference of first and last profiling sample in seconds

        """
        if not self.data:
            return 0.0
        df: pd.DataFrame = self.to_pandas()
        return (df["timestamp"].max() - df["timestamp"].min()).total_seconds()

    def get_avg_memory_usage(self, gpu_id: int | None = None) -> float:
        """Get average memory usage by gpu_id

        Args:
            gpu_id (int): the id of the gpu to calculate the avg memeory usage for (default = None, the first gpu id is used)

        Returns:
            the avgerage memory usage in MiB

        """
        if not self.data:
            return 0.0
        df: pd.DataFrame = self.to_pandas()
        gpu_id = gpu_id or df["gpu_id"].unique()[0]
        df = df[df["gpu_id"] == gpu_id]
        return df["memory"].mean()

    def get_time_series_plot(self) -> go.Figure:
        """Creates a plotly figure with the recorded time series data"""
        if not self.data:
            return None
        df: pd.DataFrame = self.to_pandas()
        fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])

        profiled_gpus: list[int] = self.get_profiled_gpus()
        n_colors: int = max(len(profiled_gpus), 2)
        color_scale: list[str] = px.colors.sample_colorscale(
            "Rainbow",
            [n / (n_colors - 1) for n in range(n_colors)],
        )
        for i, gpu_id in enumerate(profiled_gpus):
            gpu_df = df[df["gpu_id"] == gpu_id]
            for trace, unit in [("power", "W"), ("memory", "MiB")]:
                fig.add_trace(
                    go.Scatter(
                        x=gpu_df["timestamp"],
                        y=gpu_df[trace],
                        name=f"{trace.capitalize()} ({unit})",
                        mode="lines+markers",
                        legendgroup=gpu_id,
                        legendgrouptitle_text=f"GPU #{gpu_id}",
                        line=dict(
                            color=color_scale[i],
                            width=4,
                            dash="dot" if trace == "memory" else "solid",
                        ),
                    ),
                    secondary_y=(trace == "memory"),
                )

        max_timestamp: datetime = df["timestamp"].max()
        plt_record_steps: list[tuple[datetime, str]] = self.record_steps + [
            (max_timestamp, "."),
        ]
        unique_record_step_names = set([name for _, name in plt_record_steps])
        n_colors: int = max(len(unique_record_step_names), 2)
        color_scale: list[str] = px.colors.sample_colorscale(
            "viridis",
            [n / (n_colors - 1) for n in range(n_colors)],
        )
        colors = {
            name: color_scale[i] for i, name in enumerate(unique_record_step_names)
        }
        (last_ts, last_name) = plt_record_steps[0]
        for ts, name in plt_record_steps[1:]:
            fig.add_vrect(
                x0=last_ts,
                x1=ts,
                annotation_text=last_name,
                annotation_position="top left",
                line_width=0,
                opacity=0.25,
                fillcolor=colors[last_name],
            )
            last_ts, last_name = ts, name

        fig.update_layout(
            title="GPU Memory and Power Usage",
            legend=dict(groupclick="toggleitem"),
        )
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Power (W)", secondary_y=False)
        fig.update_yaxes(title_text="Memory (MiB)", secondary_y=True)

        return fig
