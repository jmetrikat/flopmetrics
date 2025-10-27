"""Multiprocessing utilities."""
from abc import ABC
from itertools import zip_longest
from multiprocessing import Manager, Process, Queue
import multiprocessing
import os
from typing import Any, Callable, List, Tuple, Union
import warnings


class ResultHandler(ABC):
    """Base class for handling multiprocessing results."""

    def __init__(self):
        self.column_names: Tuple[str, ...] = ()
        self.dtypes: Tuple[type, ...] = ()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_columns(self, names: Tuple[str, ...], dtypes: Tuple[type, ...] = None):
        """
        Set column names and types.

        Args:
            names: Column names.
            dtypes: Column types.
        """
        assert len(names) == len(dtypes), "Length of names and dtypes must be equal"
        self.column_names = names
        self.dtypes = dtypes

    def put(self, data: Tuple[Any, ...]):
        raise NotImplementedError

    def get(self) -> Union[Tuple[Any, ...], None]:
        raise NotImplementedError

    def get_all(self) -> List[Tuple[Any, ...]]:
        raise NotImplementedError


class MPQueueResultHandler(ResultHandler):
    """Queue-based result handler using multiprocessing.Queue."""

    def __init__(self):
        """Initialize queue-based result handler."""
        super().__init__()
        self.queue = Queue()

    def put(self, data):
        """Put data into queue."""
        return self.queue.put(data)

    def get(self) -> Union[Tuple[Any, ...], None]:
        """Get data from queue."""
        return self.queue.get()

    def get_all(self) -> List[Tuple[Any, ...]]:
        """Get all data from queue."""
        data = []
        for el in iter(self.queue.get, None):
            data.append(el)
        return data


class FileCacheResultHandler(ResultHandler):
    """File-based result handler that writes to and reads from disk."""

    def __init__(self, file_path: str, force: bool = False):
        """
        Initialize file-based result handler.

        Args:
            file_path: Path to file.
            force: If True, overwrite existing file.
        """
        super().__init__()
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if force and os.path.isfile(file_path):
            os.remove(file_path)
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
            warnings.warn(f"Warning: File {file_path} already exists and is not empty")

        self.file_obj = None

    def __enter__(self):
        """Open file for writing."""
        self.file_obj = open(self.file_path, "a", encoding="utf-8")
        return self

    def __exit__(self, *args):
        """Close file."""
        self.file_obj.close()

    def set_columns(self, header: Tuple[str, ...], dtypes: Tuple[type, ...] = None):
        """
        Set columns and write header if file is new.

        Args:
            header: Column names.
            dtypes: Column types.
        """
        super().set_columns(header, dtypes)
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            return
        with open(self.file_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(",".join(header) + "\n")

    def put(self, data):
        """Append data to file."""
        if not isinstance(data, tuple):
            return
        with open(self.file_path, "a", encoding="utf-8") as file_obj:
            file_obj.write(",".join([str(el) for el in data]) + "\n")

    def _read_line_tuple(self, line: str) -> Tuple[Any, ...]:
        """Parse line from CSV into typed tuple."""
        return tuple(
            t(el) for el, t in zip_longest(line.strip().split(","), self.dtypes, fillvalue=str)
        )

    def get(self) -> Union[Tuple[Any, ...], None]:
        """Get last line from file."""
        with open(self.file_path, "r", encoding="utf-8") as file_obj:
            lines = file_obj.readlines()[1 if len(self.column_names) > 0 else 0 :]
            if not lines:
                return None
            return self._read_line_tuple(lines[-1])

    def get_all(self) -> List[Tuple[Any, ...]]:
        """Get all data from file."""
        with open(self.file_path, "r", encoding="utf-8") as file_obj:
            lines = file_obj.readlines()[1 if len(self.column_names) > 0 else 0 :]
            data = [self._read_line_tuple(line) for line in lines]
        return data


def start_separate_process(target: Callable, other_args: List) -> Any:
    """
    Start a separate process for evaluation.

    Args:
        target: Target function that takes a result_queue as the first argument.
        other_args: List of additional arguments for the target function.

    Returns:
        Result from the queue.
    """
    multiprocessing.set_start_method("spawn", force=True)
    manager = Manager()
    queue = manager.Queue()
    process = Process(
        target=target,
        args=[queue] + other_args,
    )
    process.start()
    process.join()
    return queue.get()
