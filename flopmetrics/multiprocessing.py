"""Multiprocessing utilities."""
from abc import ABC
from itertools import zip_longest
from multiprocessing import Manager, Process, Queue
import multiprocessing
import os
from typing import Any, Callable, List, Tuple, Union
import warnings


class ResultHandler(ABC):
    """
    ResultHandler for Multiprocessing Results.
    Works like a Queue but can be implemented in different ways.
    """

    def __init__(self):
        self.column_names: Tuple[str, ...] = ()
        self.dtypes: Tuple[type, ...] = ()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_columns(self, names: Tuple[str, ...], dtypes: Tuple[type, ...] = None):
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
    """
    Multiprocessing Queue ResultHandler.
    Wrapper for a multiprocessing.Queue
    """

    def __init__(self):
        super().__init__()
        self.queue = Queue()

    def put(self, data):
        return self.queue.put(data)

    def get(self) -> Union[Tuple[Any, ...], None]:
        return self.queue.get()

    def get_all(self) -> List[Tuple[Any, ...]]:
        data = []
        for el in iter(self.queue.get, None):
            data.append(el)
        return data


class FileCacheResultHandler(ResultHandler):
    """
    File Cache ResultHandler.
    Writes data to a file and reads it from a file on disk.
    Args:
        file_path (str): path to the file
        force (bool): if True the file will be overwritten if it already exists
    """

    def __init__(self, file_path: str, force: bool = False):
        super().__init__()
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if force and os.path.isfile(file_path):
            os.remove(file_path)
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
            warnings.warn(f"Warning: File {file_path} already exists and is not empty")

        self.file_obj = None

    def __enter__(self):
        self.file_obj = open(self.file_path, "a", encoding="utf-8")
        return self

    def __exit__(self, *args):
        self.file_obj.close()

    def set_columns(self, header: Tuple[str, ...], dtypes: Tuple[type, ...] = None):
        super().set_columns(header, dtypes)
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            return
        with open(self.file_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(",".join(header) + "\n")

    def put(self, data):
        if not isinstance(data, tuple):
            return
        with open(self.file_path, "a", encoding="utf-8") as file_obj:
            file_obj.write(",".join([str(el) for el in data]) + "\n")

    def _read_line_tuple(self, line: str) -> Tuple[Any, ...]:
        return tuple(
            t(el) for el, t in zip_longest(line.strip().split(","), self.dtypes, fillvalue=str)
        )

    def get(self) -> Union[Tuple[Any, ...], None]:
        with open(self.file_path, "r", encoding="utf-8") as file_obj:
            lines = file_obj.readlines()[1 if len(self.column_names) > 0 else 0 :]
            if not lines:
                return None
            return self._read_line_tuple(lines[-1])

    def get_all(self) -> List[Tuple[Any, ...]]:
        with open(self.file_path, "r", encoding="utf-8") as file_obj:
            lines = file_obj.readlines()[1 if len(self.column_names) > 0 else 0 :]
            data = [self._read_line_tuple(line) for line in lines]
        return data


def start_separate_process(target: Callable, other_args: List) -> Any:
    """
    Function to start a separate process for the evaluation
    Args:
        target (Callable): target function
            the target function should take a result_queue as the first argument and put the result in the queue
        other_args (List): list of arguments for the target function after the result_queue
    Returns:
        Process: process object
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
