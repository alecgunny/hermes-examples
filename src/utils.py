import logging
import os
import re
import shutil
import subprocess
import sys
import time
from io import StringIO
from threading import Event, Thread
from typing import Sequence, Union

from rich.progress import BarColumn, Progress, TimeElapsedColumn
from rich.logging import RichHandler


def get_logger():
    logger = logging.getLogger("hermes-example")
    logger.setLevel(logging.INFO)

    handler = RichHandler(show_time=False, show_level=False, show_path=False)
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d    %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    sys.stderr = StringIO()
    return logger


def clear_repo(repo_path):
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)


class LocalProgressMonitor(Progress):
    """Progress bar subclass for measuring GPU utilization"""

    def __init__(
        self,
        num_inferences: int,
        inference_sampling_rate: float,
        gpu_ids: Union[int, Sequence[int], None] = None,
        max_throughput: int = 1000,
        *args
    ) -> None:
        if len(args) == 0:
            super().__init__(
                "[progress.description]{task.description}",
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
            )
        else:
            super().__init__(*args)

        self.percentage_task_id = self.add_task(
            "[red]Waiting for server to come online",
            total=num_inferences,
            start=False,
        )
        self.throughput_task_id = self.add_task(
            "[blue]Throughput", total=max_throughput, start=False
        )
        self.inference_sampling_rate = inference_sampling_rate

        # if we specified gpu ids to monitor, create a progress bar
        # for each of them that shows their utilization level
        self.gpu_ids = {}
        if gpu_ids is not None:
            # normalize a single integer gpu id to a sequence for simplicity
            if isinstance(gpu_ids, int):
                gpu_ids = [gpu_ids]

            host_ids = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            mapped_ids = []
            if host_ids is not None:
                host_ids = list(map(int, host_ids.split(",")))
                mapped_ids = [host_ids[i] for i in gpu_ids]
            else:
                mapped_ids = gpu_ids

            ids = sorted(zip(mapped_ids, gpu_ids))
            for host_id, gpu_id in ids:
                self.gpu_ids[host_id] = self.add_task(
                    f"[green]GPU {gpu_id} utilization", total=100, start=False
                )

        # use an event to stop the internal loop
        self._stop_event = self._gpu_monitor = self._start_time = None

    def begin(self):
        self.tasks[self.percentage_task_id].description = (
            "[red]Completed requests"
        )
        self.start_task(self.percentage_task_id)
        self.start_task(self.throughput_task_id)

        if len(self.gpu_ids) > 0:
            for task_id in self.gpu_ids.values():
                self.start_task(task_id)

            self._stop_event = Event()
            self._gpu_monitor = Thread(target=self.monitor_utilization)
            self._gpu_monitor.start()

        self._start_time = time.time()

    def __exit__(self, *exc_args):
        try:
            # if we're not exiting because anything went wrong,
            # then wait for all pending inference requests
            # to return before closing the context
            if exc_args[0] is None:
                while not self.tasks[self.percentage_task_id].finished:
                    time.sleep(0.1)

            # if we were monitoring any GPUs, stop monitoring now
            # by setting the stop event and waiting for the last
            # loop to complete
            if self._gpu_monitor is not None:
                self._stop_event.set()
                self._gpu_monitor.join()

            # reset these attributes
            self._stop_event = self._gpu_monitor = self._start_time = None
        finally:
            # now run the normal context exiting buisness
            super().__exit__(*exc_args)

    def monitor_utilization(self) -> None:
        # sort the gpu ids since nvidia-smi returns them
        # in numeric order and it makes it easier to zip later
        gpu_ids = sorted(self.gpu_ids)
        cmd = "nvidia-smi -i " + ",".join(map(str, gpu_ids))

        # now continue updating our progress bar until
        # the external thread sets our stop event
        while not self._stop_event.is_set():
            # strip the percentages from the output of nvidia-smi
            response = subprocess.run(cmd, shell=True, capture_output=True)
            nv_smi = response.stdout.decode()
            percentages = re.findall("[0-9]{1,3}(?=%)", nv_smi)

            # set this to the value of the corresponding progress bar
            for percentage, gpu_id in zip(percentages, gpu_ids):
                task_id = self.gpu_ids[gpu_id]
                self.update(task_id, completed=int(percentage))

            # sleep to avoid having this thread overwhelm everything
            time.sleep(0.2)

    def __call__(self, response, *args):
        self.update(self.percentage_task_id, advance=len(response))

        speed = self.tasks[self.percentage_task_id].speed or 0
        speed *= self.inference_sampling_rate
        self.update(self.throughput_task_id, completed=speed)
        return response[0, 0]
