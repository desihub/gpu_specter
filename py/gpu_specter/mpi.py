"""This module provides helpers classes for managing data movement when using MPI.

Example usage:

# no mpi
python py/gpu_specter/mpi.py

# mpi: serial io
mpirun -n 2 python py/gpu_specter/mpi.py --mpi

# mpi: parallel io
# parallel io requires at least 3 MPI ranks (2 for IO and at least one for processing)
mpirun -n 4 python py/gpu_specter/mpi.py --mpi --async-io

"""

from abc import ABC, abstractmethod


class AbstractIOCoordinator(ABC):
    """Abstract base class for coordinating read/process/write steps of a program."""
    @classmethod
    def is_reader(cls, rank):
        return rank == cls._read_rank

    @classmethod
    def is_writer(cls, rank):
        return rank == cls._write_rank

    @classmethod
    def is_worker_root(cls, rank):
        return rank == cls._worker_root

    @classmethod
    def is_worker(cls, rank):
        return rank >= cls._worker_root

    @abstractmethod
    def read(self, func, payload):
        """Read input via func()."""
        raise NotImplementedError()

    @abstractmethod
    def process(self, func, payload):
        """Returns the result of func()."""
        raise NotImplementedError()

    @abstractmethod
    def write(self, func, payload):
        """Writes output by calling func(payload)."""
        raise NotImplementedError()


class NoMPIIOCoordinator(AbstractIOCoordinator):
    _read_rank = 0
    _write_rank = 0
    _worker_root = 0

    def __init__(self):
        """A NoMPIIOCoordinator coordinates read/process/write steps of a program when run without MPI."""
        self.comm = None
        self.rank = 0
        self.size = 1
        self.work_comm = None

    def read(self, func, payload):
        """Reads input via func().

        Args:
            func: a callable with no arguments.
            payload: a dummy value matching the return signature of func.

        Returns:
            result: the result of func().
        """
        return func()

    def process(self, func, payload):
        """Returns the result of func().

        Args:
            func: a callable with no arguments.
            payload: a dummy value matching the return signature of func.

        Returns:
            result: the result of func().
        """
        return func()

    def write(self, func, payload):
        """Writes output by calling func(payload).

        Args:
            func: a callable that takes payload as an argument.
            payload: the argument to pass to func.

        Rertuns:
            result: the result of func(payload).
        """
        return func(payload)


class SerialIOCoordinator(AbstractIOCoordinator):
    _read_rank = 0
    _write_rank = 0
    _worker_root = 0

    def __init__(self, comm):
        """A SerialIOCoordinator coordinates read/process/write steps of a program using MPI.
        The read and write steps will performed by the root rank of the provided MPI communicator.
        All ranks will peform the process step.

        Args:
            comm: an MPI communicator.
        """
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        self.work_comm = comm

    def read(self, func, payload):
        """Reads input via func() on reader rank. Meanwhile, other ranks proceed and are not blocked.

        Args:
            func: a callable with no arguments.
            payload: a dummy value matching the return signature of func.

        Returns:
            result: all ranks return dummy payload except the reader rank which returns
                the result of func().
        """
        if SerialIOCoordinator.is_reader(self.rank):
            result = func()
        else:
            result = payload
        return result

    def process(self, func, payload):
        """Returns the result of func(). Non-worker ranks return the provided dummy payload.

        Args:
            func: a callable with no arguments.
            payload: a dummy value matching the return signature of func.

        Returns:
            result: workers return the result of func(). Non-worker ranks return the provided dummy payload.
        """
        if SerialIOCoordinator.is_worker(self.rank):
            result = func()
        else:
            result = payload
        return result

    def write(self, func, payload):
        """Writes output by calling func(payload) from the writer rank. Meanwhile, other ranks
        proceed and are not blocked.

        Args:
            func: a callable that takes payload as an argument.
            payload: on writer rank, the argument to pass to func. otherwise, a dummy
                value matching the return signature of func.

        Rertuns:
            result: all ranks return dummy payload except the writer rank, which returns
                the result of func(payload).
        """
        if SerialIOCoordinator.is_writer(self.rank):
            result = func(payload)
        else:
            result = payload
        return result


class ParallelIOCoordinator(AbstractIOCoordinator):
    _read_rank = 0
    _write_rank = 1
    _worker_root = 2

    def __init__(self, comm):
        """
        A ParallelIOCoordinator coordinates read/process/write steps of a program using MPI.
        The read and write steps are performed on dedicated ranks which allows for parallel compute and IO
        when processing multiple tasks in series. The read and write steps between tasks will be interleaved
        with processing.

        Args:
            comm: an MPI communicator.
        """
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        assert self.size >= 3, "ParallelIOCoordinator requires at least 3 MPI ranks"

        # Initialize work comm
        self.work_comm = None
        work_group = self.comm.group.Excl(
            [ParallelIOCoordinator._read_rank, ParallelIOCoordinator._write_rank]
        )
        if ParallelIOCoordinator.is_worker(self.rank):
            self.work_comm = comm.Create_group(work_group)

    def read(self, func, payload):
        """Reads input via func() on reader rank and sends the result to worker root. Meanwhile,
        other ranks proceed and are not blocked. The worker root will proceed after receiving.

        Args:
            func: a callable with no arguments.
            payload: a dummy value matching the return signature of func.

        Returns:
            result: all ranks return dummy payload except the worker root which returns
                the result of func().
        """
        if ParallelIOCoordinator.is_reader(self.rank):
            # read input via func()
            result = func()
            # send the result to worker root
            self.comm.send(result, dest=ParallelIOCoordinator._worker_root, tag=1)
            # dummy payload passes through
            result = payload
        elif ParallelIOCoordinator.is_worker_root(self.rank):
            # receive the result from reader
            result = self.comm.recv(source=ParallelIOCoordinator._read_rank, tag=1)
        else:
            # dummy payload passes through
            result = payload
        return result

    def process(self, func, payload):
        """Returns the result of func(). Non-worker ranks return the provided dummy payload.

        Args:
            func: a callable with no arguments.
            payload: a dummy value matching the return signature of func.

        Returns:
            result: workers return the result of func(). Non-worker ranks return the provided dummy payload.
        """
        if ParallelIOCoordinator.is_worker(self.rank):
            result = func()
        else:
            result = payload
        return result

    def write(self, func, payload):
        """Writes output by sending payload from worker root to writer rank. The writer rank will then
        call the provided func with the payload argument to write output. Meanwhile, other ranks
        proceed and are not blocked. The worker root will proceed after sending payload.

        Args:
            func: a callable that takes payload as an argument.
            payload: on worker_root, the argument to pass to func. otherwise, a dummy
                value matching the return signature of func.

        Rertuns:
            result: all ranks return dummy payload except the write_rank which returns
                the result of func(payload).
        """
        if ParallelIOCoordinator.is_worker_root(self.rank):
            # receive the dummy payload from writer
            result = self.comm.recv(source=ParallelIOCoordinator._write_rank, tag=2)
            # send the actual payload to writer
            self.comm.send(payload, dest=ParallelIOCoordinator._write_rank, tag=3)
        elif ParallelIOCoordinator.is_writer(self.rank):
            # send the dummy payload to worker root
            self.comm.send(payload, dest=ParallelIOCoordinator._worker_root, tag=2)
            # receive actual payload from worker root
            payload = self.comm.recv(source=ParallelIOCoordinator._worker_root, tag=3)
            # call func with payload
            result = func(payload)
        else:
            # dummy payload passes through
            result = payload
        return result


def example_program():
    """This function demonstrates how to use the IOCoordinator classes."""
    import argparse

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mpi", action="store_true", help="use mpi")
    parser.add_argument("--async-io", action="store_true", help="async io")
    args = parser.parse_args()

    # initialize IO coordinator based on arguments
    if args.mpi:
        from mpi4py import MPI

        if args.async_io:
            coordinator = ParallelIOCoordinator(MPI.COMM_WORLD)
        else:
            coordinator = SerialIOCoordinator(MPI.COMM_WORLD)
    else:
        coordinator = NoMPIIOCoordinator()
    rank, size = coordinator.rank, coordinator.size

    # define example read/process/write functions
    def generate_numbers(task_index):
        return [task_index for i in range(10)]

    def distributed_sum(numbers, comm):
        # broadcast data
        if comm is not None:
            numbers = comm.bcast(numbers, root=0)
            numbers = numbers[comm.rank :: comm.size]
        # each rank computes a subtotal
        subtotal = sum(numbers)
        # gather subtotals
        if comm is not None:
            subtotals = comm.gather(subtotal, root=0)
        else:
            subtotals = [
                subtotal,
            ]
        # combine subtotals
        if comm is not None and comm.rank > 0:
            result = None
        else:
            result = sum(subtotals)
        return result

    def print_result(task_index, result):
        print(f"{rank=} {task_index=} {result=}")

    # iterate over tasks
    for task_index in range(5):
        # generate data
        numbers = coordinator.read(lambda: generate_numbers(task_index), None)
        # distributed_sum
        result = coordinator.process(
            lambda: distributed_sum(numbers, coordinator.work_comm), None
        )
        # print result
        coordinator.write(lambda result: print_result(task_index, result), result)

    if coordinator.comm is not None:
        coordinator.comm.barrier()


if __name__ == "__main__":
    example_program()