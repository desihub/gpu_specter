import cupy.prof

class NoMPIComm(object):
    READ_RANK = 0
    WRITE_RANK = 0
    EXTRACT_ROOT = 0

    def __init__(self):
        self.comm = None
        self.rank = 0
        self.size = 1

        self.extract_comm = None

    def is_extract_rank(self):
        return True

    def is_extract_root(self):
        return True

    def read(self, func, data):
        return func()

    def write(self, func, data):
        func(data)

class SyncIOComm(object):
    READ_RANK = 0
    WRITE_RANK = 0
    EXTRACT_ROOT = 0

    def __init__(self, comm):
        """Synchronous communication/extraction manager.

        Args:
            comm: the parent MPI communicator. Must have at least 1 rank
        """
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size

        self.extract_comm = comm

    def is_extract_rank(self):
        return self.comm.rank >= SyncIOComm.EXTRACT_ROOT

    def is_extract_root(self):
        return self.comm.rank == SyncIOComm.EXTRACT_ROOT

    @cupy.prof.TimeRangeDecorator("SyncIOComm.read")
    def read(self, func, data):
        if self.comm.rank == SyncIOComm.READ_RANK:
            data = func()
        return data

    @cupy.prof.TimeRangeDecorator("SyncIOComm.write")
    def write(self, func, data):
        if self.comm.rank == SyncIOComm.WRITE_RANK:
            func(data)

class AsyncIOComm(object):
    READ_RANK = 0
    WRITE_RANK = 1
    EXTRACT_ROOT = 2

    def __init__(self, comm):
        """Asynchronous communication/extraction manager.

        Args:
            comm: the parent MPI communicator. Must have at least 3 ranks
        """
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        assert self.size >= 3

        # Initialize extraction comm
        # READ/WRITE ranks have MPI_COMM_NULL?
        self.extract_comm = None
        self.extract_group = self.comm.group.Excl(
            [AsyncIOComm.READ_RANK, AsyncIOComm.WRITE_RANK])
        if self.is_extract_rank():
            self.extract_comm = comm.Create_group(self.extract_group)

    def is_extract_rank(self):
        """Returns True if this MPI rank is part of the extraction group.
        Otherwise returns False.
        """
        return self.comm.rank >= AsyncIOComm.EXTRACT_ROOT

    def is_extract_root(self):
        return self.comm.rank == AsyncIOComm.EXTRACT_ROOT

    @cupy.prof.TimeRangeDecorator("AsyncIOComm.read")
    def read(self, func, data):
        """READ_RANK will call `func` and send the result to EXTRACT_ROOT.
        EXTRACT_ROOT returns the result from READ_RANK. All other ranks return 
        the provided default `data`.

        Args:
            func (method): function to be called by READ_RANK that reads input data.
            data: default placeholder for data. mostly here for symmetry with `write(...)`

        Returns: 
            data: either the provided default data or data returned by func on EXTRACT_ROOT/READ_RANK
        """
        if self.comm.rank == AsyncIOComm.READ_RANK:
            data = func()
            self.comm.send(data, dest=AsyncIOComm.EXTRACT_ROOT, tag=1)
        elif self.comm.rank == AsyncIOComm.EXTRACT_ROOT:
            data = self.comm.recv(source=AsyncIOComm.READ_RANK, tag=1)
        return data

    @cupy.prof.TimeRangeDecorator("AsyncIOComm.write")
    def write(self, func, data):
        """EXTRACT_ROOT sends `data` to WRITE_RANK. WRITE_RANK calls `func` to
        write `data`. This is a no-op for all other ranks.

        Args:
            func (method): function that writes `data`.
            data: the data to be written by `func`.
        """
        if self.comm.rank == AsyncIOComm.EXTRACT_ROOT:
            self.comm.send(data, dest=AsyncIOComm.WRITE_RANK, tag=2)
        elif self.comm.rank == AsyncIOComm.WRITE_RANK:
            data = self.comm.recv(source=AsyncIOComm.EXTRACT_ROOT, tag=2)
            func(data)


class AnotherAsyncIOComm(object):
    READ_RANK = 0
    WRITE_RANK = 1
    EXTRACT_READ_RANK = 2
    EXTRACT_WRITE_RANK = 3


    def __init__(self, comm):
        """Asynchronous communication/extraction manager.

        Args:
            comm: the parent MPI communicator. Must have at least 4 ranks
        """
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        assert self.size >= 4

        # Initialize extraction comm
        # READ/WRITE ranks have MPI_COMM_NULL?
        self.extract_comm = None
        self.extract_group = self.comm.group.Excl(
            [AnotherAsyncIOComm.READ_RANK, AnotherAsyncIOComm.WRITE_RANK])
        if self.is_extract_rank():
            self.extract_comm = comm.Create_group(self.extract_group)

    def is_extract_rank(self):
        """Returns True if this MPI rank is part of the extraction group.
        Otherwise returns False.
        """
        return self.comm.rank >= AnotherAsyncIOComm.EXTRACT_READ_RANK

    def is_extract_root(self):
        # WIP: what should the extract root be in this case?
        raise NotImplementedError()

    @cupy.prof.TimeRangeDecorator("AnotherAsyncIOComm.read")
    def read(self, func, data):
        """READ_RANK will call `func` and send the result to EXTRACT_ROOT.
        EXTRACT_ROOT returns the result from READ_RANK. All other ranks return 
        the provided default `data`.

        Args:
            func (method): function to be called by READ_RANK that reads input data.
            data: default placeholder for data. mostly here for symmetry with `write(...)`

        Returns: 
            data: either the provided default data or data returned by func on EXTRACT_ROOT/READ_RANK
        """
        if self.comm.rank == AnotherAsyncIOComm.READ_RANK:
            data = func()
            self.comm.send(data, dest=AnotherAsyncIOComm.EXTRACT_READ_RANK, tag=1)
        elif self.comm.rank == AnotherAsyncIOComm.EXTRACT_READ_RANK:
            data = self.comm.recv(source=AnotherAsyncIOComm.READ_RANK, tag=1)
        return data

    @cupy.prof.TimeRangeDecorator("AnotherAsyncIOComm.write")
    def write(self, func, data):
        """EXTRACT_ROOT sends `data` to WRITE_RANK. WRITE_RANK calls `func` to
        write `data`. This is a no-op for all other ranks.

        Args:
            func (method): function that writes `data`.
            data: the data to be written by `func`.
        """
        if self.comm.rank == AnotherAsyncIOComm.EXTRACT_ROOT:
            self.comm.send(data, dest=AnotherAsyncIOComm.EXTRACT_WRITE_RANK, tag=2)
        elif self.comm.rank == AnotherAsyncIOComm.EXTRACT_WRITE_RANK:
            data = self.comm.recv(source=AnotherAsyncIOComm.EXTRACT_ROOT, tag=2)
            func(data)
