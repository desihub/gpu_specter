import cupy.prof

class SyncIOComm(object):
    READ_RANK = 0
    WRITE_RANK = 0
    EXTRACT_ROOT = 0

    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size

        self.extract_comm = comm

    def is_extract_rank(self):
        return self.comm.rank >= SyncIOComm.EXTRACT_ROOT

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


class AsyncIOComm2(object):
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
            [AsyncIOComm2.READ_RANK, AsyncIOComm2.WRITE_RANK])
        if self.is_extract_rank():
            self.extract_comm = comm.Create_group(self.extract_group)

    def is_extract_rank(self):
        """Returns True if this MPI rank is part of the extraction group.
        Otherwise returns False.
        """
        return self.comm.rank >= AsyncIOComm2.EXTRACT_READ_RANK

    @cupy.prof.TimeRangeDecorator("AsyncIOComm2.read")
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
        if self.comm.rank == AsyncIOComm2.READ_RANK:
            data = func()
            self.comm.send(data, dest=AsyncIOComm2.EXTRACT_READ_RANK, tag=1)
        elif self.comm.rank == AsyncIOComm2.EXTRACT_READ_RANK:
            data = self.comm.recv(source=AsyncIOComm2.READ_RANK, tag=1)
        return data

    @cupy.prof.TimeRangeDecorator("AsyncIOComm2.write")
    def write(self, func, data):
        """EXTRACT_ROOT sends `data` to WRITE_RANK. WRITE_RANK calls `func` to
        write `data`. This is a no-op for all other ranks.

        Args:
            func (method): function that writes `data`.
            data: the data to be written by `func`.
        """
        if self.comm.rank == AsyncIOComm2.EXTRACT_ROOT:
            self.comm.send(data, dest=AsyncIOComm2.EXTRACT_WRITE_RANK, tag=2)
        elif self.comm.rank == AsyncIOComm2.EXTRACT_WRITE_RANK:
            data = self.comm.recv(source=AsyncIOComm2.EXTRACT_ROOT, tag=2)
            func(data)
