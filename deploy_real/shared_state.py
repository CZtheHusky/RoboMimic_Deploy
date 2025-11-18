from multiprocessing import shared_memory
from typing import Optional

import numpy as np


# 4 bytes seq + 4 bytes padding for alignment
_HEADER_BYTES = 8


class SharedStateBuffer:
    """
    Small helper around multiprocessing.shared_memory for a single
    float32 state vector with a seqlock-style uint32 header.

    Layout in shared memory:
        uint32 seq   (offset 0)
        uint32 pad   (offset 4, unused)
        float32[data_dim]  (offset 8)
    """

    def __init__(self, shm: shared_memory.SharedMemory, state_dim: int):
        self.shm = shm
        self.state_dim = state_dim
        # Header: 1 x uint32 for seq; the second uint32 is padding
        self._seq = np.ndarray((1,), dtype=np.uint32, buffer=self.shm.buf[:4])
        # Data region
        self.data = np.ndarray(
            (state_dim,),
            dtype=np.float32,
            buffer=self.shm.buf[_HEADER_BYTES : _HEADER_BYTES + 4 * state_dim],
        )

    @classmethod
    def create(cls, name: str, state_dim: int) -> "SharedStateBuffer":
        """Create a new shared memory segment."""
        size = _HEADER_BYTES + 4 * state_dim
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        buf = cls(shm, state_dim)
        buf._seq[0] = 0
        buf.data[...] = 0.0
        return buf

    @classmethod
    def attach(cls, name: str, state_dim: int) -> "SharedStateBuffer":
        """Attach to an existing shared memory segment."""
        shm = shared_memory.SharedMemory(name=name, create=False)
        return cls(shm, state_dim)

    def write(self, arr: np.ndarray) -> None:
        """
        Write a full state vector atomically using a simple seqlock:
          seq = seq + 1  (odd -> writing)
          copy data
          seq = seq + 1  (even -> done)
        """
        if arr.shape != (self.state_dim,):
            raise ValueError(f"Expected shape {(self.state_dim,)}, got {arr.shape}")

        seq = int(self._seq[0])
        self._seq[0] = seq + 1
        self.data[...] = arr
        self._seq[0] = seq + 2

    def read(self, out: Optional[np.ndarray] = None, max_retry: int = 3) -> Optional[np.ndarray]:
        """
        Read a consistent snapshot with up to max_retry attempts.
        Returns a numpy array (view or copy), or None if it failed
        to obtain a consistent snapshot.
        """
        if out is not None and out.shape != (self.state_dim,):
            raise ValueError(f"Expected out shape {(self.state_dim,)}, got {out.shape}")

        for _ in range(max_retry):
            seq1 = int(self._seq[0])
            if seq1 & 1:
                # Writer in progress
                continue

            if out is None:
                buf = self.data.copy()
            else:
                out[...] = self.data
                buf = out

            seq2 = int(self._seq[0])
            if seq1 == seq2 and (seq2 & 1) == 0:
                return buf

        return None

    def close(self) -> None:
        self.shm.close()

    def unlink(self) -> None:
        """
        Remove the shared memory segment from the system.
        Should only be called once by the creating process.
        """
        self.shm.unlink()

