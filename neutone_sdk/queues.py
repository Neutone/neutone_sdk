import logging
import os
from typing import Tuple

import torch as tr
from torch import Tensor

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class CircularInplaceTensorQueue:
    def __init__(self,
                 n_ch: int,
                 max_size: int,
                 use_debug_mode: bool = True) -> None:
        """
        Creates a FIFO queue designed for audio data that does not allocate any memory during normal use and performs
        as few memory operations as possible. The queue is also compatible with converting to TorchScript.
        """
        self.use_debug_mode = use_debug_mode
        self.max_size = max_size
        self.queue = tr.zeros((n_ch, max_size))
        self.start_idx = 0
        self.end_idx = 0
        self.size = 0

    def _calc_push_indices(self, in_n: int) -> Tuple[int, int, int, int]:
        """
        Calculates the indices to place new data of length in_n into the queue. Since it's a circular queue this can
        mean wrapping around once past the end of the queue depending on the contents of the queue at that moment in
        time. As a result, we define two possible index ranges for pushing data: start_1:end_1 and start_2:end_2
        if wrapping occurs, otherwise end_1 == start_2 == end_2

        Returns:
            Tuple[int, int, int, int]: start_1, end_1, start_2, end_2
        """
        if self.use_debug_mode:
            assert 0 < in_n < self.max_size
        start_1 = self.end_idx
        if start_1 == self.max_size:
            start_1 = 0
        end_2 = start_1 + in_n
        if end_2 > self.max_size:
            end_2 = end_2 % self.max_size
        end_1 = end_2
        start_2 = end_2
        if end_2 < start_1:
            end_1 = self.max_size
            start_2 = 0
        return start_1, end_1, start_2, end_2

    def push(self, x: Tensor) -> None:
        """
        Pushes the contents of x to the end of the queue. If the queue does not have adequate space left, the contents
        of the queue will be overwritten, starting at the head of the queue.
        """
        if self.use_debug_mode:
            assert x.ndim == self.queue.ndim
            assert x.size(0) == self.queue.size(0)
        in_n = x.size(1)
        if in_n >= self.max_size:
            self.queue[:, :] = x[:, -self.max_size:]
            self.start_idx = 0
            self.end_idx = self.max_size
            self.size = self.max_size
            return
        if in_n < 1:
            return
        start_1, end_1, start_2, end_2 = self._calc_push_indices(in_n)
        n_1 = end_1 - start_1
        self.queue[:, start_1:end_1] = x[:, 0:n_1]
        if n_1 < in_n:
            self.queue[:, start_2:end_2] = x[:, n_1:]
        self.end_idx = end_2
        self.size = min(self.size + in_n, self.max_size)
        if self.size == self.max_size:
            self.start_idx = self.end_idx

    def _calc_pop_indices(self, out_n: int) -> Tuple[int, int, int, int]:
        """
        Calculates the indices to pop data of length out_n from the queue. Since it's a circular queue this can
        mean wrapping around once past the end of the queue depending on the contents of the queue at that moment in
        time. As a result, we define two possible index ranges for popping data: start_1:end_1 and start_2:end_2
        if wrapping occurs, otherwise end_1 == start_2 == end_2

        Returns:
            Tuple[int, int, int, int]: start_1, end_1, start_2, end_2
        """
        out_n = min(out_n, self.size)
        if self.use_debug_mode:
            assert out_n > 0
        start_1 = self.start_idx
        if start_1 == self.max_size:
            start_1 = 0
        end_2 = start_1 + out_n
        if end_2 > self.max_size:
            end_2 = end_2 % self.max_size
        end_1 = end_2
        start_2 = end_2
        if end_2 <= start_1:
            end_1 = self.max_size
            start_2 = 0
        return start_1, end_1, start_2, end_2

    def pop(self, out: Tensor) -> int:
        """
        Attempts to fill the out tensor with data popped from the head of the queue. Begins filling the out tensor at
        index 0. If the out tensor is bigger than the number of items in the queue, fills the tensor as much as
        possible.

        Returns:
            int: the number of items successfully popped from the queue.
        """
        # TODO(cm): remove duplicate code using fill
        if self.use_debug_mode:
            assert out.ndim == self.queue.ndim
            assert out.size(0) == self.queue.size(0)
        if self.is_empty():
            return 0
        out_n = out.size(1)
        if out_n < 1:
            return 0
        start_1, end_1, start_2, end_2 = self._calc_pop_indices(out_n)
        n_1 = end_1 - start_1
        n_2 = end_2 - start_2
        removed_n = n_1 + n_2
        if self.use_debug_mode:
            assert 0 < n_1 <= self.size
            assert 0 <= n_2 < self.size
            assert removed_n <= self.size
        out[:, 0:n_1] = self.queue[:, start_1:end_1]
        if n_2 > 0:
            out[:, n_1:removed_n] = self.queue[:, start_2:end_2]
        self.start_idx = end_2
        self.size -= removed_n
        if self.use_debug_mode:
            if self.size == 0:
                assert self.start_idx == self.end_idx
        return removed_n

    def fill(self, out: Tensor) -> int:
        """
        Attempts to fill the out tensor with data from the head of the queue. Begins filling the out tensor at index 0.
        If the out tensor is bigger than the number of items in the queue, fills the tensor as much as possible. Does
        not remove any elements from the queue.

        Returns:
            int: the number of items successfully filled from the queue.
        """
        if self.use_debug_mode:
            assert out.ndim == self.queue.ndim
            assert out.size(0) == self.queue.size(0)
        if self.is_empty():
            return 0
        out_n = out.size(1)
        if out_n < 1:
            return 0
        start_1, end_1, start_2, end_2 = self._calc_pop_indices(out_n)
        n_1 = end_1 - start_1
        n_2 = end_2 - start_2
        filled_n = n_1 + n_2
        if self.use_debug_mode:
            assert 0 < n_1 <= self.size
            assert 0 <= n_2 < self.size
            assert filled_n <= self.size
        out[:, 0:n_1] = self.queue[:, start_1:end_1]
        if n_2 > 0:
            out[:, n_1:filled_n] = self.queue[:, start_2:end_2]
        return filled_n

    def is_empty(self) -> bool:
        return self.size == 0

    def is_full(self) -> bool:
        return self.size == self.max_size

    def reset(self) -> None:
        self.start_idx = 0
        self.end_idx = 0
        self.size = 0
