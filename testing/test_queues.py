import logging
import os
import random

import torch as tr
from tqdm import tqdm

from neutone_sdk import CircularInplaceTensorQueue

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def test_circular_queue() -> None:
    trials = 100
    iters = 100
    max_queue_len = 19
    random.seed(26)
    for _ in tqdm(range(trials)):
        in_list = []
        out_list = []
        queue_len = random.randint(1, max_queue_len)
        queue = CircularInplaceTensorQueue(1, queue_len)
        for idx in range(iters):
            if not queue.is_full():
                block = tr.full((1, random.randint(1, queue_len - queue.size)), idx + 1)
                queue.push(block)
                in_list += block[0, :].tolist()

            if not queue.is_empty():
                block = tr.zeros((1, random.randint(1, queue.size)))
                queue.pop(block)
                out_list += block[0, :].int().tolist()

        assert len(in_list) >= len(out_list)
        assert in_list[:len(out_list)] == out_list
        assert queue.size == len(in_list) - len(out_list)
