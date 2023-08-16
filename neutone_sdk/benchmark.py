import os
import logging
import timeit
import itertools
from typing import List
import click
import torch
from neutone_sdk.utils import load_neutone_model
import numpy as np

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


@click.group()
def cli():
    """This is needed to make a group command with click."""
    pass


@cli.command()
@click.option("--model_file", help="Path to model file")
@click.option(
    "--buffer_size",
    default=(128, 256, 512, 1024, 2048),
    multiple=True,
    help="Buffer sizes to benchmark",
)
@click.option(
    "--sample_rate",
    default=(48000,),
    multiple=True,
    help="Sample rates to benchmark",
)
@click.option("--repeat", default=10, help="How many times to repeat the benchmark")
@click.option(
    "--number",
    default=30,
    help="How many times to run each benchmark for each repetition",
)
@click.option("--num_threads", default=1, help="num_threads to use for the benchmark")
@click.option(
    "--num_interop_threads",
    default=1,
    help="num_interop_threads to use for the benchmark",
)
def benchmark_speed(
    model_file: str,
    buffer_size: List[int],
    sample_rate: List[int],
    repeat: int,
    number: int,
    num_threads: int,
    num_interop_threads: int,
) -> None:
    return benchmark_speed_(
        model_file,
        buffer_size,
        sample_rate,
        repeat,
        number,
        num_threads,
        num_interop_threads,
    )


def benchmark_speed_(
    model_file: str,
    buffer_size: List[int],
    sample_rate: List[int],
    repeat: int,
    number: int,
    num_threads: int,
    num_interop_threads: int,
) -> None:
    np.set_printoptions(precision=3)
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_interop_threads)
    with torch.inference_mode():
        m, _ = load_neutone_model(model_file)
        log.info(
            f"Running benchmark for buffer sizes {buffer_size} and sample rates {sample_rate}. Outliers will be removed from the calculation of mean and std and displayed separately if existing."
        )
        for sr, bs in itertools.product(sample_rate, buffer_size):
            m.set_daw_sample_rate_and_buffer_size(sr, bs)
            for _ in range(number):  # Warmup
                m.forward(torch.rand((2, bs)))
            m.reset()

            # Pregenerate random buffers to more accurately benchmark the model itself
            def get_random_buffer_generator():
                buffers = torch.rand(100, 2, bs)
                i = 0

                def return_next_random_buffer():
                    nonlocal i
                    i = (i + 1) % 100
                    return buffers[i]

                return return_next_random_buffer

            rbg = get_random_buffer_generator()

            durations = np.array(
                timeit.repeat(lambda: m.forward(rbg()), repeat=repeat, number=number)
            )
            m.reset()
            mean, std = np.mean(durations), np.std(durations)
            outlier_mask = np.abs(durations - mean) > 2 * std
            outliers = durations[outlier_mask]
            # Remove outliers from general benchmark
            durations = durations[~outlier_mask]
            mean, std = np.mean(durations), np.std(durations)
            log.info(
                f"Sample rate: {sr: 6} | Buffer size: {bs: 6} | duration: {mean: 6.3f}±{std:.3f} | 1/RTF: {bs/(mean/number*sr): 6.3f} | Outliers: {outliers[:3]}"
            )


@cli.command()
@click.option("--model_file", help="Path to model file")
@click.option(
    "--buffer_size",
    default=(128, 256, 512, 1024, 2048),
    multiple=True,
    help="Buffer sizes to benchmark",
)
@click.option(
    "--sample_rate",
    default=(
        44100,
        48000,
    ),
    multiple=True,
    help="Sample rates to benchmark",
)
def benchmark_latency(
    model_file: str, buffer_size: List[int], sample_rate: List[int]
) -> None:
    return benchmark_latency_(model_file, buffer_size, sample_rate)


def benchmark_latency_(
    model_file: str, buffer_size: List[int], sample_rate: List[int]
) -> None:
    m, _ = load_neutone_model(model_file)
    nbs, nsr = m.get_native_buffer_sizes(), m.get_native_sample_rates()
    log.info(f"Native buffer sizes: {nbs[:10]}, Native sample rates: {nsr[:10]}")
    if len(nbs) > 10 or len(nsr) > 10:
        log.info(f"Showing only the first 10 values in case there are more.")
    with torch.inference_mode():
        m, _ = load_neutone_model(model_file)
        delays = []
        for sr, bs in itertools.product(sample_rate, buffer_size):
            m.set_daw_sample_rate_and_buffer_size(sr, bs)
            m.reset()
            delays += [
                [
                    sr,
                    bs,
                    m.calc_buffering_delay_samples(),
                    m.calc_model_delay_samples(),
                ]
            ]
        delays = sorted(delays, key=lambda x: x[2] + x[3])
        log.info(
            f"Model {model_file} has the following delays for each sample rate / buffer size combination (lowest delay first):"
        )
        for sr, bs, bds, mds in delays:
            log.info(
                f"Sample rate: {sr: 6} | Buffer size: {bs: 6} | Total delay: {bds+mds: 6} | (Buffering delay: {bds: 6} | Model delay: {mds: 6})"
            )
        log.info(
            f"The recommended sample rate / buffer size combination is sample rate {delays[0][0]}, buffer size {delays[0][1]}"
        )


if __name__ == "__main__":
    cli()
