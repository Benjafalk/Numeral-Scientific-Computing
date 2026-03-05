import random
import time
import statistics
import math
import os
from multiprocessing import Pool
"""
Monte carlo, mostly used from slides.
Results:
Serial: 3.1415392 time: 2.0319421999156475
 1 workers: 2.128s  pi=3.141786
 2 workers: 1.165s  pi=3.141864
 3 workers: 0.937s  pi=3.141506
 4 workers: 0.788s  pi=3.141634
 5 workers: 0.755s  pi=3.141689
 6 workers: 0.705s  pi=3.140811
 7 workers: 0.667s  pi=3.140876
 8 workers: 0.722s  pi=3.141717
 9 workers: 0.727s  pi=3.141903
10 workers: 0.722s  pi=3.140751
11 workers: 0.754s  pi=3.141945
12 workers: 0.767s  pi=3.141933
13 workers: 0.755s  pi=3.142101
14 workers: 0.776s  pi=3.140362
15 workers: 0.809s  pi=3.141624
16 workers: 0.867s  pi=3.141313
"""

def estimate_pi_serial(num_samples):
    hits = 0
    for _ in range(num_samples):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1.0:
            hits += 1
    return 4 * hits / num_samples


# worker function from slides
def estimate_pi_chunk(num_samples):
    inside = 0
    for _ in range(num_samples):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1.0:
            inside += 1
    return inside


def estimate_pi_parallel(num_samples, num_processes):
    samples_per_proc = num_samples // num_processes
    tasks = [samples_per_proc] * num_processes

    with Pool(processes=num_processes) as pool:
        results = pool.map(estimate_pi_chunk, tasks)

    return 4 * sum(results) / num_samples


if __name__ == "__main__":
    num_samples = 10_000_000

    # serial timing
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        pi_est = estimate_pi_serial(num_samples)
        times.append(time.perf_counter() - t0)

    t_serial = statistics.median(times)
    print("Serial:", pi_est, "time:", t_serial)

    # parallel timing
    for num_proc in range(1, os.cpu_count() + 1):
        times = []

        for _ in range(3):
            t0 = time.perf_counter()
            pi_est = estimate_pi_parallel(num_samples, num_proc)
            times.append(time.perf_counter() - t0)

        t = statistics.median(times)

        print(f"{num_proc:2d} workers: {t:.3f}s  pi={pi_est:.6f}")