import random
import time
import statistics
import math
"""
Results:
pi estimate: 3.1416796
error: 8.694641020667859e-05
serial time: 2.089182100025937
"""
#from slides
def estimate_pi_serial(num_samples):
    hits = 0

    for _ in range(num_samples):
        x = random.random()
        y = random.random()

        if x*x + y*y <= 1.0:
            hits += 1

    return 4 * hits / num_samples


if __name__ == "__main__":
    num_samples = 10_000_000

    times = []

    for _ in range(3):
        start = time.perf_counter()
        pi_est = estimate_pi_serial(num_samples)
        times.append(time.perf_counter() - start)

    t_serial = statistics.median(times)

    print("pi estimate:", pi_est)
    print("error:", abs(pi_est - math.pi))
    print("serial time:", t_serial)


    