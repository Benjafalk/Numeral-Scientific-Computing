import time
import statistics
import numpy as np
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster
from numba import njit
import matplotlib.pyplot as plt

#MP 1:
#It is basically just the previous lectures script, but with the mandelbrot_dask function
#Result is 0.129s with 8 workers and 32 chunks

#MP 2:
#optimal chunks is 8, with a time of 0.075032s and LIF of 9.554026
#It only plots when running the script, not saving the image

@njit(cache=True)
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = 0.0
    z_imag = 0.0

    for i in range(max_iter):
        z_real2 = z_real * z_real - z_imag * z_imag + c_real
        z_imag2 = 2.0 * z_real * z_imag + c_imag
        z_real = z_real2
        z_imag = z_imag2

        if z_real * z_real + z_imag * z_imag > 4.0:
            return i

    return max_iter


@njit(cache=True)
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)

    for local_row, row in enumerate(range(row_start, row_end)):
        c_imag = y_min + (y_max - y_min) * row / (N - 1)

        for col in range(N):
            c_real = x_min + (x_max - x_min) * col / (N - 1)
            out[local_row, col] = mandelbrot_pixel(c_real, c_imag, max_iter)

    return out


def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


def mandelbrot_dask(N, x_min, x_max, y_min, y_max, max_iter=100, n_chunks=32):
    chunk_size = max(1, N // n_chunks)
    tasks = []
    row = 0

    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(
            delayed(mandelbrot_chunk)(
                row, row_end, N, x_min, x_max, y_min, y_max, max_iter
            )
        )
        row = row_end

    parts = dask.compute(*tasks)
    return np.vstack(parts)


def time_dask_runs(N, x_min, x_max, y_min, y_max, max_iter, n_chunks, repeats=3):
    times = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = mandelbrot_dask(
            N, x_min, x_max, y_min, y_max,
            max_iter=max_iter,
            n_chunks=n_chunks
        )
        times.append(time.perf_counter() - t0)

    return statistics.median(times)


if __name__ == "__main__":
    N = 1024
    max_iter = 100
    n_workers = 8

    X_MIN, X_MAX = -2.5, 1.0
    Y_MIN, Y_MAX = -1.25, 1.25

    # choose a sweep range
    n_chunks_values = [4, 8, 16, 24, 32, 40, 48, 64, 96, 128]

    # serial baseline T1
    t1_runs = []
    for _ in range(3):
        t0 = time.perf_counter()
        ref = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        t1_runs.append(time.perf_counter() - t0)

    T1 = statistics.median(t1_runs)
    print(f"Serial baseline T1: {T1:.6f} s")

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    client = Client(cluster)
    print("Dashboard:", client.dashboard_link)

    # warm up numba in all workers before timing sweep
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))

    # verify once before sweep
    test_result = mandelbrot_dask(
        N, X_MIN, X_MAX, Y_MIN, Y_MAX,
        max_iter=max_iter,
        n_chunks=32
    )
    print("Matches serial:", np.array_equal(ref, test_result))

    results = []

    print("\nn_chunks | time (s) | vs 1x | speedup | LIF")
    print("-" * 46)

    for n_chunks in n_chunks_values:
        Tp = time_dask_runs(
            N, X_MIN, X_MAX, Y_MIN, Y_MAX,
            max_iter=max_iter,
            n_chunks=n_chunks,
            repeats=3
        )

        vs_1x = Tp / T1
        speedup = T1 / Tp
        lif = n_workers * Tp / T1 - 1.0

        results.append((n_chunks, Tp, vs_1x, speedup, lif))

        print(f"{n_chunks:8d} | {Tp:8.4f} | {vs_1x:5.2f} | {speedup:7.2f} | {lif:6.2f}")

    client.close()
    cluster.close()

    # find best result
    best = min(results, key=lambda x: x[1])
    n_chunks_optimal, t_min, _, _, lif_min = best

    print("\nRecord:")
    print(f"n_chunks_optimal = {n_chunks_optimal}")
    print(f"t_min = {t_min:.6f} s")
    print(f"LIF_min = {lif_min:.6f}")

    # plot wall time vs n_chunks (log scale)
    x_vals = [r[0] for r in results]
    y_vals = [r[1] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, marker="o")
    plt.xscale("log")
    plt.xlabel("n_chunks")
    plt.ylabel("wall time (s)")
    plt.title("Dask local chunk sweep")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dask_chunk_sweep.png", dpi=150)
    plt.show()