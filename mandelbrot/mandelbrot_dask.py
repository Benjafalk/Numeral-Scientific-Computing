import time
import statistics
import numpy as np
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster
from numba import njit

#It is basically just the previous lectures script, but with the mandelbrot_dask function
#Result is 0.129s with 8 workers and 32 chunks

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

        task = delayed(mandelbrot_chunk)(
            row,
            row_end,
            N,
            x_min,
            x_max,
            y_min,
            y_max,
            max_iter,
        )
        tasks.append(task)
        row = row_end

    parts = dask.compute(*tasks)
    return np.vstack(parts)


def main():
    N = 1024
    max_iter = 100
    n_workers = 8
    n_chunks = 32

    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.25, 1.25

    # serial reference
    ref = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    client = Client(cluster)

    print("Dashboard:", client.dashboard_link)

    # warm up numba in every worker
    client.run(
        lambda: mandelbrot_chunk(
            0, 8, 8, x_min, x_max, y_min, y_max, 10
        )
    )

    # verify correctness
    result = mandelbrot_dask(
        N, x_min, x_max, y_min, y_max, max_iter=max_iter, n_chunks=n_chunks
    )
    print("Matches serial:", np.array_equal(ref, result))

    # timing: 3 runs, median
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        result = mandelbrot_dask(
            N, x_min, x_max, y_min, y_max, max_iter=max_iter, n_chunks=n_chunks
        )
        times.append(time.perf_counter() - t0)

    print(f"Dask local ({n_chunks} chunks): {statistics.median(times):.3f} s")

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()