import time
import statistics
import os
from multiprocessing import Pool

import numpy as np
from numba import njit


@njit(cache=True)
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = 0.0
    z_imag = 0.0

    for i in range(max_iter):
        z_sq = z_real * z_real + z_imag * z_imag
        if z_sq > 4.0:
            return i

        zr = z_real * z_real - z_imag * z_imag + c_real
        zi = 2.0 * z_real * z_imag + c_imag
        z_real = zr
        z_imag = zi

    return max_iter


@njit(cache=True)
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)

    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N

    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            c_real = x_min + col * dx
            out[r, col] = mandelbrot_pixel(c_real, c_imag, max_iter)

    return out


def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


def _worker(args):
    return mandelbrot_chunk(*args)


def build_chunks(N, n_chunks, x_min, x_max, y_min, y_max, max_iter):
    chunk_size = max(1, N // n_chunks)

    chunks = []
    row = 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    return chunks


def mandelbrot_parallel(
    N,
    x_min,
    x_max,
    y_min,
    y_max,
    max_iter=100,
    n_workers=4,
    n_chunks=None,
    pool=None,
):
    if n_chunks is None:
        n_chunks = n_workers

    chunks = build_chunks(N, n_chunks, x_min, x_max, y_min, y_max, max_iter)

    if pool is not None:
        parts = pool.map(_worker, chunks)
        return np.vstack(parts)

    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    with Pool(processes=n_workers) as p:
        p.map(_worker, tiny)   # warm-up in workers
        parts = p.map(_worker, chunks)

    return np.vstack(parts)


if __name__ == "__main__":
    N = 1024
    max_iter = 50
    X_MIN, X_MAX = -2.0, 1.0
    Y_MIN, Y_MAX = -1.5, 1.5

    # warm-up main process JIT
    mandelbrot_serial(32, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter=20)

    # M1 check
    img_serial = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter=max_iter)
    img_parallel = mandelbrot_parallel(
        N, X_MIN, X_MAX, Y_MIN, Y_MAX,
        max_iter=max_iter,
        n_workers=4,
        n_chunks=16
    )
    print("matches serial:", np.array_equal(img_serial, img_parallel))

    # serial baseline
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter=max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)

    print("\nserial baseline:", t_serial)

    max_workers = os.cpu_count() or 1

    for n_workers in range(1, max_workers + 1):
        n_chunks = 4 * n_workers
        chunks = build_chunks(N, n_chunks, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)

        with Pool(processes=n_workers) as pool:
            pool.map(_worker, [(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)])

            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                parts = pool.map(_worker, chunks)
                _ = np.vstack(parts)
                times.append(time.perf_counter() - t0)

        t_par = statistics.median(times)
        speedup = t_serial / t_par
        eff = speedup / n_workers * 100.0

        print(
            f"{n_workers:2d} workers, {n_chunks:3d} chunks: "
            f"{t_par:.3f}s, speedup={speedup:.2f}x, eff={eff:.0f}%"
        )