import time
import statistics
import os
from multiprocessing import Pool

import numpy as np
from numba import njit

"""

Milestone 1:
Refactor into 3 functions (pixel, chunk, serial wrapper)

Mostly copy pasted from my numba implementation, but with a parallel worker function and a parallel pool map (from slides)

Result of milestone 1:
1024x1024 and it is split up

Milestone 2:
add mandelbrot_parallel using multiprocessing Pool.map

copy pasted from slides and changed a bit

Result of milestone 2:
It matches the serial output

Milestone 3:
Benchmark parallel vs serial. Result (expected):
serial baseline: 0.03030620003119111
 1 workers: 0.044s, speedup=0.69x, eff=69%
 2 workers: 0.022s, speedup=1.39x, eff=69%
 3 workers: 0.026s, speedup=1.18x, eff=39%
 4 workers: 0.019s, speedup=1.58x, eff=40%
 5 workers: 0.022s, speedup=1.37x, eff=27%
 6 workers: 0.016s, speedup=1.87x, eff=31%
 7 workers: 0.016s, speedup=1.93x, eff=28%
 8 workers: 0.013s, speedup=2.25x, eff=28%
 9 workers: 0.013s, speedup=2.34x, eff=26%
10 workers: 0.013s, speedup=2.28x, eff=23%
11 workers: 0.012s, speedup=2.44x, eff=22%
12 workers: 0.013s, speedup=2.42x, eff=20%
13 workers: 0.017s, speedup=1.79x, eff=14%
14 workers: 0.018s, speedup=1.65x, eff=12%
15 workers: 0.011s, speedup=2.85x, eff=19%
16 workers: 0.022s, speedup=1.37x, eff=9%
"""


@njit
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = 0.0
    z_imag = 0.0

    for i in range(max_iter):
        # |z|^2 check (escape radius 2 => 4 in squared space) same as before
        z_sq = z_real * z_real + z_imag * z_imag
        if z_sq > 4.0:
            return i

        # z = z*z + c   (written out for numba)
        zr = z_real * z_real - z_imag * z_imag + c_real
        zi = 2.0 * z_real * z_imag + c_imag
        z_real = zr
        z_imag = zi

    return max_iter


@njit
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


def build_chunks(N, n_workers, x_min, x_max, y_min, y_max, max_iter):
    # same chunk building as M2, but pulled out so M3 can reuse it cleanly
    chunk_size = max(1, N // n_workers)

    chunks = []
    row = 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    return chunks


def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter=100, n_workers=4):
    chunks = build_chunks(N, n_workers, x_min, x_max, y_min, y_max, max_iter)

    with Pool(processes=n_workers) as pool:
        pool.map(_worker, chunks)  # warm-up
        parts = pool.map(_worker, chunks)

    return np.vstack(parts)


if __name__ == "__main__":
    N = 1024
    max_iter = 50
    X_MIN, X_MAX = -2.0, 1.0
    Y_MIN, Y_MAX = -1.5, 1.5

    # warm-up compile 
    mandelbrot_serial(32, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter=20)

    # M1/M2 size check
    img_serial = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter=max_iter)
    img_parallel = mandelbrot_parallel(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter=max_iter, n_workers=4)
    print("matches serial:", np.array_equal(img_serial, img_parallel))

    # M3 benchmarking
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter=max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)

    #serial baseline
    print("\nserial baseline:", t_serial)

    # sweep workers
    max_workers = os.cpu_count() or 1

    for n_workers in range(1, max_workers + 1):
        chunks = build_chunks(N, n_workers, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)

        # create Pool outside timed region (as jim-jim said in the slides)
        with Pool(processes=n_workers) as pool:
            pool.map(_worker, chunks)  # un-timed warm-up

            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                parts = pool.map(_worker, chunks)
                _ = np.vstack(parts)  # include assembly time   /asked ai what this means
                times.append(time.perf_counter() - t0)

        t_par = statistics.median(times)
        speedup = t_serial / t_par
        eff = speedup / n_workers * 100.0

        print(f"{n_workers:2d} workers: {t_par:.3f}s, speedup={speedup:.2f}x, eff={eff:.0f}%")