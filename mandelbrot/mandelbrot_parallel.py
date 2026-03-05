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


def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter=100, n_workers=4):
    chunk_size = max(1, N // n_workers)

    chunks = []
    row = 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    with Pool(processes=n_workers) as pool:
        pool.map(_worker, chunks)  # warm-up
        parts = pool.map(_worker, chunks)

    return np.vstack(parts)


if __name__ == "__main__":
    N = 1024
    max_iter = 50
    X_MIN, X_MAX = -2.0, 1.0
    Y_MIN, Y_MAX = -1.5, 1.5

    # warm-up compile (main process)
    mandelbrot_serial(32, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter=20)

    # M1: run serial once and check shape
    img_serial = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter=max_iter)
    print("serial shape:", img_serial.shape, "dtype:", img_serial.dtype)

    # M2: run parallel and verify it matches serial
    img_parallel = mandelbrot_parallel(
        N, X_MIN, X_MAX, Y_MIN, Y_MAX,
        max_iter=max_iter,
        n_workers=4
    )

    print("parallel shape:", img_parallel.shape, "dtype:", img_parallel.dtype)
    
    #check if match
    same = np.array_equal(img_serial, img_parallel)
    print("matches serial:", same)

 