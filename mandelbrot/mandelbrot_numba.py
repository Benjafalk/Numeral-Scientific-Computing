"""
Numba versions of Mandelbrot.

This is both the hybrid and fully compiled
AI helped a bit with the fully compiled version, some things in the prange loop

Hybrid result: 0.709s
Fully compiled: 0.006s
Ratio between the two is 118.5x

By looking at results from others i have 0 clue why mine is so fast if i have done something wrong dunno (i think its because maybe only 50 iterations?)
"""

import time
import statistics
import numpy as np
from numba import njit, prange


@njit
def mandelbrot_point_numba(cre, cim, max_iter=50):
    zr = 0.0
    zi = 0.0
    for n in range(max_iter):
        # z = z*z + c, written out so numba doesn't do anything weird
        zr2 = zr * zr - zi * zi + cre
        zi2 = 2.0 * zr * zi + cim
        zr = zr2
        zi = zi2

        # escape condition check
        if zr * zr + zi * zi > 4.0:
            return n
    return max_iter


# Hybrid approach
# numpy creates x/y arrays, Python loops pixels, but inner point work is JITed
def mandelbrot_hybrid(xmin, xmax, ymin, ymax, width, height, max_iter=50):
    xs = np.linspace(xmin, xmax, width)
    ys = np.linspace(ymin, ymax, height)

    out = np.empty((height, width), dtype=np.int32)
    for j in range(height):
        y = ys[j]
        for i in range(width):
            x = xs[i]
            out[j, i] = mandelbrot_point_numba(x, y, max_iter=max_iter)
    return out


# Fully compiled approach: everything in numba, no Python loops at all
@njit(parallel=True)
def mandelbrot_naive_numba(xmin, xmax, ymin, ymax, width, height, max_iter=50):
    xs = np.linspace(xmin, xmax, width)
    ys = np.linspace(ymin, ymax, height)
    out = np.empty((height, width), dtype=np.int32)

    # outer loop 
    for j in prange(height):
        y = ys[j]
        for i in range(width):
            x = xs[i]

            zr = 0.0
            zi = 0.0
            it = max_iter
            
            # inner loop
            for n in range(max_iter):
                zr2 = zr * zr - zi * zi + x
                zi2 = 2.0 * zr * zi + y
                zr = zr2
                zi = zi2

                if zr * zr + zi * zi > 4.0:
                    it = n
                    break

            out[j, i] = it

    return out


def bench(fn, *args, runs=5):
    # extra warmup (jit compile), don't count it
    fn(*args)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)

    return statistics.median(times), times


if __name__ == "__main__":
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    w = h = 1024
    max_iter = 50

    # warm up both (compile) on small grid so it doesnt take ages
    _ = mandelbrot_hybrid(xmin, xmax, ymin, ymax, 64, 64, max_iter=max_iter)
    _ = mandelbrot_naive_numba(xmin, xmax, ymin, ymax, 64, 64, max_iter=max_iter)

    t_hybrid, raw_h = bench(mandelbrot_hybrid, xmin, xmax, ymin, ymax, w, h, max_iter, runs=5)
    t_full, raw_f = bench(mandelbrot_naive_numba, xmin, xmax, ymin, ymax, w, h, max_iter, runs=5)

    print("Hybrid times:", [round(t, 4) for t in raw_h])
    print("Full   times:", [round(t, 4) for t in raw_f])
    print(f"Hybrid median: {t_hybrid:.3f}s")
    print(f"Full median:   {t_full:.3f}s")
    print(f"Ratio (hybrid/full): {t_hybrid / t_full:.1f}x")