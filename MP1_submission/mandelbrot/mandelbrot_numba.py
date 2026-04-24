"""
Numba versions of Mandelbrot.

This is both the hybrid and fully compiled
AI helped a bit with the fully compiled version, some things in the prange loop

Hybrid result: 0.709s
Fully compiled: 0.006s
Ratio between the two is 118.5x

By looking at results from others i have 0 clue why mine is so fast if i have done something wrong dunno
(i think its because maybe only 50 iterations?)

Milestone 4:
- timing float16/32/64 (float16 not supported here -> emulated as float32)
- save timings to numba_float_result.txt
- compare "precision" visually and dump max diffs
"""

import time
import statistics
import numpy as np
import matplotlib.pyplot as plt
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


# 32 and 64 typed versions for milestone 4 (16 is not supported by numba here)
# just copy-pasted versions with small changes
@njit(parallel=True)
def mandelbrot_naive_numba_f32(xmin, xmax, ymin, ymax, width, height, max_iter=50):
    xs = np.linspace(xmin, xmax, width).astype(np.float32)
    ys = np.linspace(ymin, ymax, height).astype(np.float32)
    out = np.empty((height, width), dtype=np.int32)

    four = np.float32(4.0)
    two = np.float32(2.0)
    zero = np.float32(0.0)

    for j in prange(height):
        y = ys[j]
        for i in range(width):
            x = xs[i]
            zr = zero
            zi = zero
            it = max_iter

            for n in range(max_iter):
                zr2 = zr * zr - zi * zi + x
                zi2 = two * zr * zi + y
                zr = zr2
                zi = zi2
                if zr * zr + zi * zi > four:
                    it = n
                    break

            out[j, i] = it
    return out


@njit(parallel=True)
def mandelbrot_naive_numba_f64(xmin, xmax, ymin, ymax, width, height, max_iter=50):
    xs = np.linspace(xmin, xmax, width).astype(np.float64)
    ys = np.linspace(ymin, ymax, height).astype(np.float64)
    out = np.empty((height, width), dtype=np.int32)

    four = 4.0
    two = 2.0
    zero = 0.0

    for j in prange(height):
        y = ys[j]
        for i in range(width):
            x = xs[i]
            zr = zero
            zi = zero
            it = max_iter

            for n in range(max_iter):
                zr2 = zr * zr - zi * zi + x
                zi2 = two * zr * zi + y
                zr = zr2
                zi = zi2
                if zr * zr + zi * zi > four:
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


def precision_compare_and_plot(xmin, xmax, ymin, ymax, w, h, max_iter=50, out_name="numa_float_visual.png"):
    # get results
    r64 = mandelbrot_naive_numba_f64(xmin, xmax, ymin, ymax, w, h, max_iter=max_iter)
    r32 = mandelbrot_naive_numba_f32(xmin, xmax, ymin, ymax, w, h, max_iter=max_iter)

    # "float16" isn't supported by this numba build, so this is just float32 again
    r16 = mandelbrot_naive_numba_f32(xmin, xmax, ymin, ymax, w, h, max_iter=max_iter)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, result, title in zip(
        axes,
        [r16, r32, r64],
        ["float16 (emulated)", "float32", "float64 (ref)"],
    ):
        ax.imshow(result, cmap="hot", origin="lower")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_name, dpi=150)
    plt.close(fig)

    # accuracy diffs (int arrays, but still useful)
    d32 = np.abs(r32.astype(np.int32) - r64.astype(np.int32)).max()
    d16 = np.abs(r16.astype(np.int32) - r64.astype(np.int32)).max()

    print(f"saved: {out_name}")
    print(f"Max diff float32 vs float64: {d32}")
    print(f"Max diff float16 vs float64: {d16}")

    return d16, d32


if __name__ == "__main__":
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    w = h = 1024
    max_iter = 50

    # warm up both on small grid so it doesnt take ages
    _ = mandelbrot_hybrid(xmin, xmax, ymin, ymax, 64, 64, max_iter=max_iter)
    _ = mandelbrot_naive_numba(xmin, xmax, ymin, ymax, 64, 64, max_iter=max_iter)

    t_hybrid, raw_h = bench(mandelbrot_hybrid, xmin, xmax, ymin, ymax, w, h, max_iter, runs=5)
    t_full, raw_f = bench(mandelbrot_naive_numba, xmin, xmax, ymin, ymax, w, h, max_iter, runs=5)

    print("Hybrid times:", [round(t, 4) for t in raw_h])
    print("Full   times:", [round(t, 4) for t in raw_f])
    print(f"Hybrid median: {t_hybrid:.3f}s")
    print(f"Full median:   {t_full:.3f}s")
    print(f"Ratio (hybrid/full): {t_hybrid / t_full:.1f}x")

    # milestone 4 dtype timings
    dtypes = [np.float16, np.float32, np.float64]

    # compile warmups
    _ = mandelbrot_naive_numba_f32(xmin, xmax, ymin, ymax, 32, 32, max_iter=10)
    _ = mandelbrot_naive_numba_f64(xmin, xmax, ymin, ymax, 32, 32, max_iter=10)

    lines = []
    lines.append("Mandelbrot Numba dtype test (fully compiled-ish)\n")
    lines.append(f"grid: {w}x{h}, max_iter: {max_iter}\n")


    for dt in dtypes:
        t0 = time.perf_counter()

        if dt is np.float64:
            mandelbrot_naive_numba_f64(xmin, xmax, ymin, ymax, w, h, max_iter=max_iter)
            label = "float64"
        elif dt is np.float32:
            mandelbrot_naive_numba_f32(xmin, xmax, ymin, ymax, w, h, max_iter=max_iter)
            label = "float32"
        else:
            mandelbrot_naive_numba_f32(xmin, xmax, ymin, ymax, w, h, max_iter=max_iter)
            label = "float16 (emulated as float32)"

        t1 = time.perf_counter()
        sec = t1 - t0

        s = f"{label}: {sec:.3f}s"
        print(s)
        lines.append(s + "\n")

    with open("numba_float_result.txt", "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("wrote numba_float_result.txt")

    # milestone 4 visual + diff
    precision_compare_and_plot(
        xmin, xmax, ymin, ymax, w, h,
        max_iter=max_iter,
        out_name="numa_float_visual.png"
    )