# mandelbrot_m2.py

"""
M2 version for MP3.

One cleaned-up Mandelbrot implementation with:
- type hints on all public functions
- NumPy style docstrings
- should pass ruff after normal formatting

Picked NumPy version because it is simple enough to read
and still useful for benchmarking.
"""

from __future__ import annotations

import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def mandelbrot_pixel(c: complex, max_iter: int) -> int:
    """
    Compute escape iteration count for one complex point.

    Starts from z = 0 and iterates:

        z(n+1) = z(n)^2 + c

    Iteration stops when |z| > 2 or when max_iter is reached.

    Parameters
    ----------
    c : complex
        Complex coordinate to test.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    int
        Iteration count where escape happened.
        Returns max_iter if no escape occurred.
    """
    z = 0j

    for n in range(max_iter):
        z = z * z + c
        if abs(z) > 2.0:
            return n

    return max_iter


def mandelbrot_numpy(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: int,
    height: int,
    max_iter: int,
) -> np.ndarray:
    """
    Compute a Mandelbrot grid using NumPy arrays.

    Parameters
    ----------
    xmin : float
        Minimum x-value.
    xmax : float
        Maximum x-value.
    ymin : float
        Minimum y-value.
    ymax : float
        Maximum y-value.
    width : int
        Number of columns in the grid.
    height : int
        Number of rows in the grid.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    numpy.ndarray
        2D array with escape iteration counts.
    """
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    c = x + y[:, None] * 1j
    z = np.zeros_like(c, dtype=np.complex128)

    result = np.full(c.shape, max_iter, dtype=np.int32)
    active = np.ones(c.shape, dtype=bool)

    for n in range(max_iter):
        z[active] = z[active] * z[active] + c[active]

        escaped = np.abs(z) > 2.0
        new_escape = escaped & active

        result[new_escape] = n
        active[new_escape] = False

        if not active.any():
            break

    return result


def benchmark(
    fn,
    *args,
    runs: int = 5,
) -> Tuple[float, list[float]]:
    """
    Time a function several times and return median runtime.

    Parameters
    ----------
    fn : callable
        Function to benchmark.
    *args :
        Positional arguments passed to the function.
    runs : int, default=5
        Number of timing runs.

    Returns
    -------
    tuple[float, list[float]]
        Median runtime and all measured runtimes.
    """
    times: list[float] = []

    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)

    return float(np.median(times)), times


def plot_result(data: np.ndarray, filename: str) -> None:
    """
    Save a Mandelbrot image.

    Parameters
    ----------
    data : numpy.ndarray
        Escape-count grid.
    filename : str
        Output image filename.

    Returns
    -------
    None
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(data, cmap="hot", origin="lower")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def main() -> None:
    """
    Run a simple benchmark and save one image.

    Returns
    -------
    None
    """
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    width = height = 1000
    max_iter = 100

    median, times = benchmark(
        mandelbrot_numpy,
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
        runs=5,
    )

    print("Times:", [round(t, 4) for t in times])
    print(f"Median: {median:.4f}s")

    img = mandelbrot_numpy(
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        max_iter,
    )

    plot_result(img, "mandelbrot_m2.png")
    print("Saved mandelbrot_m2.png")


if __name__ == "__main__":
    main()