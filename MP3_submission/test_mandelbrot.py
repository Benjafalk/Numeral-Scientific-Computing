# test_mandelbrot.py

import numpy as np
import pytest

from mandelbrot_numba import (
    mandelbrot_point_numba,
    mandelbrot_hybrid,
    mandelbrot_naive_numba,
    mandelbrot_naive_numba_f32,
    mandelbrot_naive_numba_f64,
)


@pytest.mark.parametrize(
    "cre, cim, max_iter, expected",
    [
        (0.0, 0.0, 100, 100),   # origin stays bounded
        (5.0, 0.0, 100, 0),     # escapes right away after first update
        (-2.5, 0.0, 100, 0),    # clearly outside on the left
        (0.0, 2.0, 100, 1),     # good edge case for the escape threshold
    ],
)
def test_mandelbrot_point_numba_known_values(cre, cim, max_iter, expected):
    assert mandelbrot_point_numba(cre, cim, max_iter) == expected


def test_mandelbrot_point_numba_result_stays_in_range():
    # Nothing fancy here, just checking the count never goes outside valid bounds.
    points = [
        (0.0, 0.0),
        (-0.75, 0.1),
        (0.3, 0.5),
        (-1.75, 0.0),
        (2.0, 2.0),
    ]
    max_iter = 80

    for cre, cim in points:
        value = mandelbrot_point_numba(cre, cim, max_iter)
        assert 0 <= value <= max_iter, f"bad count {value} for c={cre}+{cim}j"


def test_hybrid_and_fully_compiled_agree_on_small_grid():
    # Small grid on purpose: enough to catch mistakes, small enough to stay cheap.
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    width = height = 32
    max_iter = 50

    hybrid = mandelbrot_hybrid(xmin, xmax, ymin, ymax, width, height, max_iter=max_iter)
    compiled = mandelbrot_naive_numba(xmin, xmax, ymin, ymax, width, height, max_iter=max_iter)

    np.testing.assert_array_equal(compiled, hybrid)


def test_f32_and_f64_agree_on_small_grid():
    # This is safer than comparing float32 directly against some unrelated implementation.
    # Also matches the slide note about cross-validating same-precision families.
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    width = height = 32
    max_iter = 50

    arr32 = mandelbrot_naive_numba_f32(xmin, xmax, ymin, ymax, width, height, max_iter=max_iter)
    arr64 = mandelbrot_naive_numba_f64(xmin, xmax, ymin, ymax, width, height, max_iter=max_iter)

    np.testing.assert_array_equal(arr32, arr64)


def test_grid_shape_matches_requested_size():
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    width = 20
    height = 12
    max_iter = 40

    arr = mandelbrot_naive_numba(xmin, xmax, ymin, ymax, width, height, max_iter=max_iter)

    assert arr.shape == (height, width)


def test_grid_counts_stay_in_bounds():
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    width = height = 24
    max_iter = 60

    arr = mandelbrot_naive_numba(xmin, xmax, ymin, ymax, width, height, max_iter=max_iter)

    assert np.all(arr >= 0)
    assert np.all(arr <= max_iter)