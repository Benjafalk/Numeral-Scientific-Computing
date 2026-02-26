import cProfile
import pstats

from mandelbrot import mandelbrot_naive
from mandelbrot_np import mandelbrot_numpy
from mandelbrot import compute_mandelbrot_grid

#Code from jimmy rewritten to be more reusable and less copy-pastey. Also added a function to run the profiles and print the results.
def run_profiles():
    x_min, x_max, y_min, y_max = -2.0, 1.0, -1.5, 1.5
    w, h = 512, 512
    max_iter = 50

    cProfile.run(
        f"mandelbrot_naive({x_min}, {x_max}, {y_min}, {y_max}, {w}, {h}, {max_iter})",
        "naive_profile.prof",
    )

    cProfile.run(
        f"mandelbrot_numpy({x_min}, {x_max}, {y_min}, {y_max}, {w}, {h}, {max_iter})",
        "numpy_profile.prof",
    )

    for name in ("naive_profile.prof", "numpy_profile.prof"):
        print("\n====", name, "====")
        stats = pstats.Stats(name)
        stats.sort_stats("cumulative")
        stats.print_stats(10)


if __name__ == "__main__":
    run_profiles()
    compute_mandelbrot_grid(
        width=512,
        height=512,
        max_iter=50
    )