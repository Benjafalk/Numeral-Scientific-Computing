import time
#Task 1
import numpy as np

def make_complex_grid(width=1024, height=1024,
                      x_min=-2.0, x_max=1.0,
                      y_min=-1.5, y_max=1.5):
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)          # (height, width)
    C = X + 1j * Y
    return C

#Task 2
def mandelbrot_vectorized(C: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """
    Vectorized Mandelbrot:
      Z and M arrays same shape as C
      only loop over iterations
      mask tracks points that haven't escaped (|Z| <= 2)
      update Z and count M only where mask is True
    Returns:
      M: int array of iteration counts (how many iterations stayed unescaped)
    """
    Z = np.zeros_like(C)                      # complex array
    M = np.zeros(C.shape, dtype=np.int32)     # iteration counts

    for _ in range(max_iter):
        mask = np.abs(Z) <= 2                 # unescaped points
        Z[mask] = Z[mask]**2 + C[mask]        # update only those
        M[mask] += 1                          # count only those

    return M


def time_run(fn, *args, repeats=3, **kwargs):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return out, times

if __name__ == "__main__":
    C = make_complex_grid(1024, 1024)

    # Warm-up (reduces first-run effects)
    _ = mandelbrot_vectorized(C, max_iter=10)

    M, times = time_run(mandelbrot_vectorized, C, max_iter=50, repeats=5)
    print("Vectorized times (s):", [round(t, 4) for t in times])
    print("Best/avg (s):", round(min(times), 4), "/", round(sum(times)/len(times), 4))


