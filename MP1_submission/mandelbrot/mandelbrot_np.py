import time
import matplotlib.pyplot as plt
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

#timed average is 0.6858s

def time_run(fn, *args, repeats=3, **kwargs):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return out, times

# -------------------------
# Task 3: Memory access patterns
# -------------------------

def row_sums_loop(A: np.ndarray) -> float:
    """Loop over rows; each iteration sums a contiguous row slice in C-order."""
    N = A.shape[0]
    s = 0.0
    for i in range(N):
        s += np.sum(A[i, :])
    return s

def col_sums_loop(A: np.ndarray) -> float:
    """Loop over columns; each iteration sums a strided column slice in C-order."""
    N = A.shape[1]
    s = 0.0
    for j in range(N):
        s += np.sum(A[:, j])
    return s

#Usage for lecture 3
def mandelbrot_numpy(x_min, x_max, y_min, y_max, width, height, max_iter=50):
    C = make_complex_grid(
        width=width,
        height=height,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    return mandelbrot_vectorized(C, max_iter=max_iter)


if __name__ == "__main__":
    # task 1
    C = make_complex_grid(1024, 1024)

    # task 2 timing
    _ = mandelbrot_vectorized(C, max_iter=10)  # warm-up

    M, times = time_run(mandelbrot_vectorized, C, max_iter=50, repeats=5)
    print("Vectorized times (s):", [round(t, 4) for t in times])
    print("Best/avg (s):", round(min(times), 4), "/", round(sum(times)/len(times), 4))

    # task 3 memory access patterns

    N = 10_000
    print("\nTask 3:")
    print(f"Creating A with shape ({N}, {N}) ...")
    A = np.random.rand(N, N)  # default float64, C-order

    # Warm-up 
    _ = row_sums_loop(A[:1000, :1000])
    _ = col_sums_loop(A[:1000, :1000])

    # Time row vs column on C-order array
    _, t_row = time_run(row_sums_loop, A, repeats=3)
    _, t_col = time_run(col_sums_loop, A, repeats=3)

    print("\nC-order (default) array:")
    print("Row-sum times (s):", [round(t, 4) for t in t_row], " best:", round(min(t_row), 4))
    print("Col-sum times (s):", [round(t, 4) for t in t_col], " best:", round(min(t_col), 4))

    # Convert to Fortran-order 
    print("\nconvert to Fortran-order array A_f = np.asfortranarray(A) ...")
    A_f = np.asfortranarray(A)

    _, t_row_f = time_run(row_sums_loop, A_f, repeats=3)
    _, t_col_f = time_run(col_sums_loop, A_f, repeats=3)

    print("\nFortran-order array:")
    print("Row-sum times (s):", [round(t, 4) for t in t_row_f], " best:", round(min(t_row_f), 4))
    print("Col-sum times (s):", [round(t, 4) for t in t_col_f], " best:", round(min(t_col_f), 4))

    #Task 4 problem size scaling

    print("\nTask 4:")
    sizes = [256, 512, 1024, 2048, 4096]
    max_iter = 50
    repeats = 3

    results = []
    for s in sizes:
        C_s = make_complex_grid(s, s)

        _ = mandelbrot_vectorized(C_s, max_iter=5)  # warm-up

        _, t = time_run(mandelbrot_vectorized, C_s, max_iter=max_iter, repeats=repeats)
        best = min(t)
        avg = sum(t) / len(t)

        results.append((s, best, avg))
        print(f"{s}x{s}: best {best:.4f}s, avg {avg:.4f}s")

    t1024 = next((best for (s, best, _) in results if s == 1024), None)
    t2048 = next((best for (s, best, _) in results if s == 2048), None)
    if t1024 is not None and t2048 is not None:
        pred_2048 = 4 * t1024
        print(f"\nPredict 2048 from 1024 (4x pixels): {pred_2048:.4f}s")
        print(f"Measured 2048: {t2048:.4f}s")
        print(f"Ratio measured/pred: {t2048 / pred_2048:.3f}")

    x = [s for (s, _, _) in results]
    y = [best for (_, best, _) in results]

    # Shows in new window, not like picture which saves
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("Grid size (N)")
    plt.ylabel("Best runtime (s)")
    plt.title("Mandelbrot scaling (vectorized NumPy)")
    plt.grid(True)
    plt.show()

