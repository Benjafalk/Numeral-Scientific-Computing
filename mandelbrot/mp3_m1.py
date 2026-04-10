import numpy as np
import matplotlib.pyplot as plt


# MP3 M1
# Compare Mandelbrot trajectories in float32 and float64
# and mark the first iteration where they separate enough.


def make_grid(n, xmin, xmax, ymin, ymax):
    x = np.linspace(xmin, xmax, n)
    y = np.linspace(ymin, ymax, n)
    return x[np.newaxis, :] + 1j * y[:, np.newaxis]


def trajectory_divergence(c64, max_iter=1000, tau=0.01):
    # same coordinates, just different precision
    c32 = c64.astype(np.complex64)

    z64 = np.zeros_like(c64, dtype=np.complex128)
    z32 = np.zeros_like(c32, dtype=np.complex64)

    h, w = c64.shape

    # default value means did not diverge before max_iter
    first_div = np.full((h, w), max_iter, dtype=np.int32)

    # only keep working on pixels that have not diverged yet
    active = np.ones((h, w), dtype=bool)

    for k in range(max_iter):
        if not np.any(active):
            break

        z32[active] = z32[active] * z32[active] + c32[active]
        z64[active] = z64[active] * z64[active] + c64[active]

        # compare in float64 so the subtraction itself is not done in float32
        diff = (
            np.abs(z32.real.astype(np.float64) - z64.real)
            + np.abs(z32.imag.astype(np.float64) - z64.imag)
        )

        newly_diverged = active & (diff > tau)
        first_div[newly_diverged] = k
        active[newly_diverged] = False

    return first_div


def escape_count(c, max_iter=1000):
    z = np.zeros_like(c, dtype=np.complex128)
    counts = np.full(c.shape, max_iter, dtype=np.int32)
    escaped = np.zeros(c.shape, dtype=bool)

    for k in range(max_iter):
        z[~escaped] = z[~escaped] * z[~escaped] + c[~escaped]

        newly_escaped = (~escaped) & (np.abs(z) > 2.0)
        counts[newly_escaped] = k
        escaped[newly_escaped] = True

        if np.all(escaped):
            break

    return counts


def main():
    # default region from the slide
    n = 512
    max_iter = 1000
    tau = 0.01

    xmin, xmax = -0.7530, -0.7490
    ymin, ymax = 0.0990, 0.1030

    c64 = make_grid(n, xmin, xmax, ymin, ymax)

    first_div = trajectory_divergence(c64, max_iter=max_iter, tau=tau)
    esc = escape_count(c64, max_iter=max_iter)

    frac_diverged = np.mean(first_div < max_iter)

    print(f"Region: x in [{xmin}, {xmax}], y in [{ymin}, {ymax}]")
    print(f"N = {n}, max_iter = {max_iter}, tau = {tau}")
    print(f"Fraction diverged before max_iter: {frac_diverged:.4f}")

    plt.figure(figsize=(8, 6))
    plt.imshow(
        first_div,
        cmap="plasma",
        origin="lower",
        extent=[xmin, xmax, ymin, ymax]
    )
    plt.colorbar(label="First divergence iteration")
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.title(f"Trajectory divergence (tau={tau})")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(
        esc,
        cmap="magma",
        origin="lower",
        extent=[xmin, xmax, ymin, ymax]
    )
    plt.colorbar(label="Escape iteration")
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.title("Escape-count map")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()