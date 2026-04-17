import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def make_grid(n, xmin, xmax, ymin, ymax):
    x = np.linspace(xmin, xmax, n)
    y = np.linspace(ymin, ymax, n)
    return (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)


def escape_count(c, max_iter):
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
    # same region as M1
    n = 512
    max_iter = 1000

    xmin, xmax = -0.7530, -0.7490
    ymin, ymax = 0.0990, 0.1030

    c = make_grid(n, xmin, xmax, ymin, ymax)

    eps32 = float(np.finfo(np.float32).eps)

    # small perturbation based on float32 precision
    delta = np.maximum(eps32 * np.abs(c), 1e-10)

    n_base = escape_count(c, max_iter).astype(float)
    n_perturb = escape_count(c + delta, max_iter).astype(float)

    dn = np.abs(n_base - n_perturb)

    # avoid divide-by-zero just in case
    kappa = np.where(n_base > 0, dn / (eps32 * n_base), np.nan)

    print(f"Region: x in [{xmin}, {xmax}], y in [{ymin}, {ymax}]")
    print(f"N = {n}, max_iter = {max_iter}")
    print(f"eps32 = {eps32:.3e}")
    print(f"kappa min = {np.nanmin(kappa):.3e}")
    print(f"kappa max = {np.nanmax(kappa):.3e}")

    cmap = plt.cm.hot.copy()
    cmap.set_bad('0.25')

    vmax = np.nanpercentile(kappa, 99)

    plt.figure(figsize=(8, 6))
    plt.imshow(
        kappa,
        cmap=cmap,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        norm=LogNorm(vmin=1, vmax=vmax)
    )
    plt.colorbar(label=r'$\kappa(c)$ (log scale, $\kappa \geq 1$)')
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.title(r'Condition number approximation $\kappa(c)$')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()