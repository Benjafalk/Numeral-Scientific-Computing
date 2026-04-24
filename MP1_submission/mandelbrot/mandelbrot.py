"""
Mandelbrot Set Generator
Author : Benjamin Falk
Course : Numerical Scientific Computing 2026

AI has helped with any and all matplotlib as it it wacky to work with imo
Tried working with automatic docstring but it was bad (future documents are hand made docstrings)
Result is in comment on line 97
"""
#This is a comment which i need to add
import numpy as np
import time
import matplotlib.pyplot as plt


# Task 2
#Outcommented the profile decorator as it kept giving error (1)
#@profile
def mandelbrot_point(c: complex, max_iter: int = 100) -> int:
    """
    Compute the escape iteration count for a single complex number c.

    We iterate:
    z_{n+1} = z_n^2 + c
    starting from z_0 = 0.

    If |z| exceeds 2, the sequence will diverge, and we return the
    iteration number where that happens. If it never escapes within
    max_iter iterations, we return max_iter.

    Parameters:
    c: complex
    Point in the complex plane.
    
    max_iter: int
    Maximum number of iterations.

    Returns:
    int: 
    Number of iterations before escape, or max_iter if bounded.
    """

    z = 0 + 0j  # start value

    for n in range(max_iter):
        z = z * z + c

        # Check escape condition |z| > 2 squared
        if (z.real * z.real + z.imag * z.imag) > 4:
            return n

    return max_iter




# Task 3
def compute_mandelbrot_grid(
    width: int = 100,
    height: int = 100,
    max_iter: int = 50,
    x_min: float = -2.0,
    x_max: float = 1.0,
    y_min: float = -1.5,
    y_max: float = 1.5,
):
    """
    Compute a grid of Mandelbrot iteration counts.

    We create a rectangular grid in the complex plane and evaluate
    mandelbrot_point() for every location.

    Parameters:
    Grid resolution: width, height = int
    Max iterations per point: max_iter = int
    Region of the complex plane: x_min, x_max, y_min, y_max = float

    Returns: 2D numpy array of iteration counts
    """

    # Create mesh of complex numbers (hint from task lmao)

    # Create evenly spaced values along each axis 
    xs = np.linspace(x_min, x_max, width)
    ys = np.linspace(y_min, y_max, height)

    # Empty grid to store iteration counts
    grid = np.zeros((height, width), dtype=int)

    # Loop over all points
    for j in range(height):
        for i in range(width):
            c = complex(xs[i], ys[j])
            grid[j, i] = mandelbrot_point(c, max_iter)

    return grid

# Task 4 is in test
# Result is around 4.877s




# Task 5, visualize the result using plt
def plot_mandelbrot(grid, cmap="viridis", filename=None):
    """
    Visualize Mandelbrot iteration grid as an image.

    Parameters:
    grid : 2D numpy array i.e Iteration counts
    cmap : str which is just a Matplotlib colormap name (hot is just blue/green/yellow ish colors)
    filename : None (If given, saves image to file which we only do when testing)
    """

    plt.figure(figsize=(6, 6)) # size of figure

    plt.imshow(grid, cmap=cmap, origin="lower") #origin is 0,0 in left bottom
    plt.colorbar(label="Iterations")

    plt.title(f"Mandelbrot Set ({cmap} colormap)")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=150) #size in dpi, higher=better
        print(f"Saved image to {filename}")

    plt.show()

#Usage for lecture 3
def mandelbrot_naive(x_min, x_max, y_min, y_max, width, height, max_iter=50):
    return compute_mandelbrot_grid(
        width=width,
        height=height,
        max_iter=max_iter,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )




# Testing setup
if __name__ == "__main__":
    # Task 2
    # c = 0 should stay bounded and returns max_iter
    print("mandelbrot point test, c=0:", mandelbrot_point(0 + 0j, max_iter=50))

    # This point escapes quickly and should give 0
    print("mandelbrot point test, c=2+2i:", mandelbrot_point(2 + 2j, max_iter=50))

    # Task 3
    # Test the small grid actually exists
    print("\nComputing small Mandelbrot grid (100x100)...")
    grid = compute_mandelbrot_grid(width=100, height=100, max_iter=50)
    print("Grid shape:", grid.shape)
    print("Top-left value:", grid[0, 0])
    print("Center value:", grid[50, 50])

    # Task 4 (time performance)
    print("\nMeasuring performance for 1024x1024 grid...")

    start_time = time.time()

    grid_large = compute_mandelbrot_grid(
        width=1024,
        height=1024,
        max_iter=50
    )

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Time: {elapsed:.3f} seconds")


    # Task 5 (visualize)
    print("\nCreating plot, will open in a new window and save")
    plot_mandelbrot(grid_large, cmap="viridis", filename="mandelbrot_viridis.png")
