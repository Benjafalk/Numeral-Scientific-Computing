from mandelbrot import compute_mandelbrot_grid

if __name__ == "__main__":
    compute_mandelbrot_grid(
        width=512,
        height=512,
        max_iter=50,
        x_min=-2.0,
        x_max=1.0,
        y_min=-1.5,
        y_max=1.5,
    )