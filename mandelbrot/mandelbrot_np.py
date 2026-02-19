#Task 1
import numpy as np

def make_complex_grid(width=1024, height=1024,
                      x_min=-2.0, x_max=1.0,
                      y_min=-1.5, y_max=1.5):
    # 1D axes
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)

    # 2D grids
    X, Y = np.meshgrid(x, y)  #(height, width)

    # Complex grid C = X + iY
    C = X + 1j * Y
    return C

if __name__ == "__main__":
    C = make_complex_grid()
    print("Shape:", C.shape)  # (1024, 1024)
    print("Type:", C.dtype)   # should be complex128


