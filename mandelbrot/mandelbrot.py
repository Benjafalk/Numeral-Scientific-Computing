"""
Mandelbrot Set Generator
Author : Benjamin Falk
Course : Numerical Scientific Computing 2026
"""
#This is a comment which i need to add

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
    int: Number of iterations before escape, or max_iter if bounded.
    """

    z = 0 + 0j  # start value

    for n in range(max_iter):
        z = z * z + c

        # Check escape condition |z| > 2
        if (z.real * z.real + z.imag * z.imag) > 4:
            return n

    return max_iter


# Simple manual test 
if __name__ == "__main__":
    # c = 0 should stay bounded → returns max_iter
    print("Test c=0:", mandelbrot_point(0 + 0j, max_iter=50))

    # This point escapes quickly
    print("Test c=2+2i:", mandelbrot_point(2 + 2j, max_iter=50))