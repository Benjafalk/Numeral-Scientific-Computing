import time
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt


KERNEL_SRC = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void mandelbrot_f32(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    float c_real = x_min + col * (x_max - x_min) / (float)N;
    float c_imag = y_min + row * (y_max - y_min) / (float)N;

    float zr = 0.0f;
    float zi = 0.0f;
    int count = 0;

    while (count < max_iter && zr*zr + zi*zi <= 4.0f) {
        float tmp = zr*zr - zi*zi + c_real;
        zi = 2.0f * zr * zi + c_imag;
        zr = tmp;
        count++;
    }

    result[row * N + col] = count;
}


__kernel void mandelbrot_f64(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    double c_real = x_min + col * (x_max - x_min) / (double)N;
    double c_imag = y_min + row * (y_max - y_min) / (double)N;

    double zr = 0.0;
    double zi = 0.0;
    int count = 0;

    while (count < max_iter && zr*zr + zi*zi <= 4.0) {
        double tmp = zr*zr - zi*zi + c_real;
        zi = 2.0 * zr * zi + c_imag;
        zr = tmp;
        count++;
    }

    result[row * N + col] = count;
}
"""


def run_kernel(prog, queue, image_dev, image, N, max_iter, dtype):
    if dtype == "f32":
        kernel = prog.mandelbrot_f32
        scalar_type = np.float32
    else:
        kernel = prog.mandelbrot_f64
        scalar_type = np.float64

    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.25, 1.25

    # warm-up
    kernel(
        queue, (N, N), None,
        image_dev,
        scalar_type(x_min), scalar_type(x_max),
        scalar_type(y_min), scalar_type(y_max),
        np.int32(N), np.int32(max_iter),
    )
    queue.finish()

    t0 = time.perf_counter()

    kernel(
        queue, (N, N), None,
        image_dev,
        scalar_type(x_min), scalar_type(x_max),
        scalar_type(y_min), scalar_type(y_max),
        np.int32(N), np.int32(max_iter),
    )
    queue.finish()

    elapsed = time.perf_counter() - t0

    cl.enqueue_copy(queue, image, image_dev)
    queue.finish()

    return elapsed, image.copy()


def main():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    dev = ctx.devices[0]
    print("Device:", dev.name)

    if "cl_khr_fp64" not in dev.extensions:
        print("Warning: cl_khr_fp64 not found. Float64 may not work on this device.")

    prog = cl.Program(ctx, KERNEL_SRC).build()

    max_iter = 200

    for N in [1024, 2048]:
        image = np.zeros((N, N), dtype=np.int32)
        image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)

        t32, img32 = run_kernel(prog, queue, image_dev, image, N, max_iter, "f32")
        t64, img64 = run_kernel(prog, queue, image_dev, image, N, max_iter, "f64")

        ratio = t64 / t32

        print()
        print(f"N = {N}")
        print(f"float32 runtime: {t32:.6f} s")
        print(f"float64 runtime: {t64:.6f} s")
        print(f"f64/f32 speed ratio: {ratio:.2f}x slower")

        diff_pixels = np.mean(img32 != img64) * 100
        print(f"Different pixels between f32 and f64: {diff_pixels:.4f}%")

        plt.imshow(img32, cmap="hot", origin="lower")
        plt.axis("off")
        plt.savefig(f"mandelbrot_f32_N{N}.png", dpi=150, bbox_inches="tight")
        plt.close()

        plt.imshow(img64, cmap="hot", origin="lower")
        plt.axis("off")
        plt.savefig(f"mandelbrot_f64_N{N}.png", dpi=150, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()