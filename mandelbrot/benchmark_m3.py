import matplotlib.pyplot as plt

results = {
    "Naive Python": 4.877,
    "NumPy vectorized": 0.6858,
    "Numba hybrid": 0.7000,
    "Numba compiled": 0.0060,
    "Parallel opt.": 0.0110,
    "Dask local bad": 0.129,
    "Dask local opt.": 0.075,
    "Dask Strato": 1.75,
    "GPU f32": 0.006913,
    "GPU f64": 0.072742,
}

naive_time = results["Naive Python"]

print("Full MP3 comparison:")
print("| Implementation | Time (s) | Speedup vs Naive |")
print("|---|---:|---:|")

for name, t in results.items():
    speedup = naive_time / t
    print(f"| {name} | {t:.6f} | {speedup:.2f}x |")

names = list(results.keys())
times = list(results.values())

plt.figure(figsize=(11, 5))
plt.bar(names, times, log=True)
plt.ylabel("Runtime (s), log scale")
plt.xticks(rotation=30, ha="right")
plt.title("MP3 Mandelbrot performance comparison")
plt.tight_layout()
plt.savefig("benchmark_mp3.png", dpi=150)
plt.show()