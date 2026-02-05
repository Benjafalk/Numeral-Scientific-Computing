import sys

print("Python version:", sys.version)
print("-" * 40)

packages = [
    "numpy",
    "matplotlib",
    "scipy",
    "numba",
    "pytest",
    "dask"
]

for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, "__version__", "unknown")
        print(f"{pkg} imported successfully | version: {version}")
    except Exception as e:
        print(f"{pkg} FAILED:", e)

print("-" * 40)
