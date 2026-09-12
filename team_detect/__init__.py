import os

# This torch (2.13+) build JIT-compiles some ops (e.g. the bmm inside
# nn.MultiheadAttention) via a Triton-generated CUDA extension on first
# use, which needs the Python.h dev headers - not installed on this
# machine's system Python. Set before any torch import happens (i.e.
# here, at the top of the package all our entrypoints import first) so
# it takes effect regardless of which script runs.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
