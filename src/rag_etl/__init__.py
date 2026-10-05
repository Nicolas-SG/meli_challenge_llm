import os

# torch>=2.14 JIT-compiles some CUDA ops with Triton, which needs the Python
# C headers (python3-dev) at runtime. Fall back to torch's prebuilt kernels so
# the pipeline runs on machines without them. Must be set before torch is
# imported (importing any rag_etl module does it); export
# TORCH_DISABLE_NATIVE_JIT=0 to opt back in.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

# Long training sequences of varying length fragment the CUDA allocator;
# expandable segments avoid spurious out-of-memory errors.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
