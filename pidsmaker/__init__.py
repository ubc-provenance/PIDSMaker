import os
import platform


def _limit_cpu_threads_in_docker_desktop_arm64(max_threads=4):
    """Limits the OpenMP threads of torch in Docker Desktop on ARM (e.g. Apple Silicon Macs).

    The VM of Docker Desktop exposes the performance and efficiency cores of the Mac as identical
    CPUs, and OpenMP synchronizes all its threads at each operation: with one thread per CPU,
    operations on small tensors wait for the slowest cores and are up to 100x slower than with a
    few threads (e.g. 31 ms instead of 0.3 ms for a linear layer on 1024 edges). OpenMP reads
    `OMP_NUM_THREADS` when torch is imported, so this runs before any import of torch, and only
    if the variable is not already set. Linux machines (and GPUs) are not concerned.
    """
    if "OMP_NUM_THREADS" in os.environ or platform.machine() not in ("aarch64", "arm64"):
        return
    if "linuxkit" not in platform.release():  # kernel of the Docker Desktop VM
        return
    os.environ["OMP_NUM_THREADS"] = str(min(max_threads, os.cpu_count() or 1))


_limit_cpu_threads_in_docker_desktop_arm64()
