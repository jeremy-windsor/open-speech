"""Regression checks for the GPU Docker image runtime dependencies."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gpu_dockerfile_exposes_ctranslate2_cuda_libraries() -> None:
    """faster-whisper/CT2 GPU inference needs cuBLAS discoverable at runtime."""
    dockerfile = (ROOT / "Dockerfile").read_text()

    required_packages = [
        "nvidia-cublas-cu12",
        "nvidia-cuda-runtime-cu12",
    ]
    for package in required_packages:
        assert package in dockerfile

    assert "metadata.version(package)" in dockerfile
    assert "libcublas.so.12" in dockerfile
    assert "libcudart.so.12" in dockerfile
    assert "/opt/venv/cuda-libs" in dockerfile
    assert "LD_LIBRARY_PATH=/opt/venv/cuda-libs" in dockerfile
