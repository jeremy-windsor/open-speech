"""Regression checks for the GPU Docker image runtime dependencies."""

import hashlib
import importlib.util
import io
import re
import urllib.error
import urllib.parse
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COSYVOICE = ROOT / "providers" / "cosyvoice"


def _fetch_wetext():
    spec = importlib.util.spec_from_file_location("fetch_wetext", COSYVOICE / "fetch_wetext.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cosyvoice_image_bakes_its_text_normalizer_at_a_pinned_commit() -> None:
    """wetext fetched CosyVoice's text normalizer at every load in
    a new container, from ModelScope's unpinned default branch, and a refused fetch left
    CosyVoice without one. The build fetches it once, pinned, and proves it works offline."""
    fetch_wetext = _fetch_wetext()
    assert re.fullmatch(r"[0-9a-f]{40}", fetch_wetext.REVISION)
    # Exactly what CosyVoice's two Normalizers read: English and Chinese text normalization.
    assert sorted(fetch_wetext.FILES) == [
        "en/tn/tagger.fst",
        "en/tn/verbalizer.fst",
        "zh/tn/tagger.fst",
        "zh/tn/verbalizer.fst",
    ]
    for size, sha256 in fetch_wetext.FILES.values():
        assert size > 0 and re.fullmatch(r"[0-9a-f]{64}", sha256)

    app = (COSYVOICE / "app.py").read_text(encoding="utf-8")
    wetext_dir = re.search(r'^WETEXT_DIR = "([^"]+)"$', app, re.MULTILINE).group(1)
    lines = (COSYVOICE / "Dockerfile").read_text(encoding="utf-8").splitlines()
    fetched = lines.index(f"RUN python /tmp/fetch_wetext.py {wetext_dir} && rm /tmp/fetch_wetext.py")
    assert lines.index("COPY cosyvoice/fetch_wetext.py /tmp/") < fetched
    worker_copied = lines.index("COPY cosyvoice/app.py providers/cosyvoice/")
    checked = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("RUN --network=none ") and "check_text_normalizer()" in line
    )
    assert fetched < worker_copied < checked


def test_cosyvoice_normalizer_fetch_takes_the_pinned_commit_and_refuses_other_bytes(
    monkeypatch, tmp_path
) -> None:
    """Each file comes from the pinned commit and must match the
    size and SHA-256 recorded for it; ModelScope's 403 "too many requests" is waited out."""
    fetch_wetext = _fetch_wetext()
    content = b"fst bytes"
    monkeypatch.setattr(
        fetch_wetext, "FILES", {"en/tn/tagger.fst": (len(content), hashlib.sha256(content).hexdigest())}
    )
    requested, waits = [], []

    def opener(url, timeout):
        requested.append(url)
        if len(requested) == 1:
            raise urllib.error.HTTPError(url, 403, "too many requests", {}, None)
        return io.BytesIO(content)

    fetch_wetext.fetch(tmp_path, opener=opener, sleep=waits.append)

    assert (tmp_path / "en" / "tn" / "tagger.fst").read_bytes() == content
    assert waits == [fetch_wetext.RETRY_WAITS_S[0]]
    url = urllib.parse.urlsplit(requested[-1])
    assert (url.netloc, url.path) == ("modelscope.cn", "/api/v1/models/pengzhendong/wetext/repo")
    assert urllib.parse.parse_qs(url.query) == {
        "Revision": [fetch_wetext.REVISION],
        "FilePath": ["en/tn/tagger.fst"],
    }

    with pytest.raises(RuntimeError, match="expected 9 bytes"):
        fetch_wetext.fetch(tmp_path / "other", opener=lambda url, timeout: io.BytesIO(b"other"))
    assert not (tmp_path / "other").exists()


FISH = ROOT / "providers" / "fish"


def test_fish_image_pins_the_nf4_fork_and_its_torch_and_leaves_the_weights_to_the_first_load() -> None:
    """The Fish Audio S2 image installs the NF4 fork of fish-speech at one commit, torch 2.8.0 from the cu126 index
    (which carries sm_75), and the pins the fork needs. No weights are fetched at build time: the worker fetches the
    pinned snapshot at its first load, into its Hugging Face cache volume."""
    dockerfile = (FISH / "Dockerfile").read_text(encoding="utf-8")
    app = (FISH / "app.py").read_text(encoding="utf-8")
    commit = re.search(r"^ARG FISH_COMMIT=([0-9a-f]{40})$", dockerfile, re.MULTILINE).group(1)
    assert re.search(rf'^FORK_COMMIT = "{commit}"$', app, re.MULTILINE), "the worker reports another fork commit"
    assert "https://github.com/scarxity/fish-speech-int4-patch.git" in dockerfile
    assert 'git checkout "${FISH_COMMIT}"' in dockerfile
    assert 'test "$(git rev-parse HEAD)" = "${FISH_COMMIT}"' in dockerfile
    assert "--index-url https://download.pytorch.org/whl/cu126" in dockerfile
    assert "torch==2.8.0 torchaudio==2.8.0" in dockerfile
    assert "torch.version.cuda == '12.6'" in dockerfile and "'sm_75' in torch._C._cuda_getArchFlags()" in dockerfile
    assert 'pip install -c constraints.txt -r requirements.txt -e "/opt/fish-speech[bnb]"' in dockerfile
    assert "pip check" in dockerfile
    pins = dict(
        line.split("==", 1)
        for line in (FISH / "constraints.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )
    assert {name: pins.get(name) for name in ("torch", "torchaudio", "transformers", "pydantic")} == {
        "torch": "2.8.0", "torchaudio": "2.8.0", "transformers": "4.57.3", "pydantic": "2.9.2"}
    assert pins["huggingface-hub"].startswith("0."), "transformers 4.57 needs huggingface-hub below 1.0"
    assert all(re.fullmatch(r"[0-9][0-9A-Za-z.]*", version) for version in pins.values()), pins
    # Nothing in the build fetches model files.
    for fetch in ("snapshot_download", "hf_hub_download", "huggingface-cli", "hf download", "git lfs"):
        assert fetch not in dockerfile, fetch
    assert re.search(r'^MODEL_REVISION = "[0-9a-f]{40}"$', app, re.MULTILINE)
    lines = dockerfile.splitlines()
    assert lines[-1] == 'CMD ["uvicorn", "providers.fish.app:app", "--host", "0.0.0.0", "--port", "8230"]'
    assert "EXPOSE 8230" in lines and "USER fishworker" in lines


# The fork's own dependencies at FISH_COMMIT 3fd7da6a (pyproject.toml `dependencies` and the `bnb` extra).
FISH_FORK_DEPENDENCIES = (
    "numpy", "torch", "torchaudio", "transformers", "datasets", "lightning", "hydra-core", "tensorboard", "natsort",
    "einops", "librosa", "rich", "gradio", "wandb", "grpcio", "kui", "uvicorn", "loguru", "loralib", "pyrootutils",
    "resampy", "einx", "zstandard", "pydub", "pyaudio", "modelscope", "opencc-python-reimplemented", "silero-vad",
    "ormsgpack", "tiktoken", "pydantic", "cachetools", "inflect", "descript-audio-codec", "descript-audiotools",
    "safetensors", "bitsandbytes",
)


def _fish_constraints() -> dict:
    from packaging.requirements import Requirement

    pins = {}
    for line in (FISH / "constraints.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            requirement = Requirement(line)
            assert requirement.name not in pins, f"{requirement.name} is pinned twice"
            assert str(requirement.specifier).startswith("=="), line
            pins[requirement.name] = str(requirement.specifier)[2:]
    return pins


def test_fish_constraints_pin_a_protobuf_descript_audiotools_accepts_so_pip_never_backtracks_wandb() -> None:
    """The Fish image's pip step failed. descript-audiotools (every release) requires
    protobuf<3.20,>=3.9.2; the newest tensorboard (2.21, protobuf>=6.31.1) and wandb (0.31, protobuf>=5) can't share
    one, and with none of them pinned pip walked wandb back to 0.15.12, whose pathtools sdist imports `imp` (gone in
    Python 3.12), and the build stopped. constraints.txt now pins the fork's locked set (its uv.lock at 3fd7da6a) so the
    resolver has nothing to walk."""
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    pins = _fish_constraints()
    # descript-audiotools 0.7.1 and 0.7.2 (PyPI requires_dist): protobuf (<3.20,>=3.9.2).
    open_ranges = [name for name in ("protobuf", "tensorboard", "wandb") if name not in pins]
    assert not open_ranges, (
        f"constraints.txt leaves {open_ranges} open: pip takes tensorboard 2.21 and wandb 0.31 (protobuf>=5), which "
        "descript-audiotools' protobuf<3.20 rules out, and backtracks wandb into pathtools")
    protobuf = Version(pins["protobuf"])
    assert protobuf in SpecifierSet("<3.20,>=3.9.2"), pins["protobuf"]
    # The protobuf range each pinned release declares on Linux, Python 3.12 (PyPI requires_dist).
    protobuf_ranges = {
        ("tensorboard", "2.20.0"): "!=4.24.0,>=3.19.6",
        ("wandb", "0.21.4"): "!=4.21.0,!=5.28.0,<7,>=3.19.0",
    }
    for (name, version), allowed in protobuf_ranges.items():
        assert pins[name] == version, (name, pins[name])
        assert protobuf in SpecifierSet(allowed), (name, version, allowed)
    # wandb 0.15.12 and older depend on pathtools, an sdist that can't build on Python 3.12.
    assert Version(pins["wandb"]) >= Version("0.16.0")
    unpinned = [name for name in FISH_FORK_DEPENDENCIES if name not in pins]
    assert not unpinned, f"constraints.txt leaves the fork's open ranges to pip: {unpinned}"

    # gradio 6.16+ needs huggingface-hub>=1.16; 5.45.0 (the lock's) needs huggingface-hub<1.0,>=0.33.5 and
    # pydantic<2.12, and its caps on pillow, pandas and tomlkit (and gradio-client's on websockets) sit below the
    # newest releases, which pip would otherwise collect first and then backtrack.
    assert (pins["gradio"], pins["gradio-client"]) == ("5.45.0", "1.13.0")
    caps = {
        "huggingface-hub": "<1.0,>=0.33.5", "pydantic": "<2.12,>=2.0", "pillow": "<12.0,>=8.0",
        "pandas": "<3.0,>=1.0", "tomlkit": "<0.14.0,>=0.12.0", "websockets": "<16.0,>=10.0",
    }
    for name, allowed in caps.items():
        assert Version(pins[name]) in SpecifierSet(allowed), (name, pins[name], allowed)
    # The fork's lock takes numpy 1.x; the file keeps its own transformers and huggingface-hub (newer than the lock's
    # 4.56.1 and 0.34.4).
    assert pins["numpy"] == "1.26.4"
    assert (pins["transformers"], pins["huggingface-hub"]) == ("4.57.3", "0.35.3")
    # torch's own stack comes from the cu126 index in the step before; only torch and torchaudio are pinned here.
    assert not [name for name in pins if name.startswith("nvidia-") or name == "triton"]


def test_fish_compose_service_is_a_gpu_profile_on_the_private_network() -> None:
    """The `fish` service is off by default (profile `fish`), reachable only on the
    Compose network by its own alias on port 8230, reserves the GPU, keeps its weights and compile cache in volumes,
    and the core names it in TTS_EXTERNAL_PROVIDERS."""
    import json

    import yaml

    compose = yaml.safe_load((ROOT / "docker-compose.voice-models.yml").read_text(encoding="utf-8"))
    fish = compose["services"]["fish"]
    assert fish["profiles"] == ["fish"]
    assert fish["networks"]["default"]["aliases"] == ["fish.voice-workers.internal"]
    assert "ports" not in fish, "the worker publishes no port"
    assert fish["build"] == {"context": "providers", "dockerfile": "fish/Dockerfile",
                             "args": {"REVISION": "${OPEN_SPEECH_REVISION:-unknown}"}}
    assert fish["image"] == "${OPEN_SPEECH_FISH_IMAGE:-open-speech-fish:0.1.0}"
    assert fish["gpus"] == "all"
    assert fish["deploy"]["resources"]["reservations"]["devices"] == [
        {"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]
    assert sorted(fish["volumes"]) == ["fish-compile-cache:/home/fishworker/.cache/torch",
                                       "fish-hf-cache:/home/fishworker/.cache/huggingface"]
    assert {"fish-hf-cache", "fish-compile-cache"} <= set(compose["volumes"])
    health = " ".join(fish["healthcheck"]["test"])
    assert "http://localhost:8230/health" in health and "data['provider']=='fish'" in health
    environment = dict(item.split("=", 1) for item in fish["environment"])
    assert environment["FISH_RESTART_AFTER_UNLOAD_S"] == "${FISH_RESTART_AFTER_UNLOAD_S:-1}"
    assert environment["FISH_MAX_INPUT_CHARS"] == "${FISH_MAX_INPUT_CHARS:-1500}"
    core = dict(item.split("=", 1) for item in compose["services"]["open-speech"]["environment"])
    assert json.loads(core["TTS_EXTERNAL_PROVIDERS"])["fish"] == "http://fish.voice-workers.internal.:8230"


def test_gpu_dockerfile_exposes_ctranslate2_cuda_libraries() -> None:
    """faster-whisper/CT2 GPU inference needs cuBLAS discoverable at runtime."""
    dockerfile = (ROOT / "Dockerfile").read_text()

    lock = (ROOT / "uv.lock").read_text()
    required_packages = [
        "nvidia-cublas-cu12",
        "nvidia-cuda-runtime-cu12",
    ]
    for package in required_packages:
        assert package in lock

    assert '"--frozen"' in dockerfile
    assert "libcublas.so.12" in dockerfile
    assert "libcudart.so.12" in dockerfile
    assert "/opt/venv/cuda-libs" in dockerfile
    assert "LD_LIBRARY_PATH=/opt/venv/cuda-libs" in dockerfile
