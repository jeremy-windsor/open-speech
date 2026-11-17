"""Isolated Breeze TTS 2 worker for the Open Speech harness.

Breeze TTS 2 runs through the community C++/ggml port (HoppouAI/Breeze-TTS-2.cpp) with its Q8_0 GGUF weights,
about 4.4 GB of GPU memory. The worker starts the port's ``breeze-server`` as a child process on a load and stops it
on an unload, so the GPU memory goes back with the process. Each request is passed to the server's native
``/v1/audio/speech`` form, and its 16-bit PCM stream is passed on as float32 as it arrives.

A voice comes from a recording and its transcript (cloning) or from a description (voice design).
"""

from __future__ import annotations

import asyncio
import collections
import http.client
import io
import json
import os
import re
import secrets
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

import numpy as np
from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic import Field

from providers._common import worker as common
from providers._common.worker import LoadRequest, WorkerFailure

PROVIDER = "breeze"
SCHEMA_VERSION = common.SCHEMA_VERSION
MODEL_ID = "breeze/tts-2"
# The port the image builds (providers/breeze/Dockerfile, BREEZE_REF and BREEZE_COMMIT); reported in /health.
PORT_REF = "v0.1.7"
PORT_COMMIT = "4110a9fff4e47d4d1ba850d92d81f3ba86bf39c6"
WEIGHTS_REPO = "HoppouAI/Breeze-TTS-2.cpp"
WEIGHTS_REVISION = "e4fae6493ffb8bbeacc402d1b36f18dfa18fc8d5"
WEIGHTS_FILE = "breeze-tts-2-q8_0.gguf"
SAMPLE_RATE = 24000

SERVER_BIN = os.environ.get("BREEZE_SERVER_BIN", "breeze-server")
SERVER_PORT = int(os.environ.get("BREEZE_SERVER_PORT", "7863"))
VOICES_DIR = os.environ.get("BREEZE_VOICES_DIR", "/tmp/breeze-voices")
# The CUDA architectures the image's ggml was built for (the Dockerfile sets it); empty skips the check.
CUDA_ARCHITECTURES = os.environ.get("BREEZE_CUDA_ARCHITECTURES", "")
MAX_INPUT_CHARS = int(os.environ.get("BREEZE_MAX_INPUT_CHARS", "4096"))
MAX_REFERENCE_BYTES = int(os.environ.get("BREEZE_MAX_REFERENCE_MB", "100")) * 1024 * 1024
MAX_REFERENCE_SECONDS = float(os.environ.get("BREEZE_MAX_REFERENCE_SECONDS", "30"))
LOAD_TIMEOUT_S = float(os.environ.get("BREEZE_LOAD_TIMEOUT_S", "600"))
REQUEST_TIMEOUT_S = float(os.environ.get("BREEZE_REQUEST_TIMEOUT_S", "600"))
# Guidance: a description needs it to be followed, a clone does not, and an inline cue such as "(laugh)" needs
# about 2 to 3 to be voiced. Past about 3 the voice turns harsh.
DESIGN_CFG = float(os.environ.get("BREEZE_DESIGN_CFG", "3.0"))
CLONE_CFG = float(os.environ.get("BREEZE_CLONE_CFG", "1.0"))
CUE_CFG = float(os.environ.get("BREEZE_CUE_CFG", "2.5"))
SEED = int(os.environ.get("BREEZE_SEED", "42"))
MIN_CFG, MAX_CFG = 1.0, 5.0
# The model's own sampling temperature (its GGUF default), sent only when a request sets one.
MIN_TEMPERATURE, MAX_TEMPERATURE, DEFAULT_TEMPERATURE = 0.1, 1.3, 0.9
TEXT_ENCODER_CPU = common.env_flag("BREEZE_TEXT_ENCODER_CPU", "false")
# How long a request waits for the server's slot (see generate) before the worker answers worker_busy.
BUSY_WAIT_S = float(os.environ.get("BREEZE_BUSY_WAIT_S", "15"))
STOP_TIMEOUT_S = 15.0
LOG_TAIL_LINES = 40

logger = common.logger

LANGUAGES = ["en", "zh"]
LANGUAGE_CODES = {code: code for code in LANGUAGES}
# The vocal events the port's README and the model card name: English in round brackets, Chinese in square ones.
INLINE_CUE = re.compile(
    r"\((?:nervous\s+)?(?:laugh(?:s|ing)?|chuckles?|sigh(?:s|ing)?|cough(?:s|ing)?|gasps?|whisper(?:s|ing)?"
    r"|clears?\s+(?:(?:his|her|their|my)\s+)?throat)\)"
    r"|\[(?:笑|咳嗽|清嗓子|叹气|嘆氣)\]",
    re.IGNORECASE,
)
OUT_OF_MEMORY_MARKERS = ("cudamalloc failed", "out of memory", "cudaerrormemoryallocation")
# Where a crash says why: the last lines before the exit. ggml's pinned-memory warning is not a failure.
OUT_OF_MEMORY_TAIL_LINES = 5

# GPU memory once loaded, in MiB (worker contract: the manifest's gpu_memory_mb): the Q8_0 weights, the codec and
# the compute buffers, measured at 4.35 GiB on an RTX 2070 SUPER.
GPU_MEMORY_MB = {MODEL_ID: 4450}

MANIFEST = {
    "schema_version": SCHEMA_VERSION,
    "provider": PROVIDER,
    "models": [
        {
            "id": MODEL_ID,
            "provider": PROVIDER,
            "sample_rate": SAMPLE_RATE,
            "revision": WEIGHTS_REVISION,
            "max_input_chars": MAX_INPUT_CHARS,
            "gpu_memory_mb": GPU_MEMORY_MB[MODEL_ID],
            "capabilities": {
                "voice_blend": False,
                "voice_design": True,
                "voice_clone": True,
                "reference_audio": True,
                "clone_transcript": True,
                "clone_transcript_required": True,
                "streaming": True,
                # A voice description without a recording; how to speak with one.
                "instructions": True,
                "languages": list(LANGUAGES),
                "speed_control": False,
                "speed_mode": "none",
                "ssml": False,
                "batch": False,
                "cancellation": "between_chunks",
                "live_reader": True,
                "native_tags": True,
                "speakers": [],
                # Request fields only this model reads; a null default is chosen per request (guidance()).
                "options": {
                    "cfg_scale": {
                        "type": "number", "label": "Strictness", "min": MIN_CFG, "max": MAX_CFG, "step": 0.5,
                        "default": None,
                        "description": "How closely it follows the description or direction, and whether sound tags "
                                       "such as (laughs) are voiced. Auto: 3 to design, 1 to clone, 2.5 with a tag. "
                                       "Above 3 can sound harsh.",
                    },
                    "temperature": {
                        "type": "number", "label": "Expressiveness", "min": MIN_TEMPERATURE, "max": MAX_TEMPERATURE,
                        "step": 0.05, "default": DEFAULT_TEMPERATURE,
                        "description": "Lower is steadier and more even, higher is livelier but can drift.",
                    },
                },
            },
            "voices": [],
        }
    ],
}


def _validate_settings() -> None:
    if not 0 < MAX_INPUT_CHARS <= 4096:
        raise RuntimeError("BREEZE_MAX_INPUT_CHARS must be from 1 to 4096")
    for name, value in (("BREEZE_DESIGN_CFG", DESIGN_CFG), ("BREEZE_CLONE_CFG", CLONE_CFG), ("BREEZE_CUE_CFG", CUE_CFG)):
        if not MIN_CFG <= value <= MAX_CFG:
            raise RuntimeError(f"{name} must be from {MIN_CFG:g} to {MAX_CFG:g}")


class SynthesisRequest(common.SynthesisRequest):
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    cfg_scale: float | None = Field(default=None, ge=MIN_CFG, le=MAX_CFG, allow_inf_nan=False)
    temperature: float | None = Field(default=None, ge=MIN_TEMPERATURE, le=MAX_TEMPERATURE, allow_inf_nan=False)


# ── The GPU, read with nvidia-smi: the worker runs no CUDA code of its own ─────────────────────────────────────────


def _nvidia_smi(*query: str) -> list[str]:
    """One CSV row of `nvidia-smi --query-gpu` for the first GPU, or [] when there is none to read."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--id=0", f"--query-gpu={','.join(query)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    lines = output.strip().splitlines()
    return [value.strip() for value in lines[0].split(",")] if lines else []


def gpu_info() -> dict[str, Any]:
    """The /health gpu block: name, compute capability and memory, or {"available": False}."""
    row = _nvidia_smi("name", "compute_cap", "memory.total", "memory.used")
    if len(row) != 4:
        return {"available": False}
    name, capability, total, used = row
    gpu: dict[str, Any] = {"available": True, "name": name}
    try:
        major, minor = capability.split(".", 1)
        gpu["capability"] = [int(major), int(minor)]
    except ValueError:
        pass
    for key, value in (("memory_total_mb", total), ("memory_used_mb", used)):
        try:
            gpu[key] = round(float(value), 1)
        except ValueError:
            pass
    return gpu


def built_arch_list(setting: str | None = None) -> list[str]:
    """CMAKE_CUDA_ARCHITECTURES ("75", "75-real;86-virtual") as the names arch_failure reads: sm_75, compute_86."""
    names = []
    setting = CUDA_ARCHITECTURES if setting is None else setting
    for entry in re.split(r"[;,\s]+", setting.strip()):
        match = re.fullmatch(r"(\d+)(?:-(real|virtual))?", entry)
        if match is None:
            continue
        number, kind = match.groups()
        # A bare number is both: the SASS for that GPU and the PTX newer ones compile at run time.
        if kind in (None, "real"):
            names.append(f"sm_{number}")
        if kind in (None, "virtual"):
            names.append(f"compute_{number}")
    return names


def arch_failure(gpu: dict[str, Any]) -> WorkerFailure | None:
    capability = gpu.get("capability")
    arch_list = built_arch_list()
    if not CUDA_ARCHITECTURES.strip() or not capability:
        return None
    failure = common.arch_failure((capability[0], capability[1]), arch_list)
    if failure is None:
        return None
    return WorkerFailure(
        f"The Breeze image's CUDA build has no code that runs on sm_{capability[0]}{capability[1]} "
        f"(built for: {' '.join(arch_list) or 'none'}); rebuild it with "
        f"--build-arg BREEZE_CUDA_ARCHITECTURES={capability[0]}{capability[1]}",
        code="cuda_arch_unsupported",
        status_code=503,
    )


# ── Requests ───────────────────────────────────────────────────────────────────────────────────────────────────────


def _reference_wav(audio: bytes) -> bytes:
    """The reference as 16-bit PCM WAV, the format the port reads, after the contract's checks."""
    import soundfile as sf

    info = common.reference_audio_info(audio)
    if info.duration > MAX_REFERENCE_SECONDS:
        raise WorkerFailure(
            f"Breeze TTS 2 reference audio cannot exceed {MAX_REFERENCE_SECONDS:g} seconds",
            code="reference_audio_too_long",
            status_code=400,
        )
    samples, rate = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
    output = io.BytesIO()
    sf.write(output, samples.mean(axis=1), rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def guidance(text: str, *, cloning: bool, given: float | None = None) -> float:
    if given is not None:
        return given
    scale = CLONE_CFG if cloning else DESIGN_CFG
    if INLINE_CUE.search(text):
        scale = max(scale, CUE_CFG)
    return scale


def _random_seed() -> int:
    return secrets.randbelow(2**31)


def request_form(payload: SynthesisRequest) -> tuple[dict[str, str], bytes | None]:
    """The port's form fields and reference WAV for a contract request; every refusal comes before any work."""
    if payload.speed != 1.0:
        raise WorkerFailure(
            "Breeze TTS 2 does not support numeric speed control",
            code="speed_unsupported",
            status_code=400,
        )
    if payload.language and payload.language.strip():
        if common.language_name(payload.language, LANGUAGE_CODES) is None:
            raise WorkerFailure(
                f"Unsupported language '{payload.language}' for {MODEL_ID}. Supported: {', '.join(LANGUAGES)}",
                code="language_unsupported",
                status_code=400,
            )
    description = (payload.voice_design or "").strip()
    instructions = (payload.instructions or "").strip()
    fields = {"text": payload.text, "seed": str(SEED if payload.seed is None else payload.seed)}
    if payload.temperature is not None:
        fields["temperature"] = f"{payload.temperature:g}"
    if payload.reference_audio:
        if description:
            raise WorkerFailure(
                "Breeze TTS 2 takes a recording or a voice description, not both",
                code="voice_design_with_reference",
                status_code=400,
            )
        audio = common.decode_reference_audio(
            payload.reference_audio,
            max_bytes=MAX_REFERENCE_BYTES,
            missing_message="reference_audio is required for Breeze TTS 2 voice cloning",
        )
        transcript = (payload.clone_transcript or "").strip()
        if not transcript:
            raise WorkerFailure(
                "clone_transcript is required for Breeze TTS 2 voice cloning",
                code="clone_transcript_required",
                status_code=400,
            )
        reference = _reference_wav(audio)
        fields["ref_text"] = transcript
        if instructions:
            fields["instruction"] = instructions
        fields["cfg_scale"] = f"{guidance(payload.text, cloning=True, given=payload.cfg_scale):g}"
        return fields, reference
    instruction = " ".join(part for part in (description, instructions) if part)
    if not instruction:
        raise WorkerFailure(
            "Breeze TTS 2 needs a voice: a recording with its transcript, or a voice description. "
            f"GET /v1/audio/voices?model={MODEL_ID} lists the named voices it can speak in (voice:<id>)",
            code="voice_required",
            status_code=400,
        )
    # A description gives a new voice each time, unless the caller names a seed to get one back.
    if payload.seed is None:
        fields["seed"] = str(_random_seed())
    fields["instruction"] = instruction
    fields["cfg_scale"] = f"{guidance(payload.text, cloning=False, given=payload.cfg_scale):g}"
    return fields, None


def multipart_body(fields: dict[str, str], reference: bytes | None) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n'
            f"Content-Type: text/plain; charset=utf-8\r\n\r\n".encode() + value.encode("utf-8") + b"\r\n"
        )
    if reference is not None:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="ref_audio"; filename="reference.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n".encode() + reference + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def pcm16_to_float32(data: bytes) -> np.ndarray:
    return (np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0).astype("<f4", copy=False)


# ── The breeze-server process ──────────────────────────────────────────────────────────────────────────────────────


def _weights_path() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=WEIGHTS_REPO, filename=WEIGHTS_FILE, revision=WEIGHTS_REVISION)


def server_command(weights: str) -> list[str]:
    command = [
        SERVER_BIN, weights,
        "--host", "127.0.0.1",
        "--port", str(SERVER_PORT),
        "--ws-port", "-1",
        "--voices-dir", VOICES_DIR,
        "--parallel", "1",
    ]
    if TEXT_ENCODER_CPU:
        command.append("--te-cpu")
    return command


def _start_server(command: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )


def _signal(process: subprocess.Popen, signum: int) -> None:
    """Signal the server's process group (it started its own session), or the process alone."""
    try:
        os.killpg(process.pid, signum)
    except (AttributeError, OSError):
        process.send_signal(signum)


_pause = time.sleep


def _connection(timeout: float) -> http.client.HTTPConnection:
    return http.client.HTTPConnection("127.0.0.1", SERVER_PORT, timeout=timeout)


def _server_health(timeout: float = 2.0) -> dict[str, Any] | None:
    connection = _connection(timeout)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            return None
        decoded = json.loads(body.decode("utf-8"))
        return decoded if isinstance(decoded, dict) else None
    except (OSError, http.client.HTTPException, ValueError):
        return None
    finally:
        connection.close()


def _gpu_used_mb() -> float | None:
    value = gpu_info().get("memory_used_mb")
    return float(value) if isinstance(value, (int, float)) else None


class BreezeRuntime(common.WorkerRuntime):
    display_name = "Breeze TTS 2"
    gpu_memory_need_mb = GPU_MEMORY_MB

    def __init__(self) -> None:
        super().__init__()
        self.process: subprocess.Popen | None = None
        self.weights: str | None = None
        self.model_mb: float | None = None
        self.log_tail: collections.deque[str] = collections.deque(maxlen=LOG_TAIL_LINES)
        self.drainer: threading.Thread | None = None
        self.exit_code: int | None = None
        # Health, a load and a stream each read and clear the process fields, from their own threads.
        self.process_lock = threading.Lock()

    def is_known_model(self, model_id: str) -> bool:
        return model_id == MODEL_ID

    def _drain(self, process: subprocess.Popen) -> None:
        stream = process.stdout
        if stream is None:
            return
        for line in stream:
            line = line.rstrip()
            if line:
                self.log_tail.append(line)
                logger.info("breeze-server: %s", line)

    def _ran_out_of_memory(self) -> bool:
        for line in list(self.log_tail)[-OUT_OF_MEMORY_TAIL_LINES:]:
            line = line.lower()
            if "pinned memory" not in line and any(marker in line for marker in OUT_OF_MEMORY_MARKERS):
                return True
        return False

    def download(self, model_id: str) -> None:
        """Fetch the weights into the cache without starting the server, so no GPU memory is held meanwhile."""
        if not self.is_known_model(model_id):
            raise WorkerFailure(f"Unknown model: {model_id}", code="unknown_model", status_code=400)
        try:
            _weights_path()
        except Exception as exc:
            raise WorkerFailure(f"Failed to fetch {WEIGHTS_FILE}: {exc}", code="model_download_failed") from exc

    def server_exited(self) -> bool:
        """Whether the loaded server has stopped by itself; the worker then holds no model."""
        with self.process_lock:
            process, drainer = self.process, self.drainer
            if process is None or process.poll() is None:
                return False
            self.exit_code = process.returncode
        if drainer is not None:
            drainer.join(timeout=2.0)
        with self.process_lock:
            if self.process is process:
                logger.warning("breeze-server exited with %s", self.exit_code)
                self._clear()
        return True

    def _exit_failure(self, action: str) -> WorkerFailure:
        if self._ran_out_of_memory():
            return WorkerFailure(
                f"CUDA out of memory while {action} Breeze TTS 2",
                code=common.OUT_OF_MEMORY_CODE,
                status_code=common.OUT_OF_MEMORY_STATUS,
            )
        last = self.log_tail[-1] if self.log_tail else "no output"
        return WorkerFailure(
            f"breeze-server stopped while {action} the model (exit {self.exit_code}): {last}",
            code="model_load_failed" if action == "loading" else "generation_failed",
        )

    def load(self, model_id: str) -> None:
        if not self.is_known_model(model_id):
            raise WorkerFailure(f"Unknown model: {model_id}", code="unknown_model", status_code=400)
        self.server_exited()
        with self.process_lock:
            if self.model_id == model_id and self.process is not None:
                return
        self.unload()
        gpu = gpu_info()
        if not gpu["available"]:
            raise WorkerFailure("CUDA is unavailable", code="cuda_unavailable", status_code=503)
        failure = arch_failure(gpu)
        if failure is not None:
            raise failure
        started = time.perf_counter()
        try:
            weights = _weights_path()
        except Exception as exc:
            raise WorkerFailure(f"Failed to fetch {WEIGHTS_FILE}: {exc}", code="model_load_failed") from exc
        used_before = _gpu_used_mb()
        self.log_tail.clear()
        self.exit_code = None
        try:
            process = _start_server(server_command(weights))
        except OSError as exc:
            raise WorkerFailure(f"Failed to start breeze-server: {exc}", code="model_load_failed") from exc
        drainer = threading.Thread(target=self._drain, args=(process,), daemon=True, name="breeze-server-log")
        with self.process_lock:
            self.process, self.drainer = process, drainer
        drainer.start()
        deadline = time.monotonic() + LOAD_TIMEOUT_S
        health: dict[str, Any] | None = None
        while health is None:
            if process.poll() is not None:
                self.exit_code = process.returncode
                drainer.join(timeout=2.0)  # its last lines say why
                failure = self._exit_failure("loading")
                with self.process_lock:
                    if self.process is process:
                        self._clear()
                raise failure
            if time.monotonic() > deadline:
                self.unload()
                raise WorkerFailure(
                    f"breeze-server did not come up within {LOAD_TIMEOUT_S:g} seconds",
                    code="model_load_failed",
                    status_code=504,
                )
            health = _server_health()
            if health is None:
                time.sleep(0.25)
        rate = health.get("sample_rate")
        if rate != SAMPLE_RATE:
            self.unload()
            raise WorkerFailure(
                f"Model sample rate {rate} does not match {SAMPLE_RATE}",
                code="sample_rate_mismatch",
            )
        used_after = _gpu_used_mb()
        held = used_after - used_before if used_before is not None and used_after is not None else 0.0
        with self.process_lock:
            self.model_mb = round(held, 1) if held > 0 else None
            self.weights = weights
            self.model = process
            self.model_id = model_id
            self.load_seconds = time.perf_counter() - started
            self.loaded_at = time.time()
            self.last_used_at = None

    def _clear(self) -> None:
        self.process = None
        self.model = None
        self.model_id = None
        self.weights = None
        self.model_mb = None
        self.loaded_at = None
        self.last_used_at = None
        self.load_seconds = None

    def unload(self) -> None:
        """Stop the server; its GPU memory goes back to the driver when the process exits."""
        with self.process_lock:
            process = self.process
            self._clear()
        if process is None or process.poll() is not None:
            return
        _signal(process, signal.SIGTERM)
        try:
            process.wait(timeout=STOP_TIMEOUT_S)
            return
        except subprocess.TimeoutExpired:
            _signal(process, getattr(signal, "SIGKILL", signal.SIGTERM))
        try:
            process.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            # It still holds its GPU memory: kept, so the next load or unload stops it again.
            logger.warning("breeze-server (pid %s) did not stop", process.pid)
            with self.process_lock:
                if self.process is None:
                    self.process = process

    def held_mb(self) -> float | None:
        """The MiB the server holds: measured across its load, else the manifest's figure."""
        if self.model_id is None:
            return None
        return self.model_mb or float(GPU_MEMORY_MB[self.model_id])

    def generate(self, payload: SynthesisRequest) -> Iterator[np.ndarray]:
        """Yield the server's audio as it streams; every refusal comes before any generation."""
        fields, reference = request_form(payload)
        if self.server_exited():
            raise WorkerFailure(f"Model {payload.model} is not loaded", code="model_not_loaded", status_code=409)
        self.require_model(payload.model)
        body, content_type = multipart_body(fields, reference)
        # A stream cancelled just before may still hold the server's one slot: it answers 409 until that ends.
        deadline = time.monotonic() + BUSY_WAIT_S
        wait = 0.1
        while True:
            connection = _connection(REQUEST_TIMEOUT_S)
            try:
                connection.request(
                    "POST", "/v1/audio/speech", body=body,
                    headers={"Content-Type": content_type, "Accept": "audio/pcm"},
                )
                response = connection.getresponse()
            except (OSError, http.client.HTTPException) as exc:
                connection.close()
                raise self._stream_failure(exc) from exc
            if response.status != 409 or time.monotonic() + wait > deadline:
                break
            response.read(65536)
            connection.close()
            _pause(wait)
            wait = min(wait * 2, 1.0)
        try:
            if response.status != 200:
                raise self._server_error(response.status, response.read(65536))
            rate = response.getheader("X-Sample-Rate")
            if rate is not None and rate.strip() != str(SAMPLE_RATE):
                raise WorkerFailure(
                    f"Model returned {rate} Hz; manifest declares {SAMPLE_RATE} Hz",
                    code="sample_rate_mismatch",
                )
            sample_format = response.getheader("X-Sample-Format")
            if sample_format is not None and sample_format.strip().lower() != "s16le":
                raise WorkerFailure(
                    f"Model returned {sample_format} audio; s16le was expected",
                    code="audio_format_mismatch",
                )
            remainder = b""
            produced = False
            while True:
                try:
                    data = response.read1(65536)
                except (OSError, http.client.HTTPException) as exc:
                    raise self._stream_failure(exc) from exc
                if not data:
                    break
                data = remainder + data
                usable = len(data) - len(data) % 2
                remainder = data[usable:]
                if usable:
                    produced = True
                    self.last_used_at = time.time()
                    yield pcm16_to_float32(data[:usable])
            if not produced:
                raise WorkerFailure("Model returned no audio", code="invalid_audio")
        finally:
            connection.close()

    def _stream_failure(self, exc: BaseException) -> WorkerFailure:
        if self.server_exited():
            return self._exit_failure("running")
        return WorkerFailure(
            f"Breeze TTS 2 stream ended early: {exc}",
            code="generation_failed",
            status_code=502,
        )

    def _server_error(self, status: int, raw: bytes) -> WorkerFailure:
        message = f"breeze-server answered HTTP {status}"
        try:
            decoded = json.loads(raw.decode("utf-8"))
            error = decoded.get("error") if isinstance(decoded, dict) else None
            if isinstance(error, dict):
                error = error.get("message")
            if isinstance(error, str) and error:
                message = f"Breeze TTS 2 refused the request: {error}"
        except (UnicodeDecodeError, ValueError):
            pass
        if status == 409:
            return WorkerFailure("Worker is busy", code="worker_busy", status_code=429)
        if status == 400:
            return WorkerFailure(message, code="invalid_request", status_code=400)
        return WorkerFailure(message, code="generation_failed", status_code=502)


_validate_settings()
runtime = BreezeRuntime()
operation_lock = asyncio.Lock()
app = common.create_app("Open Speech Breeze TTS 2 Worker")


def health_payload(*, busy: bool) -> dict[str, Any]:
    if not busy:
        runtime.server_exited()
    process = runtime.process
    gpu = gpu_info()
    failure = arch_failure(gpu) if gpu["available"] else None
    if failure is not None:
        status = "failed"
    else:
        status = "ok" if gpu["available"] else "unavailable"
    body = common.health_body(
        PROVIDER,
        runtime,
        status=status,
        busy=busy,
        gpu=gpu,
        details={
            "runtime": "breeze.cpp",
            "quantization": "q8_0",
            "port_ref": PORT_REF,
            "port_commit": PORT_COMMIT,
            "weights": WEIGHTS_FILE,
            "server_pid": process.pid if process is not None else None,
            "last_exit_code": runtime.exit_code,
            "failure": common.error_detail(failure) if failure else None,
        },
    )
    body["gpu_memory"]["model_mb"] = runtime.held_mb()
    return body


@app.get("/health")
async def health() -> dict[str, Any]:
    # nvidia-smi takes a moment: off the event loop, which may be streaming audio.
    return await asyncio.to_thread(health_payload, busy=operation_lock.locked())


@app.get("/v1/manifest")
async def manifest() -> dict[str, Any]:
    return MANIFEST


@app.post("/v1/models/load")
async def load_model(payload: LoadRequest) -> dict[str, Any]:
    return await common.load_model(runtime, operation_lock, payload.model)


@app.post("/v1/models/download")
async def download_model(payload: LoadRequest) -> dict[str, Any]:
    return await common.download_model(runtime, payload.model)


@app.post("/v1/models/unload")
async def unload_model(payload: LoadRequest) -> dict[str, str]:
    return await common.unload_model(runtime, operation_lock, payload.model)


@app.post("/v1/audio/speech")
async def synthesize(payload: SynthesisRequest, request: Request) -> StreamingResponse:
    if payload.model != MODEL_ID:
        raise common.unknown_model_error(payload.model)
    common.ensure_idle(operation_lock)
    await operation_lock.acquire()
    return await common.stream_chunks(
        runtime.generate(payload),
        lock=operation_lock,
        request=request,
        sample_rate=SAMPLE_RATE,
        cancellation="between_chunks",
    )
