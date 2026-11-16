"""Generic HTTP adapter for isolated TTS provider workers."""

from __future__ import annotations

import base64
import contextlib
import http.client
import json
import math
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from src.tts.backends.base import (
    MAX_GPU_MEMORY_MB,
    TTSLoadedModelInfo,
    TTSModelManifest,
    VoiceInfo,
    check_declared_language,
)

MANIFEST_SCHEMA_VERSION = 1
PROVIDER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
MODEL_LOAD_TIMEOUT_S = 1800.0
# A worker stays busy until the chunk in flight when a stream was cancelled finishes, so
# the next request (such as the next Live Reader segment) waits out that window.
WORKER_BUSY_RETRY_S = 2.0
WORKER_BUSY_MAX_BACKOFF_S = 0.5
# A worker that couldn't be reached (a timeout, a refused connection, no such host) is not read
# again, by health or manifest, for the failed read's window (1 s health, 2 s manifest) or four
# times as long as the read took to fail, whichever is longer, at most 30 s. One model listing
# reads every configured worker in turn, so a slow failure must outlast the other workers' reads.
UNREACHABLE_HOLD_FACTOR = 4.0
UNREACHABLE_HOLD_MAX_S = 30.0
OUT_OF_MEMORY_CODES = {"out_of_memory", "cuda_out_of_memory"}
# A failed worker's /health reason, passed on in load answers and model rows: a code word and one bounded line.
HEALTH_FAILURE_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
HEALTH_FAILURE_MESSAGE_MAX_CHARS = 300


def _health_failure(health: dict[str, Any]) -> dict[str, str] | None:
    """The {code, message} of a /health whose status is "failed", bounded; None for any other health.

    A worker sends it for a state no load gets past (a torch build with no code for the GPU, a poisoned CUDA
    context, a dtype the GPU can't run), with the code and message its load's 503 gives. It comes from the
    worker's own health, never from a request.
    """
    failure = health.get("failure") if health.get("status") == "failed" else None
    if not isinstance(failure, dict):
        return None
    code, message = failure.get("code"), failure.get("message")
    if not isinstance(code, str) or not HEALTH_FAILURE_CODE_RE.fullmatch(code) or not isinstance(message, str):
        return None
    message = message[: HEALTH_FAILURE_MESSAGE_MAX_CHARS * 4]
    message = " ".join("".join(ch if ch.isprintable() else " " for ch in message).split())
    if len(message) > HEALTH_FAILURE_MESSAGE_MAX_CHARS:
        message = message[: HEALTH_FAILURE_MESSAGE_MAX_CHARS - 1].rstrip() + "…"
    return {"code": code, "message": message} if message else None


def _health_model_memory_mb(health: dict[str, Any]) -> float | None:
    """/health's gpu_memory.model_mb; None when it is missing or not a size."""
    memory = health.get("gpu_memory")
    value = memory.get("model_mb") if isinstance(memory, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    if not 0 < value <= MAX_GPU_MEMORY_MB:
        return None
    return float(value)


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Keep configured worker traffic on the exact configured origin."""

    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


@dataclass
class ExternalProviderError(RuntimeError):
    """Typed provider failure that API services can map without guessing."""

    message: str
    code: str = "external_provider_error"
    status_code: int = 502

    def __str__(self) -> str:
        return self.message


def _validation_problem(item: Any) -> str:
    """Describe one FastAPI request-validation entry, whose loc starts with "body"."""
    if not isinstance(item, dict):
        return str(item)
    loc = item.get("loc")
    field = ".".join(str(part) for part in loc[1:]) if isinstance(loc, list) else ""
    return f"{field}: {item.get('msg')}" if field else str(item.get("msg"))


def _retry_after_seconds(exc: urllib.error.HTTPError) -> float | None:
    """Read a Retry-After given in seconds; an HTTP date is too coarse for this window."""
    headers = getattr(exc, "headers", None)
    try:
        seconds = float(headers.get("Retry-After")) if headers is not None else None
    except (TypeError, ValueError):
        return None
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def parse_external_provider_urls(raw: str) -> dict[str, str]:
    """Parse the config-only JSON mapping of provider IDs to worker URLs."""
    if not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("TTS_EXTERNAL_PROVIDERS must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("TTS_EXTERNAL_PROVIDERS must be a JSON object")

    providers: dict[str, str] = {}
    for provider, url in payload.items():
        if not isinstance(provider, str) or not PROVIDER_ID_RE.fullmatch(provider.strip()):
            raise ValueError(
                "External provider IDs must use lowercase letters, numbers, underscores, or hyphens"
            )
        if not isinstance(url, str):
            raise ValueError(f"External provider URL for '{provider}' must be a string")
        normalized_url = url.strip().rstrip("/")
        parsed = urllib.parse.urlsplit(normalized_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"External provider URL for '{provider}' must be HTTP(S)")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                f"External provider URL for '{provider}' cannot contain credentials, query, or fragment"
            )
        try:
            valid_port = parsed.port != 0
        except ValueError:  # not a number, or outside 0-65535
            valid_port = False
        if not valid_port:
            raise ValueError(f"External provider URL for '{provider}' has an invalid port")
        providers[provider.strip()] = normalized_url
    return providers


class ExternalTTSBackend:
    """TTS backend whose ML runtime lives behind a private worker URL."""

    sample_rate = 24000
    requires_model_id = True

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        device: str = "external",
        metadata_timeout_s: float = 3.0,
        synthesis_timeout_s: float = 600.0,
        busy_retry_s: float = WORKER_BUSY_RETRY_S,
    ) -> None:
        self.name = provider
        self._base_url = base_url.rstrip("/")
        self._device = device
        self._metadata_timeout_s = metadata_timeout_s
        self._synthesis_timeout_s = synthesis_timeout_s
        self._busy_retry_s = busy_retry_s
        self._manifest_lock = threading.RLock()
        self._manifests: dict[str, TTSModelManifest] = {}
        self._manifest_error: ExternalProviderError | None = None
        self._manifest_checked_at = 0.0
        self._health: dict[str, Any] | None = None
        self._health_error: ExternalProviderError | None = None
        self._health_checked_at = 0.0
        # The last health or manifest read that couldn't reach the worker; it answers both reads
        # until _unreachable_until.
        self._unreachable_error: ExternalProviderError | None = None
        self._unreachable_until = 0.0
        # How many times the worker has answered a read, load or unload. Reads run on several threads
        # at once, so a read that couldn't reach the worker and saw an answer arrive while it ran is
        # out of date when it returns, and is not stored; the check and the store share _reach_lock
        # with the answer. A count, not a clock: GetTickCount64 ticks only every 15.6 ms on Windows.
        self._reach_lock = threading.Lock()
        self._answers = 0
        # The worker only reports use once audio comes out, so a long or failing request
        # would otherwise look idle to the TTL and LRU checks.
        self._usage_lock = threading.Lock()
        self._active_requests = 0
        self._last_request_at: float | None = None
        # time.monotonic() of the last unload: a worker that restarts to free the memory holds it a few seconds more.
        self.last_unload_at: float | None = None
        # fetch_weights: the models whose weights the worker has fetched, and whether it has the call at all.
        self._fetched: set[str] = set()
        self._downloads = True
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _RejectRedirects(),
        )

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _open(
        self,
        request: urllib.request.Request,
        *,
        timeout: float,
        model_id: str | None = None,
    ) -> Any:
        """Open a worker request, retrying briefly while the worker answers worker_busy."""
        deadline = time.monotonic() + self._busy_retry_s
        backoff = 0.1
        while True:
            try:
                return self._opener.open(request, timeout=timeout)
            except urllib.error.HTTPError as exc:
                error = self._provider_http_error(exc, model_id=model_id)
                if error.status_code != 429 or error.code != "worker_busy":
                    raise error from exc
                retry_after = _retry_after_seconds(exc)
                wait = backoff if retry_after is None else max(retry_after, 0.05)
                if wait > deadline - time.monotonic():
                    raise error from exc
                exc.close()  # release the refused response's connection before retrying
                time.sleep(wait)
                backoff = min(backoff * 2, WORKER_BUSY_MAX_BACKOFF_S)

    def _json_request(
        self,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        method = "GET"
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
            method = "POST"
        request = urllib.request.Request(
            self._url(path), data=data, headers=headers, method=method
        )
        try:
            with self._open(
                request,
                timeout=timeout or self._metadata_timeout_s,
                model_id=payload.get("model") if payload is not None else None,
            ) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise ExternalProviderError(
                f"TTS provider '{self.name}' is unavailable: {exc}",
                code="provider_unavailable",
                status_code=503,
            ) from exc
        except http.client.HTTPException as exc:
            raise ExternalProviderError(
                f"TTS provider '{self.name}' response ended unexpectedly: {exc}",
                code="provider_stream_aborted",
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExternalProviderError(
                f"TTS provider '{self.name}' returned invalid JSON",
                code="invalid_provider_response",
            ) from exc
        if not isinstance(decoded, dict):
            raise ExternalProviderError(
                f"TTS provider '{self.name}' returned a non-object response",
                code="invalid_provider_response",
            )
        return decoded

    def _provider_http_error(
        self, exc: urllib.error.HTTPError, *, model_id: str | None = None
    ) -> ExternalProviderError:
        raw = exc.read(65536)
        message = f"TTS provider '{self.name}' request failed with HTTP {exc.code}"
        code = "external_provider_error"
        try:
            payload = json.loads(raw.decode("utf-8"))
            detail = payload.get("detail") if isinstance(payload, dict) else None
            if detail is None and isinstance(payload, dict):
                # Typed worker errors may also use the {"error": {"code", "message"}} envelope.
                detail = payload.get("error")
            if isinstance(detail, dict):
                message = str(detail.get("message") or message)
                code = str(detail.get("code") or code)
            elif isinstance(detail, str):
                message = detail
            elif isinstance(detail, list) and detail:
                problems = "; ".join(_validation_problem(item) for item in detail)
                message = f"TTS provider '{self.name}' rejected the request: {problems}"
                code = "invalid_request"
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        if exc.code == 507 or code in OUT_OF_MEMORY_CODES:
            return ExternalProviderError(
                f"Not enough GPU memory to run {model_id or 'the model'} on TTS provider "
                f"'{self.name}'. Unload other GPU models or choose a smaller model.",
                code="insufficient_gpu_memory",
                status_code=503,
            )
        # A worker 422 rejects a value the caller chose (such as a speed outside the
        # model's own range), so report it like the workers' other per-model 400s.
        status_code = 400 if exc.code == 422 else exc.code
        if status_code not in {400, 404, 409, 413, 429, 503}:
            status_code = 502
        return ExternalProviderError(message, code=code, status_code=status_code)

    @staticmethod
    def _never_reached(exc: ExternalProviderError) -> bool:
        cause = exc.__cause__
        # An HTTP error is the worker answering (or sending something unreadable): it is reachable.
        return isinstance(cause, OSError) and not isinstance(cause, urllib.error.HTTPError)

    def _out_of_date(self, exc: ExternalProviderError, answers_at_start: int) -> bool:
        """A read that never reached the worker, while the worker answered another. Hold _reach_lock."""
        return self._never_reached(exc) and self._answers != answers_at_start

    def _mark_answered(self) -> None:
        """The worker answered; clear the remembered unreachable read. Hold _reach_lock."""
        self._answers += 1
        self._unreachable_error = None

    def _remember_unreachable(self, exc: ExternalProviderError, started: float, window: float) -> None:
        """Keep a read that never reached the worker, so the next reads of either endpoint don't repeat it.

        Hold _reach_lock, and call it only for a read that isn't _out_of_date.
        """
        if not self._never_reached(exc):
            return
        returned = time.monotonic()
        hold = min(UNREACHABLE_HOLD_MAX_S, max(window, UNREACHABLE_HOLD_FACTOR * (returned - started)))
        self._unreachable_error = exc
        self._unreachable_until = returned + hold

    def _raise_if_unreachable(self, now: float) -> None:
        if self._unreachable_error is not None and now < self._unreachable_until:
            raise self._unreachable_error

    def _refresh_manifests(self, *, force: bool = False) -> None:
        with self._manifest_lock:
            now = time.monotonic()
            if not force and self._manifests and now - self._manifest_checked_at < 30.0:
                return
            if not force:
                self._raise_if_unreachable(now)
            if not force and self._manifest_error and now - self._manifest_checked_at < 2.0:
                raise self._manifest_error
            answers = self._answers
            try:
                payload = self._json_request("/v1/manifest")
                if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
                    raise ExternalProviderError(
                        f"TTS provider '{self.name}' uses unsupported manifest schema "
                        f"{payload.get('schema_version')!r}",
                        code="manifest_version_unsupported",
                    )
                if payload.get("provider") != self.name:
                    raise ExternalProviderError(
                        f"Configured provider '{self.name}' identified itself as "
                        f"'{payload.get('provider')}'",
                        code="provider_identity_mismatch",
                    )
                raw_models = payload.get("models")
                if not isinstance(raw_models, list) or not raw_models:
                    raise ExternalProviderError(
                        f"TTS provider '{self.name}' returned no model manifests",
                        code="invalid_manifest",
                    )
                manifests: dict[str, TTSModelManifest] = {}
                for item in raw_models:
                    if not isinstance(item, dict):
                        raise ValueError("Model manifests must be objects")
                    manifest = TTSModelManifest.from_dict(item)
                    if manifest.provider != self.name:
                        raise ValueError(
                            f"Model '{manifest.id}' declares provider '{manifest.provider}'"
                        )
                    if manifest.id in manifests:
                        raise ValueError(f"Duplicate model manifest '{manifest.id}'")
                    manifests[manifest.id] = manifest
            # A failure is stamped when it returns, not when the read began: a read that ran
            # out its timeout is otherwise already older than the failure cache.
            except ExternalProviderError as exc:
                with self._reach_lock:
                    if not self._out_of_date(exc, answers):
                        self._manifest_error = exc
                        self._manifest_checked_at = time.monotonic()
                        self._remember_unreachable(exc, now, 2.0)
                raise
            except ValueError as exc:
                error = ExternalProviderError(
                    f"Invalid manifest from TTS provider '{self.name}': {exc}",
                    code="invalid_manifest",
                )
                self._manifest_error = error
                self._manifest_checked_at = time.monotonic()
                raise error from exc
            with self._reach_lock:
                self._manifests = manifests
                self._manifest_error = None
                self._manifest_checked_at = now
                self._mark_answered()

    def get_model_manifest(self, model_id: str) -> TTSModelManifest:
        self._refresh_manifests()
        manifest = self._manifests.get(model_id)
        if manifest is None:
            raise ValueError(f"Unknown {self.name} TTS model: {model_id}")
        return manifest

    def supports_model(self, model_id: str) -> bool:
        try:
            self._refresh_manifests()
            return model_id in self._manifests
        except ExternalProviderError:
            # Preserve typed provider-unavailable errors when a configured worker
            # is down instead of misrouting its models to STT.
            return model_id.startswith(f"{self.name}/")

    def advertises_model(self, model_id: str) -> bool | None:
        """Catalog membership; None means the worker cannot be checked right now."""
        try:
            self._refresh_manifests()
        except ExternalProviderError:
            if not self._manifests:
                return None
        return model_id in self._manifests

    def get_capabilities(self, model_id: str) -> dict[str, Any]:
        return dict(self.get_model_manifest(model_id).capabilities)

    def gpu_memory_need_mb(self, model_id: str) -> int | None:
        """The manifest's gpu_memory_mb; None when the worker doesn't say or can't be read."""
        try:
            return self.get_model_manifest(model_id).gpu_memory_mb
        except (ExternalProviderError, ValueError):
            return None

    def get_sample_rate(self, model_id: str) -> int:
        return self.get_model_manifest(model_id).sample_rate

    def list_voices(self, model_id: str | None = None) -> list[VoiceInfo]:
        if model_id is None:
            raise ValueError("External voice catalogs require a model ID")
        return list(self.get_model_manifest(model_id).voices)

    def _get_health(self, *, force: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        if not force and now - self._health_checked_at < 1.0:
            if self._health is not None:
                return self._health
            if self._health_error is not None:
                raise self._health_error
        if not force:
            self._raise_if_unreachable(now)
        answers = self._answers
        try:
            payload = self._json_request("/health")
            if payload.get("provider") != self.name:
                raise ExternalProviderError(
                    f"TTS provider '{self.name}' health identity mismatch",
                    code="provider_identity_mismatch",
                )
        except ExternalProviderError as exc:
            # Stamped when the failure returns, as for the manifest above.
            with self._reach_lock:
                if not self._out_of_date(exc, answers):
                    self._health = None
                    self._health_error = exc
                    self._health_checked_at = time.monotonic()
                    self._remember_unreachable(exc, now, 1.0)
            raise
        with self._reach_lock:
            self._health = payload
            self._health_error = None
            self._health_checked_at = now
            self._mark_answered()
        return payload

    def is_provider_available(self) -> bool:
        try:
            return self._get_health().get("status") == "ok"
        except ExternalProviderError:
            return False

    def provider_failure(self) -> dict[str, str] | None:
        """Why the worker says it can't run, {code, message} from its last /health, or None: it is ok, only
        unavailable, or doesn't answer (then it gives no reason)."""
        try:
            return _health_failure(self._get_health())
        except ExternalProviderError:
            return None

    def fetch_weights(self, model_id: str) -> bool:
        """Have the worker download the model's weights without loading it (POST /v1/models/download), so a first
        download holds no GPU memory; False for a worker without that call, which downloads as it loads."""
        if model_id in self._fetched:
            return True
        if not self._downloads:
            return False
        self.get_model_manifest(model_id)
        try:
            self._json_request(
                "/v1/models/download",
                payload={"model": model_id},
                timeout=max(self._synthesis_timeout_s, MODEL_LOAD_TIMEOUT_S),
            )
        except ExternalProviderError as exc:
            if exc.status_code == 404 and exc.code == "external_provider_error":
                self._downloads = False
                return False
            raise
        self._fetched.add(model_id)
        return True

    def load_model(self, model_id: str) -> None:
        self.get_model_manifest(model_id)
        self._json_request(
            "/v1/models/load",
            payload={"model": model_id},
            # A first load may need to download several gigabytes. Keep the
            # core request alive while the worker owns that operation so a
            # client does not receive a timeout while the model keeps loading.
            timeout=max(self._synthesis_timeout_s, MODEL_LOAD_TIMEOUT_S),
        )
        with self._reach_lock:
            self._health = None
            self._health_error = None
            self._mark_answered()
        with self._usage_lock:
            self._last_request_at = None

    def unload_model(self, model_id: str) -> None:
        self._json_request(
            "/v1/models/unload",
            payload={"model": model_id},
            timeout=self._synthesis_timeout_s,
        )
        self.last_unload_at = time.monotonic()
        with self._reach_lock:
            self._health = None
            self._health_error = None
            self._mark_answered()

    def is_model_loaded(self, model_id: str) -> bool:
        return self._get_health(force=True).get("loaded_model") == model_id

    def _health_timestamp(self, health: dict[str, Any], field: str) -> float | None:
        """Read an epoch-seconds /health field; drift is a typed error the router skips."""
        value = health.get(field)
        if value is None:
            return None
        seconds = math.nan
        # Epoch seconds sent as a numeric string are still epoch seconds.
        if isinstance(value, (int, float, str)) and not isinstance(value, bool):
            with contextlib.suppress(ValueError):
                seconds = float(value)
        if not math.isfinite(seconds):
            raise ExternalProviderError(
                f"TTS provider '{self.name}' reported an invalid {field} in /health",
                code="invalid_provider_response",
            )
        return seconds

    def loaded_models(self) -> list[TTSLoadedModelInfo]:
        health = self._get_health()
        model_id = health.get("loaded_model")
        if not isinstance(model_id, str) or not model_id:
            return []
        last_used_at = self._health_timestamp(health, "last_used_at")
        loaded_at = self._health_timestamp(health, "loaded_at")
        with self._usage_lock:
            if self._active_requests or health.get("busy"):
                last_used_at = time.time()
            elif self._last_request_at is not None:
                last_used_at = max(last_used_at or 0.0, self._last_request_at)
        return [
            TTSLoadedModelInfo(
                model=model_id,
                backend=self.name,
                device=str(health.get("device") or self._device),
                loaded_at=loaded_at or time.time(),
                last_used_at=last_used_at,
                gpu_memory_mb=_health_model_memory_mb(health),
            )
        ]

    @contextlib.contextmanager
    def _in_use(self) -> Iterator[None]:
        with self._usage_lock:
            self._active_requests += 1
        try:
            yield
        finally:
            with self._usage_lock:
                self._active_requests -= 1
                self._last_request_at = time.time()

    def validate_voice(self, voice: str, model_id: str | None = None) -> None:
        if model_id is None:
            return
        voices = self.list_voices(model_id)
        if voices and voice not in {item.id for item in voices}:
            raise ValueError(
                f"Unsupported voice '{voice}' for {model_id}. "
                f"Supported: {', '.join(item.id for item in voices)}"
            )

    def validate_language(self, language: str | None, model_id: str | None = None) -> None:
        """Reject a hint outside the model's manifest languages, before the worker is asked.

        Listed language names (Qwen3's "Auto", "English", ...) count as their codes. A manifest
        that declares no languages, or lists a name check_declared_language doesn't know, leaves
        the check to the worker.
        """
        if model_id is None:
            return
        languages = self.get_model_manifest(model_id).capabilities.get("languages")
        check_declared_language(language, languages, model_id)

    def synthesize(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        lang_code: str | None = None,
        *,
        model_id: str | None = None,
        **backend_options: Any,
    ) -> Iterator[np.ndarray]:
        if not model_id:
            raise ValueError("External synthesis requires the selected model ID")
        manifest = self.get_model_manifest(model_id)
        if manifest.max_input_chars is not None and len(text) > manifest.max_input_chars:
            raise ValueError(
                f"Input too long for {model_id}. Max: {manifest.max_input_chars} characters"
            )
        payload: dict[str, Any] = {
            "model": model_id,
            "text": text,
            "voice": voice,
            "speed": speed,
            "language": lang_code,
        }
        declared = manifest.capabilities.get("options")
        option_names = tuple(
            name for name in (declared if isinstance(declared, dict) else ())
            if name not in payload and name != "reference_audio"
        )
        for key in ("instructions", "voice_design", "clone_transcript", "seed", *option_names):
            if backend_options.get(key) is not None:
                payload[key] = backend_options[key]
        reference_audio = backend_options.get("reference_audio")
        if reference_audio is not None:
            payload["reference_audio"] = base64.b64encode(reference_audio).decode("ascii")

        with self._in_use():
            yield from self._stream_speech(payload, manifest)

    def _stream_speech(
        self, payload: dict[str, Any], manifest: TTSModelManifest
    ) -> Iterator[np.ndarray]:
        request = urllib.request.Request(
            self._url("/v1/audio/speech"),
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Accept": "application/octet-stream", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._open(
                request, timeout=self._synthesis_timeout_s, model_id=manifest.id
            ) as response:
                try:
                    sample_rate = int(response.headers.get("X-Audio-Sample-Rate", "0"))
                except ValueError as exc:
                    raise ExternalProviderError(
                        f"TTS provider '{self.name}' returned an invalid sample rate",
                        code="sample_rate_mismatch",
                    ) from exc
                if sample_rate != manifest.sample_rate:
                    raise ExternalProviderError(
                        f"TTS provider '{self.name}' returned {sample_rate} Hz for "
                        f"a {manifest.sample_rate} Hz model",
                        code="sample_rate_mismatch",
                    )
                if response.headers.get("X-Audio-Format") != "float32le":
                    raise ExternalProviderError(
                        f"TTS provider '{self.name}' returned an unsupported audio format",
                        code="audio_format_mismatch",
                    )
                remainder = b""
                while True:
                    read_chunk = getattr(response, "read1", response.read)
                    data = read_chunk(65536)
                    if not data:
                        break
                    data = remainder + data
                    usable = len(data) - (len(data) % 4)
                    if usable:
                        yield np.frombuffer(data[:usable], dtype="<f4").astype(
                            np.float32, copy=True
                        )
                    remainder = data[usable:]
                if remainder:
                    raise ExternalProviderError(
                        f"TTS provider '{self.name}' returned truncated float32 audio",
                        code="truncated_audio",
                    )
        except ExternalProviderError:
            raise
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise ExternalProviderError(
                f"TTS provider '{self.name}' is unavailable: {exc}",
                code="provider_unavailable",
                status_code=503,
            ) from exc
        except http.client.HTTPException as exc:
            raise ExternalProviderError(
                f"TTS provider '{self.name}' stream ended unexpectedly: {exc}",
                code="provider_stream_aborted",
                status_code=502,
            ) from exc
