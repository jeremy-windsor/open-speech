"""Generic HTTP adapter for isolated TTS provider workers."""

from __future__ import annotations

import base64
import http.client
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from src.tts.backends.base import TTSLoadedModelInfo, TTSModelManifest, VoiceInfo


MANIFEST_SCHEMA_VERSION = 1
PROVIDER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
MODEL_LOAD_TIMEOUT_S = 1800.0


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
    ) -> None:
        self.name = provider
        self._base_url = base_url.rstrip("/")
        self._device = device
        self._metadata_timeout_s = metadata_timeout_s
        self._synthesis_timeout_s = synthesis_timeout_s
        self._manifest_lock = threading.RLock()
        self._manifests: dict[str, TTSModelManifest] = {}
        self._manifest_error: ExternalProviderError | None = None
        self._manifest_checked_at = 0.0
        self._health: dict[str, Any] | None = None
        self._health_error: ExternalProviderError | None = None
        self._health_checked_at = 0.0
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _RejectRedirects(),
        )

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

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
            with self._opener.open(
                request, timeout=timeout or self._metadata_timeout_s
            ) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            self._raise_http_error(exc)
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

    def _raise_http_error(self, exc: urllib.error.HTTPError) -> None:
        raw = exc.read(65536)
        message = f"TTS provider '{self.name}' request failed with HTTP {exc.code}"
        code = "external_provider_error"
        try:
            payload = json.loads(raw.decode("utf-8"))
            detail = payload.get("detail") if isinstance(payload, dict) else None
            if isinstance(detail, dict):
                message = str(detail.get("message") or message)
                code = str(detail.get("code") or code)
            elif isinstance(detail, str):
                message = detail
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        status_code = exc.code if exc.code in {400, 404, 409, 413, 429, 503} else 502
        raise ExternalProviderError(message, code=code, status_code=status_code) from exc

    def _refresh_manifests(self, *, force: bool = False) -> None:
        with self._manifest_lock:
            now = time.monotonic()
            if not force and self._manifests and now - self._manifest_checked_at < 30.0:
                return
            if not force and self._manifest_error and now - self._manifest_checked_at < 2.0:
                raise self._manifest_error
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
            except ExternalProviderError as exc:
                self._manifest_error = exc
                self._manifest_checked_at = now
                raise
            except ValueError as exc:
                error = ExternalProviderError(
                    f"Invalid manifest from TTS provider '{self.name}': {exc}",
                    code="invalid_manifest",
                )
                self._manifest_error = error
                self._manifest_checked_at = now
                raise error from exc
            self._manifests = manifests
            self._manifest_error = None
            self._manifest_checked_at = now

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
        try:
            payload = self._json_request("/health")
        except ExternalProviderError as exc:
            self._health = None
            self._health_error = exc
            self._health_checked_at = now
            raise
        if payload.get("provider") != self.name:
            raise ExternalProviderError(
                f"TTS provider '{self.name}' health identity mismatch",
                code="provider_identity_mismatch",
            )
        self._health = payload
        self._health_error = None
        self._health_checked_at = now
        return payload

    def is_provider_available(self) -> bool:
        try:
            return self._get_health().get("status") == "ok"
        except ExternalProviderError:
            return False

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
        self._health = None
        self._health_error = None

    def unload_model(self, model_id: str) -> None:
        self._json_request(
            "/v1/models/unload",
            payload={"model": model_id},
            timeout=self._synthesis_timeout_s,
        )
        self._health = None
        self._health_error = None

    def is_model_loaded(self, model_id: str) -> bool:
        return self._get_health(force=True).get("loaded_model") == model_id

    def loaded_models(self) -> list[TTSLoadedModelInfo]:
        health = self._get_health()
        model_id = health.get("loaded_model")
        if not isinstance(model_id, str) or not model_id:
            return []
        return [
            TTSLoadedModelInfo(
                model=model_id,
                backend=self.name,
                device=str(health.get("device") or self._device),
                loaded_at=float(health.get("loaded_at") or time.time()),
                last_used_at=(
                    float(health["last_used_at"])
                    if health.get("last_used_at") is not None
                    else None
                ),
            )
        ]

    def validate_voice(self, voice: str, model_id: str | None = None) -> None:
        if model_id is None:
            return
        voices = self.list_voices(model_id)
        if voices and voice not in {item.id for item in voices}:
            raise ValueError(
                f"Unsupported voice '{voice}' for {model_id}. "
                f"Supported: {', '.join(item.id for item in voices)}"
            )

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
        for key in ("instructions", "voice_design", "clone_transcript"):
            if backend_options.get(key) is not None:
                payload[key] = backend_options[key]
        reference_audio = backend_options.get("reference_audio")
        if reference_audio is not None:
            payload["reference_audio"] = base64.b64encode(reference_audio).decode("ascii")

        request = urllib.request.Request(
            self._url("/v1/audio/speech"),
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Accept": "application/octet-stream", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self._synthesis_timeout_s) as response:
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
        except urllib.error.HTTPError as exc:
            self._raise_http_error(exc)
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
