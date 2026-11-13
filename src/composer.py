"""Multi-track audio composer for Studio."""

from __future__ import annotations

import json
import logging
import math
import subprocess
import threading
import wave
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from src.config import settings
from src.effects.chain import apply_chain, effect_parameter_error
from src.storage import get_db, snapshot, transaction

logger = logging.getLogger(__name__)

# The mix buffer spans up to the latest track end, so offsets alone can size it.
MAX_RENDER_SECONDS = 2 * 60 * 60


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MultiTrackComposer:
    def __init__(self) -> None:
        self.output_dir = Path(settings.os_composer_dir)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            self.output_dir = (Path.cwd() / "data" / "composer").resolve()
            self.output_dir.mkdir(parents=True, exist_ok=True)
        # One render at a time, so concurrent requests cannot multiply its memory.
        self._render_lock = threading.Lock()

    def render(self, tracks: list[dict], format: str = "wav", sample_rate: int = 24000, name: str | None = None) -> dict:
        sample_rate = int(sample_rate)
        # Number tracks as the request lists them, muted ones included.
        for n, track in enumerate(tracks, start=1):
            effect_error = effect_parameter_error(track.get("effects"))
            if effect_error:
                raise ValueError(f"Track {n}: {effect_error}")
        active = self._active_tracks(tracks)
        if not active:
            raise ValueError("No active tracks to render")
        for track in active:
            self._check_render_length(float(track.get("offset_s", 0.0)))

        with self._render_lock:
            return self._render_active(active, format, sample_rate, name)

    def _render_active(self, active: list[dict], format: str, sample_rate: int, name: str | None) -> dict:
        # Size the mix from each source's header, so nothing is decoded for a composition that is too long.
        planned = []
        for track in active:
            source_path = self._validate_source_path(track.get("source_path", ""))
            src_rate, frames = self._read_wav_header(source_path)
            start = self._track_start(track, sample_rate)
            end = start + self._resampled_length(frames, src_rate, sample_rate)
            self._check_render_length(end / sample_rate)
            planned.append((track, source_path, start, end))

        # Decode and add one track at a time: peak memory is the mix plus one track, whatever the track count.
        mixed = np.zeros(max(end for *_, end in planned), dtype=np.float32)
        persisted_tracks = []
        for track, source_path, start, _end in planned:
            volume = float(track.get("volume", 1.0))
            self._add_track(mixed, self._prepare_samples(source_path, track, volume, sample_rate), start)
            persisted_tracks.append(
                {
                    "offset_s": float(track.get("offset_s", 0.0)),
                    "source_path": str(source_path),
                    "volume": volume,
                    "muted": bool(track.get("muted", False)),
                    "solo": bool(track.get("solo", False)),
                    "effects": track.get("effects") or [],
                }
            )
        np.clip(mixed, -1.0, 1.0, out=mixed)

        composition_id = str(uuid4())
        out_ext = "mp3" if str(format).lower() == "mp3" else "wav"
        output_path = self.output_dir / f"render_{composition_id}.{out_ext}"
        if out_ext == "wav":
            wavfile.write(output_path, sample_rate, self._float_to_int16(mixed))
        else:
            self._write_mp3(output_path, mixed, sample_rate)

        rel_output_path = self._relative_to_repo(output_path)
        duration_ms = int(1000 * len(mixed) / sample_rate) if len(mixed) else 0
        self._save_composition(
            composition_id=composition_id,
            name=name,
            sample_rate=sample_rate,
            output_path=rel_output_path,
            tracks=persisted_tracks,
            meta={"format": out_ext, "duration_ms": duration_ms},
        )

        return {
            "composition_id": composition_id,
            "output_path": rel_output_path,
            "download_url": f"/api/composer/render/{composition_id}/audio",
            "duration_ms": duration_ms,
        }

    def list_renders(self, limit: int = 100, offset: int = 0) -> dict:
        with snapshot() as db:
            total = db.execute("SELECT COUNT(*) FROM compositions").fetchone()[0]
            rows = db.execute(
                "SELECT * FROM compositions ORDER BY updated_at DESC, rowid DESC LIMIT ? OFFSET ?",
                (int(limit), int(offset)),
            ).fetchall()
        items = [self._row_to_render(r) for r in rows]
        return {"items": items, "total": total, "limit": int(limit), "offset": int(offset)}

    def get_render(self, composition_id: str) -> dict | None:
        db = get_db()
        row = db.execute("SELECT * FROM compositions WHERE id = ?", (composition_id,)).fetchone()
        if not row:
            return None
        return self._row_to_render(row)

    def delete_render(self, composition_id: str) -> bool:
        with transaction() as db:
            # Read and delete together: of two deletes at once, only one finds the row.
            row = db.execute("SELECT render_output_path FROM compositions WHERE id = ?", (composition_id,)).fetchone()
            if not row:
                return False
            db.execute("DELETE FROM compositions WHERE id = ?", (composition_id,))
        output_path = self._resolve_repo_path(row["render_output_path"])
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            # Windows cannot remove a file while a download still streams it.
            logger.exception("Composition %s was deleted but its render could not be removed", composition_id)
        return True

    def _active_tracks(self, tracks: list[dict]) -> list[dict]:
        non_muted = [t for t in tracks if not bool(t.get("muted", False))]
        if any(bool(t.get("solo", False)) for t in non_muted):
            return [t for t in non_muted if bool(t.get("solo", False))]
        return non_muted

    def _validate_source_path(self, source_path: str) -> Path:
        if not source_path:
            raise ValueError("Track source_path is required")
        candidate = Path(source_path)
        resolved = self._resolve_repo_path(candidate)

        repo_data = self._resolve_repo_path(Path("data"))
        allowed_roots = {
            repo_data,
            repo_data / "conversations",
            repo_data / "voices",
            Path("/home/openspeech/data"),
            Path("/home/openspeech/data/conversations"),
            Path("/home/openspeech/data/voices"),
            Path(settings.os_voice_library_path),
            Path(settings.os_conversations_dir),
            self.output_dir,
        }
        # Check containment before existence so a rejection cannot reveal whether
        # an arbitrary server path exists.
        if not any(self._is_relative_to(resolved, root) for root in allowed_roots):
            raise PermissionError("Track source path is outside allowed roots")
        if not resolved.exists():
            raise ValueError(f"Track source not found: {source_path}")
        return resolved

    def _read_wav_header(self, source_path: Path) -> tuple[int, int]:
        """Return a WAV source's sample rate and frame count without decoding its samples."""
        try:
            sr, data = wavfile.read(str(source_path), mmap=True)
            return int(sr), len(data)
        except ValueError as exc:
            # numpy cannot map 3-byte (24-bit) samples, but those are plain PCM, which wave reads.
            try:
                with wave.open(str(source_path), "rb") as wav_file:
                    return wav_file.getframerate(), wav_file.getnframes()
            except (EOFError, wave.Error):
                raise exc from None

    def _load_audio(self, source_path: Path) -> tuple[int, np.ndarray]:
        sr, data = wavfile.read(str(source_path))
        arr = np.asarray(data)
        if arr.dtype.kind == "u":
            midpoint = float(2 ** (np.iinfo(arr.dtype).bits - 1))
            arr = (arr.astype(np.float32) - midpoint) / midpoint
        elif arr.dtype.kind == "i":
            scale = float(2 ** (np.iinfo(arr.dtype).bits - 1))
            arr = arr.astype(np.float32) / scale
        else:
            arr = arr.astype(np.float32)
        # Normalize integer PCM before averaging channels changes its dtype.
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        return int(sr), arr

    def _resample(self, samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate:
            return samples.astype(np.float32, copy=False)
        gcd = math.gcd(src_rate, dst_rate)
        up = dst_rate // gcd
        down = src_rate // gcd
        return resample_poly(samples, up, down).astype(np.float32, copy=False)

    @staticmethod
    def _track_start(track: dict, sample_rate: int) -> int:
        return int(round(max(0.0, float(track.get("offset_s", 0.0))) * sample_rate))

    @staticmethod
    def _resampled_length(frames: int, src_rate: int, dst_rate: int) -> int:
        if src_rate == dst_rate:
            return frames
        gcd = math.gcd(src_rate, dst_rate)
        # resample_poly returns ceil(frames * up / down) samples.
        return -(-frames * (dst_rate // gcd) // (src_rate // gcd))

    @staticmethod
    def _check_render_length(seconds: float) -> None:
        if seconds > MAX_RENDER_SECONDS:
            raise ValueError(f"Composition is too long. Max: {MAX_RENDER_SECONDS} seconds")

    def _prepare_samples(self, source_path: Path, track: dict, volume: float, sample_rate: int) -> np.ndarray:
        src_rate, samples = self._load_audio(source_path)
        if settings.os_effects_enabled:
            samples = apply_chain(samples, src_rate, track.get("effects") or [])
        samples = (samples * volume).astype(np.float32, copy=False)
        if src_rate != sample_rate:
            samples = self._resample(samples, src_rate, sample_rate)
        return samples

    @staticmethod
    def _add_track(mixed: np.ndarray, samples: np.ndarray, start: int) -> None:
        # The header sized the mix; a source that decodes longer must not grow it.
        end = min(len(mixed), start + len(samples))
        if end > start:
            mixed[start:end] += samples[: end - start]

    def _write_mp3(self, output_path: Path, samples: np.ndarray, sample_rate: int) -> None:
        pcm = self._float_to_int16(samples).tobytes()
        cmd = [
            "ffmpeg", "-y",
            "-f", "s16le",
            "-ar", str(sample_rate),
            "-ac", "1",
            "-i", "pipe:0",
            "-f", "mp3", "-codec:a", "libmp3lame", "-b:a", "128k",
            str(output_path),
        ]
        proc = subprocess.run(cmd, input=pcm, capture_output=True, timeout=30)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')}")

    def _save_composition(self, composition_id: str, name: str | None, sample_rate: int, output_path: str, tracks: list[dict], meta: dict) -> None:
        now = _now_iso()
        with transaction() as db:
            db.execute(
                """
                INSERT INTO compositions (id, name, sample_rate, created_at, updated_at, render_output_path, tracks_json, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    composition_id,
                    name,
                    sample_rate,
                    now,
                    now,
                    output_path,
                    json.dumps(tracks),
                    json.dumps(meta),
                ),
            )

    def _row_to_render(self, row) -> dict:
        data = dict(row)
        data["tracks"] = json.loads(data.pop("tracks_json") or "[]")
        data["meta"] = json.loads(data.pop("meta_json") or "{}")
        return data

    def _relative_to_repo(self, path: Path) -> str:
        repo_root = Path.cwd().resolve()
        try:
            return str(path.resolve().relative_to(repo_root))
        except ValueError:
            return str(path)

    def _resolve_repo_path(self, source_path: Path | str) -> Path:
        p = Path(source_path)
        if p.is_absolute():
            return p.resolve()
        return (Path.cwd() / p).resolve()

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root.resolve())
            return True
        except ValueError:
            return False

    @staticmethod
    def _float_to_int16(samples: np.ndarray) -> np.ndarray:
        clipped = np.clip(samples, -1.0, 1.0)
        return (clipped * 32767).astype(np.int16)
