"""Multi-track audio composer for Studio."""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from src.config import settings
from src.effects.chain import apply_chain
from src.storage import get_db


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

    def render(self, tracks: list[dict], format: str = "wav", sample_rate: int = 24000, name: str | None = None) -> dict:
        active = self._active_tracks(tracks)
        if not active:
            raise ValueError("No active tracks to render")

        prepared = []
        persisted_tracks = []
        for track in active:
            source_path = self._validate_source_path(track.get("source_path", ""))
            src_rate, samples = self._load_audio(source_path)
            samples = apply_chain(samples, src_rate, track.get("effects") or [])
            volume = float(track.get("volume", 1.0))
            samples = (samples * volume).astype(np.float32, copy=False)
            if int(src_rate) != int(sample_rate):
                samples = self._resample(samples, int(src_rate), int(sample_rate))
            prepared.append(
                {
                    "samples": samples,
                    "offset_s": float(track.get("offset_s", 0.0)),
                    "source_path": str(source_path),
                    "volume": volume,
                    "muted": bool(track.get("muted", False)),
                    "solo": bool(track.get("solo", False)),
                    "effects": track.get("effects") or [],
                }
            )
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

        mixed = self._mix_prepared(prepared, int(sample_rate))

        composition_id = str(uuid4())
        out_ext = "mp3" if str(format).lower() == "mp3" else "wav"
        output_path = self.output_dir / f"render_{composition_id}.{out_ext}"
        if out_ext == "wav":
            wavfile.write(output_path, int(sample_rate), self._float_to_int16(mixed))
        else:
            self._write_mp3(output_path, mixed, int(sample_rate))

        rel_output_path = self._relative_to_repo(output_path)
        duration_ms = int(1000 * len(mixed) / int(sample_rate)) if len(mixed) else 0
        self._save_composition(
            composition_id=composition_id,
            name=name,
            sample_rate=int(sample_rate),
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
        db = get_db()
        total = db.execute("SELECT COUNT(*) FROM compositions").fetchone()[0]
        rows = db.execute(
            "SELECT * FROM compositions ORDER BY updated_at DESC LIMIT ? OFFSET ?",
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
        db = get_db()
        row = db.execute("SELECT render_output_path FROM compositions WHERE id = ?", (composition_id,)).fetchone()
        if not row:
            return False
        output_path = self._resolve_repo_path(row["render_output_path"])
        if output_path.exists():
            output_path.unlink()
        db.execute("DELETE FROM compositions WHERE id = ?", (composition_id,))
        db.commit()
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
        if not resolved.exists():
            raise ValueError(f"Track source not found: {source_path}")

        repo_data = self._resolve_repo_path(Path("data"))
        allowed_roots = {
            repo_data,
            repo_data / "conversations",
            repo_data / "voices",
            Path("/home/openspeech/data"),
            Path("/home/openspeech/data/conversations"),
            Path("/home/openspeech/data/voices"),
        }
        if not any(self._is_relative_to(resolved, root) for root in allowed_roots):
            raise PermissionError(f"Track source path is outside allowed roots: {source_path}")
        return resolved

    def _load_audio(self, source_path: Path) -> tuple[int, np.ndarray]:
        sr, data = wavfile.read(str(source_path))
        arr = np.asarray(data)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if arr.dtype.kind in ("i", "u"):
            max_int = np.iinfo(arr.dtype).max
            arr = arr.astype(np.float32) / float(max_int)
        else:
            arr = arr.astype(np.float32)
        return int(sr), arr

    def _resample(self, samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate:
            return samples.astype(np.float32, copy=False)
        gcd = math.gcd(src_rate, dst_rate)
        up = dst_rate // gcd
        down = src_rate // gcd
        return resample_poly(samples, up, down).astype(np.float32, copy=False)

    def _mix_prepared(self, prepared: list[dict], sample_rate: int) -> np.ndarray:
        total_samples = 0
        for track in prepared:
            start = int(round(max(0.0, float(track.get("offset_s", 0.0))) * sample_rate))
            end = start + len(track["samples"])
            total_samples = max(total_samples, end)
        if total_samples <= 0:
            return np.zeros(0, dtype=np.float32)

        mixed = np.zeros(total_samples, dtype=np.float32)
        for track in prepared:
            start = int(round(max(0.0, float(track.get("offset_s", 0.0))) * sample_rate))
            samples = np.asarray(track["samples"], dtype=np.float32)
            mixed[start:start + len(samples)] += samples
        return np.clip(mixed, -1.0, 1.0)

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
        db = get_db()
        now = _now_iso()
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
        db.commit()

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
