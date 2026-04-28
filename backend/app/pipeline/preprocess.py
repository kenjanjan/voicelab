"""Denoise -> 24k mono -> VAD segment -> loudness norm -> ASR. Writes AudioClip rows."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
from sqlalchemy import delete, func, select

from app.core.config import settings
from app.db import AudioClip, SessionLocal

SR = 24000
TARGET_LUFS = -20.0
MIN_DUR, MAX_DUR = 3.0, 12.0

_vad = None
_asr = None
_demucs_warned = False


def _status_path(creator_id: str) -> Path:
    return settings.DATA_DIR / "processed" / creator_id / "_status.json"


def _default_status() -> dict:
    return {
        "status": "idle", "progress": 0.0, "log": "",
        "n_input": 0, "n_output": 0,
        "started_at": None, "finished_at": None,
    }


def read_status(creator_id: str) -> dict:
    p = _status_path(creator_id)
    if not p.exists():
        return _default_status()
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return _default_status()
    # backfill missing fields from older runs
    base = _default_status()
    base.update(data)
    return base


def _write_status(creator_id: str, **fields) -> None:
    p = _status_path(creator_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    current = read_status(creator_id)
    current.update(fields)
    p.write_text(json.dumps(current))


def _log(creator_id: str, line: str) -> None:
    print(f"[preprocess:{creator_id}] {line}")
    cur = read_status(creator_id)
    cur["log"] = (cur.get("log", "") + line + "\n")[-12000:]
    _status_path(creator_id).write_text(json.dumps(cur))


def _load_vad():
    global _vad
    if _vad is None:
        _vad = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    return _vad


def _load_asr():
    global _asr
    if _asr is None:
        from faster_whisper import WhisperModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        _asr = WhisperModel("large-v3", device=device, compute_type=compute)
    return _asr


def _denoise(in_path: Path, work: Path) -> Path:
    global _demucs_warned
    out = work / "demucs"
    out.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
             "-o", str(out), str(in_path)],
            check=True, capture_output=True, text=True,
        )
        cleaned = out / "htdemucs" / in_path.stem / "vocals.wav"
        return cleaned if cleaned.exists() else in_path
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        if not _demucs_warned:
            msg = (getattr(e, "stderr", "") or str(e))[-400:]
            print(
                "[preprocess] denoise unavailable — proceeding without it. "
                "Install ffmpeg + `pip install torchcodec` to enable Demucs.\n"
                f"  reason: {msg.strip()}"
            )
            _demucs_warned = True
        return in_path


def _to_mono_24k(path: Path) -> torch.Tensor:
    """Load any audio file -> mono float32 tensor at SR.

    Uses librosa, which falls back through soundfile (WAV/FLAC/OGG native) and
    audioread (mp3/m4a/aac via ffmpeg if installed). Avoids torchaudio 2.7+'s
    new torchcodec-only default backend.
    """
    arr, _ = librosa.load(str(path), sr=SR, mono=True)
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


def _vad_segments(wav: torch.Tensor) -> list[tuple[int, int]]:
    """Run silero VAD at 16 kHz (it requires 8k or multiples of 16k).

    Returns sample-index tuples in the original SR (24 kHz) so callers can
    slice the original tensor directly.
    """
    model, utils = _load_vad()
    get_speech_timestamps = utils[0]
    VAD_SR = 16000
    wav_16k_np = librosa.resample(wav.numpy().astype(np.float32), orig_sr=SR, target_sr=VAD_SR)
    wav_16k = torch.from_numpy(wav_16k_np)
    ts = get_speech_timestamps(
        wav_16k, model, sampling_rate=VAD_SR,
        min_speech_duration_ms=2500,
        max_speech_duration_s=MAX_DUR,
        min_silence_duration_ms=400,
    )
    scale = SR / VAD_SR
    return [(int(t["start"] * scale), int(t["end"] * scale)) for t in ts]


def _normalize(wav: torch.Tensor) -> torch.Tensor:
    arr = wav.numpy().astype("float32")
    meter = pyln.Meter(SR)
    loud = meter.integrated_loudness(arr)
    if loud == float("-inf"):
        return wav
    gain = TARGET_LUFS - loud
    return wav * (10 ** (gain / 20))


def _clip_path_prefix(creator_id: str, raw_stem: str) -> str:
    """SQL/path prefix for clips derived from a given raw file."""
    return f"processed/{creator_id}/{raw_stem}_"


def _wipe_partial_for_file(creator_id: str, out_dir: Path, raw_stem: str) -> None:
    """Delete any clip files + DB rows derived from this raw file (partial earlier run)."""
    for f in out_dir.glob(f"{raw_stem}_*.wav"):
        f.unlink(missing_ok=True)
    prefix = _clip_path_prefix(creator_id, raw_stem)
    with SessionLocal() as db:
        db.execute(
            delete(AudioClip).where(
                AudioClip.creator_id == creator_id,
                AudioClip.path.like(prefix + "%"),
            )
        )
        db.commit()


def preprocess_creator(creator_id: str, force: bool = False) -> dict:
    """Resumable preprocess. Each raw file gets a marker once fully committed.
    On retry, files with markers are skipped. `force=True` wipes everything first."""
    raw_dir = settings.DATA_DIR / "raw" / creator_id
    out_dir = settings.DATA_DIR / "processed" / creator_id
    work_dir = out_dir / "_work"
    completed_dir = out_dir / "_completed"

    if force and out_dir.exists():
        for sub in out_dir.iterdir():
            if sub.name == "_status.json":
                continue
            if sub.is_dir():
                shutil.rmtree(sub, ignore_errors=True)
            else:
                sub.unlink(missing_ok=True)
        with SessionLocal() as db:
            db.execute(delete(AudioClip).where(AudioClip.creator_id == creator_id))
            db.commit()

    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    completed_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted([p for p in raw_dir.iterdir() if p.is_file()]) if raw_dir.exists() else []

    # Resume detection with fallback: if a marker is missing but clips already
    # exist on disk + in DB, treat the file as done and write a marker.
    pending: list[Path] = []
    already_done: list[str] = []
    fallback_marked: list[tuple[str, int]] = []
    for raw in raw_files:
        marker = completed_dir / raw.name
        if marker.exists():
            already_done.append(raw.name)
            continue
        on_disk = list(out_dir.glob(f"{raw.stem}_*.wav"))
        if on_disk:
            prefix = _clip_path_prefix(creator_id, raw.stem)
            with SessionLocal() as db:
                existing = db.execute(
                    select(func.count()).select_from(AudioClip).where(
                        AudioClip.creator_id == creator_id,
                        AudioClip.path.like(prefix + "%"),
                    )
                ).scalar_one()
            if existing > 0:
                marker.touch()
                already_done.append(raw.name)
                fallback_marked.append((raw.name, int(existing)))
                continue
        pending.append(raw)

    n_total = len(raw_files)
    n_done = len(already_done)
    initial_progress = n_done / n_total if n_total else 0.0

    with SessionLocal() as db:
        existing_count = db.execute(
            select(func.count()).select_from(AudioClip)
            .where(AudioClip.creator_id == creator_id)
        ).scalar_one()

    _write_status(
        creator_id, status="running",
        progress=initial_progress, log="",
        n_input=n_total, n_output=int(existing_count),
        started_at=datetime.utcnow().isoformat() + "Z",
        finished_at=None,
    )
    _log(
        creator_id,
        f"Starting preprocess. Total={n_total}, already_done={n_done} "
        f"({len(fallback_marked)} via fallback), pending={len(pending)} "
        f"(force={force}, SR={SR}Hz)",
    )
    if fallback_marked:
        _log(creator_id, f"Fallback-marked (clips already in DB + on disk):")
        for name, count in fallback_marked[:20]:
            _log(creator_id, f"  - {name}: {count} clips")
        if len(fallback_marked) > 20:
            _log(creator_id, f"  ... and {len(fallback_marked) - 20} more")

    if not raw_files:
        _log(creator_id, "No raw files to process.")
        _write_status(
            creator_id, status="completed", progress=1.0,
            finished_at=datetime.utcnow().isoformat() + "Z",
        )
        return {"clips": 0}

    if not pending:
        _log(creator_id, "All files already processed. Pass force=true to redo.")
        _write_status(
            creator_id, status="completed", progress=1.0,
            finished_at=datetime.utcnow().isoformat() + "Z",
        )
        return {"clips": int(existing_count)}

    try:
        asr = _load_asr()
        _log(creator_id, "ASR (faster-whisper large-v3) loaded.")
    except Exception as e:
        _log(creator_id, f"ASR load FAILED: {e}\n{traceback.format_exc()}")
        _write_status(
            creator_id, status="failed",
            finished_at=datetime.utcnow().isoformat() + "Z",
        )
        raise

    n_files_failed = 0
    n_clips_total = int(existing_count)

    for idx, raw in enumerate(pending):
        position = n_done + idx + 1
        _log(creator_id, f"[{position}/{n_total}] {raw.name} ({raw.stat().st_size/1e6:.1f} MB)")

        # Wipe any partial work from a previous failed run on THIS raw file
        _wipe_partial_for_file(creator_id, out_dir, raw.stem)

        try:
            cleaned = _denoise(raw, work_dir)
        except Exception as ex:
            _log(creator_id, f"  denoise error (continuing on raw): {ex}")
            cleaned = raw

        try:
            wav = _to_mono_24k(cleaned)
        except Exception as ex:
            _log(creator_id, f"  load FAILED: {ex}")
            n_files_failed += 1
            continue

        total_dur = len(wav) / SR
        try:
            segs = _vad_segments(wav)
        except Exception as ex:
            _log(creator_id, f"  VAD FAILED: {ex}\n{traceback.format_exc()}")
            n_files_failed += 1
            continue

        file_clips: list[AudioClip] = []
        n_dropped = 0

        try:
            for i, (s, e) in enumerate(segs):
                seg = wav[s:e]
                dur = (e - s) / SR
                if dur < MIN_DUR:
                    n_dropped += 1
                    continue
                seg = _normalize(seg)
                clip_path = out_dir / f"{raw.stem}_{i:03}.wav"
                sf.write(str(clip_path), seg.numpy(), SR, subtype="PCM_16")
                text = ""
                try:
                    segments, _info = asr.transcribe(
                        str(clip_path), language="en", vad_filter=False,
                    )
                    text = " ".join(seg_.text.strip() for seg_ in segments)
                except Exception as ex:
                    _log(creator_id, f"  ASR error on segment {i}: {ex}")
                file_clips.append(AudioClip(
                    creator_id=creator_id,
                    path=str(clip_path.relative_to(settings.DATA_DIR).as_posix()),
                    text=text,
                    duration=round(dur, 2),
                    is_reference=False,
                ))
        except Exception as ex:
            _log(creator_id, f"  segment loop FAILED: {ex}")
            # Don't mark complete; partial files will be wiped on next retry.
            n_files_failed += 1
            continue

        # Commit this file's clips atomically + write completion marker
        with SessionLocal() as db:
            for c in file_clips:
                db.add(c)
            db.commit()
        (completed_dir / raw.name).touch()
        n_clips_total += len(file_clips)

        _log(
            creator_id,
            f"  duration={total_dur:.1f}s · VAD found {len(segs)} segments · "
            f"saved {len(file_clips)} · dropped {n_dropped} (<{MIN_DUR}s) · committed",
        )
        _write_status(
            creator_id,
            progress=position / n_total,
            n_output=n_clips_total,
        )

    shutil.rmtree(work_dir, ignore_errors=True)

    _log(
        creator_id,
        f"DONE — {n_clips_total} total clips · {len(pending) - n_files_failed}/{len(pending)} "
        f"new files committed · {n_done} skipped (already done) · {n_files_failed} failed.",
    )
    _write_status(
        creator_id, status="completed", progress=1.0, n_output=n_clips_total,
        finished_at=datetime.utcnow().isoformat() + "Z",
    )
    return {"clips": n_clips_total}
