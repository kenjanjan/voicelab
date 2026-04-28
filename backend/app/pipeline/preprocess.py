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
from sqlalchemy import delete

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


def preprocess_creator(creator_id: str) -> dict:
    raw_dir = settings.DATA_DIR / "raw" / creator_id
    out_dir = settings.DATA_DIR / "processed" / creator_id
    work_dir = out_dir / "_work"

    if out_dir.exists():
        for sub in out_dir.iterdir():
            if sub.name == "_status.json":
                continue
            if sub.is_dir():
                shutil.rmtree(sub, ignore_errors=True)
            else:
                sub.unlink(missing_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted([p for p in raw_dir.iterdir() if p.is_file()]) if raw_dir.exists() else []
    _write_status(
        creator_id, status="running", progress=0.0, log="",
        n_input=len(raw_files), n_output=0,
        started_at=datetime.utcnow().isoformat() + "Z",
        finished_at=None,
    )
    _log(creator_id, f"Starting preprocess of {len(raw_files)} files (SR={SR} Hz)")

    if not raw_files:
        _log(creator_id, "No raw files to process.")
        _write_status(
            creator_id, status="completed", progress=1.0,
            finished_at=datetime.utcnow().isoformat() + "Z",
        )
        return {"clips": 0}

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

    new_clips: list[AudioClip] = []
    n_files_failed = 0

    for idx, raw in enumerate(raw_files):
        _log(creator_id, f"[{idx+1}/{len(raw_files)}] {raw.name} ({raw.stat().st_size/1e6:.1f} MB)")
        try:
            cleaned = _denoise(raw, work_dir)
        except Exception as e:
            _log(creator_id, f"  denoise error (continuing on raw): {e}")
            cleaned = raw

        try:
            wav = _to_mono_24k(cleaned)
        except Exception as e:
            _log(creator_id, f"  load FAILED: {e}")
            n_files_failed += 1
            continue

        total_dur = len(wav) / SR
        try:
            segs = _vad_segments(wav)
        except Exception as e:
            _log(creator_id, f"  VAD FAILED: {e}\n{traceback.format_exc()}")
            n_files_failed += 1
            continue

        n_dropped = 0
        n_saved = 0
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
            new_clips.append(AudioClip(
                creator_id=creator_id,
                path=str(clip_path.relative_to(settings.DATA_DIR).as_posix()),
                text=text,
                duration=round(dur, 2),
                is_reference=False,
            ))
            n_saved += 1

        _log(
            creator_id,
            f"  duration={total_dur:.1f}s · VAD found {len(segs)} segments · "
            f"saved {n_saved} · dropped {n_dropped} (<{MIN_DUR}s)",
        )
        _write_status(
            creator_id, progress=(idx + 1) / len(raw_files), n_output=len(new_clips),
        )

    shutil.rmtree(work_dir, ignore_errors=True)

    with SessionLocal() as db:
        db.execute(delete(AudioClip).where(AudioClip.creator_id == creator_id))
        for c in new_clips:
            db.add(c)
        db.commit()

    _log(
        creator_id,
        f"DONE — {len(new_clips)} clips written from {len(raw_files)} files "
        f"({n_files_failed} failed).",
    )
    _write_status(
        creator_id, status="completed", progress=1.0, n_output=len(new_clips),
        finished_at=datetime.utcnow().isoformat() + "Z",
    )
    return {"clips": len(new_clips)}
