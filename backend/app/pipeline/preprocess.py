"""Denoise -> 24k mono -> VAD segment -> loudness norm -> ASR. Writes AudioClip rows."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pyloudnorm as pyln
import soundfile as sf
import torch
import torchaudio
from sqlalchemy import delete

from app.core.config import settings
from app.db import AudioClip, SessionLocal

SR = 24000
TARGET_LUFS = -20.0
MIN_DUR, MAX_DUR = 3.0, 12.0

_vad = None
_asr = None


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
    out = work / "demucs"
    out.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["python", "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
             "-o", str(out), str(in_path)],
            check=True, capture_output=True,
        )
        cleaned = out / "htdemucs" / in_path.stem / "vocals.wav"
        return cleaned if cleaned.exists() else in_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        return in_path


def _to_mono_24k(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.squeeze(0).float()


def _vad_segments(wav: torch.Tensor) -> list[tuple[int, int]]:
    model, utils = _load_vad()
    get_speech_timestamps = utils[0]
    ts = get_speech_timestamps(
        wav, model, sampling_rate=SR,
        min_speech_duration_ms=2500,
        max_speech_duration_s=MAX_DUR,
        min_silence_duration_ms=400,
    )
    return [(s["start"], s["end"]) for s in ts]


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
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    asr = _load_asr()
    new_clips: list[AudioClip] = []

    for raw in sorted(raw_dir.iterdir()):
        if not raw.is_file():
            continue
        cleaned = _denoise(raw, work_dir)
        wav = _to_mono_24k(cleaned)
        for i, (s, e) in enumerate(_vad_segments(wav)):
            seg = wav[s:e]
            dur = (e - s) / SR
            if dur < MIN_DUR:
                continue
            seg = _normalize(seg)
            clip_path = out_dir / f"{raw.stem}_{i:03}.wav"
            sf.write(str(clip_path), seg.numpy(), SR, subtype="PCM_16")
            text = ""
            try:
                segments, _ = asr.transcribe(str(clip_path), language="en", vad_filter=False)
                text = " ".join(seg_.text.strip() for seg_ in segments)
            except Exception:
                pass
            new_clips.append(AudioClip(
                creator_id=creator_id,
                path=str(clip_path.relative_to(settings.DATA_DIR).as_posix()),
                text=text,
                duration=round(dur, 2),
                is_reference=False,
            ))

    shutil.rmtree(work_dir, ignore_errors=True)

    with SessionLocal() as db:
        db.execute(delete(AudioClip).where(AudioClip.creator_id == creator_id))
        for c in new_clips:
            db.add(c)
        db.commit()

    return {"clips": len(new_clips)}
