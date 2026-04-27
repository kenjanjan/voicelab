"""CosyVoice 2 inference with per-creator reference clips and optional LoRA adapter.

The CosyVoice repo must be cloned to settings.COSYVOICE_PATH and the model weights
downloaded to settings.COSYVOICE_MODEL — see scripts/download_models.py.
"""
from __future__ import annotations

import io
import random
import sys
from pathlib import Path

import soundfile as sf
import torch
from sqlalchemy import select

from app.core.config import settings
from app.db import AudioClip, SessionLocal

EMOTIONS: dict[str, str] = {
    "casual":         "Speak naturally and warmly, like talking to a close friend.",
    "flirty":         "Speak in a flirty, playful tone with a smile in your voice.",
    "seductive":      "Speak slowly and softly, with a low intimate whisper.",
    "excited":        "Speak with high energy, upbeat and enthusiastic.",
    "playful_giggle": "Speak with a light, teasing tone, almost giggling.",
    "soft_intimate":  "Speak gently and close to the microphone, breathy and warm.",
}

_engine = None


def _load_engine():
    global _engine
    if _engine is not None:
        return _engine
    cosy_path = settings.COSYVOICE_PATH.resolve()
    if not cosy_path.exists():
        raise RuntimeError(
            f"CosyVoice not found at {cosy_path}. "
            "Run: python scripts/download_models.py"
        )
    if str(cosy_path) not in sys.path:
        sys.path.insert(0, str(cosy_path))
    from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore
    _engine = CosyVoice2(
        str(settings.COSYVOICE_MODEL.resolve()),
        load_jit=False, load_trt=False, fp16=torch.cuda.is_available(),
    )
    return _engine


def _pick_reference(creator_id: str, emotion: str) -> Path:
    with SessionLocal() as db:
        q = select(AudioClip).where(
            AudioClip.creator_id == creator_id, AudioClip.is_reference.is_(True),
        )
        clips = db.execute(q).scalars().all()
        same_emotion = [c for c in clips if c.emotion == emotion]
        pool = same_emotion or clips
        if not pool:
            raise RuntimeError(
                f"No reference clips marked for creator {creator_id}. "
                "Mark at least one processed clip as reference."
            )
        choice = random.choice(pool)
        return settings.DATA_DIR / choice.path


def _maybe_load_lora(creator_id: str) -> None:
    """Load most recent LoRA adapter for the creator into the engine LM, if present."""
    lora_dir = settings.DATA_DIR / "loras" / creator_id
    if not lora_dir.exists():
        return
    adapters = sorted(lora_dir.glob("adapter_*"), reverse=True)
    if not adapters:
        return
    try:
        from peft import PeftModel  # type: ignore
        eng = _load_engine()
        lm = getattr(eng.model, "llm", None)
        if lm is None:
            return
        if not isinstance(lm, PeftModel):
            eng.model.llm = PeftModel.from_pretrained(lm, str(adapters[0]))
    except Exception as exc:
        print(f"[tts_engine] LoRA load skipped: {exc}")


def synthesize(
    creator_id: str,
    text: str,
    emotion: str,
    speed: float,
    seed: int | None,
    out_path: Path,
    use_lora: bool,
) -> int:
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    torch.manual_seed(seed)
    random.seed(seed)

    engine = _load_engine()
    if use_lora:
        _maybe_load_lora(creator_id)

    ref_path = _pick_reference(creator_id, emotion)
    from cosyvoice.utils.file_utils import load_wav  # type: ignore

    prompt_wav = load_wav(str(ref_path), 16000)
    instruction = EMOTIONS[emotion]

    chunks = []
    for out in engine.inference_instruct2(
        text, instruction, prompt_wav, stream=False, speed=speed,
    ):
        chunks.append(out["tts_speech"])
    audio = torch.cat(chunks, dim=1).cpu()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio.squeeze(0).numpy(), engine.sample_rate, subtype="PCM_16")
    return seed


def synthesize_to_bytes(*args, **kwargs) -> bytes:
    buf = io.BytesIO()
    tmp = Path("/tmp/_voicelab_synth.wav")
    synthesize(*args, out_path=tmp, **kwargs)
    buf.write(tmp.read_bytes())
    return buf.getvalue()
