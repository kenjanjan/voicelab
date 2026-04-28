"""LoRA fine-tune of the CosyVoice 2 LM head on a creator's clip manifest.

This adapts only the autoregressive text->speech-token LM (the part that
controls prosody). The flow-matching decoder is left frozen — it generalises
across speakers from the reference clip alone.

For >1 hour of clean data this typically beats zero-shot reference-only quality.
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from app.core.config import settings
from app.db import AudioClip, SessionLocal, TrainingJob


def _build_manifest(creator_id: str) -> list[dict]:
    with SessionLocal() as db:
        rows = db.execute(
            select(AudioClip).where(
                AudioClip.creator_id == creator_id, AudioClip.text != "",
            )
        ).scalars().all()
        return [
            {
                "audio": str((settings.DATA_DIR / r.path).resolve()),
                "text": r.text,
                "emotion": r.emotion or "casual",
                "duration": r.duration,
            }
            for r in rows
        ]


def _set_status(job_id: str, **fields) -> None:
    with SessionLocal() as db:
        job = db.get(TrainingJob, job_id)
        if not job:
            return
        for k, v in fields.items():
            setattr(job, k, v)
        db.commit()


def _append_log(job_id: str, line: str) -> None:
    with SessionLocal() as db:
        job = db.get(TrainingJob, job_id)
        if not job:
            return
        job.log = (job.log or "") + line + "\n"
        db.commit()


def run_lora_training(
    job_id: str,
    creator_id: str,
    epochs: int,
    learning_rate: float,
    rank: int,
    device: str | None = None,
) -> None:
    _set_status(job_id, status="running", started_at=datetime.utcnow(), progress=0.01)
    _append_log(job_id, f"Starting LoRA training rank={rank} lr={learning_rate} epochs={epochs} device={device or 'auto'}")

    try:
        manifest = _build_manifest(creator_id)
        if len(manifest) < 20:
            raise RuntimeError(
                f"Need ≥20 transcribed clips, have {len(manifest)}. "
                "Run preprocessing and verify ASR transcripts."
            )
        _append_log(job_id, f"Manifest size: {len(manifest)} clips")

        cosy_path = settings.COSYVOICE_PATH.resolve()
        if str(cosy_path) not in sys.path:
            sys.path.insert(0, str(cosy_path))

        import torch
        from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore
        from peft import LoraConfig, get_peft_model  # type: ignore

        engine = CosyVoice2(str(settings.COSYVOICE_MODEL.resolve()), load_jit=False, load_trt=False, fp16=False)
        lm = engine.model.llm
        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        _append_log(job_id, f"Using device: {device}")
        lm.to(device)

        lora_cfg = LoraConfig(
            r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
            bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        lm = get_peft_model(lm, lora_cfg)
        lm.print_trainable_parameters()

        optim = torch.optim.AdamW(
            [p for p in lm.parameters() if p.requires_grad], lr=learning_rate,
        )

        total_steps = epochs * len(manifest)
        step = 0
        for epoch in range(epochs):
            for item in manifest:
                wav, sr = torch.load(item["audio"]) if item["audio"].endswith(".pt") else (None, None)
                # Tokenise text + audio via CosyVoice's tokenizers, then compute LM loss.
                # Using engine's internal helpers; exact API depends on installed CosyVoice version.
                try:
                    batch = engine.frontend.text_audio_pair_to_batch(item["text"], item["audio"])
                except AttributeError:
                    raise RuntimeError(
                        "Installed CosyVoice version does not expose "
                        "frontend.text_audio_pair_to_batch. Use the official "
                        "CosyVoice training script in vendor/CosyVoice/examples/ "
                        "and pass the dataset built by _build_manifest."
                    )
                batch = {k: v.to(device) for k, v in batch.items()}
                out = lm(**batch)
                loss = out.loss
                optim.zero_grad()
                loss.backward()
                optim.step()

                step += 1
                if step % 10 == 0:
                    _append_log(job_id, f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f}")
                    _set_status(job_id, progress=step / total_steps)

        out_dir = settings.DATA_DIR / "loras" / creator_id
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        adapter_path = out_dir / f"adapter_{ts}"
        lm.save_pretrained(str(adapter_path))

        manifest_path = adapter_path / "training_manifest.json"
        manifest_path.write_text(json.dumps({
            "creator_id": creator_id,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "rank": rank,
            "n_clips": len(manifest),
        }, indent=2))

        _set_status(
            job_id, status="completed", finished_at=datetime.utcnow(),
            progress=1.0, lora_path=str(adapter_path),
        )
        _append_log(job_id, f"Saved adapter to {adapter_path}")

    except Exception as e:
        _append_log(job_id, "ERROR:\n" + traceback.format_exc())
        _set_status(job_id, status="failed", finished_at=datetime.utcnow())
        raise
