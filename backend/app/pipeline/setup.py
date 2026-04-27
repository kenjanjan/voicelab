"""One-shot CosyVoice setup: clone repo + download CosyVoice2-0.5B weights.

Status is persisted to data/setup_status.json so the frontend can poll it
without needing a websocket. Safe to call repeatedly — each step is idempotent.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import traceback
from pathlib import Path

from app.core.config import settings

STATUS_PATH = settings.DATA_DIR / "setup_status.json"


def _write(state: dict) -> None:
    STATUS_PATH.write_text(json.dumps(state, indent=2))


def read_status() -> dict:
    if not STATUS_PATH.exists():
        return {"status": "idle", "step": None, "progress": 0.0, "log": ""}
    try:
        return json.loads(STATUS_PATH.read_text())
    except json.JSONDecodeError:
        return {"status": "idle", "step": None, "progress": 0.0, "log": ""}


def _append_log(state: dict, line: str) -> None:
    state["log"] = (state.get("log", "") + line + "\n")[-8000:]
    _write(state)


def detect_models() -> dict:
    cosy_repo = settings.COSYVOICE_PATH.resolve()
    repo_present = cosy_repo.exists() and (cosy_repo / "cosyvoice").exists()
    weights_dir = settings.COSYVOICE_MODEL.resolve()
    weights_present = weights_dir.exists() and any(weights_dir.iterdir())
    return {
        "cosyvoice_repo": repo_present,
        "cosyvoice_repo_path": str(cosy_repo),
        "cosyvoice_weights": weights_present,
        "cosyvoice_weights_path": str(weights_dir),
    }


def gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            i = 0
            props = torch.cuda.get_device_properties(i)
            return {
                "available": True,
                "kind": "cuda",
                "name": torch.cuda.get_device_name(i),
                "vram_gb": round(props.total_memory / 1e9, 1),
            }
        if torch.backends.mps.is_available():
            return {"available": True, "kind": "mps", "name": "Apple Silicon (MPS)", "vram_gb": None}
        return {"available": False, "kind": "cpu", "name": "CPU only", "vram_gb": None}
    except Exception as e:
        return {"available": False, "kind": "unknown", "name": str(e), "vram_gb": None}


def run_setup() -> None:
    state = {"status": "running", "step": "starting", "progress": 0.01, "log": ""}
    _write(state)

    try:
        # 1. Clone CosyVoice repo
        cosy = settings.COSYVOICE_PATH.resolve()
        if cosy.exists() and (cosy / "cosyvoice").exists():
            _append_log(state, f"[skip] CosyVoice repo present at {cosy}")
        else:
            state["step"] = "cloning_cosyvoice"
            state["progress"] = 0.05
            _write(state)
            cosy.parent.mkdir(parents=True, exist_ok=True)
            if cosy.exists():
                shutil.rmtree(cosy)
            r = subprocess.run(
                ["git", "clone", "--recursive",
                 "https://github.com/FunAudioLLM/CosyVoice.git", str(cosy)],
                capture_output=True, text=True,
            )
            _append_log(state, r.stdout)
            _append_log(state, r.stderr)
            if r.returncode != 0:
                raise RuntimeError("git clone failed")

        # 2. Download model weights
        weights = settings.COSYVOICE_MODEL.resolve()
        if weights.exists() and any(weights.iterdir()):
            _append_log(state, f"[skip] weights present at {weights}")
        else:
            state["step"] = "downloading_weights"
            state["progress"] = 0.3
            _write(state)
            try:
                from modelscope import snapshot_download  # type: ignore
            except ImportError as e:
                raise RuntimeError("modelscope not installed (`pip install modelscope`)") from e
            weights.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download("iic/CosyVoice2-0.5B", local_dir=str(weights))
            _append_log(state, f"weights downloaded to {weights}")

        state.update({"status": "completed", "step": "done", "progress": 1.0})
        _write(state)

    except Exception:
        _append_log(state, "ERROR:\n" + traceback.format_exc())
        state["status"] = "failed"
        _write(state)
        raise
