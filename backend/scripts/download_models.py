"""Clone CosyVoice + download CosyVoice2-0.5B weights.

Run from the backend/ directory:
    python scripts/download_models.py
"""
import subprocess
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "vendor"
MODELS = Path(__file__).resolve().parents[1] / "pretrained_models"


def clone_cosyvoice() -> None:
    VENDOR.mkdir(parents=True, exist_ok=True)
    target = VENDOR / "CosyVoice"
    if target.exists():
        print(f"[skip] {target} exists")
        return
    subprocess.run(
        ["git", "clone", "--recursive",
         "https://github.com/FunAudioLLM/CosyVoice.git", str(target)],
        check=True,
    )


def download_weights() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    target = MODELS / "CosyVoice2-0.5B"
    if target.exists():
        print(f"[skip] {target} exists")
        return
    try:
        from modelscope import snapshot_download  # type: ignore
    except ImportError:
        print("modelscope not installed. pip install modelscope", file=sys.stderr)
        sys.exit(1)
    snapshot_download("iic/CosyVoice2-0.5B", local_dir=str(target))


if __name__ == "__main__":
    clone_cosyvoice()
    download_weights()
    print("Done. Add backend/vendor/CosyVoice to your PYTHONPATH or rely on tts_engine auto-injection.")
