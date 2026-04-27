import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db import Creator, get_session
from app.pipeline.tts_engine import EMOTIONS, synthesize

router = APIRouter()


class TTSRequest(BaseModel):
    creator_id: str
    text: str = Field(..., min_length=1, max_length=600)
    emotion: str = "flirty"
    speed: float = Field(1.0, ge=0.7, le=1.3)
    seed: int | None = None
    use_lora: bool = True


class TTSResponse(BaseModel):
    url: str
    filename: str
    emotion: str
    seed: int | None


@router.get("/emotions")
def list_emotions() -> list[str]:
    return list(EMOTIONS.keys())


@router.post("", response_model=TTSResponse)
def synth(req: TTSRequest, db: Session = Depends(get_session)) -> TTSResponse:
    creator = db.get(Creator, req.creator_id)
    if not creator:
        raise HTTPException(404, "Creator not found")
    if not creator.consent_signed:
        raise HTTPException(403, "Creator consent required for synthesis")
    if req.emotion not in EMOTIONS:
        raise HTTPException(400, f"emotion must be one of {list(EMOTIONS)}")

    out_dir = settings.DATA_DIR / "outputs" / req.creator_id
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{uuid.uuid4().hex[:10]}.wav"
    out_path = out_dir / fname

    used_seed = synthesize(
        creator_id=req.creator_id,
        text=req.text,
        emotion=req.emotion,
        speed=req.speed,
        seed=req.seed,
        out_path=out_path,
        use_lora=req.use_lora,
    )

    rel = out_path.relative_to(settings.DATA_DIR).as_posix()
    return TTSResponse(
        url=f"/files/{rel}", filename=fname, emotion=req.emotion, seed=used_seed,
    )


@router.get("/file/{creator_id}/{filename}")
def get_file(creator_id: str, filename: str) -> FileResponse:
    p = settings.DATA_DIR / "outputs" / creator_id / Path(filename).name
    if not p.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(p, media_type="audio/wav", filename=filename)
