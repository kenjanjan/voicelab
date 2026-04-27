import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import Creator, TrainingJob, get_session
from app.pipeline.lora_train import run_lora_training

router = APIRouter()


class TrainStart(BaseModel):
    epochs: int = 8
    learning_rate: float = 1e-4
    rank: int = 16


class JobOut(BaseModel):
    id: str
    creator_id: str
    status: str
    progress: float
    log: str
    started_at: datetime | None
    finished_at: datetime | None
    lora_path: str | None


@router.post("/{creator_id}/start", response_model=JobOut, status_code=202)
def start_training(
    creator_id: str,
    body: TrainStart,
    bg: BackgroundTasks,
    db: Session = Depends(get_session),
) -> JobOut:
    creator = db.get(Creator, creator_id)
    if not creator:
        raise HTTPException(404, "Creator not found")
    if not creator.consent_signed:
        raise HTTPException(403, "Consent required before training")

    job = TrainingJob(
        id=uuid.uuid4().hex[:12],
        creator_id=creator_id,
        status="pending",
        progress=0.0,
    )
    db.add(job)
    db.commit()

    bg.add_task(run_lora_training, job.id, creator_id, body.epochs, body.learning_rate, body.rank)
    return _to_out(job)


@router.get("/{creator_id}/jobs", response_model=list[JobOut])
def list_jobs(creator_id: str, db: Session = Depends(get_session)) -> list[JobOut]:
    rows = db.execute(
        select(TrainingJob)
        .where(TrainingJob.creator_id == creator_id)
        .order_by(TrainingJob.started_at.desc().nulls_last())
    ).scalars().all()
    return [_to_out(j) for j in rows]


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_session)) -> JobOut:
    j = db.get(TrainingJob, job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    return _to_out(j)


def _to_out(j: TrainingJob) -> JobOut:
    return JobOut(
        id=j.id, creator_id=j.creator_id, status=j.status, progress=j.progress,
        log=j.log, started_at=j.started_at, finished_at=j.finished_at, lora_path=j.lora_path,
    )
