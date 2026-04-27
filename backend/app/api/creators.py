import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import Creator, get_session

router = APIRouter()


class CreatorIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    consent_signed: bool = False


class CreatorOut(BaseModel):
    id: str
    name: str
    consent_signed: bool


@router.get("", response_model=list[CreatorOut])
def list_creators(db: Session = Depends(get_session)) -> list[CreatorOut]:
    rows = db.execute(select(Creator).order_by(Creator.created_at.desc())).scalars().all()
    return [CreatorOut(id=c.id, name=c.name, consent_signed=c.consent_signed) for c in rows]


@router.post("", response_model=CreatorOut, status_code=201)
def create_creator(body: CreatorIn, db: Session = Depends(get_session)) -> CreatorOut:
    c = Creator(id=uuid.uuid4().hex[:10], name=body.name, consent_signed=body.consent_signed)
    db.add(c)
    db.commit()
    return CreatorOut(id=c.id, name=c.name, consent_signed=c.consent_signed)


@router.get("/{creator_id}", response_model=CreatorOut)
def get_creator(creator_id: str, db: Session = Depends(get_session)) -> CreatorOut:
    c = db.get(Creator, creator_id)
    if not c:
        raise HTTPException(404, "Creator not found")
    return CreatorOut(id=c.id, name=c.name, consent_signed=c.consent_signed)


@router.patch("/{creator_id}/consent", response_model=CreatorOut)
def sign_consent(creator_id: str, db: Session = Depends(get_session)) -> CreatorOut:
    c = db.get(Creator, creator_id)
    if not c:
        raise HTTPException(404, "Creator not found")
    c.consent_signed = True
    db.commit()
    return CreatorOut(id=c.id, name=c.name, consent_signed=c.consent_signed)
