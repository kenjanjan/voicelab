from datetime import datetime
from typing import Iterator

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, Session, mapped_column, sessionmaker,
)

from app.core.config import settings

engine = create_engine(settings.DATABASE_URL, future=True, echo=False)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Creator(Base):
    __tablename__ = "creators"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    consent_signed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AudioClip(Base):
    __tablename__ = "audio_clips"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    creator_id: Mapped[str] = mapped_column(ForeignKey("creators.id"))
    path: Mapped[str] = mapped_column(String)
    text: Mapped[str] = mapped_column(Text, default="")
    emotion: Mapped[str | None] = mapped_column(String, nullable=True)
    duration: Mapped[float] = mapped_column(Float)
    is_reference: Mapped[bool] = mapped_column(Boolean, default=False)


class TrainingJob(Base):
    __tablename__ = "training_jobs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    creator_id: Mapped[str] = mapped_column(ForeignKey("creators.id"))
    status: Mapped[str] = mapped_column(String, default="pending")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    log: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    lora_path: Mapped[str | None] = mapped_column(String, nullable=True)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
