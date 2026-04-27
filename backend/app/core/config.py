from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATA_DIR: Path = Path("./data")
    DATABASE_URL: str = "sqlite:///./data/voicelab.db"
    COSYVOICE_PATH: Path = Path("./vendor/CosyVoice")
    COSYVOICE_MODEL: Path = Path("./pretrained_models/CosyVoice2-0.5B")
    DEVICE: str = "cuda"
    CORS_ORIGIN: str = "http://localhost:3000"


settings = Settings()

for sub in ("raw", "processed", "loras", "outputs"):
    (settings.DATA_DIR / sub).mkdir(parents=True, exist_ok=True)
