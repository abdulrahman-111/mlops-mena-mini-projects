from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict







 
class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_prefix="PRODML_",  # export PRODML_MODEL_PATH=/app/models/model.pkl
        env_file=".env",
        extra="ignore",
    )

    project_root: Path = Path("../")
    data_dir: Path = project_root / "data/raw/"
    data_path: Path = data_dir  / "green_tripdata_2026-01.parquet"

    model_dir:Path = project_root / "models/"
    model_path:Path = model_dir / "model.pkl"
    onnx_path: Path = model_dir / "onnx.pkl"

    model_version: str = "0.1.0"

    test_size: float = 0.2
    random_state: int = 42

    min_duration: float = 1.0
    max_duration: float = 60.0

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    log_level: str = "INFO"


settings = Settings()