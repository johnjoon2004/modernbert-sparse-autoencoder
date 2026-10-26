from pathlib import Path
from typing import List

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Local paths, overridable through SAE_* environment variables or .env."""

    model_config = SettingsConfigDict(env_prefix="SAE_", env_file=".env", extra="ignore")

    project_root: Path = Path(__file__).resolve().parents[1]
    lm_weights_path: Path = Path("models/classifier")
    text_dataset_path: Path = Path("data/text_dataset/texts.csv")
    rs_data_dir: Path = Path("data/post_mlp_residual_stream")
    feature_record_dir: Path = Path("data/feature_records")
    feature_meta_dir: Path = Path("data/feature_meta")
    feature_explanation_dir: Path = Path("data/feature_explanations")
    feature_cluster_path: Path = Path("outputs/analysis/clusters.json")
    font_path: Path = Path("assets/NanumGothic.ttf")
    prev_bs: int = 64
    prev_seq_len: int = 1024
    prev_dtype: str = "float16"
    train_dtype: str = "float32"
    layers: List[int] = [1, 6, 11, 15, 21]

    @model_validator(mode="after")
    def resolve_paths(self):
        self.project_root = self.project_root.expanduser().resolve()
        for name in (
            "lm_weights_path", "text_dataset_path", "rs_data_dir", "feature_record_dir",
            "feature_meta_dir", "feature_explanation_dir", "feature_cluster_path", "font_path",
        ):
            path = getattr(self, name).expanduser()
            if not path.is_absolute():
                path = self.project_root / path
            setattr(self, name, path)
        return self


settings = Settings()
