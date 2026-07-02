from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Shared project configuration for deterministic synthetic workflows."""

    random_seed: int = 202601
    forecast_horizon_days: int = 14
    synthetic_data_dir: Path = Path("data/synthetic")
    report_output_dir: Path = Path("outputs/reports")


DEFAULT_CONFIG = AppConfig()

