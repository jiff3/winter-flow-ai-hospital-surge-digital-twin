from winterflow import __version__
from winterflow.config import DEFAULT_CONFIG
from winterflow.constants import APP_SUBTITLE, APP_TITLE, SCENARIO_NAMES


def test_package_imports_cleanly() -> None:
    assert __version__ == "0.1.0"


def test_default_config_is_deterministic() -> None:
    assert DEFAULT_CONFIG.random_seed == 202601
    assert DEFAULT_CONFIG.forecast_horizon_days == 14


def test_core_app_constants_exist() -> None:
    assert APP_TITLE == "WINTER-Flow Command Center"
    assert APP_SUBTITLE == "AI Hospital Surge Digital Twin"
    assert "severe_combined_surge" in SCENARIO_NAMES

