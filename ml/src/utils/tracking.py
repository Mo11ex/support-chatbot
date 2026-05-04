from __future__ import annotations
from pathlib import Path
import mlflow

# Корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Фиксированный путь к хранилищу MLflow
MLFLOW_TRACKING_URI = (
    f"sqlite:///{PROJECT_ROOT / 'ml' / 'logs' / 'mlflow.db'}"
)


def _setup():
    """Настраивает tracking URI один раз при импорте."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


_setup()


def start_run(experiment_name: str, run_name: str | None = None):
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict):
    mlflow.log_params(params)


def log_metrics(metrics: dict, step: int | None = None):
    mlflow.log_metrics(metrics, step=step)


def log_artifact(local_path: str):
    mlflow.log_artifact(local_path)


def end_run():
    mlflow.end_run()