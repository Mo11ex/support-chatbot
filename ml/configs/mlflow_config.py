from pathlib import Path

# Корень проекта — папка support-chatbot
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Все данные MLflow хранятся здесь
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'ml' / 'logs' / 'mlflow.db'}"
MLFLOW_ARTIFACT_ROOT = str(PROJECT_ROOT / "ml" / "logs" / "mlruns")