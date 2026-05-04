import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.utils.tracking import start_run, log_params, log_metrics, end_run

with start_run(experiment_name="stage0-smoke-test", run_name="baseline"):
    log_params({
        "model": "test_model",
        "seed": 42,
        "batch_size": 32,
    })
    log_metrics({
        "macro_f1": 0.91,
        "accuracy": 0.89,
        "precision_at_1": 0.92,
        "recall_at_5": 0.87,
    })

print("[OK] MLflow run logged successfully.")
print("[INFO] Run: mlflow ui — to see results at http://127.0.0.1:5000")