from pathlib import Path
import json

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = Path("ml/models/classifier/stage3-rubert-tiny2-patched")
THRESHOLD_PATH = Path("ml/logs/reports/stage3_threshold_calibration/chosen_threshold.json")

app = FastAPI(title="Local Classifier API", version="1.0.0")


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    threshold: float
    threshold_passed: bool
    fallback_recommended: bool
    top3: list


device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = None
model = None
threshold = 0.5


def softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


@app.on_event("startup")
def startup_event():
    global tokenizer, model, threshold

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model dir not found: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
            threshold_data = json.load(f)
        threshold = float(threshold_data["chosen_threshold"])
    else:
        threshold = 0.5

    print(f"[INFO] Model loaded from: {MODEL_DIR}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Threshold: {threshold}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "model_dir": str(MODEL_DIR),
        "threshold": threshold,
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    text = request.text.strip()

    inputs = tokenizer(
        text,
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits)[0]

    pred_id = int(torch.argmax(probs).item())
    pred_conf = float(probs[pred_id].item())

    id2label = model.config.id2label
    label = id2label[pred_id] if isinstance(id2label, dict) else str(pred_id)

    topk = torch.topk(probs, k=min(3, probs.shape[0]))
    top3 = []
    for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
        top3.append({
            "label": id2label[idx] if isinstance(id2label, dict) else str(idx),
            "score": float(score),
        })

    threshold_passed = pred_conf >= threshold

    return ClassifyResponse(
        label=label,
        confidence=pred_conf,
        threshold=threshold,
        threshold_passed=threshold_passed,
        fallback_recommended=not threshold_passed,
        top3=top3,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)