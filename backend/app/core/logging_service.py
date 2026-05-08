import json
import time
from pathlib import Path
from app.config import settings


class RequestLogger:
    def __init__(self):
        self.log_path = Path(settings.request_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[RequestLogger] Logging to: {self.log_path}")

    def log(
        self,
        text: str,
        branch: str,
        intent: str | None,
        intent_confidence: float | None,
        faq_score: float | None,
        rag_score: float | None,
        source_type: str | None,
        source_id: str | None,
        answer: str | None,
        latency_ms: float,
        fallback_recommended: bool = False,
    ):
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "text": text,
            "branch": branch,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "faq_score": faq_score,
            "rag_score": rag_score,
            "source_type": source_type,
            "source_id": source_id,
            "answer": answer[:300] if answer else None,
            "latency_ms": round(latency_ms, 2),
            "fallback_recommended": fallback_recommended,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")