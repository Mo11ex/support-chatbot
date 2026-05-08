import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.config import settings


class ClassifierService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_dir = settings.classifier_model_dir

        print(f"[ClassifierService] Loading model from: {model_dir}")
        print(f"[ClassifierService] Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

        print(f"[ClassifierService] Labels: {list(self.id2label.values())}")

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=settings.classifier_max_length,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        top3_indices = probs.argsort(descending=True)[:3]
        top_idx = top3_indices[0].item()

        return {
            "category": self.id2label[top_idx],
            "confidence": float(probs[top_idx].item()),
            "top_3": [
                {
                    "category": self.id2label[idx.item()],
                    "confidence": float(probs[idx].item()),
                }
                for idx in top3_indices
            ],
        }