"""
Сервис классификации обращений
Загружает fine-tuned rubert-tiny2 и выполняет предсказания
"""
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import os

CATEGORIES = [
    "account", "delivery", "general_info", "order_status", "other",
    "payment_refund", "product_info", "promo_loyalty",
    "return_exchange", "technical_issue",
]

CATEGORY_LABELS = {
    "account": "Аккаунт",
    "delivery": "Доставка",
    "general_info": "Общая информация",
    "order_status": "Статус заказа",
    "other": "Прочее",
    "payment_refund": "Оплата и возврат средств",
    "product_info": "Товар и ассортимент",
    "promo_loyalty": "Промокоды и бонусы",
    "return_exchange": "Возврат и обмен товара",
    "technical_issue": "Техническая проблема",
}


class IntentClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return self.fc(cls)


class ClassifierService:
    def __init__(self):
        self.model_name = "cointegrated/rubert-tiny2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading classifier on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = IntentClassifier(self.model_name, len(CATEGORIES))

        # Путь к весам
        weights_path = os.path.join(
            os.path.dirname(__file__), "..", "ml", "classifier", "weights", "best_model.pt"
        )
        weights_path = os.path.abspath(weights_path)

        if os.path.exists(weights_path):
            self.model.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )
            print(f"Weights loaded from {weights_path}")
        else:
            print(f"Weights not found at {weights_path}, using random weights")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.model(
                encoded["input_ids"].to(self.device),
                encoded["attention_mask"].to(self.device),
            )
            probs = torch.softmax(logits, dim=1)[0]

        top3_indices = probs.argsort(descending=True)[:3]
        top_cat = CATEGORIES[top3_indices[0]]

        return {
            "category": top_cat,
            "category_label": CATEGORY_LABELS[top_cat],
            "confidence": round(probs[top3_indices[0]].item(), 4),
            "top_3": [
                {
                    "category": CATEGORIES[idx],
                    "confidence": round(probs[idx].item(), 4),
                }
                for idx in top3_indices
            ],
        }