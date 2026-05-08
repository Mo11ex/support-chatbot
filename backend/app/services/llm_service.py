import time
from pathlib import Path
from llama_cpp import Llama
from app.config import settings

class LlmService:
    def __init__(self):
        model_path = Path(settings.llm_model_path)
        
        print(f"[LlmService] Loading model from: {model_path}")
        if not model_path.exists():
            print(f"[LlmService] WARNING: Model not found at {model_path}")
            self.llm = None
            return

        # Инициализируем LLM (пробуем задействовать GPU, если доступен)
        self.llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=-1, # -1 = все слои на GPU
            n_ctx=2048,      # Размер контекста
            verbose=False    # Отключаем спам в консоль
        )
        print("[LlmService] Model loaded successfully.")

    def generate_answer(self, query: str, context: str) -> str:
        if not self.llm:
            return "Извините, сервис генерации ответов временно недоступен."

        prompt = f"""<|im_start|>system
Ты вежливый, дружелюбный и компетентный оператор службы поддержки интернет-магазина.
Твоя задача — дать подробный, но понятный ответ клиенту, используя ТОЛЬКО факты из предоставленного ниже документа (Контекст).
Если в тексте есть перечисления, цены или шаги — обязательно используй их в ответе. Не придумывай того, чего нет в тексте. Если информации вообще нет — скажи, что переведешь на оператора.
<|im_end|>
<|im_start|>user
Контекст:
{context}

Вопрос клиента: {query}<|im_end|>
<|im_start|>assistant
"""
        
        t0 = time.perf_counter()
        response = self.llm(
            prompt,
            max_tokens=300, # Дадим ей больше места для развернутого ответа
            stop=["<|im_end|>"],
            temperature=0.3, # Чуть-чуть добавим креативности для связности текста (было 0.1)
            top_p=0.9
        )
        latency = (time.perf_counter() - t0) * 1000
        print(f"[LlmService] Generation latency: {latency:.1f} ms")
        
        return response["choices"][0]["text"].strip()