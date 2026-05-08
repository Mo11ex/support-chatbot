from pathlib import Path
from llama_cpp import Llama

MODEL_PATH = Path("ml/models/llm/qwen2.5-3b-instruct-q4_k_m.gguf")

def main():
    if not MODEL_PATH.exists():
        print(f"Файл модели не найден: {MODEL_PATH}")
        return

    print("Загружаем модель в память (ищем GPU)...")
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_gpu_layers=-1, # -1 означает "загрузить все слои в видеокарту"
        n_ctx=2048,      # размер контекстного окна
        verbose=True     # покажет логи (увидим, сработала ли CUDA)
    )

    prompt = """<|im_start|>system
Ты вежливый оператор службы поддержки интернет-магазина. Отвечай кратко.<|im_end|>
<|im_start|>user
Как оформить возврат товара?<|im_end|>
<|im_start|>assistant
"""

    print("\n--- Генерация ответа ---")
    response = llm(
        prompt,
        max_tokens=150,
        stop=["<|im_end|>"],
        temperature=0.1
    )

    print("\nОТВЕТ LLM:")
    print(response["choices"][0]["text"].strip())


if __name__ == "__main__":
    main()
    