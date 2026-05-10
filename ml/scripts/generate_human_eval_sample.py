import pandas as pd
from pathlib import Path

INPUT_PATH = Path("ml/logs/reports/stage8_e2e_evaluation/e2e_predictions.csv")
OUTPUT_PATH = Path("ml/logs/reports/stage8_e2e_evaluation/human_eval_sample.csv")

def main():
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} not found.")
        return

    df = pd.read_csv(INPUT_PATH)
    
    # Исключаем запросы, которые явно упали
    df = df[df["status"] == "ok"]

    # Берем 50 случайных примеров с фиксированным seed
    sample = df.sample(n=50, random_state=42)

    # Оставляем только нужные для разметки колонки
    cols = ["id", "text", "true_label", "pred_branch", "answer_preview"]
    sample = sample[cols]

    # Добавляем пустые колонки для ручной разметки
    sample["correctness_0_1"] = ""
    sample["helpfulness_1_5"] = ""
    sample["naturalness_1_5"] = ""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"[OK] Сгенерирован файл для ручной разметки: {OUTPUT_PATH}")
    print("Открой его в Excel/Calc и заполни последние 3 колонки.")

if __name__ == "__main__":
    main()