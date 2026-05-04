import pandas as pd
from pathlib import Path

files = [
    "ml/data/raw/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv",
    "ml/data/raw/ecommerce_intent.csv",
    "ml/data/raw/synthetic_dataset.csv",
    "ml/data/raw/final/full_dataset.csv",
]

for file in files:
    path = Path(file)
    if not path.exists():
        print(f"[MISS] {file}")
        continue

    print("\n" + "=" * 80)
    print(f"FILE: {file}")
    df = pd.read_csv(path)
    print("SHAPE:", df.shape)
    print("COLUMNS:", df.columns.tolist())
    print(df.head(3))

    for candidate in ["label", "intent", "category", "class"]:
        if candidate in df.columns:
            print(f"\nUnique values in '{candidate}':")
            print(df[candidate].value_counts(dropna=False).head(50))