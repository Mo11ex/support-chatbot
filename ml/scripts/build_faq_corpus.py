import re
from pathlib import Path
import pandas as pd

KB_DIR = Path("backend/data/knowledge_base")
OUTPUT_PATH = Path("ml/data/processed/faq_corpus.csv")

DOC_MAP = {
    "delivery_policy.md": "faq_delivery",
    "payment_methods.md": "faq_payment",
    "promo_codes.md": "faq_promo",
    "return_policy.md": "faq_return",
    "warranty.md": "faq_warranty",
}


def split_into_chunks(text: str, source_doc_id: str) -> list[dict]:
    """
    Разбивает markdown-документ на чанки по заголовкам ## и ###.
    Каждый чанк получает уникальный chunk_id.
    """
    lines = text.strip().split("\n")

    chunks = []
    current_title = "intro"
    current_lines = []

    for line in lines:
        # Новый раздел начинается с ## или ###
        if re.match(r"^#{2,3}\s+", line):
            # Сохраняем предыдущий чанк
            if current_lines:
                chunk_text = "\n".join(current_lines).strip()
                if chunk_text:
                    chunks.append({
                        "title": current_title,
                        "text": chunk_text
                    })

            current_title = re.sub(r"^#{2,3}\s+", "", line).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    # Последний чанк
    if current_lines:
        chunk_text = "\n".join(current_lines).strip()
        if chunk_text:
            chunks.append({
                "title": current_title,
                "text": chunk_text
            })

    # Формируем строки с chunk_id
    rows = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{source_doc_id}_chunk_{i}"
        # Заменяем переносы строк на пробелы для CSV-совместимости
        clean_text = " ".join(chunk["text"].split())

        rows.append({
            "doc_id": source_doc_id,
            "chunk_id": chunk_id,
            "title": chunk["title"],
            "text": clean_text
        })

    return rows


def main():
    all_rows = []

    for file_name, doc_id in DOC_MAP.items():
        file_path = KB_DIR / file_name
        if not file_path.exists():
            print(f"[WARN] File not found: {file_path}")
            continue

        raw_text = file_path.read_text(encoding="utf-8")
        chunks = split_into_chunks(raw_text, doc_id)
        all_rows.extend(chunks)
        print(f"[OK] {file_name} -> {len(chunks)} chunks")

    df = pd.DataFrame(all_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\n[OK] Saved FAQ corpus to: {OUTPUT_PATH}")
    print(f"[INFO] Total chunks: {len(df)}")
    print(f"\n[INFO] Chunks per document:")
    print(df.groupby("doc_id").size())


if __name__ == "__main__":
    main()