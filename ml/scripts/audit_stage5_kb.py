from pathlib import Path
import json
import pandas as pd


FAQ_KB_DIR = Path("backend/data/knowledge_base")
OUTPUT_DIR = Path("ml/logs/reports/stage5_kb_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_md_stats(path: Path):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    word_count = len(text.split())
    char_count = len(text)
    h1_count = sum(1 for line in lines if line.startswith("# "))
    h2_count = sum(1 for line in lines if line.startswith("## "))
    h3_count = sum(1 for line in lines if line.startswith("### "))
    faq_like_count = sum(
        1 for line in lines
        if line.strip().startswith("## ") and "?" in line
    ) + sum(
        1 for line in lines
        if line.strip().startswith("### ") and "?" in line
    )

    return {
        "file_name": path.name,
        "word_count": word_count,
        "char_count": char_count,
        "h1_count": h1_count,
        "h2_count": h2_count,
        "h3_count": h3_count,
        "faq_like_headings": faq_like_count,
    }


def main():
    if not FAQ_KB_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {FAQ_KB_DIR}")

    md_files = sorted(FAQ_KB_DIR.glob("*.md"))
    if not md_files:
        raise ValueError(f"No markdown files found in {FAQ_KB_DIR}")

    rows = []
    for path in md_files:
        rows.append(read_md_stats(path))

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "current_kb_audit.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    summary = {
        "kb_dir": str(FAQ_KB_DIR),
        "num_files": int(len(df)),
        "total_words": int(df["word_count"].sum()),
        "total_chars": int(df["char_count"].sum()),
        "files": df.to_dict(orient="records"),
    }

    summary_path = OUTPUT_DIR / "current_kb_audit_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[INFO] Current KB audit:")
    print(df.to_string(index=False))

    print(f"\n[INFO] Total files: {len(df)}")
    print(f"[INFO] Total words: {df['word_count'].sum()}")
    print(f"[INFO] Total chars: {df['char_count'].sum()}")

    print(f"\n[OK] Saved CSV: {csv_path}")
    print(f"[OK] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()