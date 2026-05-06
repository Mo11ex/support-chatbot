from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import yaml
from transformers import AutoTokenizer


CONFIG_PATH = Path("ml/configs/stage5_chunking_rag_only.yaml")


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(text: str) -> str:
    text = str(text).replace("\xa0", " ").replace("\t", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def split_markdown_sections(text: str):
    lines = text.splitlines()

    current_h1 = None
    current_h2 = None
    current_h3 = None
    current_body = []

    sections = []

    def flush():
        nonlocal current_body, sections, current_h1, current_h2, current_h3
        body_text = "\n".join(current_body).strip()
        if body_text:
            header_parts = [h for h in [current_h1, current_h2, current_h3] if h]
            sections.append({
                "header_h1": current_h1 or "",
                "header_h2": current_h2 or "",
                "header_h3": current_h3 or "",
                "header_path": " > ".join(header_parts),
                "body": normalize_text(body_text),
            })
        current_body = []

    for line in lines:
        m = re.match(r"^(#{1,3})\s+(.*)$", line.strip())
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()

            if level == 1:
                current_h1 = title
                current_h2 = None
                current_h3 = None
            elif level == 2:
                current_h2 = title
                current_h3 = None
            elif level == 3:
                current_h3 = title
        else:
            current_body.append(line)

    flush()
    return sections


def render_chunk_text(h1: str, h2_list: list[str], h3_list: list[str], bodies: list[str]) -> str:
    lines = []

    if h1:
        lines.append(f"# {h1}")

    # добавим список покрываемых заголовков
    unique_h2 = [x for i, x in enumerate(h2_list) if x and x not in h2_list[:i]]
    unique_h3 = [x for i, x in enumerate(h3_list) if x and x not in h3_list[:i]]

    if unique_h2:
        lines.append("## Covered sections")
        for h in unique_h2:
            lines.append(f"- {h}")

    if unique_h3:
        lines.append("## Covered subsections")
        for h in unique_h3:
            lines.append(f"- {h}")

    lines.append("")
    lines.extend(bodies)

    return normalize_text("\n\n".join(lines))


def get_overlap_tail(tokenizer, section_items: list[dict], overlap_tokens: int):
    selected = []
    total = 0
    for item in reversed(section_items):
        selected.insert(0, item)
        total += count_tokens(tokenizer, item["body"])
        if total >= overlap_tokens:
            break
    return selected


def chunk_document(
    tokenizer,
    sections: list[dict],
    target_min_tokens: int,
    target_max_tokens: int,
    overlap_tokens: int,
):
    chunks = []
    current = []

    i = 0
    while i < len(sections):
        section = sections[i]
        candidate = current + [section]

        candidate_text = render_chunk_text(
            h1=section["header_h1"] if not current else current[0]["header_h1"],
            h2_list=[x["header_h2"] for x in candidate],
            h3_list=[x["header_h3"] for x in candidate],
            bodies=[x["body"] for x in candidate],
        )
        candidate_tokens = count_tokens(tokenizer, candidate_text)

        if candidate_tokens <= target_max_tokens:
            current = candidate
            i += 1
            continue

        if current:
            chunk_text = render_chunk_text(
                h1=current[0]["header_h1"],
                h2_list=[x["header_h2"] for x in current],
                h3_list=[x["header_h3"] for x in current],
                bodies=[x["body"] for x in current],
            )
            chunk_tokens = count_tokens(tokenizer, chunk_text)

            if chunk_tokens >= target_min_tokens:
                chunks.append({
                    "header_h1": current[0]["header_h1"],
                    "header_h2_list": [x["header_h2"] for x in current if x["header_h2"]],
                    "header_h3_list": [x["header_h3"] for x in current if x["header_h3"]],
                    "header_path_start": current[0]["header_path"],
                    "header_path_end": current[-1]["header_path"],
                    "text": chunk_text,
                    "token_count": chunk_tokens,
                })

                current = get_overlap_tail(tokenizer, current, overlap_tokens)
            else:
                # если current слишком маленький, принудительно добавляем section, даже если выходим за max
                current = candidate
                i += 1
        else:
            # одиночная очень большая секция
            single_text = render_chunk_text(
                h1=section["header_h1"],
                h2_list=[section["header_h2"]],
                h3_list=[section["header_h3"]],
                bodies=[section["body"]],
            )
            single_tokens = count_tokens(tokenizer, single_text)

            chunks.append({
                "header_h1": section["header_h1"],
                "header_h2_list": [section["header_h2"]] if section["header_h2"] else [],
                "header_h3_list": [section["header_h3"]] if section["header_h3"] else [],
                "header_path_start": section["header_path"],
                "header_path_end": section["header_path"],
                "text": single_text,
                "token_count": single_tokens,
            })
            i += 1

    if current:
        chunk_text = render_chunk_text(
            h1=current[0]["header_h1"],
            h2_list=[x["header_h2"] for x in current],
            h3_list=[x["header_h3"] for x in current],
            bodies=[x["body"] for x in current],
        )
        chunk_tokens = count_tokens(tokenizer, chunk_text)

        chunks.append({
            "header_h1": current[0]["header_h1"],
            "header_h2_list": [x["header_h2"] for x in current if x["header_h2"]],
            "header_h3_list": [x["header_h3"] for x in current if x["header_h3"]],
            "header_path_start": current[0]["header_path"],
            "header_path_end": current[-1]["header_path"],
            "text": chunk_text,
            "token_count": chunk_tokens,
        })

    return chunks


def main():
    cfg = load_yaml(CONFIG_PATH)

    tokenizer_model_name = cfg["tokenizer_model_name"]
    target_min_tokens = int(cfg["chunking"]["target_min_tokens"])
    target_max_tokens = int(cfg["chunking"]["target_max_tokens"])
    overlap_tokens = int(cfg["chunking"]["overlap_tokens"])

    output_csv = Path(cfg["output"]["chunks_csv"])
    summary_json = Path(cfg["output"]["summary_json"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading tokenizer: {tokenizer_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)

    rows = []
    chunk_counter = 1

    for source_cfg in cfg["sources"]:
        source_path = Path(source_cfg["path"])
        category = source_cfg["category"]
        source_group = source_cfg["source_group"]

        if not source_path.exists():
            raise FileNotFoundError(f"Missing source file: {source_path}")

        text = source_path.read_text(encoding="utf-8")
        sections = split_markdown_sections(text)

        doc_chunks = chunk_document(
            tokenizer=tokenizer,
            sections=sections,
            target_min_tokens=target_min_tokens,
            target_max_tokens=target_max_tokens,
            overlap_tokens=overlap_tokens,
        )

        for chunk in doc_chunks:
            rows.append({
                "chunk_id": f"stg5_v2_chunk_{chunk_counter:05d}",
                "category": category,
                "source_group": source_group,
                "source_file": source_path.name,
                "source_path": str(source_path),
                "header_h1": chunk["header_h1"],
                "header_h2_list": json.dumps(chunk["header_h2_list"], ensure_ascii=False),
                "header_h3_list": json.dumps(chunk["header_h3_list"], ensure_ascii=False),
                "header_path_start": chunk["header_path_start"],
                "header_path_end": chunk["header_path_end"],
                "text": chunk["text"],
                "token_count": chunk["token_count"],
            })
            chunk_counter += 1

        print(f"[INFO] {source_path.name}: {len(sections)} sections -> {len(doc_chunks)} chunks")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    summary = {
        "num_chunks": int(len(df)),
        "num_files": int(df["source_file"].nunique()),
        "categories": df["category"].value_counts().sort_index().to_dict(),
        "token_stats": {
            "min": int(df["token_count"].min()),
            "max": int(df["token_count"].max()),
            "mean": float(round(df["token_count"].mean(), 2)),
            "median": float(round(df["token_count"].median(), 2)),
        },
        "output_csv": str(output_csv),
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[INFO] Chunking V2 summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[OK] Saved chunks CSV: {output_csv}")
    print(f"[OK] Saved summary JSON: {summary_json}")


if __name__ == "__main__":
    main()