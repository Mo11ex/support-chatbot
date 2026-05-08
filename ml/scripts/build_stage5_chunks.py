from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import yaml
from transformers import AutoTokenizer


CONFIG_PATH = Path("ml/configs/stage5_chunking.yaml")


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
    """
    Разбивает markdown на секции по заголовкам # / ## / ###.
    Возвращает список секций:
    {
      h1, h2, h3, header_path, body
    }
    """
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
                "body": body_text,
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


def split_into_blocks(body: str):
    """
    Делит секцию на смысловые блоки по пустым строкам.
    """
    raw_blocks = re.split(r"\n\s*\n", body)
    blocks = [normalize_text(block) for block in raw_blocks if normalize_text(block)]
    return blocks


def render_chunk_text(h1: str, h2: str, h3: str, blocks: list[str]) -> str:
    headers = []
    if h1:
        headers.append(f"# {h1}")
    if h2:
        headers.append(f"## {h2}")
    if h3:
        headers.append(f"### {h3}")

    header_text = "\n".join(headers).strip()
    body_text = "\n\n".join(blocks).strip()

    if header_text and body_text:
        return f"{header_text}\n\n{body_text}"
    elif header_text:
        return header_text
    return body_text


def get_overlap_blocks(tokenizer, blocks: list[str], overlap_tokens: int) -> list[str]:
    if not blocks:
        return []

    selected = []
    total = 0
    for block in reversed(blocks):
        selected.insert(0, block)
        total += count_tokens(tokenizer, block)
        if total >= overlap_tokens:
            break
    return selected


def chunk_section(
    tokenizer,
    section: dict,
    target_min_tokens: int,
    target_max_tokens: int,
    overlap_tokens: int,
    allow_short_faq_chunks: bool,
):
    h1 = section["header_h1"]
    h2 = section["header_h2"]
    h3 = section["header_h3"]

    blocks = split_into_blocks(section["body"])
    if not blocks:
        return []

    chunks = []
    current_blocks = []

    i = 0
    while i < len(blocks):
        block = blocks[i]

        candidate_blocks = current_blocks + [block]
        candidate_text = render_chunk_text(h1, h2, h3, candidate_blocks)
        candidate_tokens = count_tokens(tokenizer, candidate_text)

        if candidate_tokens <= target_max_tokens:
            current_blocks = candidate_blocks
            i += 1
            continue

        # Если current_blocks уже есть, сохраняем chunk
        if current_blocks:
            chunk_text = render_chunk_text(h1, h2, h3, current_blocks)
            chunk_tokens = count_tokens(tokenizer, chunk_text)

            if chunk_tokens >= target_min_tokens or allow_short_faq_chunks:
                chunks.append({
                    "header_h1": h1,
                    "header_h2": h2,
                    "header_h3": h3,
                    "header_path": section["header_path"],
                    "text": chunk_text,
                    "token_count": chunk_tokens,
                })

            overlap_blocks = get_overlap_blocks(tokenizer, current_blocks, overlap_tokens)
            current_blocks = overlap_blocks

            # если даже overlap + новый блок снова длиннее max, сбрасываем overlap
            retry_text = render_chunk_text(h1, h2, h3, current_blocks + [block])
            if count_tokens(tokenizer, retry_text) > target_max_tokens and current_blocks:
                current_blocks = []

            continue

        # Если один блок сам по себе слишком большой — сохраняем как есть
        single_text = render_chunk_text(h1, h2, h3, [block])
        single_tokens = count_tokens(tokenizer, single_text)
        chunks.append({
            "header_h1": h1,
            "header_h2": h2,
            "header_h3": h3,
            "header_path": section["header_path"],
            "text": single_text,
            "token_count": single_tokens,
        })
        i += 1

    if current_blocks:
        chunk_text = render_chunk_text(h1, h2, h3, current_blocks)
        chunk_tokens = count_tokens(tokenizer, chunk_text)
        if chunk_tokens >= target_min_tokens or allow_short_faq_chunks:
            chunks.append({
                "header_h1": h1,
                "header_h2": h2,
                "header_h3": h3,
                "header_path": section["header_path"],
                "text": chunk_text,
                "token_count": chunk_tokens,
            })

    return chunks


def main():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")

    cfg = load_yaml(CONFIG_PATH)

    tokenizer_model_name = cfg["tokenizer_model_name"]
    target_min_tokens = int(cfg["chunking"]["target_min_tokens"])
    target_max_tokens = int(cfg["chunking"]["target_max_tokens"])
    overlap_tokens = int(cfg["chunking"]["overlap_tokens"])
    allow_short_faq_chunks = bool(cfg["chunking"]["allow_short_faq_chunks"])

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

        file_chunk_count = 0

        for section in sections:
            chunks = chunk_section(
                tokenizer=tokenizer,
                section=section,
                target_min_tokens=target_min_tokens,
                target_max_tokens=target_max_tokens,
                overlap_tokens=overlap_tokens,
                allow_short_faq_chunks=allow_short_faq_chunks,
            )

            for chunk in chunks:
                rows.append({
                    "chunk_id": f"stg5_chunk_{chunk_counter:05d}",
                    "category": category,
                    "source_group": source_group,
                    "source_file": source_path.name,
                    "source_path": str(source_path),
                    "header_h1": chunk["header_h1"],
                    "header_h2": chunk["header_h2"],
                    "header_h3": chunk["header_h3"],
                    "header_path": chunk["header_path"],
                    "text": chunk["text"],
                    "token_count": chunk["token_count"],
                })
                chunk_counter += 1
                file_chunk_count += 1

        print(f"[INFO] {source_path.name}: {len(sections)} sections -> {file_chunk_count} chunks")

    chunks_df = pd.DataFrame(rows)
    chunks_df.to_csv(output_csv, index=False, encoding="utf-8")

    summary = {
        "num_chunks": int(len(chunks_df)),
        "num_files": int(chunks_df["source_file"].nunique()),
        "categories": chunks_df["category"].value_counts().sort_index().to_dict(),
        "source_groups": chunks_df["source_group"].value_counts().sort_index().to_dict(),
        "token_stats": {
            "min": int(chunks_df["token_count"].min()),
            "max": int(chunks_df["token_count"].max()),
            "mean": float(round(chunks_df["token_count"].mean(), 2)),
            "median": float(round(chunks_df["token_count"].median(), 2)),
        },
        "output_csv": str(output_csv),
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[INFO] Chunking summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\n[OK] Saved chunks CSV: {output_csv}")
    print(f"[OK] Saved summary JSON: {summary_json}")


if __name__ == "__main__":
    main()