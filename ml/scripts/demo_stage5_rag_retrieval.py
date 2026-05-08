from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer


CONFIG_PATH = Path("ml/configs/stage5_rag_retrieval.yaml")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).replace("\xa0", " ").replace("\t", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_for_bm25(text: str) -> list[str]:
    text = normalize_text(text).lower()
    return re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text)


def encode_query(model, query: str, prefix: str) -> np.ndarray:
    text = f"{prefix} {query}"
    emb = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    return emb[0]


def dense_search(index, metadata_df, q_emb, top_n):
    scores, indices = index.search(q_emb.reshape(1, -1), top_n)
    scores = scores[0]
    indices = indices[0]

    valid_mask = indices >= 0
    scores = scores[valid_mask]
    indices = indices[valid_mask]

    results = metadata_df.iloc[indices].copy().reset_index(drop=True)
    results["dense_score"] = scores
    return results.sort_values("dense_score", ascending=False).reset_index(drop=True)


def apply_category_filter(results_df, category):
    if category is None:
        return results_df
    filtered = results_df[results_df["category"] == category].copy().reset_index(drop=True)
    return filtered


def print_results(results_df, top_k: int, label: str):
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"{'='*60}")
    for i, row in results_df.head(top_k).iterrows():
        print(f"\n  Rank {i+1}")
        print(f"  chunk_id:    {row['chunk_id']}")
        print(f"  category:    {row['category']}")
        print(f"  source_file: {row['source_file']}")
        print(f"  token_count: {row['token_count']}")
        print(f"  dense_score: {row['dense_score']:.6f}")
        print(f"  header:      {row.get('header_path_start', '')}")
        snippet = str(row.get("text", ""))[:250].replace("\n", " ")
        print(f"  text[:250]:  {snippet}...")


def main():
    cfg = load_yaml(CONFIG_PATH)

    index_dir = Path(cfg["index_dir"])
    model_name = cfg["dense"]["model_name"]
    query_prefix = cfg["dense"]["query_prefix"]
    top_k_no_filter = int(cfg["retrieval"]["top_k_no_filter"])
    top_k_with_filter = int(cfg["retrieval"]["top_k_with_filter"])
    top_n_search = int(cfg["retrieval"]["top_n_search"])
    category_map = cfg["classifier_to_rag_category"]
    report_dir = Path(cfg["output"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.csv"

    for p in [faiss_path, metadata_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    metadata_df = pd.read_csv(metadata_path)
    index = faiss.read_index(str(faiss_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading model: {model_name}")
    print(f"[INFO] Device: {device}")
    model = SentenceTransformer(model_name, device=device)

    # Примеры запросов с classifier labels
    test_cases = [
        {"query": "где мой заказ?", "classifier_label": "order_status"},
        {"query": "не могу войти в личный кабинет", "classifier_label": "account"},
        {"query": "оплата не проходит", "classifier_label": "payment_refund"},
        {"query": "как отследить посылку?", "classifier_label": "delivery"},
        {"query": "сайт не работает, белый экран", "classifier_label": "technical_issue"},
        {"query": "можно ли отменить заказ?", "classifier_label": "order_status"},
        {"query": "деньги списались а заказ не создался", "classifier_label": "payment_refund"},
        {"query": "забыл пароль от аккаунта", "classifier_label": "account"},
        {"query": "какие размеры есть?", "classifier_label": "product_info"},
        {"query": "где купить со скидкой?", "classifier_label": "promo_loyalty"},
    ]

    demo_results = []

    for tc in test_cases:
        query = tc["query"]
        classifier_label = tc["classifier_label"]
        rag_category = category_map.get(classifier_label)

        print(f"\n{'#'*60}")
        print(f"Query:            {query}")
        print(f"Classifier label: {classifier_label}")
        print(f"RAG category:     {rag_category}")

        q_emb = encode_query(model, query, query_prefix)
        results_all = dense_search(index, metadata_df, q_emb, top_n_search)
        results_filtered = apply_category_filter(results_all, rag_category)

        print_results(results_all, top_k_no_filter, "NO FILTER — top-5")
        print_results(results_filtered, top_k_with_filter, f"WITH FILTER [{rag_category}] — top-5")

        top_no_filter = results_all.head(top_k_no_filter)
        top_with_filter = results_filtered.head(top_k_with_filter)

        demo_results.append({
            "query": query,
            "classifier_label": classifier_label,
            "rag_category": rag_category,
            "top1_no_filter_chunk_id": str(top_no_filter.iloc[0]["chunk_id"]) if len(top_no_filter) > 0 else None,
            "top1_no_filter_category": str(top_no_filter.iloc[0]["category"]) if len(top_no_filter) > 0 else None,
            "top1_no_filter_score": float(top_no_filter.iloc[0]["dense_score"]) if len(top_no_filter) > 0 else None,
            "top1_with_filter_chunk_id": str(top_with_filter.iloc[0]["chunk_id"]) if len(top_with_filter) > 0 else None,
            "top1_with_filter_category": str(top_with_filter.iloc[0]["category"]) if len(top_with_filter) > 0 else None,
            "top1_with_filter_score": float(top_with_filter.iloc[0]["dense_score"]) if len(top_with_filter) > 0 else None,
            "filter_category_present_in_results": rag_category in results_all["category"].values if rag_category else None,
        })

    demo_df = pd.DataFrame(demo_results)
    demo_path = report_dir / "demo_retrieval_results.csv"
    demo_df.to_csv(demo_path, index=False, encoding="utf-8")

    no_filter_correct = demo_df[
        demo_df["top1_no_filter_category"] == demo_df["rag_category"]
    ].shape[0]

    with_filter_correct = demo_df[
        demo_df.apply(
            lambda r: r["rag_category"] is not None and r["top1_with_filter_category"] == r["rag_category"],
            axis=1
        )
    ].shape[0]

    summary = {
        "total_test_cases": len(demo_results),
        "top1_category_correct_no_filter": no_filter_correct,
        "top1_category_correct_with_filter": with_filter_correct,
        "demo_path": str(demo_path),
    }

    summary_path = report_dir / "demo_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[OK] Saved demo results: {demo_path}")
    print(f"[OK] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()