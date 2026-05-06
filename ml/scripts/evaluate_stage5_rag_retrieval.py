from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.evaluation.evaluation import precision_at_k, recall_at_k, mrr  # noqa: E402
from ml.src.utils.tracking import start_run, log_params, log_metrics, log_artifact  # noqa: E402


RETRIEVAL_CONFIG = Path("ml/configs/stage5_rag_retrieval.yaml")
EVAL_PATH = Path("ml/data/kb_stage5/chunks/stage5_rag_eval.csv")


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def encode_query(model, query: str, prefix: str) -> np.ndarray:
    text = f"{prefix} {query}"
    emb = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    return emb[0]


def search_dense(index, metadata_df, q_emb, top_n: int):
    scores, indices = index.search(q_emb.reshape(1, -1), top_n)
    scores = scores[0]
    indices = indices[0]

    valid_mask = indices >= 0
    scores = scores[valid_mask]
    indices = indices[valid_mask]

    results = metadata_df.iloc[indices].copy().reset_index(drop=True)
    results["dense_score"] = scores
    results = results.sort_values("dense_score", ascending=False).reset_index(drop=True)
    return results


def filter_by_category(df: pd.DataFrame, category: str | None):
    if category is None:
        return df
    return df[df["category"] == category].copy().reset_index(drop=True)


def aggregate_to_source_file(results_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        results_df.groupby(["source_file", "category"], as_index=False)["dense_score"]
        .max()
        .sort_values("dense_score", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def evaluate_predictions(pred_df: pd.DataFrame, top_k: int = 5):
    all_relevant = pred_df["relevant_source_file"].apply(lambda x: [x]).tolist()
    all_predicted = pred_df["topk_source_files"].apply(json.loads).tolist()

    metrics = {
        "precision_at_1": float(precision_at_k(all_relevant, all_predicted, k=1)),
        "recall_at_1": float(recall_at_k(all_relevant, all_predicted, k=1)),
        "recall_at_5": float(recall_at_k(all_relevant, all_predicted, k=top_k)),
        "mrr_at_5": float(mrr(all_relevant, all_predicted, k=top_k)),
    }
    return metrics


def build_mode_predictions(eval_df, index, metadata_df, model, query_prefix, top_n_search, use_filter: bool):
    rows = []

    for _, row in eval_df.iterrows():
        query = row["query"]
        classifier_label = row["classifier_label"]
        expected_category = row["expected_category"]
        relevant_source_file = row["relevant_source_file"]

        q_emb = encode_query(model, query, query_prefix)
        retrieved = search_dense(index, metadata_df, q_emb, top_n_search)

        if use_filter:
            retrieved = filter_by_category(retrieved, expected_category)

        aggregated = aggregate_to_source_file(retrieved)

        topk_source_files = aggregated["source_file"].tolist()[:5]
        topk_categories = aggregated["category"].tolist()[:5]
        topk_scores = aggregated["dense_score"].tolist()[:5]

        top1_source_file = topk_source_files[0] if len(topk_source_files) > 0 else None
        top1_category = topk_categories[0] if len(topk_categories) > 0 else None
        top1_score = float(topk_scores[0]) if len(topk_scores) > 0 else None

        rows.append({
            "query": query,
            "classifier_label": classifier_label,
            "expected_category": expected_category,
            "relevant_source_file": relevant_source_file,
            "top1_source_file": top1_source_file,
            "top1_category": top1_category,
            "top1_score": top1_score,
            "topk_source_files": json.dumps(topk_source_files, ensure_ascii=False),
            "topk_categories": json.dumps(topk_categories, ensure_ascii=False),
            "topk_scores": json.dumps([float(x) for x in topk_scores], ensure_ascii=False),
            "top1_source_hit": int(top1_source_file == relevant_source_file),
            "top1_category_hit": int(top1_category == expected_category),
            "hit_at_5": int(relevant_source_file in topk_source_files),
            "mode": "with_filter" if use_filter else "no_filter",
        })

    return pd.DataFrame(rows)


def main():
    cfg = load_yaml(RETRIEVAL_CONFIG)

    index_dir = Path(cfg["index_dir"])
    dense_model_name = cfg["dense"]["model_name"]
    query_prefix = cfg["dense"]["query_prefix"]
    top_k_no_filter = int(cfg["retrieval"]["top_k_no_filter"])
    top_n_search = int(cfg["retrieval"]["top_n_search"])
    report_dir = Path(cfg["output"]["report_dir"])

    # Сохраним stage5 eval в отдельную подпапку
    output_dir = Path("ml/logs/reports/stage5_rag_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.csv"

    for p in [faiss_path, metadata_path, EVAL_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    metadata_df = pd.read_csv(metadata_path)
    eval_df = pd.read_csv(EVAL_PATH)
    index = faiss.read_index(str(faiss_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Dense model: {dense_model_name}")
    print(f"[INFO] Device: {device}")

    model = SentenceTransformer(dense_model_name, device=device)

    pred_no_filter = build_mode_predictions(
        eval_df=eval_df,
        index=index,
        metadata_df=metadata_df,
        model=model,
        query_prefix=query_prefix,
        top_n_search=top_n_search,
        use_filter=False,
    )

    pred_with_filter = build_mode_predictions(
        eval_df=eval_df,
        index=index,
        metadata_df=metadata_df,
        model=model,
        query_prefix=query_prefix,
        top_n_search=top_n_search,
        use_filter=True,
    )

    metrics_no_filter = evaluate_predictions(pred_no_filter, top_k=5)
    metrics_with_filter = evaluate_predictions(pred_with_filter, top_k=5)

    metrics_no_filter["top1_category_hit_rate"] = float(pred_no_filter["top1_category_hit"].mean())
    metrics_with_filter["top1_category_hit_rate"] = float(pred_with_filter["top1_category_hit"].mean())

    no_filter_path = output_dir / "rag_eval_no_filter.csv"
    with_filter_path = output_dir / "rag_eval_with_filter.csv"
    summary_path = output_dir / "rag_eval_summary.json"

    pred_no_filter.to_csv(no_filter_path, index=False, encoding="utf-8")
    pred_with_filter.to_csv(with_filter_path, index=False, encoding="utf-8")

    summary = {
        "no_filter": metrics_no_filter,
        "with_filter": metrics_with_filter,
        "num_eval_queries": int(len(eval_df)),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with start_run(experiment_name="stage5-rag-eval", run_name="dense-rag-with-category-filter"):
        log_params({
            "dense_model_name": dense_model_name,
            "device": device,
            "top_n_search": top_n_search,
            "top_k_eval": 5,
            "eval_path": str(EVAL_PATH),
            "index_dir": str(index_dir),
        })
        log_metrics({f"no_filter_{k}": v for k, v in metrics_no_filter.items()})
        log_metrics({f"with_filter_{k}": v for k, v in metrics_with_filter.items()})
        log_artifact(str(no_filter_path))
        log_artifact(str(with_filter_path))
        log_artifact(str(summary_path))

    print("\n[INFO] No-filter metrics:")
    print(json.dumps(metrics_no_filter, ensure_ascii=False, indent=2))

    print("\n[INFO] With-filter metrics:")
    print(json.dumps(metrics_with_filter, ensure_ascii=False, indent=2))

    print(f"\n[OK] Saved no-filter predictions: {no_filter_path}")
    print(f"[OK] Saved with-filter predictions: {with_filter_path}")
    print(f"[OK] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()