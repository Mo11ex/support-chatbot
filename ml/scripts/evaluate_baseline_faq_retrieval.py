from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.evaluation.evaluation import precision_at_k, recall_at_k, mrr  # noqa: E402
from ml.src.utils.tracking import start_run, log_params, log_metrics, log_artifact  # noqa: E402


CORPUS_PATH = Path("ml/data/processed/faq_corpus.csv")
EVAL_PATH = Path("ml/data/processed/faq_eval.csv")

REPORT_DIR = Path("ml/logs/reports/stage2_baseline_retrieval")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_MODEL_PATH = Path("backend/app/ml/rag/e5-small")
FALLBACK_MODEL_NAME = "intfloat/multilingual-e5-small"

TOP_K = 5
BATCH_SIZE = 64


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if LOCAL_MODEL_PATH.exists():
        model_name = str(LOCAL_MODEL_PATH)
    else:
        model_name = FALLBACK_MODEL_NAME

    print(f"[INFO] Loading embedding model: {model_name}")
    print(f"[INFO] Device: {device}")

    model = SentenceTransformer(model_name, device=device)
    return model, model_name, device


def encode_texts(model, texts, prefix: str):
    prepared = [f"{prefix} {str(t)}" for t in texts]
    emb = model.encode(
        prepared,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb


def aggregate_doc_scores(corpus_df: pd.DataFrame, chunk_scores: np.ndarray):
    """
    Агрегируем chunk-level score в doc-level ranking через max score по doc_id.
    """
    tmp = corpus_df[["doc_id", "chunk_id", "title", "text"]].copy()
    tmp["score"] = chunk_scores

    grouped = (
        tmp.sort_values("score", ascending=False)
        .groupby("doc_id", as_index=False)
        .first()
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def main():
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus file not found: {CORPUS_PATH}")
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Eval file not found: {EVAL_PATH}")

    corpus_df = pd.read_csv(CORPUS_PATH)
    eval_df = pd.read_csv(EVAL_PATH)

    required_corpus_cols = {"doc_id", "chunk_id", "title", "text"}
    required_eval_cols = {"query", "relevant_doc_id"}

    if required_corpus_cols - set(corpus_df.columns):
        raise ValueError(f"Missing corpus columns: {required_corpus_cols - set(corpus_df.columns)}")
    if required_eval_cols - set(eval_df.columns):
        raise ValueError(f"Missing eval columns: {required_eval_cols - set(eval_df.columns)}")

    model, model_name, device = load_model()

    print(f"[INFO] Corpus shape: {corpus_df.shape}")
    print(f"[INFO] Eval shape: {eval_df.shape}")

    corpus_embeddings = encode_texts(model, corpus_df["text"].tolist(), prefix="passage:")
    query_embeddings = encode_texts(model, eval_df["query"].tolist(), prefix="query:")

    print(f"[INFO] corpus_embeddings shape: {corpus_embeddings.shape}")
    print(f"[INFO] query_embeddings shape: {query_embeddings.shape}")

    all_relevant = []
    all_predicted = []
    prediction_rows = []

    for idx, row in eval_df.iterrows():
        query = row["query"]
        relevant_doc_id = row["relevant_doc_id"]

        q_emb = query_embeddings[idx]
        chunk_scores = np.matmul(corpus_embeddings, q_emb)

        doc_ranked = aggregate_doc_scores(corpus_df, chunk_scores)

        top_docs = doc_ranked["doc_id"].tolist()[:TOP_K]
        top_scores = doc_ranked["score"].tolist()[:TOP_K]
        top_chunks = doc_ranked["chunk_id"].tolist()[:TOP_K]

        all_relevant.append([relevant_doc_id])
        all_predicted.append(top_docs)

        prediction_rows.append({
            "query": query,
            "relevant_doc_id": relevant_doc_id,
            "top1_doc_id": top_docs[0] if len(top_docs) > 0 else None,
            "top1_score": top_scores[0] if len(top_scores) > 0 else None,
            "top1_chunk_id": top_chunks[0] if len(top_chunks) > 0 else None,
            "top5_doc_ids": json.dumps(top_docs, ensure_ascii=False),
            "top5_scores": json.dumps([float(x) for x in top_scores], ensure_ascii=False),
            "top5_chunk_ids": json.dumps(top_chunks, ensure_ascii=False),
            "hit_at_1": int(len(top_docs) > 0 and top_docs[0] == relevant_doc_id),
            "hit_at_5": int(relevant_doc_id in top_docs),
        })

    pred_df = pd.DataFrame(prediction_rows)

    metrics = {
        "precision_at_1": float(precision_at_k(all_relevant, all_predicted, k=1)),
        "recall_at_1": float(recall_at_k(all_relevant, all_predicted, k=1)),
        "recall_at_5": float(recall_at_k(all_relevant, all_predicted, k=5)),
        "mrr_at_5": float(mrr(all_relevant, all_predicted, k=5)),
    }

    metrics_path = REPORT_DIR / "faq_retrieval_metrics.json"
    pred_path = REPORT_DIR / "faq_retrieval_predictions.csv"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "device": device,
                "corpus_shape": list(corpus_df.shape),
                "eval_shape": list(eval_df.shape),
                "top_k": TOP_K,
                **metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    pred_df.to_csv(pred_path, index=False, encoding="utf-8")

    with start_run(experiment_name="stage2-baseline-retrieval", run_name="e5-small-faq-baseline"):
        log_params({
            "model_name": model_name,
            "device": device,
            "corpus_path": str(CORPUS_PATH),
            "eval_path": str(EVAL_PATH),
            "top_k": TOP_K,
            "query_prefix": "query:",
            "passage_prefix": "passage:",
        })
        log_metrics(metrics)
        log_artifact(str(metrics_path))
        log_artifact(str(pred_path))

    print("\n[INFO] Retrieval metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\n[OK] Saved metrics JSON: {metrics_path}")
    print(f"[OK] Saved predictions CSV: {pred_path}")

    print("\n[INFO] Top 15 predictions preview:")
    print(
        pred_df[
            ["query", "relevant_doc_id", "top1_doc_id", "top1_score", "hit_at_1", "hit_at_5"]
        ].head(15).to_string(index=False)
    )


if __name__ == "__main__":
    main()