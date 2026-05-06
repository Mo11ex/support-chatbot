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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.evaluation.evaluation import precision_at_k, recall_at_k, mrr  # noqa: E402
from ml.src.utils.tracking import start_run, log_params, log_metrics, log_artifact  # noqa: E402


CONFIG_PATH = Path("ml/configs/stage4_hybrid_index.yaml")
INDEX_DIR = Path("ml/models/retriever/stage4_hybrid_index")
EVAL_PATH = Path("ml/data/processed/faq_eval.csv")

REPORT_DIR = Path("ml/logs/reports/stage4_retrieval_ablation")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 5
TOP_N_CHUNKS_DENSE = 30
TOP_N_CHUNKS_BM25 = 30
RRF_K = 60


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).replace("\xa0", " ").replace("\t", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_for_bm25(text: str, lowercase: bool = True) -> list[str]:
    text = normalize_text(text)
    if lowercase:
        text = text.lower()
    return re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text)


def encode_queries(model, queries, prefix: str, batch_size: int = 32):
    prepared = [f"{prefix} {q}" for q in queries]
    return model.encode(
        prepared,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")


def aggregate_doc_ranking_from_chunk_scores(
    metadata_df: pd.DataFrame,
    chunk_scores: np.ndarray,
    top_n_chunks: int,
) -> pd.DataFrame:
    tmp = metadata_df.copy()
    tmp["score"] = chunk_scores
    tmp = tmp.sort_values("score", ascending=False).head(top_n_chunks)

    # агрегируем до уровня документа через max score чанка
    grouped = (
        tmp.groupby("doc_id", as_index=False)["score"]
        .max()
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def dense_rank_docs(index, metadata_df, query_emb: np.ndarray, top_n_chunks: int) -> pd.DataFrame:
    scores, indices = index.search(query_emb.reshape(1, -1), top_n_chunks)
    scores = scores[0]
    indices = indices[0]

    valid_mask = indices >= 0
    scores = scores[valid_mask]
    indices = indices[valid_mask]

    tmp = metadata_df.iloc[indices].copy().reset_index(drop=True)
    tmp["score"] = scores

    grouped = (
        tmp.groupby("doc_id", as_index=False)["score"]
        .max()
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def bm25_rank_docs(bm25, metadata_df, query_text: str, top_n_chunks: int, lowercase: bool = True) -> pd.DataFrame:
    tokens = tokenize_for_bm25(query_text, lowercase=lowercase)
    scores = np.array(bm25.get_scores(tokens), dtype=float)

    return aggregate_doc_ranking_from_chunk_scores(
        metadata_df=metadata_df,
        chunk_scores=scores,
        top_n_chunks=top_n_chunks,
    )


def rrf_fuse(rankings: dict[str, list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores = {}

    for _, ranked_doc_ids in rankings.items():
        for rank, doc_id in enumerate(ranked_doc_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


def evaluate_mode(prediction_rows: list[dict], mode_name: str):
    pred_df = pd.DataFrame(prediction_rows)

    all_relevant = pred_df["relevant_doc_id"].apply(lambda x: [x]).tolist()
    all_predicted = pred_df["top5_doc_ids"].apply(json.loads).tolist()

    metrics = {
        "precision_at_1": float(precision_at_k(all_relevant, all_predicted, k=1)),
        "recall_at_1": float(recall_at_k(all_relevant, all_predicted, k=1)),
        "recall_at_5": float(recall_at_k(all_relevant, all_predicted, k=5)),
        "mrr_at_5": float(mrr(all_relevant, all_predicted, k=5)),
    }

    pred_path = REPORT_DIR / f"{mode_name}_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")

    return metrics, pred_path, pred_df


def main():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")

    cfg = load_yaml(CONFIG_PATH)

    dense_model_name = cfg["dense"]["model_name"]
    batch_size = int(cfg["dense"]["batch_size"])
    query_prefix = "query:"
    bm25_lowercase = bool(cfg["bm25"]["lowercase"])

    faiss_path = INDEX_DIR / "faiss.index"
    metadata_path = INDEX_DIR / "metadata.csv"
    bm25_path = INDEX_DIR / "bm25.pkl"

    for p in [faiss_path, metadata_path, bm25_path, EVAL_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    metadata_df = pd.read_csv(metadata_path)
    eval_df = pd.read_csv(EVAL_PATH)

    index = faiss.read_index(str(faiss_path))
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading dense model: {dense_model_name}")
    print(f"[INFO] Device: {device}")

    model = SentenceTransformer(dense_model_name, device=device)

    query_embeddings = encode_queries(
        model=model,
        queries=eval_df["query"].tolist(),
        prefix=query_prefix,
        batch_size=batch_size,
    )

    dense_rows = []
    bm25_rows = []
    hybrid_rows = []

    for i, row in eval_df.iterrows():
        query = row["query"]
        relevant_doc_id = row["relevant_doc_id"]

        q_emb = query_embeddings[i]

        dense_ranked = dense_rank_docs(
            index=index,
            metadata_df=metadata_df,
            query_emb=q_emb,
            top_n_chunks=TOP_N_CHUNKS_DENSE,
        )
        bm25_ranked = bm25_rank_docs(
            bm25=bm25,
            metadata_df=metadata_df,
            query_text=query,
            top_n_chunks=TOP_N_CHUNKS_BM25,
            lowercase=bm25_lowercase,
        )

        dense_top = dense_ranked["doc_id"].tolist()
        bm25_top = bm25_ranked["doc_id"].tolist()

        fused = rrf_fuse(
            {
                "dense": dense_top[:20],
                "bm25": bm25_top[:20],
            },
            k=RRF_K,
        )
        hybrid_top = [doc_id for doc_id, _ in fused]

        dense_scores = dense_ranked["score"].tolist()
        dense_rows.append({
            "query": query,
            "relevant_doc_id": relevant_doc_id,
            "top1_doc_id": dense_top[0] if len(dense_top) > 0 else None,
            "top1_score": float(dense_scores[0]) if len(dense_scores) > 0 else None,
            "top5_doc_ids": json.dumps(dense_top[:TOP_K], ensure_ascii=False),
            "top5_scores": json.dumps([float(s) for s in dense_scores[:TOP_K]], ensure_ascii=False),
            "hit_at_1": int(len(dense_top) > 0 and dense_top[0] == relevant_doc_id),
            "hit_at_5": int(relevant_doc_id in dense_top[:TOP_K]),
        })

        bm25_rows.append({
            "query": query,
            "relevant_doc_id": relevant_doc_id,
            "top1_doc_id": bm25_top[0] if len(bm25_top) > 0 else None,
            "top5_doc_ids": json.dumps(bm25_top[:TOP_K], ensure_ascii=False),
            "hit_at_1": int(len(bm25_top) > 0 and bm25_top[0] == relevant_doc_id),
            "hit_at_5": int(relevant_doc_id in bm25_top[:TOP_K]),
        })

        hybrid_rows.append({
            "query": query,
            "relevant_doc_id": relevant_doc_id,
            "top1_doc_id": hybrid_top[0] if len(hybrid_top) > 0 else None,
            "top5_doc_ids": json.dumps(hybrid_top[:TOP_K], ensure_ascii=False),
            "hit_at_1": int(len(hybrid_top) > 0 and hybrid_top[0] == relevant_doc_id),
            "hit_at_5": int(relevant_doc_id in hybrid_top[:TOP_K]),
        })

    dense_metrics, dense_pred_path, dense_pred_df = evaluate_mode(dense_rows, "dense")
    bm25_metrics, bm25_pred_path, bm25_pred_df = evaluate_mode(bm25_rows, "bm25")
    hybrid_metrics, hybrid_pred_path, hybrid_pred_df = evaluate_mode(hybrid_rows, "hybrid_rrf")

    summary = {
        "dense": dense_metrics,
        "bm25": bm25_metrics,
        "hybrid_rrf": hybrid_metrics,
        "config": {
            "dense_model_name": dense_model_name,
            "device": device,
            "top_k": TOP_K,
            "top_n_chunks_dense": TOP_N_CHUNKS_DENSE,
            "top_n_chunks_bm25": TOP_N_CHUNKS_BM25,
            "rrf_k": RRF_K,
        }
    }

    summary_path = REPORT_DIR / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with start_run(experiment_name="stage4-retrieval-ablation", run_name="dense-bm25-hybrid"):
        log_params({
            "dense_model_name": dense_model_name,
            "device": device,
            "top_k": TOP_K,
            "top_n_chunks_dense": TOP_N_CHUNKS_DENSE,
            "top_n_chunks_bm25": TOP_N_CHUNKS_BM25,
            "rrf_k": RRF_K,
        })

        log_metrics({f"dense_{k}": v for k, v in dense_metrics.items()})
        log_metrics({f"bm25_{k}": v for k, v in bm25_metrics.items()})
        log_metrics({f"hybrid_{k}": v for k, v in hybrid_metrics.items()})

        for p in [summary_path, dense_pred_path, bm25_pred_path, hybrid_pred_path]:
            log_artifact(str(p))

    print("\n[INFO] Dense metrics:")
    print(json.dumps(dense_metrics, ensure_ascii=False, indent=2))

    print("\n[INFO] BM25 metrics:")
    print(json.dumps(bm25_metrics, ensure_ascii=False, indent=2))

    print("\n[INFO] Hybrid RRF metrics:")
    print(json.dumps(hybrid_metrics, ensure_ascii=False, indent=2))

    print(f"\n[OK] Saved summary: {summary_path}")
    print(f"[OK] Saved dense predictions: {dense_pred_path}")
    print(f"[OK] Saved bm25 predictions: {bm25_pred_path}")
    print(f"[OK] Saved hybrid predictions: {hybrid_pred_path}")

    print("\n[INFO] Hybrid preview:")
    print(
        hybrid_pred_df[
            ["query", "relevant_doc_id", "top1_doc_id", "hit_at_1", "hit_at_5"]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()