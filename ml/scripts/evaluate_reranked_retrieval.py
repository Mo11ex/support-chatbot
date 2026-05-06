from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.evaluation.evaluation import precision_at_k, recall_at_k, mrr  # noqa: E402
from ml.src.utils.tracking import start_run, log_artifact, log_metrics, log_params  # noqa: E402


CONFIG_PATH = Path("ml/configs/stage4_reranker.yaml")


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def encode_queries(model, queries, prefix: str, batch_size: int):
    prepared = [f"{prefix} {q}" for q in queries]
    return model.encode(
        prepared,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")


def dense_rank_docs(index, metadata_df, query_emb: np.ndarray, top_n_chunks: int) -> pd.DataFrame:
    scores, indices = index.search(query_emb.reshape(1, -1), top_n_chunks)
    scores = scores[0]
    indices = indices[0]

    valid_mask = indices >= 0
    scores = scores[valid_mask]
    indices = indices[valid_mask]

    tmp = metadata_df.iloc[indices].copy().reset_index(drop=True)
    tmp["dense_chunk_score"] = scores

    # Оставляем лучший чанк на каждый doc_id
    tmp = tmp.sort_values("dense_chunk_score", ascending=False)
    best_chunks = tmp.groupby("doc_id", as_index=False).first()

    best_chunks = best_chunks.sort_values("dense_chunk_score", ascending=False).reset_index(drop=True)
    return best_chunks


def score_reranker_pairs(model, tokenizer, pairs, batch_size: int, max_length: int, device: str):
    scores = []

    model.eval()

    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start:start + batch_size]

        queries = [p[0] for p in batch_pairs]
        passages = [p[1] for p in batch_pairs]

        inputs = tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            # Универсальная обработка:
            # если shape [B, 1] -> берём logits[:, 0]
            # если shape [B, 2] -> берём "positive" колонку logits[:, -1]
            if logits.ndim == 2 and logits.shape[1] == 1:
                batch_scores = logits[:, 0]
            elif logits.ndim == 2:
                batch_scores = logits[:, -1]
            else:
                batch_scores = logits.view(-1)

        scores.extend(batch_scores.detach().cpu().numpy().tolist())

    return scores


def main():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")

    cfg = load_yaml(CONFIG_PATH)

    eval_path = Path(cfg["eval_path"])
    index_dir = Path(cfg["index_dir"])
    report_dir = Path(cfg["output"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    dense_model_name = cfg["dense"]["model_name"]
    query_prefix = cfg["dense"]["query_prefix"]
    dense_batch_size = int(cfg["dense"]["batch_size"])

    top_n_chunks_dense = int(cfg["retrieval"]["top_n_chunks_dense"])
    top_n_docs_rerank = int(cfg["retrieval"]["top_n_docs_rerank"])
    top_k_final = int(cfg["retrieval"]["top_k_final"])

    reranker_model_name = cfg["reranker"]["model_name"]
    reranker_batch_size = int(cfg["reranker"]["batch_size"])
    reranker_max_length = int(cfg["reranker"]["max_length"])

    faiss_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.csv"

    for p in [eval_path, faiss_path, metadata_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    eval_df = pd.read_csv(eval_path)
    metadata_df = pd.read_csv(metadata_path)
    index = faiss.read_index(str(faiss_path))

    required_eval_cols = {"query", "relevant_doc_id"}
    required_meta_cols = {"faiss_row_id", "doc_id", "chunk_id", "title", "text"}
    if required_eval_cols - set(eval_df.columns):
        raise ValueError(f"Missing eval columns: {required_eval_cols - set(eval_df.columns)}")
    if required_meta_cols - set(metadata_df.columns):
        raise ValueError(f"Missing metadata columns: {required_meta_cols - set(metadata_df.columns)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Dense model: {dense_model_name}")
    print(f"[INFO] Reranker model: {reranker_model_name}")
    print(f"[INFO] Device: {device}")

    dense_model = SentenceTransformer(dense_model_name, device=device)
    reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
    reranker_model.to(device)

    query_embeddings = encode_queries(
        model=dense_model,
        queries=eval_df["query"].tolist(),
        prefix=query_prefix,
        batch_size=dense_batch_size,
    )

    prediction_rows = []
    all_relevant = []
    all_predicted = []

    for i, row in eval_df.iterrows():
        query = row["query"]
        relevant_doc_id = row["relevant_doc_id"]

        q_emb = query_embeddings[i]

        best_chunks = dense_rank_docs(
            index=index,
            metadata_df=metadata_df,
            query_emb=q_emb,
            top_n_chunks=top_n_chunks_dense,
        )

        candidates = best_chunks.head(top_n_docs_rerank).copy()

        pairs = [(query, text) for text in candidates["text"].tolist()]
        rerank_scores = score_reranker_pairs(
            model=reranker_model,
            tokenizer=reranker_tokenizer,
            pairs=pairs,
            batch_size=reranker_batch_size,
            max_length=reranker_max_length,
            device=device,
        )

        candidates["rerank_score"] = rerank_scores
        candidates = candidates.sort_values("rerank_score", ascending=False).reset_index(drop=True)

        top_docs = candidates["doc_id"].tolist()[:top_k_final]
        top_scores = candidates["rerank_score"].tolist()[:top_k_final]
        top_chunks = candidates["chunk_id"].tolist()[:top_k_final]

        all_relevant.append([relevant_doc_id])
        all_predicted.append(top_docs)

        prediction_rows.append({
            "query": query,
            "relevant_doc_id": relevant_doc_id,
            "top1_doc_id": top_docs[0] if len(top_docs) > 0 else None,
            "top1_rerank_score": top_scores[0] if len(top_scores) > 0 else None,
            "top1_chunk_id": top_chunks[0] if len(top_chunks) > 0 else None,
            "top5_doc_ids": json.dumps(top_docs, ensure_ascii=False),
            "top5_rerank_scores": json.dumps([float(x) for x in top_scores], ensure_ascii=False),
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

    pred_path = report_dir / "reranked_predictions.csv"
    metrics_path = report_dir / "reranked_metrics.json"

    pred_df.to_csv(pred_path, index=False, encoding="utf-8")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dense_model_name": dense_model_name,
                "reranker_model_name": reranker_model_name,
                "device": device,
                "top_n_chunks_dense": top_n_chunks_dense,
                "top_n_docs_rerank": top_n_docs_rerank,
                "top_k_final": top_k_final,
                **metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with start_run(experiment_name="stage4-reranked-retrieval", run_name="dense-top10-bge-reranker"):
        log_params({
            "dense_model_name": dense_model_name,
            "reranker_model_name": reranker_model_name,
            "device": device,
            "top_n_chunks_dense": top_n_chunks_dense,
            "top_n_docs_rerank": top_n_docs_rerank,
            "top_k_final": top_k_final,
        })
        log_metrics(metrics)
        log_artifact(str(pred_path))
        log_artifact(str(metrics_path))

    print("\n[INFO] Reranked retrieval metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\n[OK] Saved predictions: {pred_path}")
    print(f"[OK] Saved metrics: {metrics_path}")

    print("\n[INFO] Top 20 preview:")
    print(
        pred_df[
            ["query", "relevant_doc_id", "top1_doc_id", "hit_at_1", "hit_at_5"]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()