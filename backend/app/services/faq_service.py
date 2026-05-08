import re
import faiss
import pandas as pd
from pathlib import Path
from app.config import settings


def normalize_text(text: str) -> str:
    return str(text).replace("\xa0", " ").replace("\t", " ").strip()


def clean_faq_answer(text: str) -> str:
    text = normalize_text(text)

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.lstrip("#").strip()
        if line == "---":
            continue
        lines.append(line)

    text = " ".join(lines)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def is_too_short_answer(text: str) -> bool:
    text = clean_faq_answer(text)
    words = text.split()
    return len(words) <= 3 or len(text) < 25


def faq_text_score(text: str, query: str) -> int:
    t = clean_faq_answer(text).lower()
    q = normalize_text(query).lower()

    score = 0
    q_words = [w for w in re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", q) if len(w) >= 4]

    for w in q_words:
        if w in t:
            score += 1

    # эвристики под твои частые FAQ-кейсы
    if "москв" in q and "москв" in t:
        score += 3
    if "санкт" in q and "санкт" in t:
        score += 3
    if "питер" in q and ("санкт" in t or "петербург" in t):
        score += 3
    if "стоит" in q and ("стоим" in t or "₽" in t or "руб" in t):
        score += 2
    if "доставк" in q and "доставк" in t:
        score += 2
    if "самовывоз" in q and "самовывоз" in t:
        score += 3
    if "курьер" in q and "курьер" in t:
        score += 3
    if "оплат" in q and "оплат" in t:
        score += 2
    if "получении" in q and "получении" in t:
        score += 3
    if "рассроч" in q and "рассроч" in t:
        score += 3
    if "вернуть" in q and "вернуть" in t:
        score += 2
    if "возврат" in q and "возврат" in t:
        score += 2
    if "гарант" in q and "гарант" in t:
        score += 3
    if "промокод" in q and "промокод" in t:
        score += 3
    if "скидк" in q and "скидк" in t:
        score += 2
    if "акци" in q and "акци" in t:
        score += 2

    return score


class FaqService:
    def __init__(self, embedder_service):
        index_dir = Path(settings.faq_index_dir)
        corpus_path = Path(settings.faq_corpus_path)

        print(f"[FaqService] Loading index from: {index_dir}")
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        self.metadata = pd.read_csv(index_dir / "metadata.csv")
        self.corpus = pd.read_csv(corpus_path)
        self.embedder = embedder_service

        print(f"[FaqService] Loaded {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = 5) -> dict:
        # Ищем не только top_k, а побольше чанков, чтобы была возможность
        # выбрать лучший текст внутри одного и того же документа
        search_k = max(top_k * 4, 12)

        q_emb = self.embedder.embed_query(query).astype("float32")
        scores, indices = self.index.search(q_emb.reshape(1, -1), search_k)

        scores = scores[0]
        indices = indices[0]
        valid = indices >= 0

        if not valid.any():
            return {
                "top1_score": 0.0,
                "top1_doc_id": None,
                "top1_text": None,
                "results": [],
            }

        meta = self.metadata.iloc[indices[valid]].copy().reset_index(drop=True)
        meta["score"] = scores[valid]
        meta["text"] = meta["text"].fillna("").astype(str)

        # Документы ранжируем по лучшему чанку
        aggregated = (
            meta.sort_values("score", ascending=False)
            .groupby("doc_id", as_index=False)
            .first()
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

        top1 = aggregated.iloc[0]
        top1_doc_id = str(top1["doc_id"])
        top1_score = float(top1["score"])

        # Для top1 выбираем лучший содержательный chunk внутри документа
        same_doc_chunks = meta[meta["doc_id"] == top1_doc_id].copy().sort_values("score", ascending=False)

        candidates = []
        for _, row in same_doc_chunks.iterrows():
            candidate_text = clean_faq_answer(row["text"])
            if not candidate_text:
                continue
            candidates.append({
                "text": candidate_text,
                "score": faq_text_score(candidate_text, query),
                "dense_score": float(row["score"]),
                "is_short": is_too_short_answer(candidate_text),
            })

        if candidates:
            candidates = sorted(
                candidates,
                key=lambda x: (x["score"], x["dense_score"], -int(x["is_short"])),
                reverse=True
            )
            selected_text = candidates[0]["text"]
        else:
            selected_text = clean_faq_answer(str(same_doc_chunks.iloc[0]["text"]))

        # Формируем top-3 документов с лучшими user-facing текстами
        results = []
        for _, row in aggregated.head(3).iterrows():
            doc_id = row["doc_id"]
            doc_chunks = meta[meta["doc_id"] == doc_id].copy().sort_values("score", ascending=False)

            doc_candidates = []
            for _, c_row in doc_chunks.iterrows():
                c_text = clean_faq_answer(c_row["text"])
                if not c_text:
                    continue
                doc_candidates.append({
                    "text": c_text,
                    "score": faq_text_score(c_text, query),
                    "dense_score": float(c_row["score"]),
                    "is_short": is_too_short_answer(c_text),
                })

            if doc_candidates:
                doc_candidates = sorted(
                    doc_candidates,
                    key=lambda x: (x["score"], x["dense_score"], -int(x["is_short"])),
                    reverse=True
                )
                best_text = doc_candidates[0]["text"]
            else:
                best_text = ""

            results.append({
                "doc_id": str(doc_id),
                "score": float(row["score"]),
                "text": best_text,
            })

        return {
            "top1_score": top1_score,
            "top1_doc_id": top1_doc_id,
            "top1_text": selected_text,
            "results": results,
        }