import os
import time
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.db.database import get_db

router = APIRouter()

RAG_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.70"))

class RagQueryRequest(BaseModel):
    query: str
    top_k: int = 3
    session_id: str | None = None

class RagSource(BaseModel):
    chunk_id: int
    document: str
    similarity: float
    text: str

class RagQueryResponse(BaseModel):
    answer: str | None
    sources: list[RagSource]
    max_similarity: float
    is_confident: bool
    processing_time_ms: int

import re

_WORD_RE = re.compile(r"[a-zа-яё0-9]+")

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("ё", "е")).strip()

def _clean_markdown(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        if ln.lstrip().startswith("#"):
            continue
        lines.append(ln)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def _truncate_to_boundary(text: str, max_len: int = 500) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    p = max(cut.rfind("\n"), cut.rfind("."), cut.rfind("!"), cut.rfind("?"), cut.rfind("…"))
    if p < int(max_len * 0.6):
        return cut.rstrip() + "..."
    return cut[:p+1].rstrip()

def _score_paragraph(p: str, query: str) -> int:
    q = _norm(query)
    pn = _norm(p)
    score = 0

    # Базовые ключевые слова (очень лёгкая “морфология” через подстроки)
    if "срок" in q:
        # в тексте часто "в течение" / "дней", а не слово "срок"
        if "в течение" in pn or "дн" in pn:
            score += 5

    if "электрон" in q:
        if "электрон" in pn:
            score += 4

    # возврат/вернуть
    if "возврат" in q or "вернут" in q or "вернуть" in q:
        if "возврат" in pn or "вернут" in pn:
            score += 2

    # Доп. плюс за пересечение токенов длиной >=4
    qt = {t for t in _WORD_RE.findall(q) if len(t) >= 4}
    pt = set(_WORD_RE.findall(pn))
    score += sum(1 for t in qt if t in pt)

    return score

def _best_snippet(text: str, query: str, max_len: int = 500) -> str:
    text = _clean_markdown(text)
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not parts:
        return _truncate_to_boundary(text, max_len)

    parts_sorted = sorted(parts, key=lambda p: _score_paragraph(p, query), reverse=True)

    snippet = parts_sorted[0]
    # если место позволяет, добавим второй абзац (часто полезно)
    if len(parts_sorted) > 1 and len(snippet) < max_len * 0.6:
        snippet = snippet + "\n\n" + parts_sorted[1]

    return _truncate_to_boundary(snippet, max_len)

def vec_to_pgvector_literal(vec: list[float]) -> str:
    # pgvector принимает строку формата: [0.1,0.2,...]
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

RAG_SQL = text("""
SELECT
  c.id AS chunk_id,
  d.source_file AS document,
  c.content AS text,
  (1 - (c.embedding <=> (:vec)::vector)) AS similarity
FROM kb_chunks c
JOIN kb_documents d ON d.id = c.document_id
WHERE d.is_active = TRUE
ORDER BY c.embedding <=> (:vec)::vector
LIMIT :k;
""")

@router.post("/rag/query", response_model=RagQueryResponse)
async def rag_query(payload: RagQueryRequest, request: Request, db: AsyncSession = Depends(get_db)):
    t0 = time.perf_counter()
    q = payload.query.strip()
    if not q:
        return RagQueryResponse(
            answer=None, sources=[], max_similarity=0.0, is_confident=False, processing_time_ms=0
        )

    embedder = request.app.state.embedder
    query_vec = embedder.embed_query(q)
    vec_literal = vec_to_pgvector_literal(query_vec)

    k = max(1, min(payload.top_k, 10))
    result = await db.execute(RAG_SQL, {"vec": vec_literal, "k": k})
    rows = result.mappings().all()

    sources: list[RagSource] = []
    max_sim = 0.0
    for r in rows:
        sim = float(r["similarity"] or 0.0)
        max_sim = max(max_sim, sim)
        raw_txt = (r["text"] or "")
        snippet = _truncate_to_boundary(_clean_markdown(raw_txt), 900)
        sources.append(RagSource(
            chunk_id=int(r["chunk_id"]),
            document=r["document"],
            similarity=sim,
            text=snippet,
        ))

    is_confident = (len(sources) > 0) and (max_sim >= RAG_THRESHOLD)

    if not is_confident:
        answer = None
        sources_out = []
    else:
        # Template-MVP: отдаём top-1 chunk как "grounded answer"
        best_raw_text = rows[0]["text"] or ""
        answer = _best_snippet(best_raw_text, q, max_len=500)
        sources_out = sources[:2]  # можно 1-2 источника для демонстрации

    ms = int((time.perf_counter() - t0) * 1000)

    return RagQueryResponse(
        answer=answer,
        sources=sources_out,
        max_similarity=max_sim,
        is_confident=is_confident,
        processing_time_ms=ms,
    )