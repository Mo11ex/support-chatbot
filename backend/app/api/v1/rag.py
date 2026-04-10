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
        sources.append(RagSource(
            chunk_id=int(r["chunk_id"]),
            document=r["document"],
            similarity=sim,
            text=(r["text"] or "")[:900],
        ))

    is_confident = (len(sources) > 0) and (max_sim >= RAG_THRESHOLD)

    if not is_confident:
        answer = None
        sources_out = []
    else:
        # Template-MVP: отдаём top-1 chunk как "grounded answer"
        answer = sources[0].text.strip()
        answer = answer[:500]  # NFR UX лимит
        sources_out = sources[:2]  # можно 1-2 источника для демонстрации

    ms = int((time.perf_counter() - t0) * 1000)

    return RagQueryResponse(
        answer=answer,
        sources=sources_out,
        max_similarity=max_sim,
        is_confident=is_confident,
        processing_time_ms=ms,
    )