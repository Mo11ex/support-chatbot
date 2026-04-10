import os
import hashlib
from pathlib import Path
import asyncio

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text

from app.services.embedder_service import EmbedderService

from dotenv import load_dotenv
load_dotenv()

CHUNK_SIZE = 1000
OVERLAP = 120

def chunk_text(s: str) -> list[str]:
    s = s.strip()
    chunks = []
    i = 0
    while i < len(s):
        chunks.append(s[i:i+CHUNK_SIZE])
        i += CHUNK_SIZE - OVERLAP
    return [c.strip() for c in chunks if len(c.strip()) > 80]

def vec_to_pgvector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

async def main():
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise RuntimeError("DATABASE_URL is not set (not found in env/.env)")
        
    kb_dir = Path("data/knowledge_base")
    engine = create_async_engine(db_url, echo=False)

    embedder = EmbedderService("intfloat/multilingual-e5-small")

    async with engine.begin() as conn:
        # для MVP проще пересоздавать индекс
        await conn.execute(text("DELETE FROM kb_chunks;"))
        await conn.execute(text("DELETE FROM kb_documents;"))

    async with AsyncSession(engine) as session:
        for f in kb_dir.glob("*.md"):
            content = f.read_text(encoding="utf-8")
            h = hashlib.sha256(content.encode("utf-8")).hexdigest()
            title = f.stem

            doc_row = (await session.execute(text("""
                INSERT INTO kb_documents(title, source_file, content_hash, chunk_count, is_active)
                VALUES (:title, :source_file, :hash, 0, TRUE)
                RETURNING id;
            """), {"title": title, "source_file": f.name, "hash": h})).first()
            doc_id = doc_row[0]

            chunks = chunk_text(content)

            for idx, ch in enumerate(chunks):
                vec = embedder.embed_passage(ch)
                vec_literal = vec_to_pgvector_literal(vec)

                await session.execute(text("""
                    INSERT INTO kb_chunks(document_id, chunk_index, content, embedding, token_count, char_count)
                    VALUES (:doc, :idx, :content, (:emb)::vector, :tok, :chars)
                """), {
                    "doc": doc_id,
                    "idx": idx,
                    "content": ch,
                    "emb": vec_literal,
                    "tok": max(1, len(ch.split())),
                    "chars": len(ch),
                })

            await session.execute(text("""
                UPDATE kb_documents SET chunk_count = :cnt WHERE id = :id
            """), {"cnt": len(chunks), "id": doc_id})

        await session.commit()

    print("Knowledge base indexed OK")

if __name__ == "__main__":
    asyncio.run(main())