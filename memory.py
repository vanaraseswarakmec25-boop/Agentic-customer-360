import os
from datetime import datetime
import chromadb

# 1. Initialize persistent ChromaDB client
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

# 2. Get or create collections
collection = chroma_client.get_or_create_collection(
    name="customer_episodic_memory", metadata={"hnsw:space": "cosine"}
)
semantic_collection = chroma_client.get_or_create_collection(
    name="customer_semantic_memory", metadata={"hnsw:space": "cosine"}
)


# 3. Base function to insert episodic memories
def add_episodic_memory(
    customer_id: str,
    event_type: str,
    summary: str,
    category: str = "general",
    metadata: dict = None,
):
    meta = metadata.copy() if metadata else {}
    meta.update({
        "customer_id": str(customer_id),
        "event_type": event_type,
        "category": category,
        "timestamp": str(datetime.now().isoformat()),
    })
    doc_id = f"mem_{customer_id}_{datetime.now().timestamp()}"

    collection.add(documents=[summary], metadatas=[meta], ids=[doc_id])


# 4. Base function to query episodic memories
def query_episodic_memory(
    customer_id: str,
    query_text: str,
    category: str = "general",
    similarity_threshold: float = 0.78,
):
    max_distance = 1.0 - similarity_threshold

    results = collection.query(
        query_texts=[query_text],
        where={"customer_id": str(customer_id)},
        n_results=20,
    )

    memories = []
    if results and results.get("documents") and results["documents"][0]:
        docs = results["documents"][0]
        distances = (
            results["distances"][0] if results.get("distances") else []
        )
        metas = (
            results["metadatas"][0] if results.get("metadatas") else []
        )

        for doc, meta, dist in zip(docs, metas, distances):
            computed_similarity = 1.0 - dist
            if dist <= max_distance:
                memories.append({
                    "summary": doc,
                    "metadata": meta,
                    "similarity": round(computed_similarity, 4),
                })

    return memories


# 5. Base function to insert semantic memories
def add_semantic_memory(
    customer_id: str,
    fact: str,
    category: str = "preference",
    metadata: dict = None,
):
    doc_id = f"sem_{customer_id}_{category}_{datetime.now().timestamp()}"
    meta = {"customer_id": str(customer_id), "category": category}
    if metadata:
        meta.update(metadata)

    semantic_collection.upsert(
        ids=[doc_id],
        documents=[fact],
        metadatas=[meta],
    )


# 6. Retrieve ALL semantic facts belonging to customer directly
def query_semantic_memory(customer_id: str):
    results = semantic_collection.get(
        where={"customer_id": str(customer_id)}
    )

    facts = []
    if results and results.get("documents"):
        docs = results["documents"]
        metas = results.get("metadatas", [])

        for doc, meta in zip(docs, metas):
            facts.append({
                "fact": doc,
                "metadata": meta,
            })

    return facts


# 7. Context Retriever (Called by main.py)
async def retrieve_customer_context(
    customer_id: str, query_text: str, category: str = "general"
):
    episodic_memories = query_episodic_memory(
        customer_id=customer_id,
        query_text=query_text,
        category=category,
        similarity_threshold=0.78,
    )

    # Direct fetch without vector distance filtering
    semantic_facts = query_semantic_memory(customer_id=customer_id)

    return {
        "episodic": episodic_memories,
        "semantic": semantic_facts,
    }


# 8. Async Wrapper: Log pipeline outcome
async def log_pipeline_outcome(
    customer_id: str, event_type: str, output_summary: str, category: str
):
    add_episodic_memory(
        customer_id=customer_id,
        event_type=event_type,
        summary=output_summary,
        category=category,
    )