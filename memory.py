import chromadb
from datetime import datetime

# Initialize persistent ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="customer_episodic_memory")

# 1. Base function to insert memories into ChromaDB
def add_episodic_memory(customer_id: str, event_type: str, summary: str, category: str = "general", metadata: dict = None):
    meta = metadata.copy() if metadata else {}
    meta.update({
        "customer_id": customer_id,
        "event_type": event_type,
        "category": category,
        "timestamp": str(datetime.now().isoformat())
    })
    doc_id = f"mem_{customer_id}_{datetime.now().timestamp()}"
    
    collection.add(
        documents=[summary],
        metadatas=[meta],
        ids=[doc_id]
    )

# 2. Base function to query memories with similarity filtering
def query_episodic_memory(customer_id: str, query_text: str, category: str = "general", similarity_threshold: float = 0.78):
    results = collection.query(
        query_texts=[query_text],
        where={"customer_id": customer_id},
        n_results=5
    )
    
    memories = []
    if results and results.get("documents"):
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for doc, meta, dist in zip(docs, metas, distances):
            similarity = 1.0 - dist if dist <= 1.0 else 1.0 / (1.0 + dist)
            if similarity >= similarity_threshold:
                memories.append({
                    "summary": doc,
                    "metadata": meta,
                    "similarity": round(similarity, 4)
                })
                
    return memories

# 3. Async Wrapper: Retrieve context
async def retrieve_customer_context(customer_id: str, current_issue: str, category: str):
    return query_episodic_memory(
        customer_id=customer_id,
        query_text=current_issue,
        category=category,
        similarity_threshold=0.78
    )

# 4. Async Wrapper: Log pipeline outcome
async def log_pipeline_outcome(customer_id: str, event_type: str, output_summary: str, category: str):
    add_episodic_memory(
        customer_id=customer_id,
        event_type=event_type,
        summary=output_summary,
        category=category
    )