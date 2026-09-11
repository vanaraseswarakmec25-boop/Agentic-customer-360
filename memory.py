from memory import add_episodic_memory, query_episodic_memory

# Inside your pipeline orchestration or run_parallel_swarm function:
async def retrieve_customer_context(customer_id: str, current_issue: str, category: str):
    # Retrieve past memories meeting similarity >= 0.78 and scoped to this customer
    relevant_memories = query_episodic_memory(
        customer_id=customer_id,
        query_text=current_issue,
        category=category,
        similarity_threshold=0.78
    )
    return relevant_memories

async def log_pipeline_outcome(customer_id: str, event_type: str, output_summary: str, category: str):
    # Save the processed outcome into ChromaDB for future context
    add_episodic_memory(
        customer_id=customer_id,
        event_type=event_type,
        summary=output_summary,
        category=category
    )