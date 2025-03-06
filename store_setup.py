from uuid import uuid4
from langgraph.store.memory import InMemoryStore

def get_semantic_store():
    # Here you could configure an embedding for semantic search
    # For demo, we set up a basic store without advanced embeddings
    store = InMemoryStore()
    return store
