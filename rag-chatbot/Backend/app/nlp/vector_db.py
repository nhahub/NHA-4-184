import chromadb
from typing import List, Dict


class VectorDB:
    """Handles ChromaDB operations for storing and searching chunks."""

    def __init__(self, db_path: str = "data/vector_db", collection_name: str = "qa_chunks"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(self, ids: List[str], documents: List[str],
                   embeddings: List[List[float]], metadatas: List[Dict],
                   batch_size: int = 500):
        """Add chunks to the vector database in batches."""
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end]
            )

    def search(self, query_text: str, n_results: int = 3) -> Dict:
        """Search for similar chunks given a query."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results

    def count(self) -> int:
        """Return total number of chunks stored."""
        return self.collection.count()