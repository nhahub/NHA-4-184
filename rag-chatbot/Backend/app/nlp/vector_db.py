import chromadb
from typing import List, Dict
import logging
import time

logger = logging.getLogger(__name__)


class VectorDB:
    """Handles ChromaDB operations for storing and searching chunks."""

    def __init__(self, db_path: str = "../data/vector_db", collection_name: str = "qa_chunks"):
        self.db_path = db_path
        self.collection_name = collection_name
        logger.info(f"Initializing ChromaDB: db_path={db_path}, collection={collection_name}")
        try:
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(name=collection_name)
            collection_count = self.collection.count()
            logger.info(f"ChromaDB initialized successfully: collection={collection_name}, doc_count={collection_count}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}, db_path={db_path}", exc_info=True)
            raise

    def add_chunks(self, ids: List[str], documents: List[str],
                   embeddings: List[List[float]], metadatas: List[Dict],
                   batch_size: int = 500):
        """Add chunks to the vector database in batches."""
        logger.info(f"Adding chunks to ChromaDB: count={len(documents)}, collection={self.collection_name}, batch_size={batch_size}")
        start_time = time.time()
        try:
            for i in range(0, len(ids), batch_size):
                end = min(i + batch_size, len(ids))
                self.collection.add(
                    ids=ids[i:end],
                    documents=documents[i:end],
                    embeddings=embeddings[i:end],
                    metadatas=metadatas[i:end]
                )
                logger.debug(f"Batch added to ChromaDB: {i}-{end}/{len(ids)}")
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.info(f"Successfully added {len(documents)} chunks to ChromaDB: duration_ms={duration_ms}")
        except Exception as e:
            logger.error(f"Failed to add chunks to ChromaDB: {str(e)}", exc_info=True)
            raise

    def search(self, query_text: str, n_results: int = 3) -> Dict:
        """Search for similar chunks given a query."""
        logger.debug(f"ChromaDB search: query_len={len(query_text)}, n_results={n_results}")
        start_time = time.time()
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            duration_ms = round((time.time() - start_time) * 1000, 2)
            results_found = len(results.get("ids", [[]])[0]) if results.get("ids") else 0
            logger.info(f"ChromaDB search completed: results_found={results_found}, duration_ms={duration_ms}")
            return results
        except Exception as e:
            logger.error(f"ChromaDB search failed: {str(e)}", exc_info=True)
            raise

    def count(self) -> int:
        """Return total number of chunks stored."""
        start_time = time.time()
        try:
            count = self.collection.count()
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.info(f"ChromaDB collection stats: doc_count={count}, collection={self.collection_name}, duration_ms={duration_ms}")
            return count
        except Exception as e:
            logger.error(f"ChromaDB count failed: {str(e)}", exc_info=True)
            raise