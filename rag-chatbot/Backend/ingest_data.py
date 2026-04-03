from app.nlp.ingestion import create_qa_chunks
from app.nlp.embedder import Embedder
from app.nlp.vector_db import VectorDB


def main():
    # Step 1: Create QA chunks from cleaned data
    print("📄 Creating QA chunks...")
    chunks = create_qa_chunks("../data/processed/cleaned_data.csv")
    print(f"✅ Created {len(chunks)} chunks")

    # Step 2: Initialize embedder and vector DB
    print("🔧 Loading embedder model...")
    embedder = Embedder()
    vector_db = VectorDB(db_path="../data/vector_db", collection_name="qa_chunks")

    # Step 3: Prepare data
    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Step 4: Generate embeddings
    print("🧠 Generating embeddings (this may take a few minutes)...")
    embeddings = embedder.embed_texts(documents).tolist()

    # Step 5: Store in ChromaDB
    print("💾 Storing in ChromaDB...")
    vector_db.add_chunks(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    print(f"🎉 Done! Total chunks in DB: {vector_db.count()}")


if __name__ == "__main__":
    main()
