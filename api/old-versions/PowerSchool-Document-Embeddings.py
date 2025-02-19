import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# --- Configuration ---
PRODUCT_JSON_FILE = "powerschool_docs_product.json"
FAISS_INDEX_FILE = "powerschool_vector_index.faiss"
METADATA_FILE = "chunk_metadata.json"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Lightweight, free, and effective model

# --- Step 1: Load and Prepare Data ---
# Load the product-grade documents (with chunks) from JSON.
with open(PRODUCT_JSON_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)

# Prepare lists for texts and metadata.
chunk_texts = []
metadata = []  # List to store mapping info for each chunk.

for doc in documents:
    doc_url = doc.get("url", "")
    doc_title = doc.get("title", "")
    # Each document should have a list of chunks.
    for chunk in doc.get("chunks", []):
        text = chunk.get("text", "")
        chunk_index = chunk.get("chunk_index", 0)
        chunk_texts.append(text)
        metadata.append({
            "url": doc_url,
            "title": doc_title,
            "chunk_index": chunk_index,
        })

print(f"Total chunks to embed: {len(chunk_texts)}")

# --- Step 2: Compute Embeddings ---
# Initialize the SentenceTransformer model.
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
# Compute embeddings for all chunks. The model returns a list that we convert to a NumPy array.
embeddings = model.encode(chunk_texts, show_progress_bar=True, convert_to_numpy=True)

# Convert embeddings to float32 type if necessary (FAISS requires float32 arrays)
embeddings = np.array(embeddings).astype("float32")

# Optional: Normalize embeddings if you plan to use cosine similarity.
faiss.normalize_L2(embeddings)

print("Embeddings computed.")

# --- Step 3: Create and Populate the FAISS Index ---
# Determine the dimensionality of the embeddings.
dimension = embeddings.shape[1]
# Use an IndexFlatIP which performs inner-product search. With normalized vectors, this is equivalent to cosine similarity.
index = faiss.IndexFlatIP(dimension)
# Add all embeddings to the index.
index.add(embeddings)
print(f"FAISS index created with {index.ntotal} vectors.")

# --- Step 4: Save the Index and Metadata Mapping ---
# Save the FAISS index to disk.
faiss.write_index(index, FAISS_INDEX_FILE)
print(f"FAISS index saved to '{FAISS_INDEX_FILE}'.")

# Save the metadata mapping to a JSON file.
with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"Chunk metadata saved to '{METADATA_FILE}'.")
