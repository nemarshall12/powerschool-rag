import os
import sys
import json
import time
import hashlib
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from readability import Document

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

# ----------------- Configuration -----------------
# Base settings for scraping:
BASE_URL = "https://ps.powerschool-docs.com"
START_URL = f"{BASE_URL}/?l=en"
MAX_DEPTH = 100
USER_AGENT = "Mozilla/5.0 (compatible; ProductGradeScraper/1.0)"
REQUEST_DELAY = 0.3

# Output files:
DOCS_FILE = "powerschool_docs_product.json"
INDEX_FILE = "powerschool_vector_index.faiss"
METADATA_FILE = "chunk_metadata.json"

# Chunking parameters:
CHUNK_MAX_WORDS = 200
CHUNK_OVERLAP = 50

# FAISS and Embedding settings:
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

# OpenAI settings (for answer generation via API):
OPENAI_MODEL = "gpt-3.5-turbo"   # You can change this if needed.
TEMPERATURE = 0.7
MAX_TOKENS = 512

# ----------------- Scraping and Document Processing -----------------
visited_urls = set()
scraped_documents = []  # list of document dicts

def is_valid_url(url):
    """Only follow URLs within our domain and skip common binary file extensions."""
    parsed = urlparse(url)
    if parsed.netloc and parsed.netloc != urlparse(BASE_URL).netloc:
        return False
    if any(url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".pdf", ".doc", ".xls", ".zip"]):
        return False
    return True

def scrape_page(url):
    """Fetch a page and use readability to extract its main content."""
    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch {url} (status code: {response.status_code})")
            return None
        html = response.text
        doc = Document(html)
        title = doc.short_title()
        # Use readability to extract a summary (HTML) then clean it with BeautifulSoup:
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split())  # collapse whitespace
        # Only consider pages with substantive content:
        if len(text) < 100:
            return None
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return {"url": url, "title": title, "content": text, "hash": content_hash}
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def crawl(url, depth=0):
    """Recursively crawl pages starting at 'url' up to MAX_DEPTH."""
    if depth > MAX_DEPTH or url in visited_urls:
        return
    visited_urls.add(url)
    print(f"Scraping {url} (depth {depth})")
    doc = scrape_page(url)
    if doc:
        scraped_documents.append(doc)
    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a", href=True):
            next_url = urljoin(url, link["href"]).split("#")[0]
            if is_valid_url(next_url) and next_url not in visited_urls:
                time.sleep(REQUEST_DELAY)
                crawl(next_url, depth+1)
    except Exception as e:
        print(f"Error crawling {url}: {e}")

def chunk_text(text, max_words=CHUNK_MAX_WORDS, overlap=CHUNK_OVERLAP):
    """Split text into chunks using a sliding window."""
    words = text.split()
    chunks = []
    start = 0
    chunk_index = 0
    while start < len(words):
        end = start + max_words
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)
        chunks.append({
            "chunk_index": chunk_index,
            "text": chunk_text_str,
            "start": start,
            "end": min(end, len(words))
        })
        chunk_index += 1
        start = end - overlap  # slide window with overlap
    return chunks

def process_documents(docs):
    """Add chunked text to each document."""
    processed = []
    for doc in docs:
        doc["chunks"] = chunk_text(doc["content"])
        processed.append(doc)
    return processed

def run_scraping():
    """Run the crawler starting from the START_URL and save output to DOCS_FILE."""
    crawl(START_URL)
    print(f"Scraped {len(scraped_documents)} documents.")
    processed_docs = process_documents(scraped_documents)
    with open(DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_docs, f, indent=2, ensure_ascii=False)
    print(f"Saved processed documents to {DOCS_FILE}.")

# ----------------- FAISS Index Building -----------------
def build_faiss_index(docs_file=DOCS_FILE, index_file=INDEX_FILE, metadata_file=METADATA_FILE, embedding_model_name=EMBEDDING_MODEL_NAME):
    """Load processed documents, compute embeddings for each chunk, and build a FAISS index."""
    with open(docs_file, "r", encoding="utf-8") as f:
        docs = json.load(f)
    
    chunk_texts = []
    metadata_list = []
    for doc in docs:
        url = doc.get("url", "")
        title = doc.get("title", "")
        for chunk in doc.get("chunks", []):
            chunk_texts.append(chunk.get("text", ""))
            metadata_list.append({
                "url": url,
                "title": title,
                "chunk_index": chunk.get("chunk_index", 0)
            })
    
    print(f"Total chunks to index: {len(chunk_texts)}")
    
    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(chunk_texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, index_file)
    print(f"Saved FAISS index to {index_file}.")
    
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
    print(f"Saved chunk metadata to {metadata_file}.")

# ----------------- Query Interface and Answer Generation -----------------
def load_index_and_metadata(index_file=INDEX_FILE, metadata_file=METADATA_FILE):
    try:
        idx = faiss.read_index(index_file)
    except Exception as e:
        sys.exit(f"Error loading FAISS index: {e}")
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        sys.exit(f"Error loading metadata: {e}")
    return idx, meta

def build_chunk_mapping(docs_file=DOCS_FILE):
    """Construct a mapping from (url_chunk_index) to chunk text using the processed documents."""
    with open(docs_file, "r", encoding="utf-8") as f:
        docs = json.load(f)
    mapping = {}
    for doc in docs:
        url = doc.get("url", "")
        for chunk in doc.get("chunks", []):
            key = f"{url}_{chunk.get('chunk_index', 0)}"
            mapping[key] = chunk.get("text", "")
    return mapping

def retrieve_chunks(query, top_k=TOP_K):
    """Embed the query, search the FAISS index, and return the indices of matching chunks."""
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    query_embedding = np.array([query_embedding]).astype("float32")
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    return indices[0]

# ----------------- OpenAI Client and Answer Generation -----------------
# Use the new OpenAI client interface (openai>=1.0.0)
from openai import OpenAI
api_key = os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY"
client = OpenAI(api_key=api_key)

def generate_answer(query, context):
    prompt = (
        "You are a knowledgeable assistant for PowerSchool documentation. "
        "Below is some context retrieved from the documentation, followed by a question. "
        "Please think step-by-step and provide a clear, accurate answer.\n\n"
        "Context:\n--------------------\n"
        f"{context}\n--------------------\n\n"
        f"Question: {query}\n\n"
        "Answer (explain your reasoning step by step):"
    )
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,  # e.g., "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about PowerSchool documentation."},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
    except Exception as e:
        sys.exit(f"OpenAI API error: {e}\nEnsure you've updated your code to use the new API interface.")
    return response.choices[0].message.content.strip()

def answer_query(query):
    """Retrieve relevant chunks, build context, and generate an answer."""
    indices = retrieve_chunks(query)
    print("Retrieved document chunks:")
    for idx in indices:
        if idx < len(metadata_list):
            m = metadata_list[idx]
            print(f" - {m.get('title', 'No Title')} (Chunk {m.get('chunk_index', 0)} from {m.get('url', '')})")
    # Build context using the chunk mapping:
    mapping = build_chunk_mapping()
    context_parts = []
    for idx in indices:
        if idx < len(metadata_list):
            m = metadata_list[idx]
            key = f"{m.get('url','')}_{m.get('chunk_index',0)}"
            text = mapping.get(key, "")
            if text:
                header = f"Title: {m.get('title','No Title')}\nSource: {m.get('url','')}\nChunk: {m.get('chunk_index',0)}\n"
                context_parts.append(header + text)
    context = "\n\n".join(context_parts)
    return generate_answer(query, context)

def query_interface():
    print("=== PowerSchool RAG Application ===")
    print("Type your query (or type 'exit' to quit):")
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Exiting.")
            break
        answer = answer_query(query)
        print("\n--- Generated Answer ---")
        print(answer)
        print("------------------------")

# ----------------- Main Program -----------------
if __name__ == "__main__":
    # Use command-line arguments to choose mode.
    # "--scrape": run scraping from scratch.
    # "--index": build FAISS index from DOCS_FILE.
    # Otherwise, run query interface.
    if len(sys.argv) > 1:
        if sys.argv[1] == "--scrape":
            print("Starting scraping...")
            run_scraping()
        elif sys.argv[1] == "--index":
            print("Building FAISS index from documents...")
            build_faiss_index()
        else:
            print("Unknown argument. Use '--scrape' to scrape, '--index' to build index, or no argument to query.")
    else:
        # Before querying, load the FAISS index, metadata, and embedding model.
        print("Loading FAISS index and metadata for query interface...")
        index, metadata_list = load_index_and_metadata()
        print("Loading embedding model for query interface...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        query_interface()
