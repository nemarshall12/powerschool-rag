import requests
from bs4 import BeautifulSoup
import time
import json
import re
import hashlib
from urllib.parse import urljoin, urlparse
from datetime import datetime

# --- Configuration ---
BASE_URL = "https://ps.powerschool-docs.com"
START_URL = f"{BASE_URL}/?l=en"
USER_AGENT = "Mozilla/5.0 (compatible; ProductGradeScraper/1.0; +http://yourdomain.com/bot)"
CRAWL_MAX_DEPTH = 2       # Adjust based on your needs
REQUEST_DELAY = 0.5       # Seconds between requests for politeness
CHUNK_MAX_WORDS = 200     # Maximum number of words per chunk

# Global sets and lists for tracking
visited_urls = set()
documents = []  # List to hold enriched document data

# --- Helper Functions ---

def is_valid_url(url):
    """Check that URL is within our target domain and not a binary file."""
    parsed = urlparse(url)
    if parsed.netloc and parsed.netloc != urlparse(BASE_URL).netloc:
        return False
    if any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.pdf', '.doc', '.xls', '.xlsx']):
        return False
    return True

def clean_html(soup):
    """
    Remove unwanted elements such as scripts, styles, navs, headers, and footers.
    This helps focus on the primary content.
    """
    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()
    return soup

def extract_main_content(soup):
    """
    Try to extract the main content using common containers.
    Adjust the selectors as needed for the target site's structure.
    """
    # First attempt: look for a div with id "main-content"
    content_container = soup.find("div", {"id": "main-content"})
    if content_container is None:
        # Fallback to an article tag
        content_container = soup.find("article")
    if content_container is None:
        # Fallback to the body tag
        content_container = soup.body
    return content_container.get_text(separator="\n", strip=True) if content_container else ""

def collapse_whitespace(text):
    """Replace multiple whitespace characters with a single space."""
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, max_words=CHUNK_MAX_WORDS):
    """
    Split the cleaned text into smaller chunks.
    This implementation splits by paragraphs (newlines) and then groups paragraphs until the chunk
    reaches the maximum word count.
    """
    paragraphs = [para.strip() for para in text.split("\n") if para.strip()]
    chunks = []
    current_chunk = ""
    current_word_count = 0
    chunk_index = 0

    for para in paragraphs:
        para_word_count = len(para.split())
        # If adding the paragraph exceeds max_words and there is already content in the current chunk, store it
        if current_word_count + para_word_count > max_words and current_chunk:
            chunks.append({
                "chunk_index": chunk_index,
                "text": current_chunk.strip()
            })
            chunk_index += 1
            current_chunk = para + "\n"
            current_word_count = para_word_count
        else:
            current_chunk += para + "\n"
            current_word_count += para_word_count
    if current_chunk:
        chunks.append({
            "chunk_index": chunk_index,
            "text": current_chunk.strip()
        })
    return chunks

def compute_hash(text):
    """Compute a SHA256 hash of the text for versioning and change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def scrape_page(url):
    """Fetch the page, clean it, extract content, chunk the text, and add metadata."""
    headers = {"User-Agent": USER_AGENT}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Failed to retrieve {url}: Status {response.status_code}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        soup = clean_html(soup)

        # Extract title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else "No Title"

        # Extract and clean main content
        full_text = extract_main_content(soup)
        cleaned_text = collapse_whitespace(full_text)

        # Chunk the cleaned text into manageable pieces
        chunks = chunk_text(cleaned_text, max_words=CHUNK_MAX_WORDS)

        # Compute a hash for versioning
        doc_hash = compute_hash(cleaned_text)

        # Build the document structure with metadata
        document = {
            "url": url,
            "title": title,
            "scraped_at": datetime.utcnow().isoformat() + "Z",
            "hash": doc_hash,
            "chunks": chunks,
            "full_text": cleaned_text  # Optional: store full text if needed for debugging or re-processing
        }
        return document
    except Exception as e:
        print(f"Exception while scraping {url}: {e}")
        return None

def crawl(url, max_depth=CRAWL_MAX_DEPTH, current_depth=0):
    """
    Recursively crawl the site starting from the provided URL.
    Respects the maximum depth and avoids revisiting URLs.
    """
    if current_depth > max_depth or url in visited_urls:
        return

    visited_urls.add(url)
    print(f"Scraping {url} at depth {current_depth}")
    doc = scrape_page(url)
    if doc:
        documents.append(doc)

    try:
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        for link in soup.find_all("a", href=True):
            next_url = urljoin(url, link['href'])
            # Remove URL fragments for consistency
            next_url = next_url.split("#")[0]
            if is_valid_url(next_url) and next_url not in visited_urls:
                crawl(next_url, max_depth=max_depth, current_depth=current_depth + 1)
                time.sleep(REQUEST_DELAY)
    except Exception as e:
        print(f"Error crawling links from {url}: {e}")

def main():
    # Start the crawl from the designated URL
    crawl(START_URL)
    # Save the enriched, product-grade document data to a JSON file
    output_file = "powerschool_docs_product.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    print(f"Scraping complete. {len(documents)} documents saved to '{output_file}'.")

if __name__ == "__main__":
    main()
