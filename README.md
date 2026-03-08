# semantic-search-trademarkia
# Semantic Search System with Fuzzy Clustering and Semantic Cache

This project implements a lightweight semantic search system using the
20 Newsgroups dataset.

The system demonstrates three key components:

1. Vector embeddings for semantic understanding
2. Fuzzy clustering to uncover latent topic structure
3. Semantic caching to avoid redundant query computation

---

## Dataset

20 Newsgroups Dataset (~20,000 documents across 20 topics)

Noise such as headers, footers, and quotes were removed to focus on
the semantic content of the messages.

---

## Embedding Model

We use SentenceTransformer model:

all-MiniLM-L6-v2

Reasons:

- Lightweight
- Fast inference
- Strong semantic similarity performance

Embeddings allow semantic comparison between queries and documents.

---

## Fuzzy Clustering

Instead of hard clustering, we use Gaussian Mixture Models.

This allows each document to belong to multiple clusters with
different probabilities.

Example:

Document → [Cluster1:0.3, Cluster2:0.5, Cluster3:0.2]

This better represents real semantic overlap between topics.

---

## Semantic Cache

Traditional caches fail when users ask the same question differently.

This system implements a semantic cache using:

- Query embeddings
- Cosine similarity
- Threshold-based matching

If similarity > threshold (0.85), the cached result is returned.

This avoids redundant computation.

---

## FastAPI Endpoints

POST /query

Input:

{
 "query": "space exploration satellites"
}

Output:

{
 "query": "...",
 "cache_hit": false,
 "matched_query": null,
 "similarity_score": 0.21,
 "result": "...",
 "dominant_cluster": 3
}

---

GET /cache/stats

Returns cache statistics.

---

DELETE /cache

Clears the cache.

---

## Running the Project

Install dependencies:

pip install -r requirements.txt

Run server:

uvicorn api.main:app --reload

Open API docs:

http://127.0.0.1:8000/docs

---

## Key Design Decisions

Embedding model chosen for speed and semantic quality.

Gaussian Mixture Model chosen because it supports soft clustering.

Semantic cache implemented from scratch without Redis as required.

---

## Future Improvements

Vector database integration
Approximate nearest neighbor search
Adaptive similarity thresholds
Distributed caching

---
