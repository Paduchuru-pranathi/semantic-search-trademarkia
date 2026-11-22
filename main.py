
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from embedder import embed_query, embed_text
from dataset_loader import load_dataset
from fuzzy_cluster import FuzzyCluster
from semantic_cache import SemanticCache

app = FastAPI()

print("Loading dataset...")

texts = load_dataset()

print("Creating embeddings...")

embeddings = embed_text(texts)

print("Training clustering model...")

cluster_model = FuzzyCluster(n_clusters=10)
cluster_model.fit(embeddings)

cache = SemanticCache(threshold=0.85)


class Query(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


@app.get("/")
def home():
    return {
        "message": "Semantic Search API is running",
        "docs": "/docs"
    }


@app.post("/query")
def query(q: Query):

    search_text = q.query.strip()

    if not search_text:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    query_embedding = embed_query(search_text)

    hit, item, sim = cache.search(query_embedding, q.top_k)
    
    if hit:
        return {
            "query": search_text,
            "cache_hit": True,
            "matched_query": item["query"],
            "similarity_score": float(sim),
            "result": item["result"],
            "dominant_cluster": item["cluster"]
        }

    cluster = cluster_model.dominant_cluster(query_embedding)

    # Calculate cosine similarity with all document embeddings.
    # Embeddings are normalized, so dot product gives cosine similarity.
    scores = embeddings @ query_embedding

    # Get the indexes of the top matching documents.
    top_indices = scores.argsort()[-q.top_k:][::-1]

    results = []

    for index in top_indices:
        results.append({
            "document_id": int(index),
            "similarity_score": float(scores[index]),
            "text": texts[index][:1000]
        })

    cache.add(
        search_text,
        query_embedding,
        results,
        int(cluster),
        q.top_k
    )

    return {
        "query": search_text,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": float(sim),
        "result": results,
        "dominant_cluster": int(cluster)
    }


@app.get("/cache/stats")
def cache_stats():
    return cache.stats()


@app.delete("/cache")
def clear_cache():
    cache.clear()

    return {
        "message": "Cache cleared successfully"
    }
