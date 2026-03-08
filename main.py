from fastapi import FastAPI
from pydantic import BaseModel

from embeddings.embedder import embed_query, embed_text
from data.dataset_loader import load_dataset
from clustering.fuzzy_cluster import FuzzyCluster
from cache.semantic_cache import SemanticCache

app = FastAPI()

print("Loading dataset...")

texts = load_dataset()

print("Creating embeddings...")

embeddings = embed_text(texts)

cluster_model = FuzzyCluster(n_clusters=10)

cluster_model.fit(embeddings)

cache = SemanticCache()


class Query(BaseModel):

    query: str


@app.post("/query")
def query(q: Query):

    query_embedding = embed_query(q.query)

    hit, item, sim = cache.search(query_embedding)

    if hit:

        return {

            "query": q.query,
            "cache_hit": True,
            "matched_query": item["query"],
            "similarity_score": float(sim),
            "result": item["result"],
            "dominant_cluster": item["cluster"]

        }

    cluster = cluster_model.dominant_cluster(query_embedding)

    result = f"Documents related to cluster {cluster}"

    cache.add(q.query, query_embedding, result, cluster)

    return {

        "query": q.query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": float(sim),
        "result": result,
        "dominant_cluster": cluster

    }


@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "Cache cleared"}