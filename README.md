# Semantic Search System with Fuzzy Clustering and Semantic Cache

## Project Overview

This project implements a semantic search system using the **20 Newsgroups Dataset**.

The system uses sentence embeddings to understand the meaning of user queries and retrieve relevant documents based on semantic similarity.

It combines three main components:

1. **Vector Embeddings:** Convert text documents and user queries into numerical vectors.
2. **Fuzzy Clustering:** Use Gaussian Mixture Model (GMM) to identify the dominant cluster of a query.
3. **Semantic Cache:** Store previous search results and reuse them when a semantically similar query is received.

The application provides REST API endpoints using FastAPI.

**Note:** This project demonstrates semantic search using the 20 Newsgroups dataset. It does not connect to a live trademark registration database.

---

## Dataset

The project uses the **20 Newsgroups Dataset**, which contains approximately 20,000 documents belonging to 20 different topics.

The dataset is loaded using Scikit-learn.

The following preprocessing steps are applied:

- Headers are removed.
- Footers are removed.
- Quoted text is removed.
- Empty documents are filtered out.

This helps the system focus on the actual content of the messages.

---

## Embedding Model

The project uses SentenceTransformers with the following pre-trained model:

`all-MiniLM-L6-v2`

### Why this model?

- Lightweight and efficient.
- Generates meaningful sentence embeddings.
- Supports semantic similarity comparison.
- Suitable for document and query representation.

The model converts documents and user queries into normalized numerical vectors.

These vectors allow the system to compare text based on meaning rather than relying only on exact keyword matches.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| FastAPI | Building REST API endpoints |
| Uvicorn | Running the API server |
| SentenceTransformers | Generating text embeddings |
| Scikit-learn | Dataset loading and Gaussian Mixture Model |
| NumPy | Vector operations and cosine similarity |
| Pandas | Data processing support |
| SciPy | Scientific computing support |

---

## System Architecture

The system follows the pipeline below:

```text
             Application Startup
                     |
                     v
             Load Dataset
                     |
                     v
          Generate Document Embeddings
                     |
                     v
           Train Gaussian Mixture Model
                     |
                     v
                API Ready
                     |
                     v
                User Query
                     |
                     v
            Generate Query Embedding
                     |
                     v
             Semantic Cache Check
                     |
              +------+------+
              |             |
              v             v
          Cache Hit     Cache Miss
              |             |
              v             v
        Return Cached   Find Dominant
           Results         Cluster
                            |
                            v
                  Calculate Similarity
                   with All Documents
                            |
                            v
                    Retrieve Top-K
                    Matching Documents
                            |
                            v
                    Store Results
                      in Cache
                            |
                            v
                       API Response
```

### Architecture Explanation

**1. Dataset Loading**

The system loads the 20 Newsgroups dataset and removes unnecessary text such as headers, footers, quoted messages, and empty documents.

**2. Document Embedding Generation**

The SentenceTransformer model converts all loaded documents into normalized numerical vectors.

**3. Fuzzy Clustering**

A Gaussian Mixture Model with 10 components is trained using a sample of up to 5,000 document embeddings to reduce clustering overhead.

**4. Query Embedding**

When a user submits a query, the same embedding model converts it into a numerical vector.

**5. Semantic Cache Lookup**

The system compares the query embedding with previously stored query embeddings using cosine similarity.

If the similarity reaches the configured threshold of **0.85**, the system returns the cached search results.

**6. Document Retrieval**

If there is no suitable cache match, the system calculates cosine similarity between the query embedding and all document embeddings.

The documents with the highest similarity scores are selected as the search results.

**7. Cache Storage**

The query, its embedding, retrieved documents, dominant cluster, and requested result count are stored in the semantic cache.

**8. API Response**

The API returns the matching documents, their similarity scores, cache information, and the dominant cluster.

---

## Fuzzy Clustering

The project uses a **Gaussian Mixture Model (GMM)** for fuzzy clustering.

Traditional hard clustering assigns each data point to one cluster.

GMM provides probability distributions that represent the membership of a data point across different clusters.

For example:

```text
Document:
Cluster 1 = 0.30
Cluster 2 = 0.50
Cluster 3 = 0.20
```

This means the document has different membership probabilities across the clusters.

### Implementation

- Number of clusters: 10
- Algorithm: Gaussian Mixture Model
- Covariance type: Diagonal
- Training sample: Up to 5,000 document embeddings
- Random state: 42

The model identifies the dominant cluster for each user query based on its highest membership probability.

The clustering model is used to identify the query's dominant cluster. Document retrieval is performed separately by comparing the query against all document embeddings.

---

## Semantic Cache

The project implements a semantic cache to avoid repeating document retrieval for similar queries.

Traditional caching generally depends on exact query matching.

This project compares query embeddings using cosine similarity, allowing semantically similar queries to reuse previously stored results.

### How it works

1. A user submits a search query.
2. The query is converted into an embedding.
3. The embedding is compared with previously cached query embeddings.
4. If similarity is at least **0.85**, the cached result is returned.
5. Otherwise, the system performs document retrieval and stores the new result.

### Cache Features

- Cosine similarity-based matching.
- Configurable similarity threshold.
- Stores query embeddings and search results.
- Tracks cache hits and misses.
- Provides cache statistics.
- Supports clearing the cache.

The cache is maintained in application memory and is reset when the application restarts.

---

## API Endpoints

The API is developed using FastAPI.

### 1. GET /

Checks whether the API application is running.

Example response:

```json
{
  "message": "Semantic Search API is running",
  "docs": "/docs"
}
```

### 2. POST /query

Accepts a user query and retrieves the most relevant documents.

**Request body:**

```json
{
  "query": "space exploration and satellites",
  "top_k": 5
}
```

**Parameters:**

- `query`: Text entered by the user.
- `top_k`: Maximum number of matching documents to return. Default value is 5. Allowed range is 1 to 10.

**Example response:**

```json
{
  "query": "space exploration and satellites",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": 0.0,
  "result": [
    {
      "document_id": 125,
      "similarity_score": 0.82,
      "text": "Example matching document text..."
    }
  ],
  "dominant_cluster": 3
}
```

The above response is illustrative. Actual document IDs, similarity scores, document text, and cluster numbers depend on the query and dataset.

**Response fields:**

- `query`: The submitted query.
- `cache_hit`: Indicates whether the result was retrieved from the cache.
- `matched_query`: Previously cached query when a cache hit occurs.
- `similarity_score`: Similarity with the best cached query. On a cache miss, it represents the best cache similarity found, or 0 if there are no cached entries.
- `result`: Matching documents with their IDs, similarity scores, and text.
- `dominant_cluster`: Cluster with the highest membership probability for the query.

### 3. GET /cache/stats

Returns the current semantic cache statistics.

**Example response:**

```json
{
  "total_entries": 10,
  "hit_count": 5,
  "miss_count": 10,
  "hit_rate": 0.3333333333333333
}
```

### 4. DELETE /cache

Clears all stored cache entries and resets the cache statistics.

**Example response:**

```json
{
  "message": "Cache cleared successfully"
}
```

## Running the Project

### Prerequisites

- Python installed on your system.
- Internet connection for downloading the dataset and embedding model during the first execution.
- VS Code or another Python-compatible editor.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Paduchuru-pranathi/semantic-search-trademarkia.git
```

### Step 2: Open the Project Folder

```bash
cd semantic-search-trademarkia
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Start the FastAPI Server

```bash
uvicorn main:app --reload
```

During the first execution, the system downloads the dataset and embedding model, generates document embeddings, and trains the clustering model. This initialization can take several minutes.

### Step 7: Open API Documentation

Open the following URL in your browser:

```text
http://127.0.0.1:8000/docs
```

Use the interactive Swagger UI to test the API endpoints.

---

## Testing the API

1. Open the API documentation.
2. Select `POST /query`.
3. Click **Try it out**.
4. Enter a search query and the required `top_k` value.
5. Click **Execute**.
6. Check the matching documents and similarity scores in the response.

To test semantic caching, submit the same query or a semantically similar query again and check whether `cache_hit` becomes `true`.

You can also test the cache statistics using `GET /cache/stats` and clear the cache using `DELETE /cache`.

---

## Key Design Decisions

- **SentenceTransformers:** Selected to generate meaningful text embeddings.
- **Gaussian Mixture Model:** Selected because it provides probability-based cluster membership.
- **Cosine Similarity:** Used to compare query and document embeddings.
- **Semantic Cache:** Implemented in Python to reuse results for similar queries without requiring an external cache service.
- **FastAPI:** Used to provide simple and interactive REST API endpoints.

---

## Future Improvements

The project can be extended with the following features:

- Integration with a vector database such as FAISS.
- Approximate nearest-neighbor search for faster retrieval.
- Integration with an actual trademark dataset.
- Improved cache management and persistence.
- Adaptive similarity thresholds based on query patterns.
- Deployment using Docker.
- Cloud deployment for public API access.

---

## Conclusion

This project demonstrates how semantic embeddings, Gaussian Mixture Model clustering, cosine similarity, and semantic caching can be combined to build a document search application.

It retrieves documents based on semantic similarity rather than exact keyword matching and uses caching to reuse results for similar queries.
