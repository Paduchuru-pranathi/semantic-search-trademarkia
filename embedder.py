from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(texts):

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    return embeddings


def embed_query(query):

    embedding = model.encode([query])[0]

    return embedding