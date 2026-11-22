
import numpy as np


class SemanticCache:

    def __init__(self, threshold=0.85):

        self.cache = []
        self.threshold = threshold

        self.hit_count = 0
        self.miss_count = 0

    def search(self, embedding, top_k=5):

        best_similarity = 0.0
        best_match = None

        query_vector = np.asarray(
            embedding,
            dtype=np.float32
        )

        for item in self.cache:
            if item.get("top_k", 5) < top_k:
                continue
            cached_vector = np.asarray(
                item["embedding"],
                dtype=np.float32
            )

            denominator = (
                np.linalg.norm(query_vector)
                * np.linalg.norm(cached_vector)
            )

            if denominator == 0:
                similarity = 0.0
            else:
                similarity = float(
                    np.dot(query_vector, cached_vector)
                    / denominator
                )

            if similarity > best_similarity:

                best_similarity = similarity
                best_match = item

        if (
            best_match is not None
            and best_similarity >= self.threshold
        ):

            self.hit_count += 1

            return True, best_match, best_similarity

        self.miss_count += 1

        return False, None, best_similarity

    def add(self, query, embedding, result, cluster, top_k=5):
        self.cache.append({

            "query": query,

            "embedding": np.asarray(
                embedding,
                dtype=np.float32
            ).copy(),

            "result": result,

            "cluster": cluster
            "top_k": top_k
            
        })

    def stats(self):

        total = self.hit_count + self.miss_count

        hit_rate = (
            self.hit_count / total
            if total > 0
            else 0.0
        )

        return {

            "total_entries": len(self.cache),

            "hit_count": self.hit_count,

            "miss_count": self.miss_count,

            "hit_rate": hit_rate

        }

    def clear(self):

        self.cache.clear()

        self.hit_count = 0

        self.miss_count = 0
