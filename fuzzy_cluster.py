from sklearn.mixture import GaussianMixture
import numpy as np

class FuzzyCluster:

    def __init__(self, n_clusters=10):

        self.model = GaussianMixture(
            n_components=n_clusters
        )

    def fit(self, embeddings):

        self.model.fit(embeddings)

    def membership(self, embeddings):

        probabilities = self.model.predict_proba(embeddings)

        return probabilities

    def dominant_cluster(self, embedding):

        probs = self.model.predict_proba([embedding])[0]

        cluster = np.argmax(probs)

        return cluster