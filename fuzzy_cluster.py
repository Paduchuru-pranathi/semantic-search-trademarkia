
from sklearn.mixture import GaussianMixture
import numpy as np


class FuzzyCluster:

    def __init__(self, n_clusters=10):

        self.model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="diag",
            random_state=42,
            reg_covar=1e-5,
            max_iter=100
        )

        self.is_fitted = False

    def fit(self, embeddings):

        # Use a sample to reduce clustering time.
        if len(embeddings) > 5000:

            rng = np.random.default_rng(42)

            indices = rng.choice(
                len(embeddings),
                size=5000,
                replace=False
            )

            training_data = embeddings[indices]

        else:
            training_data = embeddings

        self.model.fit(training_data)

        self.is_fitted = True

    def membership(self, embeddings):

        if not self.is_fitted:
            raise RuntimeError("Model is not trained.")

        probabilities = self.model.predict_proba(embeddings)

        return probabilities

    def dominant_cluster(self, embedding):

        if not self.is_fitted:
            raise RuntimeError("Model is not trained.")

        probabilities = self.model.predict_proba(
            np.asarray(embedding).reshape(1, -1)
        )[0]

        cluster = np.argmax(probabilities)

        return int(cluster)
