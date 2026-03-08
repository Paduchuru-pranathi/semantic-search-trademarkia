from collections import defaultdict
import numpy as np

def analyze_clusters(texts, embeddings, cluster_model):

    memberships = cluster_model.membership(embeddings)

    clusters = defaultdict(list)

    for i, probs in enumerate(memberships):

        cluster_id = np.argmax(probs)

        clusters[cluster_id].append(texts[i])

    print("\nCluster Summary\n")

    for cid, docs in clusters.items():

        print(f"\nCluster {cid} : {len(docs)} documents")

        for example in docs[:3]:
            print("-", example[:120].replace("\n"," "))