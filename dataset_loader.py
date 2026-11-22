
from sklearn.datasets import fetch_20newsgroups


def load_dataset():
    print("Fetching 20 Newsgroups dataset...")

    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes")
    )

    texts = [
        text.strip()
        for text in dataset.data
        if text and text.strip()
    ]

    print(f"Loaded {len(texts)} documents.")

    return texts
