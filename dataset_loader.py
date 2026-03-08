from sklearn.datasets import fetch_20newsgroups

def load_dataset():
    
    dataset = fetch_20newsgroups(
        remove=('headers', 'footers', 'quotes')
    )

    texts = dataset.data

    return texts