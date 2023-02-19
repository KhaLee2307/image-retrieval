import faiss


def get_faiss_indexer(shape):

    indexer = faiss.IndexFlatL2(shape) # features.shape[1]

    return indexer