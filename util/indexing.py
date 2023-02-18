import faiss


def get_faiss_indexer():

    indexer = faiss.IndexFlatL2(25088) # features.shape[1]

    return indexer