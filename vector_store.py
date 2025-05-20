import faiss


class VectorStore:
    def __init__(self, dim, num_neighbors=32):
        self.index = faiss.IndexHNSWFlat(dim, num_neighbors)
        self.index.hnsw.efConstruction = 64
        self.metadata = []

    def add(self, embeddings, metadata):
        self.index.add(embeddings)
        self.metadata = metadata

    def search(self, query_embedding, top_k=8):
        D, I = self.index.search(query_embedding, top_k)
        return D[0], I[0]
