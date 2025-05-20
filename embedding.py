import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

    def embed_single(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0]

    def mean_pool_embeddings(self, query, text_splitter):
        query_chunks = text_splitter.split_text(query)
        query_embeddings = []
        for chunk in query_chunks:
            chunk_embeddings = self.embed_single(chunk)
            query_embeddings.append(chunk_embeddings)
        query_embeddings = np.array(query_embeddings)
        mean_pooled_embeddings = np.mean(query_embeddings, axis=0)
        mean_pooled_embeddings = mean_pooled_embeddings.reshape(1, -1)
        return mean_pooled_embeddings
