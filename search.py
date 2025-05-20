import numpy as np
from utils import similarity_score


class LyricsSearchEngine:
    def __init__(self, vector_store, chunked_dataset, id_to_lyrics):
        self.vs = vector_store
        self.chunked_dataset = chunked_dataset
        self.id_to_lyrics = id_to_lyrics

    def search(self, query_embedding, top_k=15):
        D, I = self.vs.search(query_embedding, top_k)
        D = D.flatten()
        I = I.flatten()
        results = []
        for rank, idx in enumerate(I):
            chunk = self.chunked_dataset[idx]
            results.append({
                "id": chunk["id"],
                "part": chunk["part"],
                "chunked_lyrics": chunk["lyrics"],
                "total_lyrics": self.id_to_lyrics[chunk["id"]],
                "similarity": 1 / (1 + float(D[rank]))
            })
        return results

    def compute_overall_similarity(self, embedder, query, id_to_chunks, id_to_lyrics, similarity_results):
        unique_top_ids = list(set(elem["id"] for elem in similarity_results))
        scores_by_id = {}
        for lyric_id in unique_top_ids:
            chunks = id_to_chunks[lyric_id]
            scores_by_chunk = [
                (similarity_score(embedder, query, chunk), chunk) for chunk in chunks
            ]
            scores_by_chunk.sort(key=lambda x: x[0], reverse=True)
            mean_score = sum(score for score, _ in scores_by_chunk) / len(scores_by_chunk)
            scores_by_id[lyric_id] = (mean_score, scores_by_chunk, id_to_lyrics[lyric_id])
        return scores_by_id

    def filter_results(self, scores_by_id, threshold=0.5):
        unique_lyrics = set()
        unique_lyrics_start = set()
        filtered = []

        for val in scores_by_id.values():
            full_lyrics = val[2]
            lyrics_start = full_lyrics[:200].strip()

            if (
                val[0] >= threshold and
                full_lyrics not in unique_lyrics and
                lyrics_start not in unique_lyrics_start
            ):
                unique_lyrics.add(full_lyrics)
                unique_lyrics_start.add(lyrics_start)
                filtered.append(val)
     
        filtered.sort(key=lambda x: x[0], reverse=True)
        return filtered[:5]
