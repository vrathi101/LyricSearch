import os
import pickle
import faiss
import numpy as np
from dataset import LyricsDataset
from utils import get_chunks_by_id
from embedding import Embedder


DATASET_NAME = "brunokreiner/genius-lyrics"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 0
MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_DIR = "index_data"


def preprocess_lyrics(dataset_name, chunk_size, chunk_overlap):
    ds = LyricsDataset(dataset_name, chunk_size, chunk_overlap)
    chunked = ds.get_chunked_dataset()
    text_splitter = ds.splitter
    id_to_lyrics = ds.get_lyrics_dict()
    id_to_chunks = get_chunks_by_id(chunked)
    return chunked, text_splitter, id_to_lyrics, id_to_chunks


def generate_embeddings(model_name, chunked_dataset):
    embedder = Embedder(model_name)
    texts = [chunk["lyrics"] for chunk in chunked_dataset]
    embeddings = embedder.embed(texts)
    metadata = [(chunk["id"], chunk["part"]) for chunk in chunked_dataset]
    return embeddings, metadata


def build_faiss_index(embeddings, dim=384):
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 64
    index.add(embeddings)
    return index


def save_data(index, metadata, id_to_lyrics, id_to_chunks, chunked_dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(output_dir, "lyrics_index.faiss"))

    with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    with open(os.path.join(output_dir, "id_to_lyrics.pkl"), "wb") as f:
        pickle.dump(id_to_lyrics, f)

    with open(os.path.join(output_dir, "id_to_chunks.pkl"), "wb") as f:
        pickle.dump(id_to_chunks, f)

    with open(os.path.join(output_dir, "chunked_dataset.pkl"), "wb") as f:
        pickle.dump(chunked_dataset, f)


def main():
    chunked_dataset, text_splitter, id_to_lyrics, id_to_chunks = preprocess_lyrics(
        DATASET_NAME, CHUNK_SIZE, CHUNK_OVERLAP
    )

    embeddings, metadata = generate_embeddings(MODEL_NAME, chunked_dataset)

    index = build_faiss_index(np.array(embeddings))

    save_data(index, metadata, id_to_lyrics, id_to_chunks, chunked_dataset, OUTPUT_DIR)


if __name__ == "__main__":
    main()
