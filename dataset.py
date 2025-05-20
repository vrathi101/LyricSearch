from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter


class LyricsDataset:
    def __init__(self, dataset_name, chunk_size, chunk_overlap):
        self.dataset = load_dataset(dataset_name)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def get_lyrics_dict(self):
        return {entry["id"]: entry["lyrics"] for entry in self.dataset["train"]}

    def get_chunked_dataset(self):
        chunked = []
        for entry in self.dataset["train"]:
            chunks = self.splitter.split_text(entry["lyrics"])
            for i, chunk in enumerate(chunks):
                chunked.append({
                    "id": entry["id"],
                    "part": i,
                    "lyrics": chunk
                })
        return chunked
