import numpy as np
import openai
from system_prompt import SYSTEM_PROMPT


def similarity_score(embedder, text1, text2):
    emb1 = embedder.embed_single(text1)
    emb2 = embedder.embed_single(text2)
    return 1 / (1 + np.sum((emb1 - emb2)**2))


def get_chunks_by_id(chunked_dataset):
    id_to_chunks = {}
    for chunk in chunked_dataset:
        id_to_chunks.setdefault(chunk["id"], []).append(chunk["lyrics"])
    return id_to_chunks


def ai_lyrics(description, theme, tone, genre):
    description = description.strip() if description else "No description provided"
    theme = theme.strip() if theme else "No specific theme"
    tone = tone.strip() if tone else "No specific tone"
    genre = genre.strip() if genre else "No specific genre"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Description: {description}\n"
                f"Theme: {theme}\n"
                f"Tone: {tone}\n"
                f"Genre: {genre}"
            )},
        ],
    )
    return response.choices[0].message.content
