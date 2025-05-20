from flask import Flask, render_template, request
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import openai
from embedding import Embedder
from search import LyricsSearchEngine
from utils import ai_lyrics as generate_lyrics

openai.api_key = "YOUR_OPENAI_API_KEY"
DATASET_NAME = "brunokreiner/genius-lyrics"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 0
MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_DIR = "index_data"


def load_index_and_data():
    index = faiss.read_index(f"{OUTPUT_DIR}/lyrics_index.faiss")

    with open(f"{OUTPUT_DIR}/metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    with open(f"{OUTPUT_DIR}/id_to_lyrics.pkl", "rb") as f:
        lyrics_map = pickle.load(f)
    with open(f"{OUTPUT_DIR}/id_to_chunks.pkl", "rb") as f:
        chunks_map = pickle.load(f)
    with open(f"{OUTPUT_DIR}/chunked_dataset.pkl", "rb") as f:
        dataset = pickle.load(f)

    return index, meta, lyrics_map, chunks_map, dataset


def initialize_components():
    transformer = SentenceTransformer(MODEL_NAME)
    text_embedder = Embedder()
    return transformer, text_embedder


def get_chunk_positions(total_lyrics, top_chunks):
    positions = []
    for _, chunk_text in top_chunks:
        idx = total_lyrics.find(chunk_text)
        if idx != -1:
            positions.append((idx, idx + len(chunk_text)))
        else:
            positions.append(None)
    return positions


def create_highlighted_text(total_lyrics, positions_with_color):
    highlighted_lyrics = total_lyrics
    for (start, end), color, (score, _) in positions_with_color:
        tooltip_text = f"Similarity Score: {score:.2f}"
        span_html = (
            f'<span class="highlighted-text" '
            f'style="background-color:{color};" '
            f'title="{tooltip_text}">'
        )
        highlighted_lyrics = (
            highlighted_lyrics[:start]
            + span_html
            + highlighted_lyrics[start:end]
            + '</span>'
            + highlighted_lyrics[end:]
        )
    return highlighted_lyrics


def process_analysis_results(res):
    display_results = []
    colors = ["#ff4d4d", "#ffa500", "#ffff66"]

    for mean_score, scores_by_chunk, total_lyrics in res:
        top_chunks = scores_by_chunk[:3]
        positions = get_chunk_positions(total_lyrics, top_chunks)

        positions_with_color = [
            p for p in zip(positions, colors, top_chunks) if p[0]
        ]
        positions_with_color.sort(
            key=lambda x: x[0][0] if x[0] else -1, reverse=True
        )

        highlighted_lyrics = create_highlighted_text(
            total_lyrics, positions_with_color
        )

        display_results.append({
            "mean_score": mean_score,
            "highlighted_lyrics": highlighted_lyrics,
        })

    return display_results


app = Flask(__name__)
index, _, lyrics_map, chunks_map, dataset = load_index_and_data()
_, text_embedder = initialize_components()
search_engine = LyricsSearchEngine(index, dataset, lyrics_map)


@app.route("/")
def home():
    return render_template("main.html")


@app.route("/generate_ai_lyrics", methods=["GET", "POST"])
def generate_ai_lyrics():
    if request.method == "POST":
        description = request.form.get("description", "")
        themes = request.form.get("themes", "")
        tone = request.form.get("tone", "")
        genre = request.form.get("genre", "")
        generated_lyrics = generate_lyrics(
            description, themes, tone, genre)
        return render_template(
            "generate_ai_lyrics.html", message=generated_lyrics
        )

    return render_template("generate_ai_lyrics.html")


@app.route("/submit_original_lyrics", methods=["GET", "POST"])
def submit_original_lyrics():
    if request.method == "POST":
        lyrics = request.form.get("original_lyrics", "")
        if not lyrics.strip():
            return render_template(
                "submit_original_lyrics.html",
                message="Please enter some lyrics to analyze."
            )
        return render_template(
            "analyze_results.html", res=[],
            message="Analyzing lyrics..."
        )
    return render_template("submit_original_lyrics.html")


@app.route("/analyze_lyrics", methods=["POST"])
def analyze_lyrics():
    lyrics = request.form.get("original_lyrics", "")
    if not lyrics.strip():
        return render_template(
            "analyze_results.html", res=[], message="No lyrics submitted."
        )

    query_embedding = text_embedder.embed_single(lyrics).reshape(1, -1)
    similarity_results = search_engine.search(query_embedding)
    scores_by_id = search_engine.compute_overall_similarity(
        text_embedder, lyrics, chunks_map, lyrics_map, similarity_results
    )

    res = search_engine.filter_results(scores_by_id, threshold=0.5)
    if not res:
        return render_template(
            "analyze_results.html", res=[], message="This song is authentic!"
        )

    display_results = process_analysis_results(res)
    return render_template(
        "analyze_results.html", res=display_results, message=None
    )


if __name__ == "__main__":
    app.run(debug=True)
