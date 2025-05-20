# LyricSearch

LyricSearch is an AI-powered web application that helps users generate unique lyrics and analyze the authenticity of existing lyrics. The application uses advanced natural language processing and machine learning techniques to provide intelligent lyrics generation and similarity analysis.

## Features

### 1. AI Lyrics Generation
- Generate unique lyrics based on your specifications
- Customize lyrics by providing:
  - Brief description
  - Theme(s)
  - Tone/Mood
  - Genre
- Powered by OpenAI's GPT-3.5 Turbo model

### 2. Lyrics Authenticity Analysis
- Submit your own lyrics for analysis
- Compare against a database of existing songs
- Get similarity scores and highlighted matches
- Identify potential similarities with existing songs

## Technical Stack

- **Backend**: Python with Flask
- **AI/ML**: 
  - OpenAI GPT-3.5 Turbo for lyrics generation
  - Sentence Transformers for text embeddings
  - FAISS for efficient similarity search
- **Frontend**: HTML, CSS with modern UI design
- **Data Storage**: FAISS index with pickle files for fast inference

## Prerequisites

- Python
- OpenAI API key
- Required Python packages (see Installation section)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LyricSearch.git
cd LyricSearch
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
   - Replace `YOUR_OPENAI_API_KEY` in `app.py` with your actual OpenAI API key

4. Build the search index:
```bash
python build_index.py
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to the local host URL

3. Choose between:
   - Generating AI lyrics
   - Analyzing existing lyrics
